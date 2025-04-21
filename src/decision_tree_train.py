import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from pycocotools.coco import COCO
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.autonotebook import tqdm
from models_loader import initialize_models

#Path for the files
CONFIG_PATH = "config/models_config.json"
COCO_ANN    = "coco/annotations/instances_train2017.json"
IMG_DIR     = Path("coco/train2017")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF          = transforms.ToTensor()
IOU_THRESH  = 0.5
CONF_THRESH = 0.5
N_SAMPLES   = 5000

#feature extraction function
def extract_image_features(info, coco):
    ann_ids = coco.getAnnIds(imgIds=info["id"])
    anns    = coco.loadAnns(ann_ids)
    areas   = [a["area"] for a in anns] if anns else [0]
    return {
        "num_objects": len(anns),
        "mean_box_area": float(np.mean(areas)),
        "num_categories": len({a["category_id"] for a in anns}),
        "ratio_hw": info["height"] / info["width"]
    }

#IoU computation function
def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB -xA) * max(0, yB - yA)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])- inter)
    return inter / union if union > 0 else 0

#COCO dataset loading
coco = COCO(COCO_ANN)
all_ids = coco.getImgIds()
img_ids = all_ids[:N_SAMPLES]
imgs = coco.loadImgs(img_ids)

#models initialization
models = initialize_models(CONFIG_PATH)
for m in models.values():
    m["model"] = m["model"].to(DEVICE).eval()

#image features extraction and model evaluation
records = []
for info in tqdm(imgs, desc="Images"):
    feats = extract_image_features(info, coco)
    gt      = coco.loadAnns(coco.getAnnIds(imgIds=info["id"]))
    gt_boxes = [[a["bbox"][0], a["bbox"][1],
                 a["bbox"][0] + a["bbox"][2],
                 a["bbox"][1] + a["bbox"][3]] for a in gt]

    img    = Image.open(IMG_DIR / info["file_name"]).convert("RGB")
    tensor = TF(img).unsqueeze(0).to(DEVICE)

    scores = {}
    for key, info_m in tqdm(models.items(), desc="Models", leave=False):
        t0 = time.time()
        with torch.no_grad():
            pred = info_m["model"](tensor)[0]
        t_inf = time.time() - t0

        boxes = pred["boxes"].cpu().numpy()
        confs = pred["scores"].cpu().numpy()

        tp = 0
        for gb in gt_boxes:
            if any((confs[i] >= CONF_THRESH
                    and compute_iou(gb, boxes[i]) >= IOU_THRESH)
                   for i in range(len(boxes))):
                tp += 1
        acc = tp / len(gt_boxes) if gt_boxes else 0.0
        scores[key] = acc / t_inf

    best = max(scores, key=scores.get)
    rec  = {**feats,
            **{f"score_{k}": v for k, v in scores.items()},
            "best_model": best}
    records.append(rec)

df = pd.DataFrame(records)
min_cnt = df["best_model"].value_counts().min()
balanced = pd.concat([
    df[df["best_model"] == m].sample(min_cnt, random_state=42)
    for m in df["best_model"].unique()
], ignore_index=True)

X = balanced[["num_objects", "mean_box_area", "num_categories", "ratio_hw"]]
y = balanced["best_model"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#Decision tree fitting
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)

#model results
y_pred = dt.predict(X_val)
print(classification_report(y_val, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))

#model saving
joblib.dump(dt, "models/model_selector_dt.joblib")