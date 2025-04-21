import json
from torchvision import models
from pathlib import Path

def load_models_config(config_path):
    config_path = Path(config_path)
    print(f"Loading models config from: {config_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_weights_and_functions():
    fasterrcnn_weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    retinanet_weights = models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
    maskrcnn_weights = models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    weights_dict = {
        "fasterrcnn_weights": fasterrcnn_weights,
        "retinanet_weights": retinanet_weights,
        "maskrcnn_weights": maskrcnn_weights,
    }
    load_function_dict = {
        "fasterrcnn_resnet50_fpn": models.detection.fasterrcnn_resnet50_fpn,
        "retinanet_resnet50_fpn": models.detection.retinanet_resnet50_fpn,
        "maskrcnn_resnet50_fpn": models.detection.maskrcnn_resnet50_fpn,
    }
    return weights_dict, load_function_dict

def initialize_models(config_path):
    models_config = load_models_config(config_path)
    weights_dict, load_function_dict = get_weights_and_functions()
    models_available = {}
    for model_key, config in models_config.items():
        load_func = load_function_dict[config["load_function"]]
        weight_obj = weights_dict[config["weights"]]
        model_obj = load_func(weights=weight_obj)
        models_available[model_key] = {
            "name": config["name"],
            "description": config["description"],
            "model": model_obj,
            "categories": weight_obj.meta["categories"]
        }
    return models_available