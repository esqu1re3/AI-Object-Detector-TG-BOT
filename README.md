# AI Object Detection Model Selector

An intelligent system that automatically selects the optimal object detection model for each image using a Decision Tree classifier. Built with PyTorch and deployed as a Telegram bot, it integrates three state-of-the-art detectors—Faster R‑CNN, RetinaNet, and Mask R‑CNN (all with ResNet50 + FPN)—and provides a simple interface for real‑time object detection.

## 🚀 Quick Start (under 10 minutes)

1. **Clone the repository**
   ```bash
   git clone https://github.com/esqu1re3/AI-Object-Detector-TG-BOT.git
   cd AI-Object-Detector-TG-BOT
   ```

2. **Create & activate a Python virtual environment**
   ```bash
   python3 -m venv venv      # Linux/macOS
   source venv/bin/activate  # Linux/macOS
   python -m venv venv       # Windows
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the bot**
   - Create a new bot via BotFather on Telegram and copy its token.
   - Edit `config/settings.yaml` with the following example:
     ```yaml
     telegram_bot_token: "<YOUR_TOKEN>"
     model_config_file: "models_config.json"
     font_path: "comic.ttf"
     bot_polling_timeout: 10
     logging_level: "INFO"
     ```

5. **Run the Telegram bot**
   ```bash
   python3 bot.py # Linux/macOS
   python bot.py  # Windows
   ```

6. **Test the bot**
   - Open Telegram and send `/start` to your bot.
   - Upload any image.
   - You’ll receive back the same image annotated with detected objects and a caption indicating which model was used.

---

## 📦 Project Structure

```
project-root/
├── config/
│   ├── models_config.json
│   └── settings.yaml
│
├── models/
│   ├── model_selector_dt.joblib
│   ├── predictions_fasterrcnn.json
│   ├── predictions_maskrcnn.json
│   └── predictions_retinanet.json
│
├── coco/
│   ├── annotations/
│   │   ├── captions_train2017.json
│   │   ├── captions_val2017.json
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   ├── person_keypoints_train2017.json
│   │   └── person_keypoints_val2017.json
│   ├── test2017/
│   ├── train2017/
│   └── val2017/
│
├── models_test/
│   └── models_test.ipynb
│
├── src/
│   ├── __pycache__/
│   ├── decision_tree_train.py
│   ├── models_loader.py
│   └── utils.py
│
├── .gitignore
├── bot.py
├── README.md
└── requirements.txt
```

---

If you want to train the selector yourself, download the COCO 2017 dataset:
https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset?resource=download

After downloading, your directory will look like:

```
project-root/
├── coco/
│   ├── annotations/
│   │   ├── captions_train2017.json
│   │   ├── captions_val2017.json
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   ├── person_keypoints_train2017.json
│   │   └── person_keypoints_val2017.json
│   ├── test2017/
│   ├── train2017/
│   └── val2017/
│
├── config/
│   ├── models_config.json
│   └── settings.yaml
│
├── models/
│   ├── model_selector_dt.joblib
│   ├── predictions_fasterrcnn.json
│   ├── predictions_maskrcnn.json
│   └── predictions_retinanet.json
│
├── models_test/
│   └── models_test.ipynb
│
├── src/
│   ├── __pycache__/
│   ├── decision_tree_train.py
│   ├── models_loader.py
│   └── utils.py
│
├── .gitignore
├── bot.py
├── README.md
└── requirements.txt
```

---

## 🛠️ Training the Model Selector

Once the COCO files are in place, train the Decision Tree selector:
```bash
python src/decision_tree_train.py --config config/models_config.json \
                                  --annotations coco/annotations/instances_train2017.json \
                                  --images-dir coco/train2017 \
                                  --output models/model_selector_dt.joblib
```  
This script will:
- Extract image features from the first 5 000 train images
- Evaluate each detector’s speed and accuracy
- Fit a Decision Tree on these features
- Save the trained selector for use by the bot

---

## 📚 Credits

- **Frameworks:** PyTorch / TorchVision, scikit-learn, pyTelegramBotAPI
- **Dataset:** COCO (Microsoft Common Objects in Context)

*Developed for the Introduction to Artificial Intelligence course at AUCA, Spring 2025.*