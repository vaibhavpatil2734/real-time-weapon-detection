# train_handgun_yolo.py
# ============================================================
# YOLOv8 Handgun Fine-Tuning Script (GPU enabled, Windows-safe)
# ============================================================

from ultralytics import YOLO
import torch
import os

# --- CONFIG ---
DATA_YAML = "dataset_yolo/dataset.yaml"  # path to your dataset.yaml
PRETRAINED_MODEL = "runs/train/weapon_v2_safe_training444/weights/best.pt"  # previous model weights
EPOCHS = 50
IMG_SIZE = 416
BATCH_SIZE = 16
OUTPUT_DIR = "runs/train/handgun_yolo"
# --------------

def main():
    # Check for GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ GPU detected! Using {torch.cuda.get_device_name(0)} for training.")
    else:
        device = "cpu"
        print("⚠️ GPU not detected. Using CPU (slower).")

    # Create output directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load YOLOv8 model
    model = YOLO(PRETRAINED_MODEL)

    # Start training / fine-tune on your handgun dataset
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,           # automatically uses GPU if available
        project="runs/train",
        name="handgun_yolo5555",
        exist_ok=True,
        pretrained=True,
        workers=0  # safe option for Windows multiprocessing
    )

    print("✅ YOLOv8 training started! Check logs in runs/train/handgun_yolo/")

if __name__ == "__main__":
    main()
