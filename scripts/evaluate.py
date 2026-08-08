import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    ConfusionMatrixDisplay
)

from ai.utils.config import PROCESSED_DATASET, MODEL_DIR, REPORT_DIR
from ai.models.dataset import DeepfakeSequenceDataset
from ai.models.deepfake_detector import DeepfakeDetector

# Hyperparameters
BATCH_SIZE = 4
SEQUENCE_LENGTH = 10
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model():
    print("=" * 60)
    print("STARTING EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # 1. Paths configuration
    metadata_csv = PROCESSED_DATASET / "metadata.csv"
    model_path = MODEL_DIR / "best_model.pth"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.exists():
        print(f"✖ Error: Metadata file does not exist at {metadata_csv}.")
        return
        
    if not model_path.exists():
        print(f"✖ Error: Trained model weights not found at {model_path}.")
        print("Please train the model first using: python -m scripts.train")
        return

    # 2. Image Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 3. Create Dataset and extract the exact same validation split (using seed 42)
    dataset = DeepfakeSequenceDataset(
        metadata_path=metadata_csv,
        dataset_dir=PROCESSED_DATASET,
        sequence_length=SEQUENCE_LENGTH,
        transform=transform
    )
    
    val_size = int(len(dataset) * VAL_SPLIT)
    if val_size == 0 and len(dataset) > 1:
        val_size = 1
    train_size = len(dataset) - val_size
    
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    print(f"Loaded {len(val_dataset)} validation videos.")

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )

    # 4. Load trained model
    device_name = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"
    print(f"Loading trained model weights to GPU device: {device_name}...")
    model = DeepfakeDetector(sequence_length=SEQUENCE_LENGTH).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    # 5. Inference loop
    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # 6. Compute metrics
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    
    rep = classification_report(
        all_labels, 
        all_preds, 
        target_names=["Real", "Fake"],
        zero_division=0
    )
    print(rep)
    
    # Save text report
    report_txt_path = REPORT_DIR / "classification_report.txt"
    with open(report_txt_path, "w") as f:
        f.write("AI DEEPFAKE DETECTOR - EVALUATION REPORT\n")
        f.write("=" * 45 + "\n\n")
        f.write(rep)
    print(f"✔ Saved text report to: {report_txt_path}")

    # 7. Generate and save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")
    
    cm_path = REPORT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"✔ Saved Confusion Matrix plot to: {cm_path}")

    # 8. Generate and save ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    
    roc_path = REPORT_DIR / "roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"✔ Saved ROC Curve plot to: {roc_path}")
    print("=" * 40)

if __name__ == "__main__":
    evaluate_model()
