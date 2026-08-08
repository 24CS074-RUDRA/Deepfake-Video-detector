import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path
import copy

from ai.utils.config import PROCESSED_DATASET, MODEL_DIR
from ai.models.dataset import DeepfakeSequenceDataset
from ai.models.deepfake_detector import DeepfakeDetector

# Hyperparameters
BATCH_SIZE = 4
SEQUENCE_LENGTH = 10
EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    print("=" * 60)
    print("STARTING TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # 1. Paths configuration
    metadata_csv = PROCESSED_DATASET / "metadata.csv"
    
    if not metadata_csv.exists():
        print(f"✖ Error: Metadata file does not exist at {metadata_csv}.")
        print("Please run the preprocessing pipeline first using: python -m scripts.preprocess")
        return

    # 2. Set up Image transforms (Standard ImageNet Normalization for EfficientNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 3. Create Dataset and Split into Train/Val sets
    dataset = DeepfakeSequenceDataset(
        metadata_path=metadata_csv,
        dataset_dir=PROCESSED_DATASET,
        sequence_length=SEQUENCE_LENGTH,
        transform=transform
    )
    
    if len(dataset) == 0:
        print("✖ Error: The processed dataset is empty. Check your face detection step output.")
        return
        
    val_size = int(len(dataset) * VAL_SPLIT)
    if val_size == 0 and len(dataset) > 1:
        val_size = 1
    train_size = len(dataset) - val_size
    
    # Use generator with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    print(f"Train samples (videos): {len(train_dataset)}")
    print(f"Validation samples (videos): {len(val_dataset)}")

    # 4. Set up DataLoaders
    # Note: num_workers=0 is recommended on Windows to prevent multiprocessing issues.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )

    # 5. Initialize Model, Loss Function, and Optimizer
    device_name = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"
    print(f"\nInitializing DeepfakeDetector model on GPU device: {device_name}...")
    model = DeepfakeDetector(sequence_length=SEQUENCE_LENGTH).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-4
    )
    
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # 6. Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 30)
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_corrects += torch.sum(preds == labels.data)
            train_total += inputs.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_corrects.double() / train_total
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                logits = model(inputs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_corrects += torch.sum(preds == labels.data)
                val_total += inputs.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_corrects.double() / val_total
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")
        
        # Save model if validation loss improves
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            save_path = MODEL_DIR / "best_model.pth"
            torch.save(best_model_wts, save_path)
            print(f"⭐ Best model saved to: {save_path}")
            
    print("\n🎉 Training completed successfully.")

if __name__ == "__main__":
    train_model()
