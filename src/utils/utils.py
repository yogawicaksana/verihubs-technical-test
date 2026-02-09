from pathlib import Path
import torch
import gdown

def collate_fn(batch):
    """Custom collate function for dataloader"""
    return tuple(zip(*batch))

def save_model(model, output_dir, model_save_name='best_model.pth'):
    """Save the model state dictionary to a file"""
    model_path = Path(output_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path / model_save_name)
    print(f"Model saved to {model_path / model_save_name}")
    
def download_model_weights():
    """Download best model weights."""
    
    MODEL_ID = "1TixfeXe2NUNDVDGECY5YeVEobhTAefq9"  # ← REPLACE WITH YOUR FILE ID
    MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
    MODEL_PATH = Path("../../models/fasterrcnn_facemask/best_model.pth")
    
    if MODEL_PATH.exists():
        print(f"Weights already exist: {MODEL_PATH}")
        return
    
    print("Downloading pretrained weights (~150MB)...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)
    print(f"Downloaded to: {MODEL_PATH}")

if __name__ == "__main__":
    download_model_weights()