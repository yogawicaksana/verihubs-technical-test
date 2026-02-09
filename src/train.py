from data.dataset import FaceMaskDataset
from data.transform import stratified_split
from models.faster_rcnn import get_model
from utils.utils import collate_fn, save_model
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import mlflow
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on Face Mask Detection Dataset')
    parser.add_argument('--data_dir', type=str, default='../data/', help='Path to data directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    # Export-related args
    parser.add_argument('--output_dir', type=str, default='../models/fasterrcnn_facemask',
                        help='Directory to save trained model')
    parser.add_argument('--model_save_name', type=str, default='best_model.pth',
                        help='Filename for saving the trained model')
    
    # Training-related args
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes (including background)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')


    args = parser.parse_args()

    # logging setup
    mlflow.set_experiment("fasterrcnn-facemask-detection-frcnnv2")
    mlflow.start_run(run_name="fasterrcnn-facemask-training")
    mlflow.log_params(vars(args))

    dataset = FaceMaskDataset(args.data_dir, transforms=None)

    # get only first 100 for debug
    # dataset = torch.utils.data.Subset(dataset, np.arange(100))

    # Data split (stratified)
    print("Performing stratified train-val split...")
    train_idx, val_idx = stratified_split(dataset, val_ratio=0.2, seed=42)

    print("Generate training dataset...")
    train_dataset = torch.utils.data.Subset(
        FaceMaskDataset(args.data_dir, transforms=FaceMaskDataset.get_train_transforms()), 
        train_idx
    )
    
    print("Generate validation dataset...")
    val_dataset = torch.utils.data.Subset(
        FaceMaskDataset(args.data_dir, transforms=FaceMaskDataset.get_val_transforms()), 
        val_idx
    )
    
    # implement weighted random sampler to address class imbalance
    image_weights = []
    for i in range(len(train_dataset)):
        _, target = train_dataset[i]
        labels = set(target["labels"].tolist())

        w = 1.0
        if 2 in labels:      # without_mask
            w *= 3.0
        if 3 in labels:      # mask_weared_incorrect
            w *= 5.0         # upweight hardest minority
        image_weights.append(w)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(image_weights),
        num_samples=len(train_dataset),  # or larger if you want effective oversampling
        replacement=True,
    )

    print(f"Total dataset size: {len(dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    model = get_model(args.num_classes)
    model.to(args.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }

    metric = MeanAveragePrecision(class_metrics=True)

    best_map = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Training")
        for images, targets in pbar:
            images = [img.to(args.device) for img in images]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            pbar.set_postfix({'loss': losses.item()})
            mlflow.log_metric("train_loss", losses.item(), step=epoch)

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # Validation loss (add after train_loss calculation)
        model.eval()
        val_loss = 0.0
        metric.reset()

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Validation")
            for images, targets in pbar_val:
                images = [img.to(args.device) for img in images]
                targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
                predictions = model(images)

                model.train()
                val_loss_dict = model(images, targets)
                model.eval()

                losses = sum(loss for loss in val_loss_dict.values())
                val_loss += losses.item()
                mlflow.log_metric("val_loss", losses.item(), step=epoch)

                # Convert targets to torchmetrics format
                target_dicts = []
                for tgt in targets:
                    target_dicts.append({
                        'boxes': tgt['boxes'],
                        'labels': tgt['labels']
                    })
                
                metric.update(predictions, target_dicts)
                # Show progress (use mAP or skip)
                current_metric = metric.compute()
                pbar_val.set_postfix({'mAP': f"{current_metric['map']:.3f}"})

        print(f"mAP: {current_metric['map']:.4f} | mAP50: {current_metric['map_50']:.4f} | mAP75: {current_metric['map_75']:.4f}")

        # scheduler step
        lr_scheduler.step(current_metric['map'])
        
        # log metrics to mlflow
        mlflow.log_metric("epoch_train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("epoch_val_loss", val_loss / len(val_loader), step=epoch)
        mlflow.log_metric("val_map", current_metric['map'], step=epoch)
        mlflow.log_metric("val_map50", current_metric['map_50'], step=epoch)
        mlflow.log_metric("val_map75", current_metric['map_75'], step=epoch)
        # log per class metrics
        for i in range(1, args.num_classes):
            mlflow.log_metric(f"val_map_class_{i}", current_metric['map_per_class'][i-1].item(), step=epoch)
        
        # ---------- EARLY STOP + SAVE BEST ----------
        if current_metric['map'] > best_map:
            best_map = current_metric['map']
            patience_counter = 0                  # reset patience
            print(f"New best mAP: {best_map:.4f}. Saving model...")
            save_model(model, output_dir=args.output_dir, model_save_name=args.model_save_name)
        else:
            patience_counter += 1
            print(f"mAP did not improve from {best_map:.4f} (patience {patience_counter}/{patience})")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    mlflow.end_run()