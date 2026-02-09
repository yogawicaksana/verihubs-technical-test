from collections import defaultdict
import random

def stratified_split(dataset, val_ratio=0.2, seed=42):
    """
    Stratified split: val = non-mosaic only, train = everything (including mosaic).
    """
    random.seed(seed)
    
    # Group indices by most frequent label per image
    label_to_indices = defaultdict(list)
    
    for i in range(len(dataset)):
        img_name = dataset.imgs[i]
        
        # Skip mosaic images for validation (only use original)
        if img_name.startswith('mosaic_'):
            continue
        
        _, target = dataset[i]
        most_freq = target['labels'].bincount().argmax().item()
        label_to_indices[most_freq].append(i)
    
    # Stratified split (only original images)
    train_idx, val_idx = [], []
    for label_indices in label_to_indices.values():
        n_val = max(1, int(len(label_indices) * val_ratio))
        val_sample = random.sample(label_indices, n_val)
        train_sample = [idx for idx in label_indices if idx not in val_sample]
        train_idx.extend(train_sample)
        val_idx.extend(val_sample)
    
    # Add ALL mosaic indices to train
    for i in range(len(dataset)):
        if dataset.imgs[i].startswith('mosaic_'):
            train_idx.append(i)
    
    # Shuffle
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    
    print(f"Train: {len(train_idx)} total ({len([i for i in train_idx if dataset.imgs[i].startswith('mosaic_')])} mosaic)")
    print(f"Val:   {len(val_idx)} original only")
    
    return train_idx, val_idx
