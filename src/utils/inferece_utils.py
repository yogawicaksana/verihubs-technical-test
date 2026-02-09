from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as T

def predict(model, image_path, device, threshold=0.5):
    """Run inference on a single image and return boxes, labels, and scores above the threshold"""
    model.eval()
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model([img_tensor])
    
    # Process outputs
    outputs = outputs[0]
    boxes = outputs['boxes'].cpu()
    labels = outputs['labels'].cpu()
    scores = outputs['scores'].cpu()
    
    # Filter by threshold
    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    return img, boxes, labels, scores

def visualize_predictions(img, boxes, labels, scores, class_names=None, save_path=None):
    """Visualize predictions on the image"""
    if class_names is None:
        class_names = {1: 'with_mask', 2: 'without_mask', 3: 'mask_weared_incorrect'}

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    colors = {1: 'green', 2: 'red', 3: 'orange'}

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        
        color = colors.get(label.item(), 'blue')
        rect = plt.Rectangle((x1, y1), width, height,
                             fill=False, color=color, linewidth=2)
        ax.add_patch(rect)
        
        class_name = class_names.get(label.item(), 'Unknown')
        label_text = f"{class_name}: {score:.2f}"
        ax.text(x1, y1-5, label_text,
               bbox={'facecolor': color, 'alpha': 0.7},
               fontsize=10, color='white')

    plt.axis('off')
    if save_path:
        # create dir if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()