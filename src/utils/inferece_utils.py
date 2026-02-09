from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw
import os
import json
import torch
import numpy as np
import albumentations as A

def predict(model, image_path, device, threshold=0.5):
    """Run inference with val_transforms and return scaled-back bboxes for original image"""
    model.eval()
    
    # Load original image (save dims for scaling back)
    orig_img = Image.open(image_path).convert("RGB")
    orig_h, orig_w = orig_img.size[1], orig_img.size[0]  # PIL: (w,h)
    orig_array = np.array(orig_img)
    
    # Apply val_transforms
    val_transform = A.Compose([
        A.Resize(height=800, width=800),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = val_transform(image=orig_array)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)[0]  # Single image
    
    # Get predictions in transformed space
    boxes = outputs['boxes'].cpu()  # [N, 4] xyxy 0-800
    labels = outputs['labels'].cpu()
    scores = outputs['scores'].cpu()
    
    # Filter
    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    # Scale bboxes back to original image size
    scale_x = orig_w / 800.0
    scale_y = orig_h / 800.0
    boxes[:, [0, 2]] *= scale_x  # x1, x2
    boxes[:, [1, 3]] *= scale_y  # y1, y2
    
    return orig_array, boxes, labels, scores  # orig_array for drawing

def draw_predictions(image_array, boxes_pred, labels_pred, scores_pred, true_annotations=None, class_names=None):
    """Draw TRUE (left) + PREDICTED (right) side-by-side with small gap"""
    if class_names is None:
        class_names = {'with_mask': 1, 'mask_incorrect': 2, 'without_mask': 3}
    
    h, w = image_array.shape[:2]
    gap = 20  # Small 20px space between
    
    # Canvas: original + gap + original
    combined_w = w * 2 + gap
    combined = np.zeros((h, combined_w, 3), dtype=np.uint8)
    combined[:, :w] = image_array           # Left
    combined[:, w+gap:] = image_array       # Right (shifted by gap)
    
    # === LEFT: TRUE ===
    true_img = Image.fromarray(combined[:, :w])
    draw_true = ImageDraw.Draw(true_img)
    
    if true_annotations:
        for ann in true_annotations:
            name = ann['name']
            if name not in class_names: continue
            
            bbox = ann['bndbox']
            x1, y1, x2, y2 = map(int, [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])
            
            color = 'green' if name == 'with_mask' else 'yellow' if name == 'mask_incorrect' else 'red'
            draw_true.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw_true.text((x1, y1-20), name[:12], fill=color)
    
    combined[:, :w] = np.array(true_img)
    
    # === RIGHT: PRED ===
    right_start = w + gap
    pred_img = Image.fromarray(combined[:, right_start:])
    draw_pred = ImageDraw.Draw(pred_img)
    
    for box, label, score in zip(boxes_pred, labels_pred, scores_pred):
        label_id = label.item()
        if label_id == 0 or label_id not in class_names.values(): continue
        
        pred_name = next(name for name, lid in class_names.items() if lid == label_id)
        color = 'green' if label_id == 1 else 'red' if label_id == 3 else 'yellow'
        
        x1, y1, x2, y2 = box.numpy().astype(int)
        draw_pred.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw_pred.text((x1, y1-20), f"{pred_name[:8]} {score:.1f}", fill=color)
    
    combined[:, right_start:] = np.array(pred_img)
    
    return combined

def save_img(image_array, path):
    """Save image array to specified path"""
    img = Image.fromarray(image_array)
    # create dir if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def save_json_predictions(boxes, labels, scores, json_path):
    """Save predictions to a JSON file"""
    
    predictions = []
    for box, label, score in zip(boxes, labels, scores):
        pred = {
            'box': box.numpy().tolist(),
            'label': label.item(),
            'score': score.item()
        }
        predictions.append(pred)
    
    # create dir if not exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    with open(json_path, 'w') as f:
        json.dump(predictions, f, indent=4)

def load_annotations(xml_path):
    """Parse Pascal VOC XML to annotations list"""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    
    for obj in root.findall('object'):
        ann = {
            'name': obj.find('name').text,
            'bndbox': {
                'xmin': obj.find('bndbox/xmin').text,
                'ymin': obj.find('bndbox/ymin').text,
                'xmax': obj.find('bndbox/xmax').text,
                'ymax': obj.find('bndbox/ymax').text
            }
        }
        annotations.append(ann)
    
    return annotations