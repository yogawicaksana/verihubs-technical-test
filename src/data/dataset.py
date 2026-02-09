from PIL import Image
from pathlib import Path
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import torch
import os 
import torchvision.transforms as T
import xml.etree.ElementTree as ET

class FaceMaskDataset(torch.utils.data.Dataset):
    """Face Mask Detection Dataset in Pascal VOC format"""
    
    def __init__(self, root_dir, transforms=None):
        """
        Args:
            root_dir: Root directory containing 'images' and 'annotations' folders
            transforms: Optional transforms to be applied on images
        """
        self.root_dir = Path(root_dir)
        if transforms is None:
            self.transforms = self.get_default_transforms()
        else:
            self.transforms = transforms
        self.imgs_dir = self.root_dir / 'images'
        self.anns_dir = self.root_dir / 'annotations'
        
        # Get all image files
        self.imgs = sorted([f for f in os.listdir(self.imgs_dir) if f.endswith('.png') or f.endswith('.jpg')])
        print(f"Loaded {len(self.imgs)} images from {self.imgs_dir}")
        # Class mapping (background is 0)
        self.class_to_idx = {
            'with_mask': 1,
            'without_mask': 2,
            'mask_weared_incorrect': 3
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.imgs)
    
    def parse_xml(self, xml_path):
        """Parse Pascal VOC XML annotation file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # Get class label
            class_name = obj.find('name').text
            if class_name not in self.class_to_idx:
                continue
            
            label = self.class_to_idx[class_name]
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        return boxes, labels
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.imgs[idx]
        img_path = self.imgs_dir / img_name
        img = Image.open(img_path).convert('RGB')
        
        # Load annotation
        xml_name = img_name.replace('.png', '.xml').replace('.jpg', '.xml')
        xml_path = self.anns_dir / xml_name
        
        boxes, labels = self.parse_xml(xml_path)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Apply transforms
        if hasattr(self, 'transforms') and self.transforms is not None:
            if 'albumentations' in str(type(self.transforms)):  # Albumentations
                transformed = self.transforms(
                    image=np.array(img),
                    bboxes=target['boxes'].numpy(),
                    labels=target['labels'].numpy()
                )
                img = transformed['image']
                target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            else:  # Torchvision transforms
                img = self.transforms(img)
        else:
            # Raw tensor (no transforms)
            img = T.ToTensor()(img)
        
        return img, target

    def __get_all_labels__(self):
        """Extract all labels for stratification"""
        labels = []
        for i in range(len(self)):
            _, target = self[i]
            labels.extend(target['labels'].numpy())
        return np.array(labels)

    def get_default_transforms(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet
        ])

    @staticmethod
    def get_train_transforms():
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.LongestMaxSize(max_size=800, p=1.0), 
            A.PadIfNeeded(min_height=800, min_width=800, border_mode=0, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.1, clip=True))

    @staticmethod
    def get_val_transforms():
        return A.Compose([
            A.Resize(height=800, width=800),  # ✅ Same size for val
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.1, clip=True))
