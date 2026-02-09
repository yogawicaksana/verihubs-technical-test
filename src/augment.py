import os
import cv2
import numpy as np
import xmltodict
import argparse


def read_all_xmls(xml_list, xml_dir):
    """Read all XML annotation files from a directory."""
    final_list = []
    for xml_file in xml_list:
        with open(os.path.join(xml_dir, xml_file), 'r') as f:
            xml_content = f.read()
            xml_dict = xmltodict.parse(xml_content)
            final_list.append(xml_dict)
    return final_list


def parse_annotation(data_dict, img_path):
    """
    Parse annotation from XML dict and load image.
    
    Returns:
        img: loaded image with cv2 (BGR)
        boxes: list of [x_min, y_min, x_max, y_max] (Pascal VOC style)
        labels: list of str (class names)
    """
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape

    annotation = data_dict["annotation"]
    obj = annotation["object"]
    if not isinstance(obj, list):
        obj = [obj]

    boxes, labels = [], []
    for o in obj:
        bbox = o["bndbox"]
        x_min = int(bbox["xmin"])
        y_min = int(bbox["ymin"])
        x_max = int(bbox["xmax"])
        y_max = int(bbox["ymax"])

        # clip to image size
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(o["name"])

    return img, boxes, labels


def make_mosaic(indices, all_data, img_dir, mosaic_size=640):
    """
    Create a 2x2 mosaic from 4 images with proper box transformation.
    
    Args:
        indices: list of 4 indices to select from all_data
        all_data: list of annotation dictionaries
        img_dir: directory containing images
        mosaic_size: size of each cell in the mosaic (default 640)
    
    Returns:
        mosaic: 2x2 mosaic image (mosaic_size*2 x mosaic_size*2)
        merged_boxes: list of boxes in mosaic coordinates
        merged_labels: list of corresponding labels
    """
    result = []
    for i in indices:
        img_fname = all_data[i]["annotation"]["filename"]
        img_path = os.path.join(img_dir, img_fname)
        img, boxes, labels = parse_annotation(all_data[i], img_path)

        # 1: resize to mosaic cell size
        old_h, old_w = img.shape[:2]
        img = cv2.resize(img, (mosaic_size, mosaic_size))

        # 2: transform each box to cell coords (mosaic_size x mosaic_size)
        cell_boxes = []
        cell_labels = []
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box

            # scale from original image to mosaic_size x mosaic_size cell
            x_min = int(x_min * (mosaic_size / old_w))
            y_min = int(y_min * (mosaic_size / old_h))
            x_max = int(x_max * (mosaic_size / old_w))
            y_max = int(y_max * (mosaic_size / old_h))

            # CLIP to cell bounds (0, mosaic_size)
            x_min = max(0, min(mosaic_size, x_min))
            y_min = max(0, min(mosaic_size, y_min))
            x_max = max(0, min(mosaic_size, x_max))
            y_max = max(0, min(mosaic_size, y_max))

            # FILTER degenerate boxes (width/height <= 0)
            if x_max > x_min and y_max > y_min:
                cell_boxes.append([x_min, y_min, x_max, y_max])
                cell_labels.append(label)

        result.append((img, cell_boxes, cell_labels))

    # build 2×2 mosaic
    s = mosaic_size
    mosaic = np.full((2*s, 2*s, 3), 114, dtype=np.uint8)
    merged_boxes = []
    merged_labels = []

    # offsets: [top‑left, top‑right, bottom‑left, bottom‑right]
    for k, (img, boxes, labels) in enumerate(result):
        if k == 0:  # top left
            x1, y1, x2, y2 = 0, 0, s, s
        elif k == 1:  # top right
            x1, y1, x2, y2 = s, 0, 2*s, s
        elif k == 2:  # bottom left
            x1, y1, x2, y2 = 0, s, s, 2*s
        else:  # bottom right
            x1, y1, x2, y2 = s, s, 2*s, 2*s

        mosaic[y1:y2, x1:x2] = img

        # shift boxes to mosaic coordinates
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            x_min += x1
            y_min += y1
            x_max += x1
            y_max += y1
            merged_boxes.append([x_min, y_min, x_max, y_max])
            merged_labels.append(label)

    return mosaic, merged_boxes, merged_labels


def save_mosaic_annotation(mosaic_dict, out_xml_path):
    """Save mosaic annotation as XML file."""
    # dict => XML string
    xml_str = xmltodict.unparse(mosaic_dict, pretty=True)

    # create dir if not exists
    os.makedirs(os.path.dirname(out_xml_path), exist_ok=True)
    # write to file
    with open(out_xml_path, "w") as f:
        f.write(xml_str)


def filter_data_by_class(all_data, target_class, exclude_classes=None):
    """
    Filter data to get indices containing target_class and optionally excluding other classes.
    
    Args:
        all_data: list of annotation dictionaries
        target_class: class name to filter for
        exclude_classes: list of class names to exclude (optional)
    
    Returns:
        List of indices matching the criteria
    """
    filtered_idx = []
    if exclude_classes is None:
        exclude_classes = []
    
    for i in range(len(all_data)):
        obj = all_data[i]['annotation']['object']
        if isinstance(obj, dict):
            obj = [obj]
        elif not isinstance(obj, list):
            continue

        # Check if target class is present
        has_target = any(isinstance(j, dict) and j.get('name') == target_class for j in obj)
        
        # Check if any exclude classes are present
        has_excluded = any(
            isinstance(j, dict) and j.get('name') in exclude_classes 
            for j in obj
        )
        
        if has_target and not has_excluded:
            filtered_idx.append(i)
    
    return filtered_idx


def generate_mosaics(data_subset, all_data, img_dir, out_dir, num_mosaics, 
                     start_idx=0, mosaic_size=640, seed=42):
    """
    Generate mosaic augmentations for a specific data subset.
    
    Args:
        data_subset: subset of all_data to use for mosaic generation
        all_data: full list of annotation dictionaries
        img_dir: directory containing source images
        out_dir: directory to save generated images
        num_mosaics: number of mosaics to generate
        start_idx: starting index for naming mosaics
        mosaic_size: size of each cell in the mosaic
        seed: random seed for reproducibility
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir.replace('images', 'annotations'), exist_ok=True)
    
    np.random.seed(seed)

    for i in range(num_mosaics):
        # pick 4 images from the subset
        all_indices = list(range(len(data_subset)))
        chosen_idx = np.random.choice(all_indices, size=4, replace=True)

        mosaic_img, boxes, labels = make_mosaic(
            indices=chosen_idx,
            all_data=data_subset,
            img_dir=img_dir,
            mosaic_size=mosaic_size
        )

        idx_to_save = start_idx + i
        out_img_path = os.path.join(out_dir, f"mosaic_{idx_to_save}.jpg")
        cv2.imwrite(out_img_path, mosaic_img)

        # create annotation dict
        mosaic_dict = {
            "annotation": {
                "folder": "mosaic",
                "filename": f"mosaic_{idx_to_save}.jpg",
                "size": {
                    "width": mosaic_size * 2,
                    "height": mosaic_size * 2,
                    "depth": 3,
                },
                "segmented": "0",
                "object": []
            }
        }

        obj_list = []
        for box, label in zip(boxes, labels):
            obj = {
                "name": label,
                "pose": "Unspecified",
                "truncated": "0",
                "occluded": "0",
                "difficult": "0",
                "bndbox": {
                    "xmin": int(box[0]),
                    "ymin": int(box[1]),
                    "xmax": int(box[2]),
                    "ymax": int(box[3]),
                }
            }
            obj_list.append(obj)

        mosaic_dict["annotation"]["object"] = obj_list

        out_xml = out_img_path.replace(".jpg", ".xml").replace('images', 'annotations')
        save_mosaic_annotation(mosaic_dict, out_xml)
    
    print(f"Generated {num_mosaics} mosaics (indices {start_idx} to {start_idx + num_mosaics - 1})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Augmentation via Mosaic for Face Mask Detection')
    parser.add_argument('--data_dir', type=str, default='../data/',
                        help='Path to data directory')
    parser.add_argument('--annotations_dir', type=str, default=None,
                        help='Path to annotations directory (default: data_dir/annotations)')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Path to images directory (default: data_dir/images)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for augmented images (default: images_dir)')
    parser.add_argument('--mosaic_size', type=int, default=640,
                        help='Size of each cell in the mosaic (default: 640)')
    parser.add_argument('--num_incorrect_mask', type=int, default=300,
                        help='Number of mosaics to generate for mask_weared_incorrect class')
    parser.add_argument('--num_without_mask', type=int, default=1000,
                        help='Number of mosaics to generate for without_mask class')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()

    # Set default paths
    annotations_dir = args.annotations_dir or os.path.join(args.data_dir, 'annotations')
    images_dir = args.images_dir or os.path.join(args.data_dir, 'images')
    output_dir = args.output_dir or images_dir

    # Read all XML annotations
    print("Reading annotations...")
    xml_list = os.listdir(annotations_dir)
    all_data = read_all_xmls(xml_list, annotations_dir)
    print(f"Loaded {len(all_data)} annotations")

    # Filter data by class
    print("\nFiltering data by class...")
    mask_weared_incorrect_idx = filter_data_by_class(
        all_data, 
        target_class='mask_weared_incorrect',
        exclude_classes=['with_mask', 'without_mask']
    )
    
    without_mask_idx = filter_data_by_class(
        all_data,
        target_class='without_mask',
        exclude_classes=['with_mask', 'mask_weared_incorrect']
    )
    
    print(f"Found {len(mask_weared_incorrect_idx)} images with mask_weared_incorrect only")
    print(f"Found {len(without_mask_idx)} images with without_mask only")

    # Create subsets
    mask_weared_incorrect_data = [all_data[i] for i in mask_weared_incorrect_idx]
    without_mask_data = [all_data[i] for i in without_mask_idx]

    # Generate mosaics for mask_weared_incorrect
    print(f"\nGenerating {args.num_incorrect_mask} mosaics for mask_weared_incorrect...")
    generate_mosaics(
        data_subset=mask_weared_incorrect_data,
        all_data=all_data,
        img_dir=images_dir,
        out_dir=output_dir,
        num_mosaics=args.num_incorrect_mask,
        start_idx=0,
        mosaic_size=args.mosaic_size,
        seed=args.seed
    )

    # Generate mosaics for without_mask
    print(f"\nGenerating {args.num_without_mask} mosaics for without_mask...")
    generate_mosaics(
        data_subset=without_mask_data,
        all_data=all_data,
        img_dir=images_dir,
        out_dir=output_dir,
        num_mosaics=args.num_without_mask,
        start_idx=args.num_incorrect_mask,
        mosaic_size=args.mosaic_size,
        seed=args.seed
    )

    print("\n✓ Augmentation complete!")
    print(f"Total mosaics generated: {args.num_incorrect_mask + args.num_without_mask}")
    print(f"Images saved to: {output_dir}")
    print(f"Annotations saved to: {output_dir.replace('images', 'annotations')}")
