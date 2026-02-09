from models.faster_rcnn import get_model
from utils.inferece_utils import load_annotations, predict, draw_predictions, save_img, save_json_predictions
from utils.utils import download_model_weights
import argparse
import torch 
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with Faster R-CNN on Face Mask Detection Dataset')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes (including background)')
    parser.add_argument('--model_dir', type=str, default='../models/fasterrcnn_facemask',
                        help='Directory containing model checkpoint')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Checkpoint file to load (.pt)')
    parser.add_argument('--image', type=str, default='../data/test/images/maksssksksss803.png',
                        help='Path to single image for inference')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images for batch inference')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for single image visualization')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Output directory for batch inference results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference (cuda or cpu)')
    args = parser.parse_args()


    # Download weights if not existing
    download_model_weights()

    # Load model and its best checkpoint
    model = get_model(args.num_classes)
    checkpoint = torch.load(f"{args.model_dir}/{args.checkpoint}", map_location=args.device)
    model.load_state_dict(checkpoint)

    model.to(args.device)

    model.eval()
    print(f"Model loaded from {args.model_dir}/{args.checkpoint} and ready for inference on {args.device}")

    try:
        true_labels = load_annotations(args.image.replace('images', 'annotations').replace('.png', '.xml'))
    except:
        true_labels = None

    # Run inference
    if args.image:
        # Single image inference
        print(f"\nRunning inference on {args.image}")
        img_array, boxes, labels, scores = predict(model, args.image, args.device, args.threshold)

        # Visualize
        if true_labels:
            result_img = draw_predictions(img_array, boxes, labels, scores, true_annotations=true_labels)
        else:
            result_img = draw_predictions(img_array, boxes, labels, scores)

        # Save
        save_path = args.output if args.output else os.path.join(args.output_dir, os.path.basename(args.image))
        save_img(result_img, save_path)
        save_json_predictions(boxes, labels, scores, save_path.replace('.png', '.json'))

        print(f"Prediction completed and saved in {save_path}")