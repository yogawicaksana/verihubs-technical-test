from models.faster_rcnn import get_model
from utils.inferece_utils import predict, visualize_predictions
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
    parser.add_argument('--image', type=str, default='../data/test/maksssksksss558.png',
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

    # Run inference
    if args.image:
        # Single image inference
        print(f"\nRunning inference on {args.image}")
        img, boxes, labels, scores = predict(model, args.image, args.device, args.threshold)

        print(f"Found {len(boxes)} detections")
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            print(f"  {i+1}. Label: {label} - Score: {score:.3f} - Box: {box}")

        # Visualize
        save_path = os.path.join(args.output_dir, os.path.basename(args.image)) if args.output_dir else None
        visualize_predictions(img, boxes, labels, scores, save_path=save_path)