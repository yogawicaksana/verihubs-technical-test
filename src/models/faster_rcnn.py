from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    """
    Load Faster R-CNN model with ResNet50 FPN backbone
    Replace the classifier head with custom number of classes
    """
    # Load pretrained model weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # Plug the weights into the model
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model