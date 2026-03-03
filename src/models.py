import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

def get_fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_yolo_model(version='yolov8n.pt'):
    model = YOLO(version) 
    return model