import time
import torch
import pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def evaluate_faster_rcnn(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds = model(images)
            metric.update(preds, targets)
            
    res = metric.compute()
    return res['map_50'].item(), res['map_75'].item(), res['mar_100'].item()

def print_results_table(results_list):
    df = pd.DataFrame(results_list, columns=["Dataset", "Model", "mAP@0.5", "Precision", "Recall", "Train Time (s)", "Inf (img/s)"])
    print("\n" + "="*85)
    print("FINAL COMPARISON TABLE")
    print("="*85)
    print(df.to_string(index=False))
    print("="*85)