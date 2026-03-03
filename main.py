import time
import torch
import yaml
from src.dataset_prep import prep_penn_fudan, prep_oxford_pets
from src.models import get_fasterrcnn_model, get_yolo_model
from src.dataset import FasterRCNNDataset
from src.utils import evaluate_faster_rcnn, print_results_table
from torch.utils.data import DataLoader

def train_faster_rcnn(config, dataset_name, yolo_dir, num_classes):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"\n--- Training Faster R-CNN on {dataset_name} ---")
    
    dataset = FasterRCNNDataset(f"{yolo_dir}/images/train", f"{yolo_dir}/labels/train", transforms=True)
    val_dataset = FasterRCNNDataset(f"{yolo_dir}/images/val", f"{yolo_dir}/labels/val", transforms=True)
    
    data_loader = DataLoader(dataset, batch_size=config['models']['faster_rcnn']['batch_size'], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    model = get_fasterrcnn_model(num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    epochs = config['datasets'][dataset_name.lower()]['epochs']
    
    start_train = time.time()
    model.train()
    for epoch in range(epochs): 
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'): 
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
    train_time = time.time() - start_train

    map_50, precision, recall = evaluate_faster_rcnn(model, val_loader, device)
    
    start_inf = time.time()
    model.eval()
    with torch.no_grad():
        for images, _ in val_loader:
            images = list(img.to(device) for img in images)
            _ = model(images)
    inf_speed = len(val_dataset) / (time.time() - start_inf)
    
    return [dataset_name, "Faster R-CNN", round(map_50, 3), round(precision, 3), round(recall, 3), round(train_time, 2), round(inf_speed, 2)]

def train_yolo(config, dataset_name, yaml_path):
    print(f"\n--- Training YOLOv8n on {dataset_name} ---")
    model = get_yolo_model('yolov8n.pt')
    
    start_train = time.time()
    results = model.train(
        data=yaml_path,
        epochs=config['datasets'][dataset_name.lower()]['epochs'],
        imgsz=config['image_size'],
        batch=config['models']['yolo']['batch_size'],
        project='runs',
        name=f"yolo_{dataset_name}"
    )
    train_time = time.time() - start_train
    
    metrics = model.val()
    map_50 = metrics.box.map50
    precision = metrics.box.mp
    recall = metrics.box.mr
    inf_speed = 1000 / metrics.speed['inference'] if metrics.speed['inference'] > 0 else 0
    
    return [dataset_name, "YOLOv8n", round(map_50, 3), round(precision, 3), round(recall, 3), round(train_time, 2), round(inf_speed, 2)]

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    penn_yaml, _ = prep_penn_fudan(config)
    pets_yaml = prep_oxford_pets(config)
    
    final_results = []
    
    final_results.append(train_faster_rcnn(config, "Penn_Fudan", "data/yolo_penn", num_classes=2))
    final_results.append(train_yolo(config, "Penn_Fudan", penn_yaml))
    
    num_pet_classes = len(config['datasets']['pets']['breeds']) + 1
    final_results.append(train_faster_rcnn(config, "Pets", "data/yolo_pets", num_classes=num_pet_classes))
    final_results.append(train_yolo(config, "Pets", pets_yaml))
    
    print_results_table(final_results)