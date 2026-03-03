import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import glob

class FasterRCNNDataset(Dataset):
    """A generic dataset loader for Faster RCNN using the YOLO folders we created."""
    def __init__(self, yolo_images_dir, yolo_labels_dir, transforms=None):
        self.img_paths = sorted(glob.glob(os.path.join(yolo_images_dir, "*.jpg")) + glob.glob(os.path.join(yolo_images_dir, "*.png")))
        self.labels_dir = yolo_labels_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        label_path = os.path.join(self.labels_dir, os.path.basename(img_path).rsplit('.', 1)[0] + '.txt')
        
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, xc, yc, bw, bh = map(float, line.strip().split())
                    # Convert YOLO format back to Pascal VOC format (xmin, ymin, xmax, ymax)
                    xmin = (xc - bw / 2) * w
                    xmax = (xc + bw / 2) * w
                    ymin = (yc - bh / 2) * h
                    ymax = (yc + bh / 2) * h
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(class_id) + 1)

        if not boxes: 
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        target = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            img = img.resize((512, 512))
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (512 / w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (512 / h)
            target["boxes"] = boxes
            img = T.ToTensor()(img)

        return img, target

    def __len__(self):
        return len(self.img_paths)