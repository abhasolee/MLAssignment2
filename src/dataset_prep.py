import os
import requests
import zipfile
import tarfile
import shutil
import random
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np

def download_and_extract(url, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_to, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")

        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status() 
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: 
                    f.write(chunk)
        
        print(f"Extracting {filename}...")
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif filename.endswith('.tar.gz'):
            with tarfile.open(filepath, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
    return filepath

def create_yolo_yaml(yaml_path, train_path, val_path, test_path, classes):
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.abspath(train_path)}\n")
        f.write(f"val: {os.path.abspath(val_path)}\n")
        f.write(f"test: {os.path.abspath(test_path)}\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")

def prep_penn_fudan(config):
    print("\nPreparing Penn-Fudan Dataset...")
    base_dir = "data/PennFudan"
    download_and_extract(config['datasets']['penn_fudan']['url'], base_dir)
    
    # We will format this for YOLO. Faster R-CNN will use its PyTorch Dataset class.
    yolo_dir = "data/yolo_penn"
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(yolo_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, 'labels', split), exist_ok=True)

    img_dir = os.path.join(base_dir, "PennFudanPed", "PNGImages")
    mask_dir = os.path.join(base_dir, "PennFudanPed", "PedMasks")
    images = sorted(os.listdir(img_dir))
    
    random.seed(42)
    random.shuffle(images)
    
    n = len(images)
    train_end = int(n * config['split']['train'])
    val_end = train_end + int(n * config['split']['val'])
    
    splits = {'train': images[:train_end], 'val': images[train_end:val_end], 'test': images[val_end:]}
    
    for split_name, imgs in splits.items():
        for img_name in imgs:
            # Copy Image
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(yolo_dir, 'images', split_name, img_name)
            shutil.copy(src_img, dst_img)
            
            # Process Mask to YOLO BBox
            mask_name = img_name.replace('.png', '_mask.png')
            mask = Image.open(os.path.join(mask_dir, mask_name))
            mask_np = np.array(mask)
            obj_ids = np.unique(mask_np)[1:] # Exclude background (0)
            
            img_w, img_h = mask.size
            labels_content = []
            
            for obj_id in obj_ids:
                pos = np.where(mask_np == obj_id)
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                
                # YOLO format: class_id center_x center_y width height (normalized)
                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h
                labels_content.append(f"0 {x_center} {y_center} {w} {h}")
                
            with open(os.path.join(yolo_dir, 'labels', split_name, img_name.replace('.png', '.txt')), 'w') as f:
                f.write("\n".join(labels_content))

    yaml_path = "data/penn.yaml"
    create_yolo_yaml(yaml_path, f"{yolo_dir}/images/train", f"{yolo_dir}/images/val", f"{yolo_dir}/images/test", ['person'])
    return yaml_path, os.path.join(base_dir, "PennFudanPed")

def prep_oxford_pets(config):
    print("\nPreparing Oxford Pets Dataset (Subset)...")
    base_dir = "data/OxfordPets"
    download_and_extract(config['datasets']['pets']['url_images'], base_dir)
    download_and_extract(config['datasets']['pets']['url_annotations'], base_dir)
    
    target_breeds = config['datasets']['pets']['breeds']
    breed_to_id = {breed: i for i, breed in enumerate(target_breeds)}
    
    yolo_dir = "data/yolo_pets"
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(yolo_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, 'labels', split), exist_ok=True)

    img_dir = os.path.join(base_dir, "images")
    anno_dir = os.path.join(base_dir, "annotations", "xmls")
    
    valid_images = []
    for xml_file in os.listdir(anno_dir):
        if not xml_file.endswith('.xml'): continue
        breed_name = "_".join(xml_file.split('_')[:-1]) # Extrapolate breed from filename
        if breed_name in target_breeds:
            valid_images.append(xml_file.replace('.xml', '.jpg'))
            
    random.seed(42)
    random.shuffle(valid_images)
    
    n = len(valid_images)
    train_end = int(n * config['split']['train'])
    val_end = train_end + int(n * config['split']['val'])
    splits = {'train': valid_images[:train_end], 'val': valid_images[train_end:val_end], 'test': valid_images[val_end:]}
    
    for split_name, imgs in splits.items():
        for img_name in imgs:
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(yolo_dir, 'images', split_name, img_name)
            if not os.path.exists(src_img): continue
            shutil.copy(src_img, dst_img)
            
            xml_path = os.path.join(anno_dir, img_name.replace('.jpg', '.xml'))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
            
            breed_name = "_".join(img_name.split('_')[:-1])
            class_id = breed_to_id[breed_name]
            
            labels_content = []
            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)
                
                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h
                labels_content.append(f"{class_id} {x_center} {y_center} {w} {h}")
                
            with open(os.path.join(yolo_dir, 'labels', split_name, img_name.replace('.jpg', '.txt')), 'w') as f:
                f.write("\n".join(labels_content))

    yaml_path = "data/pets.yaml"
    create_yolo_yaml(yaml_path, f"{yolo_dir}/images/train", f"{yolo_dir}/images/val", f"{yolo_dir}/images/test", target_breeds)
    return yaml_path