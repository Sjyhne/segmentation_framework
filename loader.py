import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from pathlib import Path

from transform import TrainingTransform, ValidationTransform

NEW_IMAGE_SIZE = [512, 512] # (width, height) 4:3
NUM_CLASSES = 2

class AerialDataset(Dataset):
    def __init__(self, root: str, type: str, percentage: float = 1.0, image_size: int = 256, image_suffix: str = "png", mask_suffix: str = "png", transform=None):
        self.root = root
        self.dir = Path(root, type)
        
        self.image_size = image_size
        
        self.images = sorted(Path(self.dir, "images").glob(f"*.{image_suffix}"))
        self.masks = sorted(Path(self.dir, "masks").glob(f"*.{mask_suffix}"))
        
        if percentage != 1.0:
            self.images = self.images[:int(len(self.images) * percentage)]
            self.masks = self.masks[:int(len(self.masks) * percentage)]
        
        assert len(self.images) == len(self.masks), "Number of images and masks must be equal"

        
        self.transform = TrainingTransform([self.image_size, self.image_size], NUM_CLASSES) if type == "train" else ValidationTransform([self.image_size, self.image_size], NUM_CLASSES)
        
        self.num_classes = NUM_CLASSES

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        imagepath = self.images[idx]
        maskpath = self.masks[idx]

        image = Image.open(imagepath)
        mask = Image.open(maskpath)
        
        if image.mode in ["L", "P", "1"]:
            image = image.convert("RGB")
        
        if mask.mode not in ["L", "P", "1"]:
            mask = mask.convert("L")
        
        image, mask = self.transform(image, mask)
        
        mask[mask == 255] = 1

        return image, mask


def get_dataloader(dataset_name: str, 
                   dataset_type: str, 
                   batch_size: int, 
                   percentage: float,
                   image_size: int = 256, 
                   num_workers: int = 1) -> DataLoader:
    
    shuffle = dataset_type == "train"
    
    if dataset_name == "veg":
        dataset = AerialDataset(root="data/veg", type=dataset_type, percentage=percentage, image_size=image_size)
    else:
        return NotImplementedError("Dataset not implemented")
    
    num_classes = dataset.num_classes
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader, num_classes

if __name__ == "__main__":
    
    train_dataset = AerialDataset(root="data/veg", type="train", percentage=1.0)
    val_dataset = AerialDataset(root="data/veg", type="val", percentage=1.0)
    
    print("train:", len(train_dataset))
    print("val:", len(val_dataset))
    
    for i in range(len(val_dataset)):
    
      image, mask = train_dataset[i]
      image, mask = val_dataset[i]
      print(mask.unique())
    