import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset , DataLoader , random_split
from glob import glob
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from utils import load_yolo_path







class RDD_split_dataset(Dataset):
    def __init__(self , image_dir , label_dir , transform = True):
        self.image_paths = sorted(glob(os.path.join(image_dir , "*.jpg")))
        self.label_paths = [ os.path.join(label_dir , os.path.basename(p).replace(".jpg" , ".txt")) for p in self.image_paths]

        self.transfrom = transforms.Compose([
            transforms.Resize((640,640)),
            transforms.ToTensor()
        ]) 

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        boxes = load_yolo_path(self.label_paths[index])
        image = self.transfrom(image)
        return image , boxes