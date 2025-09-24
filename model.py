import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np


class YOLO_model(nn.Module):
    def __init__(self ,pretrained_model_path = 'yolov8n.pt', num_class = 3 , freeze_layer = True):
        super().__init__()
        self.num_class = num_class
        self.freeze_layer = freeze_layer
        self.yolo = YOLO(pretrained_model_path)
        self.model = self.yolo.model
        self.model_layers = self.model

        # Updated class names for 3 classes
        self.names = [
            "crack",      # class 0: all types of cracks combined
            "pothole",    # class 1: pothole
            "other"       # class 2: other corruption
        ]

        if freeze_layer:
            self.freeze_backbone_layer()

        self.modify_detection_head()

        
    def freeze_backbone_layer(self):
        total_parameters = list(self.model.parameters())
        freeze_ratio = 0.75
        freeze_len = int(freeze_ratio * len(total_parameters))

        frozen_count = 0
        trainable_count = 0

        for i ,  parameter in enumerate(total_parameters):
            
            if i<freeze_len:
                parameter.requires_grad = False
                frozen_count += 1
            
            else:
                parameter.requires_grad = True
                trainable_count += 1
            
             
        print(f'total number of parameters: {len(total_parameters)}')
        print(f'number of frozen layer: {frozen_count}')
        print(f'total number of trainable parametr: {trainable_count}')

    def modify_detection_head(self):
        detect_layer = self.model_layers.model[-1]
        for i, conv_sequence in enumerate(detect_layer.cv3):
            
            old_conv = conv_sequence[-1]

            new_conv = nn.Conv2d(
                in_channels=old_conv.in_channels,  
                out_channels=self.num_class,     
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride,           
                padding=old_conv.padding,         
                bias=old_conv.bias is not None    
            )

            print(old_conv , new_conv)

            nn.init.normal_(new_conv.weight, mean=0.0, std=0.01)
            if new_conv.bias is not None:
                nn.init.constant_(new_conv.bias, -np.log((1 - 0.01) / 0.01))
            
            detect_layer.cv3[i][-1] = new_conv
        
        detect_layer.nc = self.num_class
        detect_layer.no = self.num_class + detect_layer.reg_max*4

        self.model.names = {i: self.names[i] for i in range(self.num_class)}




    def yolo_loss(self , pred , target):
        device = pred[0].device  

        batch = {
        "img": target["img"].to(device, dtype=torch.float32),
        "cls": target["cls"].view(-1).to(device, dtype=torch.long),   # flatten (N,1) -> (N,)
        "bboxes": target["bboxes"].to(device, dtype=torch.float32),   
        "batch_idx": target["batch_idx"].to(device, dtype=torch.long) 
        }

            
        loss, loss_items = self.model.loss(batch , pred)

        return loss, loss_items
    
    def clone_parameters(self):

        return { name : p.clone() for name , p in self.model.named_parameters() if p.requires_grad}
    
    
    def forward(self , x):
        return self.model(x)
    
    def train(self , mode: bool = True):
        self.model.train(mode)
        
        return self