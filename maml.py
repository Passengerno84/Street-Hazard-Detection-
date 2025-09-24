import torch
import torch.nn as nn
import random
from utils import yolo_batch_label , load_yolo_path

def yolo_batch_label( x , y):

    box_labels = []

    for i , boxes in enumerate(y):
        box_idx  = torch.full((boxes.size(0) , 1) , i)
        boxes = torch.cat([box_idx,boxes] , dim=1)
        box_labels.append(boxes)
    
    box_labels = torch.cat(box_labels , dim = 0) if box_labels else torch.zeros((0,6))
    

    return { 
        "img": x ,           
        "batch_idx": box_labels[:,0], 
        "cls": box_labels[:,1:2].long(),           
        "bboxes": box_labels[:,2:] 
    }


class FewShot_sampler:
    def __init__(self , dataset , n_way = 3 , k_shot = 5 , query = 5):
        self.dataset = dataset
        self.n_way = n_way  
        self.k_shot = k_shot
        self.query = query 

        self.class_group = {0: [], 1: [], 2: []}  # Initialize for 3 classes

        # Group images by class
        for indx ,  label_path in enumerate(self.dataset.label_paths):
            boxes = load_yolo_path(label_path)

            if len(boxes) > 0:  # Only process if there are annotations
                for cls in boxes[:,0].unique().tolist():
                    cls_id = int(cls)
                    if cls_id in self.class_group:
                        self.class_group[cls_id].append(indx)
        
        print("Class distribution:")
        for cls_id, indices in self.class_group.items():
            class_names = ["crack", "pothole", "other"]
            print(f"Class {cls_id} ({class_names[cls_id]}): {len(indices)} images")
    
    def sampler(self):
        selected_classes = [0, 1, 2]  # crack, pothole, other
        
        support_images , support_labels = [] , []
        query_images , query_labels = [] , []

        for cls in selected_classes:
            available_samples = len(self.class_group[cls])
            required_samples = self.k_shot + self.query
            
            if available_samples < required_samples:
                # If not enough samples, use all available and repeat some
                print(f"Warning: Class {cls} has only {available_samples} samples, need {required_samples}")
                # Sample with replacement if necessary
                shuffled_images = random.choices(self.class_group[cls], k=required_samples)
            else:
                shuffled_images = random.sample(self.class_group[cls] , required_samples)

            support_indx , query_indx = shuffled_images[:self.k_shot] , shuffled_images[self.k_shot:]

            for idx in support_indx:
                image , label = self.dataset[idx]
                support_images.append(image)
                support_labels.append(label)
                
            for idx in query_indx:
                image , label = self.dataset[idx]
                query_images.append(image)
                query_labels.append(label)

        support_images = torch.stack(support_images)
        query_images = torch.stack(query_images)

        support_labels = yolo_batch_label(support_images , support_labels)
        query_labels = yolo_batch_label(query_images , query_labels)

        return (support_images , support_labels , query_images , query_labels)