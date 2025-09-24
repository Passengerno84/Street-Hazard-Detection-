import torch
import torch.nn as nn
import os
import cv2
import numpy as np


def convert_rdd_to_3class(class_id):
    if class_id in [0, 1, 2]:  # all crack types become class 0
        return 0
    elif class_id == 4:        # pothole becomes class 1
        return 1
    elif class_id == 3:        # other corruption becomes class 2
        return 2
    else:
        print(f"Warning: Unknown class_id {class_id}, mapping to 'other'")
        return 2  # default to "other" for any unexpected class
    

def load_yolo_path(label_path):
    boxes = []
    
    if os.path.exists(label_path):
        with open(label_path , "r") as f:
            for line in f.readlines():
                cls , x , y , w , h = map(float , line.strip().split())
                # Convert original 5-class to 3-class
                new_cls = convert_rdd_to_3class(int(cls))
                boxes.append([new_cls , x , y , w , h])
    
    return torch.tensor(boxes) if boxes else torch.zeros((0,5))

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


def visualize_predictions(imgs, preds_nms, batch, indx , names, save_dir="runs/vis_test", max_images=10):

    os.makedirs(save_dir, exist_ok=True)
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # B,H,W,C
    B = imgs.shape[0]

    for si in range(min(B, max_images)):
        img = (imgs[si] * 255).astype(np.uint8).copy()
        h, w = img.shape[:2]

        # --- Draw GT boxes (green) ---
        gt_cls = batch["cls"][batch["batch_idx"] == si].cpu().numpy()
        gt_boxes = batch["bboxes"][batch["batch_idx"] == si].cpu().numpy()

        for cls, box in zip(gt_cls, gt_boxes):
            
            cx, cy, bw, bh = box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cls_name = names[int(cls)] if int(cls) < len(names) else str(int(cls))
            cv2.putText(img, f"GT:{cls_name}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- Draw Predictions ---
        pred = preds_nms[si]
        if len(pred):
            for *xyxy, conf, cls in pred.cpu().numpy():
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cls_name = names[int(cls)] if int(cls) < len(names) else str(int(cls))
                cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # --- Save ---
        save_path = os.path.join(save_dir, f"batch_{indx}_img_{si}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"Saved visualization: {save_path}")

