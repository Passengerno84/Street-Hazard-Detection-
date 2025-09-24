import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader , random_split
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.metrics import ConfusionMatrix , DetMetrics , box_iou
import numpy as np
from utils import visualize_predictions , yolo_batch_label
from data import RDD_split_dataset
from model import YOLO_model


test_image_dir = 'D:\Python_Projects\yolomaml\\new\RDD_SPLIT\\test\images'
test_label_dir = 'D:\Python_Projects\yolomaml\\new\RDD_SPLIT\\test\labels'


def detection_collection_fn(batch):
    images = []
    target = []

    for (image,label) in batch:
        images.append(image)
        target.append(label)
    
    images = torch.stack(images , 0)
    
    return images , target



def evaluate_model(model, test_dataset, batch_size=4, conf_threshold=0.025, iou_threshold=0.45):
    
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                          collate_fn=lambda x: detection_collection_fn(x))
    
    # Initialize metrics tracker
    metrics = DetMetrics(names={i: model.names[i] for i in range(model.num_class)})
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # Limit evaluation for testing
                break

            imgs, labels = batch 
                
            batch = yolo_batch_label(imgs, labels)
                
            imgs = batch["img"]
            
            
            predictions = model(imgs)[0]  
            
            print(f"Batch {batch_idx}:")
            print(f"  Raw predictions shape: {predictions.shape}")
            print(f"  Max conf across all boxes: {predictions[:, 4:].max().item():.4f}")
            print(f"  Min conf: {predictions[:, 4:].min().item():.4f}")
            
            # Apply NMS
            predictions_nms = non_max_suppression(
                predictions, conf_threshold, iou_threshold, 
                max_det=300, nc=model.num_class
            )
            
            print(f"  After NMS: {[len(pred) for pred in predictions_nms]} detections per image")
            
            
            gt_cls = batch["cls"].view(-1).cpu()
            gt_bboxes = batch["bboxes"].cpu()
            gt_batch_idx = batch["batch_idx"].cpu()
            
            
            for si, pred in enumerate(predictions_nms):
                
                gt_mask = gt_batch_idx == si
                g_cls = gt_cls[gt_mask]
                g_boxes = gt_bboxes[gt_mask]
                
                if len(pred) == 0 and len(g_cls) == 0:
                    continue
                
                if len(pred) > 0:
                    pred_boxes = pred[:, :4].cpu() 
                    pred_conf = pred[:, 4].cpu()
                    pred_cls = pred[:, 5].long().cpu()
                    
                    # Convert ground truth from xywh (normalized) to xyxy (pixel coordinates)
                    if len(g_boxes) > 0:
                        img_h, img_w = 640, 640  
                        g_boxes_xyxy = torch.zeros_like(g_boxes)
                        g_boxes_xyxy[:, 0] = (g_boxes[:, 0] - g_boxes[:, 2] / 2) * img_w  # x1
                        g_boxes_xyxy[:, 1] = (g_boxes[:, 1] - g_boxes[:, 3] / 2) * img_h  # y1
                        g_boxes_xyxy[:, 2] = (g_boxes[:, 0] + g_boxes[:, 2] / 2) * img_w  # x2
                        g_boxes_xyxy[:, 3] = (g_boxes[:, 1] + g_boxes[:, 3] / 2) * img_h  # y2
                        
                        
                        ious = box_iou(pred_boxes, g_boxes_xyxy)
                    else:
                        ious = torch.zeros((len(pred_boxes), 0))
                    
                    
                    tp = []
                    if len(g_boxes) > 0:
                        iou_thresholds = np.linspace(0.5, 0.95, 10)
                        tp_list = []
                        for i in range(len(pred_boxes)):
                            ious_i = ious[i]
                            if len(ious_i) > 0:
                                max_iou = ious_i.max().item()
                                # Find the ground truth with max IoU
                                max_idx = ious_i.argmax()
                                # Check if class matches and IoU exceeds thresholds
                                if pred_cls[i] == g_cls[max_idx]:
                                    tp_i = [(max_iou > t) for t in iou_thresholds]
                                else:
                                    tp_i = [False] * 10
                            else:
                                tp_i = [False] * 10
                            tp_list.append(tp_i)
                        tp = np.array(tp_list, dtype=bool)  # [num_preds, 10]
                    else:
                        tp = np.zeros((len(pred_boxes), 10), dtype=bool)
                    
                    detections = {
                        "tp": tp,
                        "conf": pred_conf.numpy(),
                        "pred_cls": pred_cls.numpy(),
                    }
                else:
                    detections = {
                        "tp": np.array([]),
                        "conf": np.array([]),
                        "pred_cls": np.array([])
                    }
                
                targets = {
                    "target_cls": g_cls.numpy() if len(g_cls) > 0 else np.array([]),
                    "target_img": np.full(len(g_cls), si) if len(g_cls) > 0 else np.array([]),
                }
                
                metrics.update_stats({
                    "tp": detections["tp"],
                    "conf": detections["conf"],
                    "pred_cls": detections["pred_cls"],
                    "target_cls": targets["target_cls"],
                    "target_img": targets["target_img"],
                })
    
    # Check if we have any valid data
    if len(metrics.stats.get("tp", [])) == 0:
        print("⚠️ No valid predictions/GT found, skipping metrics.")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
        }
    
    
    try:
        metrics.process()
        precision, recall, mAP50, mAP50_95 = metrics.mean_results()
        
        print(f"\n📊 Evaluation Results:")
        print(f"  Precision:    {precision:.4f}")
        print(f"  Recall:       {recall:.4f}")
        print(f"  mAP@0.5:      {mAP50:.4f}")
        print(f"  mAP@0.5:0.95: {mAP50_95:.4f}")
        
        if hasattr(metrics, 'results') and metrics.results is not None:
            print("\n📋 Per-class Results:")
            class_names = ["crack", "pothole", "other"]
            for i, name in enumerate(class_names):
                if i < len(metrics.results):
                    print(f"  {name}: P={metrics.results[i][0]:.3f}, R={metrics.results[i][1]:.3f}, "
                          f"mAP50={metrics.results[i][2]:.3f}, mAP50-95={metrics.results[i][3]:.3f}")
        
        return {
            "precision": precision,
            "recall": recall,
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
        }
        
    except Exception as e:
        print(f"Error processing metrics: {e}")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
        }





def test():

    x = RDD_split_dataset(test_image_dir , test_label_dir)

    # Initialize with 3 classes
    meta_model = YOLO_model(num_class=3, freeze_layer=False)
    meta_model.load_state_dict(torch.load('yoloMaml.pt'))

    print( evaluate_model(meta_model , x) )
