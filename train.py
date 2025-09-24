import torch
import torch.nn as nn
from torch.func import functional_call
from types import SimpleNamespace
from model import YOLO_model
from data import RDD_split_dataset
from maml import FewShot_sampler

train_image_dir = 'D:\Python_Projects\yolomaml\\new\RDD_SPLIT\\train\images'
train_label_dir = 'D:\Python_Projects\yolomaml\\new\RDD_SPLIT\\train\labels'



def forward_with_parameters(model , param_dict , x):
    return functional_call(model , param_dict , (x,) )


def MAMLtrain(model , dataset , outer_optimizer , num_innerloop = 5 , num_episodes = 1000 , inner_lr = 0.01 ):
    model.train()

    for episode in range(num_episodes):
        try:
            support_x , support_y , query_x , query_y = dataset.sampler()

            param = model.clone_parameters()

            for _ in range(num_innerloop):
                pred = forward_with_parameters(model.model , param , support_x )
                            
                loss , _ = model.yolo_loss(pred , support_y)
                loss = loss.sum()
                grad = torch.autograd.grad(loss , param.values() , create_graph=True , allow_unused=True)

                param = {name: (param[name] - inner_lr * g) if g is not None else param[name] for (name, g) in zip(param.keys(), grad)}
                
                

            pred_q = forward_with_parameters(model.model , param , query_x)
            query_loss , _ = model.yolo_loss(pred_q , query_y )

            query_loss = query_loss.sum() 

            query_loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            outer_optimizer.step()
            outer_optimizer.zero_grad()

            if episode % 5 == 0:
                print(f"[Episode {episode}] Query Loss: {query_loss.item():.4f}")

        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            continue

def train():
                    
    x = RDD_split_dataset(train_image_dir , train_label_dir)

    # Updated for 3-way, 5-shot learning
    dataset_sampler = FewShot_sampler(x , n_way=3, k_shot=5, query=5)

    # Initialize with 3 classes
    meta_model = YOLO_model(num_class=3 , freeze_layer=True)

    if isinstance(meta_model.model.args, dict):
        meta_model.model.args = SimpleNamespace(**meta_model.model.args)

    if not hasattr(meta_model.model.args, "box"):
        meta_model.model.args.box = 7.5     # default YOLOv8 value
    if not hasattr(meta_model.model.args, "cls"):
        meta_model.model.args.cls = 0.5     # default YOLOv8 value
    if not hasattr(meta_model.model.args, "dfl"):
        meta_model.model.args.dfl = 1.5     # default YOLOv8 value


    outer_optim = torch.optim.Adam(filter(lambda p : p.requires_grad , meta_model.parameters()) , lr = 1e-4 , weight_decay=1e-5)

    MAMLtrain(meta_model , dataset_sampler , outer_optim)

    save_path = 'yoloMaml.pt'
    torch.save(meta_model.state_dict() , save_path)
    print(f"✅ Model saved at {save_path}")

