import torch
import numpy as np
from config import parameters_landmark_annotation


def callback(epoch,metrics, monitor='val_loss', mode='min', patience=5,verbose=1):
    
    current_metric=metrics.get(monitor)
    
    if current_metric is None:
        ValueError(f"No key called {monitor} available in metric dictionary")
    
    best_metric_key=f'best_{monitor}'
    
    if best_metric_key not in metrics:
        metrics[best_metric_key]=float("inf") if mode=="min" else float("-inf")
    
    if (mode=="min" and current_metric < metric[best_metric_key]) or (mode=="max" and current_metric > metric[best_metric_key]):
        
        metrics[best_metric_key] = current_metric
        
        metrics[early_stopping_counter]=0
        
        checkpoint_name = f"best_model_for_{parameters_landmark_annotation['model_name']}.pt"
        checkpoint_path = os.path.join(parameters_landmark_annotation['checkpoint_dir'], checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)
        
    
    else:
        
        metrics[early_stopping_counter]+=1
        
        if patience==metrics[early_stopping_counter]:
            print(f'Early stoppping triggered after a patience of {patience} epochs of no improvment')
            break
        
    if verbose:
        print(f'Epoch {epoch}: {monitor} = {current_metric}')
        