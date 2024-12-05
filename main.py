

import os
import torch
from glob import glob
from tqdm import tqdm

from ultralytics import YOLO

from utils import result_postprocess, plot_metric

def main(args):
    save_dir = os.path.join(args.root_dir, "plots")
    model_dir = os.path.join(args.root_dir, "models")
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device is not None else "cpu"
    model_paths = glob(os.path.join(model_dir, "*.pt"))
    models = {os.path.basename(path).split(".")[0]:YOLO(path).to(device) for path in model_paths}
    
    results_dict = {}
    for model_name, model in models.items():
        results = model.val(
            data=args.test_yaml,
            conf=0.25,
            iou=0.3
        )
        processed_results = result_postprocess(results)
        results_dict[model_name] = processed_results
        
        ap_data = results.box.all_ap  # Shape (num_classes, num_iou_thresholds)
        results_dict[model_name]['ap'] = ap_data
    
    target_metric = ["ap", "pr", "precision", "recall"]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for metric in target_metric:
        plot_metric(results_dict, metric_key=metric, grid_flag=True, save_path=os.path.join(save_dir, f"{metric}.png"))


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./")
    parser.add_argument("--test_yaml", type=str, default="/data2/jh/labeled_pudding_testset/test_data.yaml")
    parser.add_argument("--device", type=str, default="4")
    args = parser.parse_args()
    
    main(args)
## python main.py --model_dir /home/ai04/jh/codes/yolo_train/test/6_exp7_exp8/models/ --save_dir /home/ai04/jh/codes/yolo_train/test/6_exp7_exp8/