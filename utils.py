
import numpy as np
import matplotlib.pyplot as plt

def result_postprocess(result):
    def _get_list(data):
        data_list = list(zip(data[0], data[1][0], data[1][1]))
        return data_list
    
    pr, _, precision, recall = result.curves_results
    
    pr_list = _get_list(pr)
    precision_list = _get_list(precision)
    recall_list = _get_list(recall)
    
    return {
        "pr": pr_list,
        "precision": precision_list,
        "recall": recall_list
    }

def plot_metric(result_dict, metric_key, class_names=["Food", "Processed"], grid_flag=False, save_path=None):
    """
    Plots a specific metric (PR curve, precision, recall, or AP) for multiple models.

    Parameters:
        result_dict (dict): Nested dictionary containing model results.
                            Top-level keys are model names, second-level keys are metric types
                            (e.g., "pr", "precision", "recall", "ap"), and values are lists or arrays.
                            For "ap", values should be a 2D numpy array of shape (2, 10).
                            For others, values are lists of tuples.
        metric_key (str): The specific metric to plot ("pr", "precision", "recall", "ap").
        class_names (list or None): A list of class names for labeling (default: None).
                                    If None, defaults to ["Class 0", "Class 1"].
        grid_flag (bool): Whether to show grid lines (default: False).
        save_path (str or None): Path to save the plot as an image file (default: None).
    """
    if class_names is None:
        class_names = ["Class 0", "Class 1"]

    if len(class_names) != 2:
        raise ValueError("class_names must contain exactly two class names.")

    linestyles = ['-', '--', '-.', ':']  # Line styles for models
    colors = ["blue", "orange"]  # Colors for different classes
    plt.figure(figsize=(12, 8))

    for model_idx, (model_name, model_result_dict) in enumerate(result_dict.items()):
        if metric_key not in model_result_dict:
            print(f"Metric '{metric_key}' not found in model '{model_name}'. Skipping.")
            continue

        if metric_key == "ap":  # Average Precision (AP)
            ap_values = model_result_dict[metric_key]
            iou_values = np.linspace(0.5, 0.95, 10)
            for class_idx, class_name in enumerate(class_names):
                plt.plot(iou_values, ap_values[class_idx],
                         label=f"{class_name} ({model_name})",
                         color=colors[class_idx],
                         linestyle=linestyles[model_idx % len(linestyles)])
        else:
            metric_values = model_result_dict[metric_key]
            if metric_key == "pr":  # Precision-Recall curve
                for class_idx, class_name in enumerate(class_names):
                    recall_values = [item[0] for item in metric_values]
                    precision_values = [item[1 + class_idx] for item in metric_values]
                    plt.plot(recall_values, precision_values,
                             label=f"{class_name} ({model_name})",
                             color=colors[class_idx],
                             linestyle=linestyles[model_idx % len(linestyles)])
            else:  # Precision or Recall vs px
                px_values = [item[0] for item in metric_values]
                for class_idx, class_name in enumerate(class_names):
                    class_values = [item[1 + class_idx] for item in metric_values]
                    plt.plot(px_values, class_values,
                             label=f"{class_name} ({model_name})",
                             color=colors[class_idx],
                             linestyle=linestyles[model_idx % len(linestyles)])

    # Formatting
    if metric_key == "ap":
        plt.xlabel("IoU Threshold")
        plt.ylabel("AP")
        plt.title("AP by IoU Threshold for Models")
        plt.xlim(0.5, 0.95)
        plt.ylim(0.0, 1.0)
    elif metric_key == "pr":
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves by Model and Class")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
    else:
        plt.xlabel("Confidence Threshold")
        plt.ylabel(metric_key.capitalize())
        plt.title(f"{metric_key.capitalize()} by Confidence Threshold")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

    
    plt.legend()
    plt.grid(grid_flag)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()