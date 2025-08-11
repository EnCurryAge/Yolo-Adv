from pytorchyolo.models import load_model
import torch
from pytorchyolo.train import _create_data_loader
from pytorchyolo.utils.datasets import ImageFolder, ListDataset
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
import random
import tqdm
import torch.optim as optim
import numpy as np
from pytorchyolo.utils.loss import compute_loss
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import os


def random_prune_model(model, prune_ratio=0.2):
    """
    随机剪枝模型参数
    
    Args:
        model: 要剪枝的模型
        prune_ratio: 要剪枝的参数比例 (0-1)
        
    Returns:
        剪枝后的模型
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 统计参数
    param_count = 0
    prunable_params = []
    
    # 统计总参数数量并找出可剪枝的参数
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:  # 通常只剪枝权重，不剪枝偏置
            param_count += param.numel()
            prunable_params.append((name, param))
    
    # 计算要剪枝的参数总数
    total_prune_count = int(param_count * prune_ratio)
    print(f"模型总参数数量: {param_count}")
    print(f"将剪枝 {total_prune_count} 个参数 ({prune_ratio*100:.1f}%)")
    
    # 对每个参数层随机选择要剪枝的参数
    params_pruned = 0
    param_indices = {}
    
    # 第一轮：为每层计算要剪枝的比例
    for name, param in prunable_params:
        param_indices[name] = []
        
        # 计算本层要剪枝的参数数量（按比例）
        layer_prune_count = int(param.numel() * prune_ratio)
        
        # 随机选择要剪枝的参数索引
        indices = np.random.choice(param.numel(), layer_prune_count, replace=False)
        param_indices[name] = indices
        params_pruned += layer_prune_count
    
    # 实际执行剪枝操作
    for name, param in model.named_parameters():
        if name in param_indices:
            # 获取张量的原始形状
            original_shape = param.data.shape
            
            # 将张量展平，置零选定的参数，然后恢复形状
            flattened = param.data.view(-1)
            flattened[param_indices[name]] = 0
            param.data = flattened.view(original_shape)
    
    print(f"实际剪枝参数数量: {params_pruned}")
    
    # 验证剪枝后的模型参数
    zero_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            zero_count += (param == 0).sum().item()
            total_count += param.numel()
    
    actual_prune_ratio = zero_count / total_count
    print(f"剪枝后的零参数比例: {actual_prune_ratio*100:.2f}%")
    
    return model


device = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load model
print("=== 加载原始模型 ===")
model_path = "config/dota-yolov3-416.cfg"
# weights_path = "DOTA/dota-yolov3-416_150000.weights"
weights_path = './config/yolo_dota_ckpt_pruned_diff_later_layers_0.1pct.pth'
# weights_path = './config/yolo_dota_ckpt_repair.pth'
model = load_model(model_path, weights_path)

# 在加载模型后立即执行剪枝
# model = random_prune_model(model, prune_ratio=0.01)
# torch.save(model.state_dict(), './config/yolo_dota_ckpt_pruned.pth')

img_path = "DOTA/adv_train/images"
output_path = "DOTA/adv_train/detections"
# img_path = './enhanced_images'
# output_path = './enhanced_images'
# img_path = './DMP/DOTA/results'
# output_path = './DMP/DOTA/results'
from pytorchyolo.detect import detect_directory, detect_image

# class_names = [
#             "small-vehicle", "large-vehicle", "plane", "storage-tank",
#             "ship", "harbor", "ground-track-field", "soccer-ball-field",
#             "baseball-diamond", "roundabout", "basketball-court", "bridge",
#             "helicopter", "container-crane"
#         ]

with open('DOTA/dota.names', 'r') as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]
print(class_names)

detect_directory(model_path, weights_path, img_path, class_names, output_path, batch_size=1, img_size=608, n_cpu=0)