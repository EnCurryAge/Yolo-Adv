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


device = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load dataset train
ori_img_path = "DOTA/adv_train/record_copx.txt" # normal image
img_size = 608
ori_dataset = ListDataset(
        ori_img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))

ori_dataloader = DataLoader(
        # subset_dataset,
        ori_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=ori_dataset.collate_fn,
        pin_memory=True)
 
adv_img_path = "DOTA/adv_train/record_copy.txt" # noised image
img_size = 608
adv_dataset = ListDataset(
        adv_img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))

adv_dataloader = DataLoader(
        # subset_dataset,
        adv_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=adv_dataset.collate_fn,
        pin_memory=True)

# load model
print("=== 加载原始模型 ===")
model_path = "config/dota-yolov3-416.cfg"
weights_path = "DOTA/dota-yolov3-416_150000.weights"
model = load_model(model_path, weights_path)
model = model.to(device)
model.eval()


def prune_based_on_activation_difference(model, ori_dataloader, adv_dataloader, prune_ratio=0.001, top_k_layers=5, skip_early_layers=True):
    """
    找出原始图片和对抗图片激活值差异最大的几层，然后剪枝这些层的参数
    跳过模型前半部分的层，以保持基础特征提取能力
    
    Args:
        model: 要剪枝的模型
        ori_dataloader: 原始图片的数据加载器
        adv_dataloader: 对抗图片的数据加载器
        prune_ratio: 要剪枝的总参数比例 (0-1)
        top_k_layers: 要剪枝的激活值差异最大的层数量
        skip_early_layers: 是否跳过模型前半部分的层
        
    Returns:
        剪枝后的模型
    """
    print("=== 开始基于原始图片和对抗图片激活值差异的剪枝 ===")
    print(f"将跳过模型前半部分的层以保持基础特征提取能力: {skip_early_layers}")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 用于存储每层的激活值统计
    ori_layer_activations = {}
    adv_layer_activations = {}
    
    # 检查原始模型中已有的零参数
    existing_zeros = 0
    all_params_count = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            zeros = (param.data == 0).sum().item()
            existing_zeros += zeros
            all_params_count += param.numel()
    
    existing_zero_ratio = existing_zeros / all_params_count
    print(f"模型已有零参数比例: {existing_zero_ratio*100:.4f}% ({existing_zeros}/{all_params_count})")
    
    # 1. 收集所有层的名称，用于后续确定层的位置
    all_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.BatchNorm2d):
            all_layers.append(name)
    
    total_layers = len(all_layers)
    mid_point = total_layers // 2
    print(f"模型总共有 {total_layers} 层, 中点位置在第 {mid_point} 层")
    
    # 将每个层映射到其在模型中的位置索引
    layer_positions = {layer: idx for idx, layer in enumerate(all_layers)}
    
    # 2. 收集原始图片的层级激活值
    hooks = []
    
    def hook_fn(name, activation_dict):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # 计算L2范数
                norm = torch.norm(output).item()
                if name in activation_dict:
                    activation_dict[name].append(norm)
                else:
                    activation_dict[name] = [norm]
        return hook
    
    # 为模型的每一层注册钩子
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(hook_fn(name, ori_layer_activations)))
    
    # 用原始图片进行前向传播，收集激活值
    print("执行前向传播，收集原始图片的层级激活值...")
    with torch.no_grad():
        for i, (_, imgs, targets) in enumerate(tqdm.tqdm(ori_dataloader, desc="原始图片")):
            if i >= 10:  # 限制样本数量，避免过长时间
                break
            imgs = imgs.to(device)
            _ = model(imgs)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 3. 收集对抗图片的层级激活值
    hooks = []
    
    # 为模型的每一层注册新钩子
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(hook_fn(name, adv_layer_activations)))
    
    # 用对抗图片进行前向传播，收集激活值
    print("执行前向传播，收集对抗图片的层级激活值...")
    with torch.no_grad():
        for i, (_, imgs, targets) in enumerate(tqdm.tqdm(adv_dataloader, desc="对抗图片")):
            if i >= 10:  # 限制样本数量，避免过长时间
                break
            imgs = imgs.to(device)
            _ = model(imgs)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 4. 计算每一层激活值的差异
    layer_differences = {}
    
    for name in ori_layer_activations:
        if name in adv_layer_activations:
            # 计算平均激活值
            ori_avg = np.mean(ori_layer_activations[name])
            adv_avg = np.mean(adv_layer_activations[name])
            
            # 计算激活值差异 (可以是绝对差异或相对差异)
            # 这里使用相对差异：(adv - ori) / ori
            if ori_avg > 0:
                rel_diff = (adv_avg - ori_avg) / ori_avg
                layer_differences[name] = abs(rel_diff)  # 取绝对值，我们只关心差异大小
            else:
                # 如果原始激活值接近0，直接使用对抗激活值作为差异
                layer_differences[name] = adv_avg
    
    # 5. 按差异大小排序层
    sorted_layers = sorted(layer_differences.items(), key=lambda x: x[1], reverse=True)
    
    # 6. 筛选层，跳过前半部分的层（如果需要）
    if skip_early_layers:
        filtered_sorted_layers = []
        for layer_name, diff in sorted_layers:
            # 如果该层在模型的后半部分，则保留
            if layer_positions.get(layer_name, 0) >= mid_point:
                filtered_sorted_layers.append((layer_name, diff))
        
        # 如果后半部分的层不足top_k_layers个，从前半部分中选择差异最大的层补充
        if len(filtered_sorted_layers) < top_k_layers:
            remaining_needed = top_k_layers - len(filtered_sorted_layers)
            print(f"后半部分层不足{top_k_layers}个，需要从前半部分选择{remaining_needed}个层补充")
            for layer_name, diff in sorted_layers:
                if layer_positions.get(layer_name, 0) < mid_point:
                    filtered_sorted_layers.append((layer_name, diff))
                    remaining_needed -= 1
                    if remaining_needed <= 0:
                        break
        
        # 使用筛选后的层列表
        sorted_layers = filtered_sorted_layers[:top_k_layers]
    else:
        # 不跳过前半部分，直接选择差异最大的top_k_layers层
        sorted_layers = sorted_layers[:top_k_layers]
    
    # 7. 选择最终的目标层
    target_layers = [layer_name for layer_name, _ in sorted_layers]
    
    print(f"激活值差异最大的{len(target_layers)}层:")
    for layer_name, diff in sorted_layers:
        ori_avg = np.mean(ori_layer_activations[layer_name])
        adv_avg = np.mean(adv_layer_activations[layer_name])
        layer_pos = layer_positions.get(layer_name, -1)
        print(f"  层 '{layer_name}' (位置 {layer_pos}/{total_layers}): 差异 = {diff:.4f} (原始 = {ori_avg:.4f}, 对抗 = {adv_avg:.4f})")
    
    # 8. 找出这些层对应的权重参数
    target_weights = []
    param_count = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            # 检查该权重是否属于目标层
            is_target = False
            for layer_name in target_layers:
                if layer_name in name:
                    is_target = True
                    break
            
            if is_target:
                # 检查非零参数
                non_zero_mask = param.data != 0
                non_zero_count = non_zero_mask.sum().item()
                
                if non_zero_count > 0:
                    target_weights.append((name, param, non_zero_mask))
                    param_count += non_zero_count
    
    # 9. 计算要剪枝的参数总数 (仅针对非零参数)
    non_zero_params = all_params_count - existing_zeros
    total_prune_count = int(non_zero_params * prune_ratio)
    
    if param_count == 0:
        print("警告: 目标层中没有非零参数可剪枝!")
        return model
    
    layer_prune_ratio = min(1.0, total_prune_count / param_count)  # 确保不超过100%
    
    print(f"总参数数量: {all_params_count}")
    print(f"当前非零参数数量: {non_zero_params}")
    print(f"目标层非零参数数量: {param_count}")
    print(f"将在这些层中剪枝约 {layer_prune_ratio*100:.4f}% 的非零参数")
    print(f"总共将剪枝 {total_prune_count} 个参数 ({prune_ratio*100:.4f}%)")
    
    if total_prune_count == 0:
        print("警告: 剪枝比例过小，没有参数会被剪枝!")
        return model
    
    # 10. 执行剪枝
    params_pruned = 0
    
    for name, param, non_zero_mask in target_weights:
        # 获取非零参数的索引
        non_zero_indices = torch.nonzero(non_zero_mask.view(-1), as_tuple=True)[0]
        
        if len(non_zero_indices) == 0:
            continue
        
        # 计算该层要剪枝的参数数量
        layer_non_zero_count = len(non_zero_indices)
        layer_count = min(int(layer_non_zero_count * layer_prune_ratio), layer_non_zero_count)
        
        if layer_count == 0:
            continue
        
        # 从非零参数中随机选择要剪枝的参数
        indices_to_prune = np.random.choice(non_zero_indices.cpu().numpy(), layer_count, replace=False)
        
        # 记录被剪枝的参数值
        flat_param = param.data.view(-1)
        pruned_values = flat_param[indices_to_prune].cpu().numpy()
        mean_val = np.abs(pruned_values).mean()
        max_val = np.abs(pruned_values).max()
        
        # 执行剪枝
        flat_param[indices_to_prune] = 0
        param.data = flat_param.view(param.shape)
        
        params_pruned += layer_count
        print(f"层 '{name}' 剪枝了 {layer_count}/{layer_non_zero_count} 个参数, 平均值={mean_val:.6f}, 最大值={max_val:.6f}")
    
    # 11. 验证剪枝结果
    new_zeros = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            new_zeros += (param.data == 0).sum().item()
    
    actual_new_prune_ratio = (new_zeros - existing_zeros) / all_params_count
    total_zero_ratio = new_zeros / all_params_count
    
    print(f"新增剪枝参数比例: {actual_new_prune_ratio*100:.6f}% ({new_zeros - existing_zeros}/{all_params_count})")
    print(f"剪枝后的总零参数比例: {total_zero_ratio*100:.4f}% ({new_zeros}/{all_params_count})")
    print("=== 剪枝完成 ===")
    
    return model

# 调用剪枝函数并保存
model = prune_based_on_activation_difference(
    model, 
    ori_dataloader, 
    adv_dataloader, 
    prune_ratio=0.01,  
    top_k_layers=50,
    skip_early_layers=False  # 启用跳过前半部分层的功能
)

# 添加版本标记以避免覆盖
torch.save(model.state_dict(), './config/yolo_dota_ckpt_pruned_diff_later_layers_0.1pct.pth')

print('finished.')