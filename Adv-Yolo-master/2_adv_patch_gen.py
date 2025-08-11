from pytorchyolo.models import load_model
import torch
from pytorchyolo.train import _create_data_loader
from pytorchyolo.utils.datasets import ImageFolder, ListDataset
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import random
import tqdm
import torch.optim as optim
import numpy as np
from pytorchyolo.utils.loss import compute_loss
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

class_names = ['small-vehicle',
  'large-vehicle',
  'plane',
  'storage-tank',
  'ship',
  'harbor',
  'ground-track-field',
  'soccer-ball-field',
  'tennis-court',
  'swimming-pool',
  'baseball-diamond',
  'roundabout',
  'basketball-court',
  'bridge',
  'helicopter',
  'container-crane']


def generate_adversarial_patch(record_txt_path, adv_patch_start, adv_patch_end, 
                              model_config_path="config/dota-yolov3-416.cfg",
                              model_weights_path="DOTA/dota-yolov3-416_150000.weights",
                              output_dir="./DMP/dota/",
                              epochs=3,
                              batch_size=4,
                              adv_target_class=12,
                              learning_rate=0.01,
                              pgd_steps=3,
                              epsilon=0.1,
                              alpha=0.05,
                              img_size=608,
                              device=None):
    """
    生成对抗噪声patch的完整函数
    
    Args:
        record_txt_path: 数据集record.txt文件路径
        adv_patch_start: 噪声区域起始位置 [y_start, x_start]
        adv_patch_end: 噪声区域结束位置 [y_end, x_end]
        model_config_path: YOLO模型配置文件路径
        model_weights_path: YOLO模型权重文件路径
        output_dir: 输出文件保存目录
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        pgd_steps: PGD攻击步数
        epsilon: PGD攻击的epsilon参数
        alpha: PGD攻击的alpha参数
        img_size: 图像尺寸
        device: 设备cuda
    
    Returns:
        dict: 包含生成的噪声、原始图像、加噪图像等信息的字典
    """
    
    # 设备设置
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    def create_adv_noise(patch_width, patch_height, channels=3):
        """创建指定尺寸的对抗噪声"""
        noise = torch.zeros(channels, patch_height, patch_width, requires_grad=True, device=device)
        noise = Variable(noise, requires_grad=True)
        return noise

    def apply_noise_to_images(images, noise, grid_position_start, grid_position_end):
        """在图像的指定区域添加噪声"""
        modified_images = images.clone()
        
        # 计算每个patch的大小
        image_height, image_width = images.shape[2], images.shape[3]
        grid_size = 19  # 19×19网格
        patch_height = image_height // grid_size  # 应该是32
        patch_width = image_width // grid_size    # 应该是32
        
        # 获取网格范围对应的像素坐标
        grid_y_start, grid_x_start = grid_position_start
        grid_y_end, grid_x_end = grid_position_end
        
        # 确保网格索引在有效范围内
        assert 0 <= grid_y_start <= grid_y_end < grid_size, f"纵向网格位置范围[{grid_y_start}, {grid_y_end}]超出有效范围[0-18]"
        assert 0 <= grid_x_start <= grid_x_end < grid_size, f"横向网格位置范围[{grid_x_start}, {grid_x_end}]超出有效范围[0-18]"
        
        # 计算像素位置
        pixel_y_start = grid_y_start * patch_height
        pixel_x_start = grid_x_start * patch_width
        pixel_y_end = (grid_y_end + 1) * patch_height  # 加1是为了包含结束网格
        pixel_x_end = (grid_x_end + 1) * patch_width
        
        # 计算区域的高度和宽度
        region_height = pixel_y_end - pixel_y_start
        region_width = pixel_x_end - pixel_x_start
        
        # 确保噪声大小与区域大小匹配
        assert noise.shape[1] == region_height and noise.shape[2] == region_width, \
            f"噪声大小({noise.shape[1]}×{noise.shape[2]})与区域大小({region_height}×{region_width})不匹配"
        
        # 应用噪声到指定区域
        for i in range(modified_images.size(0)):  # 遍历批次中的每张图像
            modified_images[i, :, pixel_y_start:pixel_y_end, pixel_x_start:pixel_x_end] += noise
        
        # 确保像素值在有效范围内 [0, 1]
        modified_images = torch.clamp(modified_images, 0, 1)
        
        return modified_images

    def update_noise_pgd(noise, grad, epsilon=0.05, alpha=0.01, original_noise=None):
        """使用PGD方法更新噪声"""
        if original_noise is None:
            original_noise = torch.zeros_like(noise)
        
        # 按梯度方向更新
        noise_updated = noise - alpha * grad.sign()
        
        # 投影回epsilon球
        delta = noise_updated - original_noise
        mask = delta.abs() > epsilon
        delta[mask] = epsilon * delta[mask].sign()
        
        noise_updated = original_noise + delta
        
        return noise_updated

    def visualize_results(original, noised, noise, position_start, position_end, grid_size=19, save_path=None):
        """可视化结果并保存"""
        if save_path is None:
            save_path = os.path.join(output_dir, 'adv_patch_visualization.png')
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 转换为numpy并调整通道顺序
        original_np = original.permute(1, 2, 0).numpy()
        noised_np = noised.permute(1, 2, 0).numpy()
        noise_np = noise.permute(1, 2, 0).numpy()
        
        # 放大噪声以便可视化
        noise_vis = (noise_np - noise_np.min()) / (noise_np.max() - noise_np.min() + 1e-8)
        
        # 计算实际像素位置
        image_height, image_width = original_np.shape[0], original_np.shape[1]
        patch_height = image_height // grid_size
        patch_width = image_width // grid_size
        
        grid_y_start, grid_x_start = position_start
        grid_y_end, grid_x_end = position_end
        
        pixel_y_start = grid_y_start * patch_height
        pixel_x_start = grid_x_start * patch_width
        pixel_y_end = (grid_y_end + 1) * patch_height
        pixel_x_end = (grid_x_end + 1) * patch_width
        
        # 创建白色背景图像显示噪声位置
        blank_np = np.ones((image_height, image_width, 3))
        blank_np[pixel_y_start:pixel_y_end, pixel_x_start:pixel_x_end] = noise_vis
        
        # 绘制网格线
        for i in range(1, grid_size):
            y_pos = i * patch_height
            blank_np[y_pos-1:y_pos+1, :] = [0.8, 0.8, 0.8]
            x_pos = i * patch_width
            blank_np[:, x_pos-1:x_pos+1] = [0.8, 0.8, 0.8]
        
        # 绘制三个子图
        axes[0].imshow(original_np)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(blank_np)
        axes[1].set_title("Noise Patch", fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(noised_np)
        axes[2].set_title("Noised Image", fontsize=14)
        axes[2].axis('off')
        
        # 在所有图像上标记噪声位置
        for ax in axes:
            rect = plt.Rectangle((pixel_x_start, pixel_y_start), 
                               pixel_x_end-pixel_x_start, pixel_y_end-pixel_y_start, 
                               linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        fig.suptitle(f"target_class is: {class_names[adv_target_class]}", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化结果已保存到: {save_path}")
        return save_path

    def save_noised_image(noised_tensor, original_path=None, save_path=None):
        """保存添加噪声后的图像"""
        if save_path is None:
            save_path = './DOTA/adv_train/images/adversarial_image.png'
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 转换为PIL图像
        noised_np = noised_tensor.permute(1, 2, 0).cpu().numpy()
        noised_np = np.clip(noised_np, 0, 1)
        noised_np = (noised_np * 255).astype(np.uint8)
        noised_pil = Image.fromarray(noised_np)
        
        # 如果提供了原始图像路径，调整大小以匹配原始图像
        if original_path:
            try:
                if isinstance(original_path, int):
                    original_path = dataset.img_files[original_path]
                
                original_img = Image.open(original_path)
                original_size = original_img.size
                noised_pil = noised_pil.resize(original_size, Image.LANCZOS)
                print(f"已将噪声图像调整为原始图像尺寸: {original_size}")
            except Exception as e:
                print(f"无法调整为原始尺寸: {e}")
        
        noised_pil.save(save_path, format='PNG')
        print(f"噪声图像已保存至: {save_path}")
        return save_path

    # 主要执行流程
    print("=== 开始生成对抗噪声patch ===")
    
    # 加载模型
    print("=== 加载YOLO模型 ===")
    model = load_model(model_config_path, model_weights_path)
    model.to(device)
    model = model.eval()

    # 加载数据集
    print("=== 加载数据集 ===")
    dataset = ListDataset(
        record_txt_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        pin_memory=True)
    
    def extract_image_index_from_record(record_path):
        """
        从record.txt文件中提取图片索引
        """
        try:
            with open(record_path, 'r') as f:
                first_line = f.readline().strip()
                # 假设record.txt格式为: /path/to/image_index.jpg
                # 或者 DOTA/adv_train/images/P0001.png 这样的格式
                import os
                filename = os.path.basename(first_line)
                # 提取文件名中的数字部分
                import re
                # 匹配文件名中的数字
                match = re.search(r'(\d+)', filename)
                if match:
                    return match.group(1)
                else:
                    return "0000"  # 默认索引
        except Exception as e:
            print(f"提取图片索引失败: {e}")
            return "0000"  # 默认索引

    # 提取图片索引
    image_index = extract_image_index_from_record(record_txt_path)
    print(f"提取到的图片索引: {image_index}")

    # 计算噪声区域大小
    grid_size = 19
    patch_size = img_size // grid_size
    region_height = (adv_patch_end[0] - adv_patch_start[0] + 1) * patch_size
    region_width = (adv_patch_end[1] - adv_patch_start[1] + 1) * patch_size
    
    # 创建对抗噪声
    adv_noise = create_adv_noise(region_width, region_height, channels=3)
    optimizer = optim.Adam([adv_noise], lr=learning_rate)
    original_noise = adv_noise.clone().detach()

    # 存储结果
    original_images = []
    noise_added_images = []
    noise_values = []
    loss_history = []

    print("=== 开始对抗优化 ===")
    print(f"噪声区域网格范围: [{adv_patch_start[0]}-{adv_patch_end[0]}, {adv_patch_start[1]}-{adv_patch_end[1]}]")
    print(f"噪声区域像素大小: {region_height}×{region_width}")
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"训练 Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i
            
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device) # targets -> torch.Size([class_length * bs, 6])
            # re-labeling for adversarial patch
            targets[:, 1] = adv_target_class
            # print(targets)
            # 保存第一批次的原始图像
            if epoch == 1 and batch_i == 0:
                original_images.append(imgs[0].cpu().detach().clone())
            
            # PGD攻击步骤
            for pgd_step in range(pgd_steps):
                # 前向传播
                imgs_with_noise = apply_noise_to_images(imgs, adv_noise, adv_patch_start, adv_patch_end)
                outputs = model(imgs_with_noise)
                
                # 计算损失
                _, _, cls_loss = compute_loss(outputs, targets, model)
                
                # 反向传播
                optimizer.zero_grad()
                cls_loss.backward()
                
                # PGD更新
                noise_grad = adv_noise.grad.clone()
                with torch.no_grad():
                    adv_noise.copy_(update_noise_pgd(adv_noise, noise_grad, 
                                                epsilon=epsilon, alpha=alpha, 
                                                original_noise=original_noise))
            
            # 重置梯度
            optimizer.zero_grad()
            
            # 记录损失
            total_loss += cls_loss.item()
            loss_history.append(cls_loss.item())
            
            # 定期保存结果
            if batch_i % 2 == 0:
                noise_values.append(adv_noise.cpu().detach().clone())
                new_img_with_noise = apply_noise_to_images(imgs, adv_noise, adv_patch_start, adv_patch_end)
                noise_added_images.append(new_img_with_noise[0].cpu().detach().clone())
                
                print(f"Batch {batch_i}, 分类损失: {cls_loss.item():.4f}, 噪声范围: [{adv_noise.min().item():.4f}, {adv_noise.max().item():.4f}]")
        
        # 打印epoch平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} 平均分类损失: {avg_loss:.4f}")

    print("=== 对抗优化完成 ===")
    
    # 生成可视化结果
    # 生成可视化结果
    if original_images and noise_added_images and noise_values:
        # 使用图片索引生成文件名
        visualization_filename = f'adv_patch_visualization_{image_index}.png'
        visualization_path = visualize_results(
            original_images[-1], 
            noise_added_images[-1], 
            noise_values[-1], 
            position_start=adv_patch_start,
            position_end=adv_patch_end,
            grid_size=19,
            save_path=os.path.join(output_dir, visualization_filename)
        )
        
        # 保存对抗图像
        adversarial_filename = f'adversarial_image_{image_index}.png'
        if len(dataloader.dataset.img_files) > 0:
            original_img_path = dataloader.dataset.img_files[0]
            adversarial_img_path = save_noised_image(
                noise_added_images[-1],
                original_path=original_img_path,
                save_path=f'./DOTA/adv_train/images/{adversarial_filename}'
            )
        else:
            adversarial_img_path = save_noised_image(
                noise_added_images[-1],
                save_path=f'./DOTA/adv_train/images/{adversarial_filename}'
            )
    
    # 返回结果
    return {
        'adversarial_noise': adv_noise.cpu().detach(),
        'original_image': original_images[-1] if original_images else None,
        'adversarial_image': noise_added_images[-1] if noise_added_images else None,
        'loss_history': loss_history,
        'noise_region': {
            'start': adv_patch_start,
            'end': adv_patch_end,
            'pixel_size': (region_height, region_width)
        },
        'output_paths': {
            'visualization': visualization_path if 'visualization_path' in locals() else None,
            'adversarial_image': adversarial_img_path if 'adversarial_img_path' in locals() else None
        }
    }

# 使用示例
if __name__ == "__main__":
    start_time = time.time()
    selected_txt = './DMP/dota/selected_top5_images.txt'
    contents = np.loadtxt(selected_txt, dtype=str, delimiter=' ', encoding='utf-8')
    for i in range(22):
        record_txt_path = f'DOTA/adv_train/record_{i}.txt'
        height, weight, target_cls = int(contents[i][1]), int(contents[i][2]), int(contents[i][3])
        if height >= 16 or weight >= 16:
            # 调用函数生成对抗噪声patch
            result = generate_adversarial_patch(
                record_txt_path=record_txt_path,
                adv_patch_start=[height - 2, weight -2],
                adv_patch_end=[height, weight],
                output_dir="./DMP/dota/",
                epochs=3,
                adv_target_class=target_cls,
                batch_size=4
            )
            print(f"噪声区域: {result['noise_region']}")
            print(f"输出文件: {result['output_paths']}")
        else:
            # 调用函数生成对抗噪声patch
            result = generate_adversarial_patch(
                record_txt_path=record_txt_path,
                adv_patch_start=[height, weight],
                adv_patch_end=[height + 2, weight + 2],
                output_dir="./DMP/dota/",
                epochs=3,
                adv_target_class=target_cls,
                batch_size=4
            )
            print(f"噪声区域: {result['noise_region']}")
            print(f"输出文件: {result['output_paths']}")
    end_time = time.time()
    print("全部对抗噪声生成完成！")
    print(f"总耗时: {end_time - start_time:.2f}秒")