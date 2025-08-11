import torch
import numpy as np
import random
import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class_dbm = np.load("./DMP/dota/DBM.npy")
print(class_dbm.shape)

import torch
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class_dbm = np.load("./DMP/dota/DBM.npy")
print(f"numpy shape: {class_dbm.shape}")

# 任务1: 找出所有不同的类别
unique_classes = np.unique(class_dbm)
print(f"totally {len(unique_classes)} classes")
print(f"classes: {unique_classes}")

# 任务2: 找到所有决策边界点
# 任务2: 找到所有决策边界点（简化版本）
def find_boundary_points(array):
    """
    找到决策边界上的所有点
    返回: boundary_points (坐标列表), boundary_classes (对应的相邻不同类别列表)
    """
    height, width = array.shape
    boundary_points = []
    boundary_classes = []
    
    # 遍历所有点
    for i in tqdm.tqdm(range(height), desc='searching for boundary points...'):
        for j in range(width):
            current_class = array[i, j]
            is_boundary = False
            first_different_class = None
            
            # 检查四个相邻的点
            neighbors = [
                (i-1, j),  # 上
                (i+1, j),  # 下
                (i, j-1),  # 左
                (i, j+1)   # 右
            ]
            
            for ni, nj in neighbors:
                # 检查邻居是否在边界内
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_class = array[ni, nj]
                    if neighbor_class != current_class:
                        is_boundary = True
                        if first_different_class is None:
                            first_different_class = neighbor_class
                        break  # 找到第一个不同类别就退出
                else:
                    # 如果当前点在图像边界上，也认为是决策边界
                    is_boundary = True
                    break
            
            if is_boundary:
                boundary_points.append([i, j])
                # 如果找到了相邻的不同类别，记录它；否则记录自身类别
                boundary_classes.append(first_different_class if first_different_class is not None else current_class)
    
    return boundary_points, boundary_classes

# 执行边界检测
boundary_points, boundary_classes = find_boundary_points(class_dbm)

tsne_points = np.load('./DMP/dota/X_proj.npy')
print(tsne_points.shape) # [690, 2]
tsne_classes = np.load('./DMP/dota/y_classif_train.npy')
print(tsne_classes.shape) # [690]

# ...existing code...
boundary_points, boundary_classes = find_boundary_points(class_dbm)

tsne_points = np.load('./DMP/dota/X_proj.npy') # [690, 2]
tsne_classes = np.load('./DMP/dota/y_classif_train.npy') # [690]

scaled_tsne_points = np.round(tsne_points * 300).astype(int)

# 任务2: 为每个测试样本找到距离最近的不同类别边界点
def find_nearest_different_class_boundary(sample_point, sample_class, boundary_points, boundary_classes):
    """
    为单个样本点找到距离最近的不同类别边界点
    """
    min_distance = float('inf')
    
    for i, boundary_point in enumerate(boundary_points):
        boundary_class = boundary_classes[i]
        
        # 只考虑不同类别的边界点
        if boundary_class != sample_class:
            # 计算L2距离
            distance = np.sqrt((sample_point[0] - boundary_point[0])**2 + 
                             (sample_point[1] - boundary_point[1])**2)
            
            if distance < min_distance:
                min_distance = distance
    
    return min_distance

# 计算所有690个点到最近不同类别边界点的距离
distances = []
boundary_points_array = np.array(boundary_points)
boundary_classes_array = np.array(boundary_classes)

for i in tqdm.tqdm(range(len(scaled_tsne_points)), desc="Distance Calculation..."):
    sample_point = scaled_tsne_points[i]
    sample_class = tsne_classes[i]
    
    # 找到所有不同类别的边界点
    different_class_mask = boundary_classes_array != sample_class
    different_class_boundaries = boundary_points_array[different_class_mask]
    
    if len(different_class_boundaries) > 0:
        # 计算到所有不同类别边界点的距离
        distances_to_boundaries = np.sqrt(
            np.sum((different_class_boundaries - sample_point)**2, axis=1)
        )
        min_distance = np.min(distances_to_boundaries)
    else:
        # 如果没有不同类别的边界点，设置为一个大值
        min_distance = float('inf')
    
    distances.append(min_distance)

distances = np.array(distances)
print(distances.shape)


# 计算所有690个点到最近不同类别边界点的距离和目标类别
distances = []
target_classes = []  # 记录最近不同类别样本的类别
boundary_points_array = np.array(boundary_points)
boundary_classes_array = np.array(boundary_classes)

for i in tqdm.tqdm(range(len(scaled_tsne_points)), desc="Distance Sorting..."):
    sample_point = scaled_tsne_points[i]
    sample_class = tsne_classes[i]
    
    # 找到所有不同类别的边界点
    different_class_mask = boundary_classes_array != sample_class
    different_class_boundaries = boundary_points_array[different_class_mask]
    different_class_labels = boundary_classes_array[different_class_mask]
    
    if len(different_class_boundaries) > 0:
        # 计算到所有不同类别边界点的距离
        distances_to_boundaries = np.sqrt(
            np.sum((different_class_boundaries - sample_point)**2, axis=1)
        )
        min_distance_idx = np.argmin(distances_to_boundaries)
        min_distance = distances_to_boundaries[min_distance_idx]
        target_class = different_class_labels[min_distance_idx]
    else:
        # 如果没有不同类别的边界点，设置为一个大值
        min_distance = float('inf')
        target_class = -1  # 无效类别
    
    distances.append(min_distance)
    target_classes.append(target_class)

distances = np.array(distances)
target_classes = np.array(target_classes)

# 找出距离最小的前5%样本
num_samples = len(distances)
top_5_percent_count = int(np.ceil(num_samples * 0.05))
print(f"前5%样本数量: {top_5_percent_count}")

# 获取距离最小的前5%样本的索引
sorted_indices = np.argsort(distances)
top_5_percent_indices = sorted_indices[:top_5_percent_count]

# 加载原始图片信息
image_index = torch.load('./DMP/dota/all_b.pth').numpy()
height_index = torch.load('./DMP/dota/all_h.pth').numpy()
width_index = torch.load('./DMP/dota/all_w.pth').numpy()

# 创建包含四个维度的数组：[图片索引, 高度, 宽度, 目标类别]
# ...existing code...

# 创建包含四个维度的数组：[图片索引, 高度, 宽度, 目标类别]
top_5_percent_data = np.column_stack([
    image_index[top_5_percent_indices],      # 第0列：原始图片索引
    height_index[top_5_percent_indices],     # 第1列：高度
    width_index[top_5_percent_indices],      # 第2列：宽度
    target_classes[top_5_percent_indices]    # 第3列：目标类别
])

print(f"去重前数据数组形状: {top_5_percent_data.shape}")
print(f"去重前数组内容 [图片索引, 高度, 宽度, 目标类别]:")
print(top_5_percent_data)

# 对重复的图片索引进行去重，保留距离最小的样本
def remove_duplicates_keep_min_distance(data, indices, distances):
    """
    去除重复的图片索引，保留距离最小的样本
    
    Args:
        data: 包含[图片索引, 高度, 宽度, 目标类别]的数组
        indices: 对应的原始索引
        distances: 对应的距离值
    
    Returns:
        去重后的数据数组
    """
    unique_data = []
    unique_indices = []
    unique_distances = []
    
    # 获取所有唯一的图片索引
    unique_image_indices = np.unique(data[:, 0])
    
    for img_idx in unique_image_indices:
        # 找到所有具有相同图片索引的样本
        mask = data[:, 0] == img_idx
        same_img_data = data[mask]
        same_img_indices = indices[mask]
        same_img_distances = distances[mask]
        
        # 找到距离最小的样本
        min_distance_idx = np.argmin(same_img_distances)
        
        # 保留距离最小的样本
        unique_data.append(same_img_data[min_distance_idx])
        unique_indices.append(same_img_indices[min_distance_idx])
        unique_distances.append(same_img_distances[min_distance_idx])
    
    return np.array(unique_data), np.array(unique_indices), np.array(unique_distances)

# 执行去重操作
top_5_percent_distances = distances[top_5_percent_indices]
unique_top_5_percent_data, unique_indices, unique_distances = remove_duplicates_keep_min_distance(
    top_5_percent_data, top_5_percent_indices, top_5_percent_distances
)

print(f"\n去重后数据数组形状: {unique_top_5_percent_data.shape}")
print(f"去重后数组内容 [图片索引, 高度, 宽度, 目标类别]:")
print(unique_top_5_percent_data)

# 显示去重前后的对比
print(f"\n去重统计:")
print(f"去重前样本数量: {len(top_5_percent_data)}")
print(f"去重后样本数量: {len(unique_top_5_percent_data)}")
print(f"去除了 {len(top_5_percent_data) - len(unique_top_5_percent_data)} 个重复样本")

# 检查是否还有重复的图片索引
unique_img_indices_after = np.unique(unique_top_5_percent_data[:, 0])
print(f"去重后唯一图片索引数量: {len(unique_img_indices_after)}")
print(f"是否完全去重: {len(unique_img_indices_after) == len(unique_top_5_percent_data)}")

# 显示详细信息
print(f"\n去重后各维度数据:")
print(f"图片索引: {unique_top_5_percent_data[:, 0]}")
print(f"高度: {unique_top_5_percent_data[:, 1]}")
print(f"宽度: {unique_top_5_percent_data[:, 2]}")
print(f"目标类别: {unique_top_5_percent_data[:, 3]}")
print(f"对应距离: {unique_distances}")

# 保存去重后的结果
np.save('./top_5_percent_samples_unique.npy', unique_top_5_percent_data)
print(f"\n去重后结果已保存到 'top_5_percent_samples_unique.npy'")

# 最终使用去重后的数据
top_5_percent_data = unique_top_5_percent_data

print(f"前5%样本数据数组形状: {top_5_percent_data.shape}")
print(f"数组内容 [图片索引, 高度, 宽度, 目标类别]:")
print(top_5_percent_data)

# 也可以分别访问各个维度
print(f"\n各维度数据:")
print(f"图片索引: {top_5_percent_data[:, 0]}")
print(f"高度: {top_5_percent_data[:, 1]}")
print(f"宽度: {top_5_percent_data[:, 2]}")
print(f"目标类别: {top_5_percent_data[:, 3]}")

# 保存结果（可选）
np.save('./top_5_percent_samples.npy', top_5_percent_data)

# ...existing code...

img_path = "DOTA/train/aaimages/part1/images"

import os
import glob

# 任务1: 读取目录中所有PNG图片的绝对路径
def get_png_files(directory):
    """
    获取指定目录下所有PNG文件的绝对路径
    """
    png_pattern = os.path.join(directory, "*.png")
    png_files = glob.glob(png_pattern)
    # 转换为绝对路径并排序
    png_files = [os.path.abspath(file) for file in png_files]
    png_files.sort()  # 确保顺序一致
    return png_files

# 获取所有PNG文件路径
all_png_files = get_png_files(img_path)
print(f"找到 {len(all_png_files)} 个PNG文件")

# 调试信息：显示前几个文件路径
if len(all_png_files) > 0:
    print(f"前3个PNG文件:")
    for i in range(min(3, len(all_png_files))):
        print(f"  {all_png_files[i]}")
else:
    print(f"警告: 在目录 {img_path} 中未找到PNG文件")
    print(f"请检查目录路径是否正确")

# 将所有PNG文件路径写入txt文件
all_images_txt_path = "./DMP/dota/all_png_images.txt"
with open(all_images_txt_path, 'w', encoding='utf-8') as f:
    for png_file in all_png_files:
        f.write(png_file + '\n')

print(f"所有PNG文件路径已保存到: {all_images_txt_path}")

# 任务2: 根据top_5_percent_data提取对应的图片信息
def create_selected_images_file(png_files, data, output_file):
    """
    根据索引提取对应的图片信息并保存到txt文件
    
    Args:
        png_files: 所有PNG文件路径列表
        data: top_5_percent_data数组 [图片索引, 高度, 宽度, 目标类别]
        output_file: 输出文件路径
    """
    print(f"准备写入 {len(data)} 个样本到 {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入文件头部说明
        f.write("# 格式: 图片路径 高度 宽度 目标类别\n")
        
        valid_count = 0
        for i, row in enumerate(data):
            img_idx = int(row[0])  # 图片索引
            height = int(row[1])   # 高度
            width = int(row[2])    # 宽度
            target_class = int(row[3])  # 目标类别
            
            # 检查索引是否有效
            if 0 <= img_idx < len(png_files):
                image_path = png_files[img_idx]  # 使用不同的变量名避免冲突
                # 写入格式: 图片路径 高度 宽度 目标类别
                f.write(f"{image_path} {height} {width} {target_class}\n")
                valid_count += 1
                
                # 显示前几个样本的详细信息
                if i < 3:
                    img_name = os.path.basename(image_path)
                    print(f"  样本 {i+1}: {img_name} -> 索引:{img_idx}, 高度:{height}, 宽度:{width}, 目标类别:{target_class}")
            else:
                print(f"警告: 图片索引 {img_idx} 超出范围 (0-{len(png_files)-1})")
        
        print(f"成功写入 {valid_count} 个有效样本")

# 创建选中图片的信息文件
selected_images_txt_path = "./DMP/dota/selected_top5_images.txt"
create_selected_images_file(all_png_files, top_5_percent_data, selected_images_txt_path)

print(f"选中的前5%图片信息已保存到: {selected_images_txt_path}")

# 显示结果统计
print(f"\n结果统计:")
print(f"总PNG文件数: {len(all_png_files)}")
print(f"选中的样本数: {len(top_5_percent_data)}")

# 验证输出文件
print(f"\n验证输出文件:")
try:
    with open(all_images_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"all_png_images.txt 包含 {len(lines)} 行")
        if len(lines) > 0:
            print(f"第一行: {lines[0].strip()}")
        
    with open(selected_images_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"selected_top5_images.txt 包含 {len(lines)} 行")
        if len(lines) > 1:  # 跳过注释行
            print(f"第一个数据行: {lines[1].strip()}")
            
except Exception as e:
    print(f"读取文件时出错: {e}")

# 检查目录是否存在
if not os.path.exists(img_path):
    print(f"错误: 目录 {img_path} 不存在!")
    print(f"当前工作目录: {os.getcwd()}")
    print("请检查路径是否正确")

