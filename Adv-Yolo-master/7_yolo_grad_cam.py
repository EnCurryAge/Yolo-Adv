import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torchvision.transforms as transforms

class YOLOGradCAM:
    def __init__(self, model, device="cuda:0", img_size=608, num_classes=16):
        """
        PyTorch版本的YOLOv3 GradCAM实现 - 16分类任务
        
        Args:
            model: 加载的YOLOv3模型
            device: 设备 (cuda:0 或 cpu)
            img_size: 输入图像尺寸
            num_classes: 类别数量 (16)
        """
        self.model = model
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.model.eval()
        
        # 存储特征图和梯度
        self.feature_maps = []
        self.gradients = []
        
        # 注册hook来获取特征图和梯度
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向传播的hook"""
        def forward_hook(module, input, output):
            self.feature_maps.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients.append(grad_output[0])
        
        # 获取YOLOv3的三个输出层
        target_layers = self._get_target_layers()
        
        for layer in target_layers:
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)
    
    def _get_target_layers(self):
        """获取目标层，需要根据具体模型结构调整"""
        target_layers = []
        
        # 遍历模型找到最后的卷积层
        # 通常YOLOv3有三个输出头，对应不同尺度
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 查找输出通道数为 3*(5+num_classes) = 3*21 = 63 的卷积层
                if module.out_channels == 3 * (5 + self.num_classes):
                    target_layers.append(module)
        
        # 如果没找到特定的层，尝试获取最后几个卷积层
        if not target_layers:
            all_conv_layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    all_conv_layers.append(module)
            target_layers = all_conv_layers[-3:]  # 取最后三个卷积层
        
        return target_layers
    
    def preprocess_image(self, image_path):
        """预处理输入图像"""
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        # 调整尺寸
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # 转换为tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        # boxes = np.zeros((1, 5))
        # transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(self.img_size)])
        # prep_img = np.array(
        #             Image.open(image_path).convert('RGB'),
        #             dtype=np.uint8)
        # prep_img,_ = transform((prep_img, boxes))
        # image_tensor = prep_img.unsqueeze(0).to(self.device)
        
        return image_tensor, original_image
    
    def generate_gradcam(self, image_path, target_class=None):
        """生成GradCAM热力图"""
        # 清空之前的特征图和梯度
        self.feature_maps = []
        self.gradients = []
        
        # 预处理图像
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # 前向传播
        image_tensor.requires_grad_(True)
        
        outputs = self.model(image_tensor)
        
        # 处理输出格式: [torch.Size([1, 3, 19, 19, 21]), ...]
        if target_class is not None:
            # 如果指定了目标类别，计算该类别的置信度
            target_scores = self._extract_class_scores(outputs, target_class)
        else:
            # 否则使用objectness scores
            target_scores = self._extract_objectness_scores(outputs)
        
        # 对所有尺度的分数求和作为目标
        total_score = sum(torch.sum(score) for score in target_scores)
        
        # 反向传播
        total_score.backward()
        
        # 生成热力图
        heatmaps = self._generate_heatmaps()

        # 前向传播
        # with torch.enable_grad():
        #     outputs = self.model(image_tensor)
            
        #     # 计算objectness scores
        #     objectness_scores = self._extract_objectness_scores(outputs)
            
        #     # 对每个尺度的objectness score求和作为目标
        #     total_objectness = sum(torch.sum(score) for score in objectness_scores)
            
        #     # 反向传播
        #     total_objectness.backward()
        
        # # 生成热力图
        # heatmaps = self._generate_heatmaps()

        
        return heatmaps, original_image
    
    def _extract_objectness_scores(self, outputs):
        """从模型输出中提取objectness scores"""
        objectness_scores = []
        
        for output in outputs:
            # 输出格式: (batch=1, anchors=3, grid_h, grid_w, predictions=21)
            # predictions = [x, y, w, h, objectness, class1, class2, ..., class16]
            # objectness在索引4的位置
            
            batch_size, num_anchors, grid_h, grid_w, num_predictions = output.shape
            
            # 提取objectness (索引4)
            obj_score = output[:, :, :, :, 4]  # shape: (1, 3, grid_h, grid_w)
            obj_score = torch.sigmoid(obj_score)  # 应用sigmoid激活
            
            # 在anchor维度上取最大值或平均值
            # obj_score = torch.mean(obj_score, dim=1)  # shape: (1, grid_h, grid_w)
            obj_score = torch.max(obj_score, dim=1)[0]
            
            objectness_scores.append(obj_score)
        
        return objectness_scores
    
    def _extract_class_scores(self, outputs, target_class, use_objectness=True):
        """从模型输出中提取特定类别的分数"""
        class_scores = []
        
        # for output in outputs:
        for i, output in enumerate(outputs):
            # 输出格式: (batch=1, anchors=3, grid_h, grid_w, predictions=21)
            batch_size, num_anchors, grid_h, grid_w, num_predictions = output.shape
            
            # 提取类别分数 (索引5到20，共16个类别)
            class_probs = output[:, :, :, :, 5:]  # shape: (1, 3, grid_h, grid_w, 16)
            class_probs = torch.sigmoid(class_probs)  # 应用sigmoid激活
            
            # 提取目标类别的分数
            target_class_score = class_probs[:, :, :, :, target_class]  # shape: (1, 3, grid_h, grid_w)

            # print(f"Class {target_class} raw scores scale {i}: mean={target_class_score.mean().item():.6f}, max={target_class_score.max().item():.6f}")
            
            if use_objectness:
                # 结合objectness
                obj_score = torch.sigmoid(output[:, :, :, :, 4])  # objectness
                combined_score = obj_score * target_class_score
                # print(f"Combined with objectness scale {i}: mean={combined_score.mean().item():.6f}, max={combined_score.max().item():.6f}")
            else:
                combined_score = target_class_score
            
            # 检查是否所有值都为0或很小
            # if combined_score.max().item() < 1e-6:
            #     print(f"Warning: Very small scores detected for scale {i}, using alternative approach")
            #     # 使用softmax来增强差异
            #     class_probs_softmax = F.softmax(output[:, :, :, :, 5:], dim=-1)
            #     target_class_score = class_probs_softmax[:, :, :, :, target_class]
            #     if use_objectness:
            #         obj_score = torch.sigmoid(output[:, :, :, :, 4])
            #         combined_score = obj_score * target_class_score + 1e-6  # 添加小的常数避免0梯度
            #     else:
            #         combined_score = target_class_score + 1e-6
            
            # 在anchor维度上取最大值
            combined_score = torch.max(combined_score, dim=1)[0]  # shape: (1, grid_h, grid_w)
            
            class_scores.append(combined_score)


            
            # 结合objectness
            # obj_score = torch.sigmoid(output[:, :, :, :, 4])  # objectness
            # combined_score = obj_score * target_class_score
            
            # # 在anchor维度上取最大值
            # # combined_score = torch.mean(combined_score, dim=1)  # shape: (1, grid_h, grid_w)
            # combined_score = torch.max(combined_score, dim=1)[0]
            
            # class_scores.append(combined_score)
        
        return class_scores
    
    def _generate_heatmaps(self):
        """生成GradCAM热力图"""
        heatmaps = []
        
        # # 确保特征图和梯度数量匹配
        num_maps = min(len(self.feature_maps), len(self.gradients))
        
        for i in range(num_maps):
            feature_map = self.feature_maps[i]
            gradient = self.gradients[-(i+1)]  # 梯度是反向顺序
            
            # 计算权重（全局平均池化）
            weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
            
            # 加权求和
            heatmap = torch.sum(weights * feature_map, dim=1, keepdim=True)
            
            # ReLU激活
            heatmap = F.relu(heatmap)
            
            # 归一化
            heatmap = heatmap.squeeze().cpu().detach().numpy()
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            else:
                heatmap = np.zeros_like(heatmap)
            
            heatmaps.append(heatmap)

        # 反转梯度列表（因为backward hook是反向顺序）
        # gradients = self.gradients[::-1]
        
        # for i, (feature_map, gradient) in enumerate(zip(self.feature_maps, gradients)):
        #     # 计算权重（全局平均池化）
        #     weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
            
        #     # 加权求和
        #     heatmap = torch.sum(weights * feature_map, dim=1, keepdim=True)
            
        #     # ReLU激活
        #     heatmap = F.relu(heatmap)
            
        #     # 归一化
        #     heatmap = heatmap.squeeze().cpu().detach().numpy()
        #     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
        #     heatmaps.append(heatmap)
        

        
        return heatmaps
    
    def visualize_gradcam(self, heatmaps, original_image, alpha=0.4):
        """可视化GradCAM结果"""
        superimposed_imgs = []
        
        for i, heatmap in enumerate(heatmaps):
            # 调整热力图尺寸到原图大小
            heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            # 转换为0-255范围
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            
            # 应用颜色映射
            jet = cm.get_cmap("jet")
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap_uint8]
            
            # 调整尺寸并转换为uint8
            jet_heatmap = (jet_heatmap * 255).astype(np.uint8)
            
            # 叠加到原图
            superimposed_img = jet_heatmap * alpha + original_image * (1 - alpha)
            superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
            
            superimposed_imgs.append(superimposed_img)
        
        return superimposed_imgs
    
    def plot_results(self, superimposed_imgs, save_path=None):
        """绘制结果"""
        plt.rcParams["font.serif"] = ["Times New Roman"]
        fig = plt.figure(figsize=(15, 5))
        fig.tight_layout()
        fig.set_dpi(112)
        fig.suptitle('GradCAM Visualization for Noised Image (Original Class: large-vehicle)', fontsize=16)

        num_imgs = len(superimposed_imgs)
        axs = fig.subplots(1, num_imgs)
        
        if num_imgs == 1:
            axs = [axs]
        
        scale_names = ["Large Scale (19x19)", "Medium Scale (38x38)", "Small Scale (76x76)"]
        
        for i, img in enumerate(superimposed_imgs):
            scale_name = scale_names[i] if i < len(scale_names) else f"Scale {i+1}"
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].set_title(scale_name, fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()


from pytorchyolo.models import load_model
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
# import torchvision.transforms as transforms



# 设置参数
device = "cuda:0"
img_size = 608


# 设置绘制的类别
with open('DOTA/dota.names', 'r') as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]
print(class_names)
target_class = 14
# file_name_to_export = "plane"
file_name_to_export = class_names[target_class]


model_path = "config/dota-yolov3-416.cfg"
weights_path = "DOTA/dota-yolov3-416_150000.weights"

# 加载模型（需要你提供的load_model函数）
model = load_model(model_path, weights_path)
model = model.to(device)

# 创建GradCAM对象
# grad_cam = YOLOGradCAM(model, device, img_size)
grad_cam = YOLOGradCAM(model, device, img_size, num_classes=16)

# 生成GradCAM
# image_path = "./DOTA/adv_train/images/P1900.png"  # 替换为实际图像路径
image_path = 'DOTA/adv_train/images/P1900_n.png'
heatmaps, original_image = grad_cam.generate_gradcam(image_path,target_class= target_class)
# heatmaps, original_image = grad_cam.generate_gradcam(image_path)

# 可视化结果
superimposed_imgs = grad_cam.visualize_gradcam(heatmaps, original_image)
grad_cam.plot_results(superimposed_imgs, save_path=f"./DMP/dota/gradcam.png")

print(f"GradCAM visualization completed for {len(heatmaps)} scales")
print(f"Output shapes: {[hm.shape for hm in heatmaps]}")
