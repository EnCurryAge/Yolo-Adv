import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from PIL.ImageFilter import GaussianBlur, SHARPEN, EDGE_ENHANCE

image_root = './P1148.png'
img = Image.open(image_root)
img = img.convert('RGB')

# 创建保存目录
output_dir = './enhanced_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 图像旋转
def rotate_image(image, angle=None):
    if angle is None:
        angle = random.randint(-30, 30)  # 随机旋转角度
    rotated = image.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    return rotated

# 2. 添加高斯噪声
def add_gaussian_noise(image, mean=0, std=1.0):
    img_array = np.array(image)
    noise = np.random.normal(mean, std, img_array.shape).astype(np.uint8)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

# 3. 更改对比度
def change_contrast(image, factor=None):
    if factor is None:
        factor = random.uniform(2.0, 4.0)  # 随机对比度因子
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

# 4. 更改饱和度
def change_saturation(image, factor=None):
    if factor is None:
        factor = random.uniform(2.0, 4.0)  # 随机饱和度因子
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

# 5. 随机切割再替换成原图大小
def random_crop_resize(image, crop_ratio=None):
    if crop_ratio is None:
        crop_ratio = random.uniform(0.3, 0.6)  # 随机裁剪比例
    
    width, height = image.size
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)
    
    # 随机选择裁剪位置
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    resized = cropped.resize((width, height), Image.Resampling.LANCZOS)
    return resized

# 6. 模糊处理
def blur_image(image, radius=None):
    if radius is None:
        radius = random.uniform(1.0, 2.0)  # 随机模糊半径
    return image.filter(GaussianBlur(radius=radius))

# 7. 锐化处理
def sharpen_image(image):
    return image.filter(SHARPEN)

# 8. 边缘增强处理
def edge_enhance_image(image):
    return image.filter(EDGE_ENHANCE)

# 9. 更改图片透明度
def change_transparency(image, alpha=None):
    """将图片调整为高透明度效果"""
    if alpha is None:
        alpha = random.uniform(0.3, 0.4)  # 随机透明度因子，0.3-0.4表示较高透明度

    # 创建一个与原图相同大小的白色背景
    background = Image.new('RGB', image.size, (255, 255, 255))
    
    # 将原图转换为RGBA模式以支持透明度
    if image.mode != 'RGBA':
        image_rgba = image.convert('RGBA')
    else:
        image_rgba = image.copy()
    
    # 调整透明度
    transparent_img = Image.blend(background, image, alpha)
    
    return transparent_img

# 执行所有数据增强操作并保存
print("开始进行数据增强操作...")

# 1. 保存旋转图像
rotated_img = rotate_image(img, 15)
rotated_img.save(os.path.join(output_dir, 'P1148_rotated.png'))
print("✓ 图像旋转完成")

# 2. 保存高斯噪声图像
noisy_img = add_gaussian_noise(img)
noisy_img.save(os.path.join(output_dir, 'P1148_gaussian_noise.png'))
print("✓ 添加高斯噪声完成")

# 3. 保存对比度调整图像
contrast_img = change_contrast(img, 1.5)
contrast_img.save(os.path.join(output_dir, 'P1148_contrast_enhanced.png'))
print("✓ 对比度调整完成")

# 4. 保存饱和度调整图像
saturation_img = change_saturation(img, 1.3)
saturation_img.save(os.path.join(output_dir, 'P1148_saturation_enhanced.png'))
print("✓ 饱和度调整完成")

# 5. 保存随机裁剪调整图像
crop_resize_img = random_crop_resize(img)
crop_resize_img.save(os.path.join(output_dir, 'P1148_crop_resize.png'))
print("✓ 随机裁剪调整完成")

# 6. 保存模糊处理图像
blurred_img = blur_image(img, 1.2)
blurred_img.save(os.path.join(output_dir, 'P1148_blurred.png'))
print("✓ 模糊处理完成")

# 7. 保存锐化处理图像
sharpened_img = sharpen_image(img)
sharpened_img.save(os.path.join(output_dir, 'P1148_sharpened.png'))
print("✓ 锐化处理完成")

# 8. 保存边缘增强处理图像
edge_enhanced_img = edge_enhance_image(img)
edge_enhanced_img.save(os.path.join(output_dir, 'P1148_edge_enhanced.png'))
print("✓ 边缘增强处理完成")

# 9. 保存透明度调整图像（压缩）
transparent_img = change_transparency(img)
transparent_img.save(os.path.join(output_dir, 'P1148_transparent.png'))
print("✓ 透明度调整完成")

print(f"\n所有数据增强操作完成！增强后的图片已保存到: {output_dir}")
print("生成的文件列表:")
for filename in os.listdir(output_dir):
    print(f"  - {filename}")