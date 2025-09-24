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
from pytorchyolo.utils.utils import rescale_boxes, non_max_suppression
from network import modify_all_yolo_output_layers, restore_model_to_original_classes
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"

start_time = time.time()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load model
print("=== 加载原始模型 ===")
model_path = "config/dota-yolov3-416.cfg"
weights_path = "DOTA/dota-yolov3-416_150000.weights"
# weights_path = './config/yolo_dota_ckpt_repair.pth'
model = load_model(model_path, weights_path)
# print(model)
new_model, _ = modify_all_yolo_output_layers(model, old_num_classes=16, new_num_classes=17)
new_model = new_model.to(device)
del model

for yolo_layer in new_model.yolo_layers:
    if hasattr(yolo_layer, 'num_classes'):
        yolo_layer.num_classes = 17
        yolo_layer.no = 17 + 5  # 更新输出数
print("=== 更改模型分类层完成 ===")
# print(model)
# model = model.train()

# load dataset eval
# img_path = "DOTA/train/images"
# img_size = 608
# dataset = ImageFolder(
#         img_path,
#         transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))

# load dataset train
img_path = "DOTA/adv_train/record.txt"
img_size = 608
dataset = ListDataset(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))

num_samples = int(len(dataset) * 0.25)
indices = random.sample(range(len(dataset)), num_samples)
subset_dataset = Subset(dataset, indices)

dataloader = DataLoader(
        subset_dataset,
        # dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        pin_memory=True)

# print(len(dataloader))
data_iter = iter(dataloader)
imgs_root, imgs, targets = next(data_iter)
# print(targets.size())
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
input_imgs = Variable(imgs.type(Tensor))

imgs = imgs.to(device)

# with torch.no_grad():
#     outputs = new_model(imgs)
#     print(outputs[0].size(), outputs[1].size(), outputs[2].size())
#     detections = non_max_suppression(outputs)
#     detections = rescale_boxes(detections[0], img_size, imgs.shape[:2])
#     print(detections)

# 冻结除输出层外的所有参数
def freeze_model_except_output_layers(model):
    """
    冻结除了conv_81, conv_93, conv_105之外的所有参数
    """
    target_layers = ['conv_81', 'conv_93', 'conv_105']
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        # 检查参数是否属于目标输出层
        is_output_layer = False
        for target_layer in target_layers:
            if target_layer in name:
                is_output_layer = True
                break
        
        if is_output_layer:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            frozen_params.append(param)
    
    # 计算参数总数
    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = sum(p.numel() for p in frozen_params)
    total_count = trainable_count + frozen_count
    
    return trainable_params

# 执行参数冻结
print("=== 冻结模型参数仅微调分类层 ===")
trainable_params = freeze_model_except_output_layers(new_model)

# 设置优化器，只优化可训练参数
optimizer = optim.Adam(
        trainable_params,
        lr=new_model.hyperparams['learning_rate'],
        weight_decay=new_model.hyperparams['decay'],
                       )

# -------------------------training----------------------------------
print("=== 开始模型边界修补 ===")
for epoch in range(1, 3):

        new_model.train()  # Set model to training mode

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
        #     print(targets.size())
            targets = targets.to(device)
            outputs = new_model(imgs)
            targets[:, 1] = 16
            _, _, cls_loss = compute_loss(outputs, targets, new_model)

            cls_loss.backward()

            if batches_done % new_model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = new_model.hyperparams['learning_rate']
                if batches_done < new_model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / new_model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in new_model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()
                print('\n Optimizer Step Done.')
# ---------------------------------------------------------------------------------------
checkpoint_path = f"./config/yolo_dota_ckpt_repair.pth"
new_model = restore_model_to_original_classes(new_model, original_classes=16, shadow_classes=17)
for yolo_layer in new_model.yolo_layers:
    if hasattr(yolo_layer, 'num_classes'):
        yolo_layer.num_classes = 16
        yolo_layer.no = 16 + 5  # 更新输出数
torch.save(new_model.state_dict(), checkpoint_path)
print("=== 模型结构还原并保存 ===")

end_time = time.time()
print(f"=== 模型边界修补完成，耗时: {end_time - start_time:.2f} 秒 ===")