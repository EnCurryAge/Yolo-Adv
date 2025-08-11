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
from pytorchyolo.test import evaluate_model_file

model_path = "config/dota-yolov3-416.cfg"
weights_path = "DOTA/dota-yolov3-416_150000.weights"
img_path = "DOTA/adv_train/record.txt" # NORMAL IMAGES
img_path = "DOTA/adv_train/record_copy.txt" # NOISED IMAGES
# with open('DOTA/dota.names', 'r') as f:
#     class_names = [line.strip() for line in f.readlines() if line.strip()]
# print(class_names)
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

outputs = evaluate_model_file(
    model_path=model_path,
    weights_path=weights_path,
    img_path=img_path,
    class_names=class_names,
    batch_size=4,
    img_size=416,
    n_cpu=0,
    iou_thres=0.25,
    conf_thres=0.25,
    nms_thres=0.25,
)