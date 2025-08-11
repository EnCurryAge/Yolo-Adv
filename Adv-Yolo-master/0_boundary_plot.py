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
from pytorchyolo.utils.loss import compute_loss
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from pytorchyolo.utils.utils import rescale_boxes, non_max_suppression
from sklearn.manifold import TSNE
from scipy.spatial import distance
from core_adversarial_dbm.projection import nninv, qmetrics
import matplotlib.pyplot as plt
import numpy as np
import time

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"

start_time = time.time()

model_path = "config/dota-yolov3-416.cfg"
# weights_path = "DOTA/dota-yolov3-416_150000.weights"
weights_path = './config/yolo_dota_ckpt_repair.pth'
model = load_model(model_path, weights_path)
model = model.to(device)
model = model.eval()
# print(model)

# img_path = "DOTA/train/record.txt"
img_path = "DOTA/train/aaimages/part1/images"
img_size = 608
dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
dataloader = DataLoader(
        # subset_dataset,
        dataset,
        batch_size=16,
        shuffle=False,
        pin_memory=True)

inputs = []

def hook_fn(module, input, output):
    # input 是一个tuple
    inputs.append(input[0].detach().cpu())

# 注册钩子到第81层的Conv2d
handle = model.module_list[81][0].register_forward_hook(hook_fn)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
for (_, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))
    input_imgs = input_imgs.to(device)
    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        # detections = non_max_suppression(detections, conf_thres, nms_thres)
handle.remove() 
inputs = torch.cat(inputs)
output_patch = model.module_list[81][0](inputs.to(device)) 
print(output_patch.shape) # torch.Size([N, 255, 19, 19])

all_b = []
all_h = []
all_w = []
y_classif_train = []

for b in tqdm.tqdm(range(output_patch.shape[0]), desc="Processing patches"):
    for h in range(output_patch.shape[2]):
        for w in range(output_patch.shape[3]):
            patch = output_patch[b:b+1, :, h:h+1, w:w+1] #[1, 255, 1, 1]
            # print(patch.shape)
            yolo_output_patch = model.module_list[82][0](patch,input_imgs.size(2)) # torch.Size([1, 3, 85])
            # print(yolo_output_patch.shape) 

            out = non_max_suppression(yolo_output_patch, conf_thres=0.5, iou_thres=0.4) #[tensor[],...,...] -> (x1, y1, x2, y2, conf, cls)
            # filter
            if out[0].shape[0] != 0:
                all_b.append(b)
                all_h.append(h)
                all_w.append(w)
                y_classif_train.append(out[0][0][-1].item()) # 这里取的是第一个anchor的类别

all_b = torch.tensor(all_b, dtype=torch.long)
all_h = torch.tensor(all_h, dtype=torch.long)
all_w = torch.tensor(all_w, dtype=torch.long)
y_classif_train = torch.tensor(y_classif_train, dtype=torch.long)
torch.save(all_b, "./DMP/dota/all_b.pth")
torch.save(all_h, "./DMP/dota/all_h.pth")   
torch.save(all_w, "./DMP/dota/all_w.pth")
torch.save(y_classif_train, "./DMP/dota/y_classif_train.pth")
selected_features = inputs[all_b, :, all_h, all_w]  # shape: [N, 255]

X_proj = TSNE(random_state=420).fit_transform(
                # X_classif_train
                selected_features
            )

X_proj = minmax_scale(X_proj).astype(np.float32)
print(X_proj)
# 计算高维空间中样本之间的成对距离（压缩形式的距离矩阵）
# D_high = distance.pdist(X_classif_train)
D_high = distance.pdist(selected_features)
D_low = distance.pdist(X_proj)
# 计算每个样本点在高维与低维空间中的“连续性”得分（即邻居保持程度）
conts = qmetrics.per_point_continuity(D_high, D_low)

keep_percent = 0.9  # 设置保留得分前80%的样本（即丢弃连续性最差的20%）
# 根据连续性得分升序排序，选出得分高的那80%的样本索引
c_keep_ixs = np.argsort(conts)[int((1 - keep_percent) * len(conts)) :]
# 根据选出的索引从低维投影数据中过滤出高连续性子集
X_proj_filtered = X_proj[c_keep_ixs]

X_proj_train = X_proj_filtered.copy()
# X_high_train = X_classif_train[c_keep_ixs].copy()
# y_high_train = y_classif_train[c_keep_ixs].copy()


import numpy as np
y_classif_train = np.array(y_classif_train)
X_high_train = selected_features[c_keep_ixs].clone()
y_high_train = y_classif_train[c_keep_ixs].copy()

np.save(f"./DMP/dota/X_proj.npy", X_proj)
np.save(f"./DMP/dota/X_proj_train.npy", X_proj_train)
np.save(f"./DMP/dota/y_classif_train.npy", y_classif_train)
np.save(f"./DMP/dota/X_high_train.npy", X_high_train)
np.save(f"./DMP/dota/y_high_train.npy", y_high_train)

def train_nninv(
    X_proj,
    X_high,
    X_proj_val=None,
    X_high_val=None,
    epochs=1000,
    *,
    device: str = device,
) -> nninv.NNInv:
    model = nninv.NNInv(X_proj.shape[1], X_high.shape[1]).to(device=device)
    model.init_parameters()

    model.fit(
        TensorDataset(torch.tensor(X_proj, device=device), torch.tensor(X_high, device=device)),
        epochs=epochs,
        validation_data=None
        if X_proj_val is None
        else TensorDataset(
            torch.tensor(X_proj_val, device=device), torch.tensor(X_high_val, device=device)
        ),
        optim_kwargs={"lr": 1e-3},
    )
    return model
(
    X_proj_train,
    X_proj_val,
    X_high_train,
    X_high_val,
    y_high_train,
    _,
) = train_test_split(
    X_proj_train,
    X_high_train,
    y_high_train,
    stratify=y_high_train,
    train_size=0.8,
    random_state=420,
    # shuffle=True,
)
nninv_model = train_nninv(
    X_proj_train,
    X_high_train,
    X_proj_val,
    X_high_val,
)
nninv_model = nninv_model



torch.save(
    nninv_model.state_dict(),
    f"./DMP/dota/nninv_model.pth",
)

def load_nniv_model(
    X_proj,
    X_high,
    device: str = device,
):
    model = nninv.NNInv(X_proj.shape[1], X_high.shape[1]).to(device=device)
    model.load_state_dict(torch.load(f"./DMP/dota/nninv_model.pth"))
    return model
nninv_model = load_nniv_model(X_proj_train,
    X_high_train)

from core_adversarial_dbm.compute import dbm_manager, gradient, neighbors
def setup_figure():
    fig, ax = plt.subplots(
        1, 1, subplot_kw={"aspect": "equal"}, figsize=(8, 8), dpi=256
    )
    ax.axis("off")
    ax.set_autoscale_on(False)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 1.0)
    return fig, ax

class MapGenerator:
    def __init__(self, dbm_resolution: int) -> None:

        # self.n_classes = len(np.unique(y_classif_train))
        self.n_classes = 16

        self.scalar_map_imshow_kwargs = {
            "extent": (0.0, 1.0, 0.0, 1.0),
            "interpolation": "none",
            "origin": "lower",
            "cmap": "viridis",
        }
        self.categorical_map_imshow_kwargs = {
            **self.scalar_map_imshow_kwargs,
            "vmin": 0,
            "vmax": self.n_classes - 1,
            "cmap": "tab10" if self.n_classes <= 10 else "tab20",
        }

        self.dbm_resolution = dbm_resolution
        self.dbm_shape = (self.dbm_resolution, self.dbm_resolution)

        xx, yy = torch.meshgrid(
            torch.linspace(0.0, 1.0, dbm_resolution, device=device),
            torch.linspace(0.0, 1.0, dbm_resolution, device=device),
            indexing="xy",
        )
        self.grid_points = torch.stack([xx.ravel(), yy.ravel()], dim=1)
        self.dbm_data = dbm_manager.DBMManager(
            model,
            nninv_model,
            self.grid_points,
            self.n_classes,
        )
    
    def projection(self):
        fig, ax = setup_figure()
        ax.scatter(
            *X_proj.T,
            c=y_classif_train,
            cmap="tab10",
            vmin=0,
            vmax=self.n_classes - 1,
        )

        fig.savefig(f"./DMP/dota/Projected.png")
        plt.close(fig)
        np.save("Projected.npy", X_proj)

        fig, ax = setup_figure()
        ax.scatter(
            *X_proj.T,
            c=y_classif_train,
            s=20,
            edgecolors="#FFFFFF",
            linewidths=0.3,
            vmin=0,
            vmax=self.n_classes - 1,
            cmap=self.categorical_map_imshow_kwargs["cmap"],
        )
        fig.savefig(f"./DMP/dota/Projected_Colored_Transparent.png", transparent=True)
        plt.close(fig)

        fig, ax = setup_figure()
        ax.scatter(
            *X_proj.T,
            c="k",
            s=20,
        )
        fig.savefig(f"./DMP/dota/Projected_Black_Transparent.png", transparent=True)
        plt.close(fig)

    def dbm(self):
        dbm = self.dbm_data.get_dbm_data_yolo(blocking=True, imgsize = input_imgs.size(2))
        # print(dbm.shape)
        fig, ax = setup_figure()
        ax.imshow(
            dbm,
            **self.categorical_map_imshow_kwargs,
        )
        fig.savefig(f"./DMP/dota/DBM.png")
        np.save(f"./DMP/dota/DBM.npy", dbm)
        print(dbm.size)
        plt.close(fig)
        return dbm
    
    def overlay_dbm_on_projection(self):
        dbm = self.dbm_data.get_dbm_data(blocking=True)

        fig, ax = setup_figure()

        # 先画 DBM 背景（分类）
        ax.imshow(
            dbm,
            **self.categorical_map_imshow_kwargs,
        )
        ax.scatter(
            *X_proj.T,
            c=y_classif_train, # notice 300 ~ 400
            s=20,
            edgecolors="#FFFFFF",
            linewidths=0.3,
            vmin=0,
            vmax=self.n_classes - 1,
            cmap=self.categorical_map_imshow_kwargs["cmap"],
        )

        # 在TSNE图中添加数字
        # for i, (x, y) in enumerate(X_proj):
        #     ax.annotate(str(i), (x, y), fontsize=2)

        fig.savefig(f"./DMP/dota/DBM_Overlay_Projected_color.png", transparent=True)
        # fig.savefig(f"./DMP/dota/DBM_Overlay_Projected_color_repair.png", transparent=True)
        plt.close(fig)


map_gen = MapGenerator(
    256,
)

map_gen.projection()

dbm = map_gen.dbm()
map_gen.overlay_dbm_on_projection()

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")