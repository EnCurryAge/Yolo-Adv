## 项目简介
航天领域视觉模型的运行环境苛刻，时常面临**强干扰**和**强延时**的极端条件，因此常用的对抗漏洞测试方法很难直接套用。综合考虑现有对抗技术和漏洞定位方法的**计算资源消耗大**且**效率低**、**依赖大量原始数据分布**、以及难以快速实定位和防御等缺陷，本项目设计了一种新型的对抗敏感性评估和修补技术，包含下列核心技术组件：

- 对抗性决策边界可视化
- 对抗敏感性样本定位
- 对抗补丁攻击
- 对抗路径揭示
- 对抗边界修补

经过实验验证，本技术显著优化现有方法的**计算效率**、**数据依赖**、和**漏洞覆盖广度**。
<br>

## 准备工作
1. 首先借助Anaconda环境管理软件安装虚拟环境：
``` python
conda create -n adv-yolo python==3.11
```

2. 检查主机CUDA版本，激活环境，并安装指定版本的Pytorch和Torchvision:
``` python
nvidia-smi
conda activate adv-yolo
pip install torch==2.6.0 torchvision==0.21.0 -f https://mirrors.aliyun.com/pytorch-wheels/cu126
```

3. 按照requirements.txt安装全部特定版本的依赖库：
``` python
pip install -r requirements.txt
```

4. 下载预训练的模型和数据(以DOTA v1.5数据集和对应的Yolo v3模型为例)。由于上传文件大小限制，此处仅仅提供数据集及模型的下载链接：

- DOTA v1.5数据集：<https://captain-whu.github.io/DOTA/index.html>
- iSAID数据集：<https://captain-whu.github.io/iSAID/dataset.html>
- COCO数据集：<https://cocodataset.org>
- 使用DOTA v1.5预训练的Yolo v3模型：<https://drive.google.com/drive/folders/1Y-W2npeaqflfO8IUA7gx9PzmesaSl9rY?usp=sharing>

## 代码结构
```
Adv-Yolo-master
├── pytorchyolo
|   ├── utils
│   ├── detect.py
│   ├── models.py
│   ├── test.py
│   └── train.py
└── 0_boundary_plot.py
└── 1_sensitive_location.py
└── 2_adv_patch_gen.py
└── 3_yolo_detection.py
└── 4_yolo_map_cal.py
└── 5_adv_layer_location.py
└── 6_boundary_repair.py
└── 7_yolo_grad_cam.py
└── 8_input_noise.py
└── dota_utils.py
└── dota2yolo.py
```
以下详细介绍核心代码的组成部分:

- `dota2yolo.py`: 由于DOTA数据集的原始标注方式与COCO不同，因此需要借助此代码文件将DOTA数据集**转换**为COCO即原生Yolo模型可以读取的数据格式。
<br>

- `0_boundary_plot.py`: 此代码文件由于借助原始模型和它10%以下的原始训练数据完成决策边界可视化，观察和分析样本特征点的分布与决策边界的关系，即可**定位出高对抗敏感性的样本**。
<br>

- `1_sensitive_location.py`: 此代码文件在决策边界可视化的基础上，借助邻域类别差异检测筛选出高对抗敏感性的样本。核心依据是：*非常靠近本类别决策边界或跨越决策边界的样本点更容易被噪声干扰，以至于被模型错误分类，因此具有更高的对抗敏感性*。此步骤会生成一个.txt文件，其中包含定位出的敏感样本的**绝对路径**，敏感patch的**横纵坐标**，以及距离其最近的**对抗目标类别**。
<br>

- `2_adv_patch_gen.py`: 本代码文件分批次并行处理上述筛选出的高对抗敏感性样本，并在每张图片的目标横纵坐标处添加和优化对抗噪声，使其被模型错误分类为目标类别。注意：此环节中，原始模型的**全部参数是冻结的**。
<br>

- `3_yolo_detection.py`: 此代码文件使用原始模型，对原始图片和对抗加噪图片同时**进行Yolo目标检测**，并将结果保存为.jpg格式，便于对比和分析对抗结果。
<br>

- `4_yolo_map_cal.py`: 此代码文件借助原始模型，计算模型在特定批次原始图片和对抗加噪图片上的**mAP指标**，根据对抗前后的ΔmAP指标可大致评估对抗攻击的效果。
<br>

- `5_adv_layer_location.py`: 此代码文件借助正常-加噪图片对，根据模型各层的激活值差别，筛选出对于对抗噪声奏效贡献最大的部分参数，据此**完成触发路径定位**。若实验验证中，对筛选出的触发参数部分进行剪枝后，对抗噪声的误导效果消除，则表明触发路径检测正确。
<br>

- `6_boundary_repair.py`: 此代码文件借助Boundary Expanding技术[1](此技术为团队原创，相关论文发表于CVPR 2023和ACM SIGKDD 2025)，对于之前筛选出的高对抗敏感性进行边界扩张，**定向提升模型整体的对抗鲁棒性**。具体实现方法为：引入一个"影子"类别对敏感样本进行微调，重塑这部分样本的预测分布，**在10个训练step之内**即可提升该部分样本的对抗鲁棒性。
<br>

- `7_yolo_grad_cam.py`: 此代码文件借助Grad-Cam可视化技术，对于对抗攻击和防御前后的模型分类梯度以及核心注意力进行可视化，有助于分析模型在对抗攻击和防御下的预测行为。
<br>

- `8_input_noise.py`: 此代码文件模拟航天视觉检测领域的常见输入噪声，包括模糊处理、高斯噪声、高透明度处理、高对比度处理、高饱和度处理、随机旋转裁剪等。可用于识别输入噪声引发的对抗漏洞。
<br>

- `pytorchyolo/detect.py`: 此代码文件可帮助用户自行对任意图片进行Yolo目标检测测试，结果直接输出为.jpg文件。
<br>

- `pytorchyolo/test.py`: 此代码文件可帮助用户自行对任意模型进行推理测试，评估模型正常训练效果及实际可用性。
<br>

- `pytorchyolo/train.py`: 此代码文件可帮助用户自行对多种Yolo模型的loss进行改造，以对模型进行保护，或探索不同类别的Yolo对抗攻击。


## 项目团队论文参考
```
@inproceedings{chen2023boundary,
  title={Boundary unlearning: Rapid forgetting of deep networks via shifting the decision boundary},
  author={Chen, Min and Gao, Weizhuo and Liu, Gaoyang and Peng, Kai and Wang, Chen},
  booktitle={Proceedings of the IEEE CVPR},
  pages={7766--7775},
  year={2023}
}
```
<br>


```
@inproceedings{,
  title={United We Stand, Divided We Fall: Fingerprinting Deep Neural Networks via Adversarial Trajectories}, 
  author={Xu, Tianlong and Wang, Chen and Liu, Gaoyang and Yang, Yang and Peng, Kai and Liu, Wei},
  year={2024},
  booktitle={Proceedings of NeurIPS}
}
```
<br>

```
@inproceedings{,
  title={From Expansion to Retraction: Long-tailed Machine Unlearning via Boundary Manipulation}, 
  author={Chen, Min and Gao, Weizhuo and Wang, Chen Liu, Gaoyang and Ahmed M. Abdelmoniem and Peng, Kai},
  year={2025},
  booktitle={Proceedings of ACM SIGKDD}
}
```
