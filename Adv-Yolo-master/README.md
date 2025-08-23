# 2025第十九届“挑战杯-揭榜挂帅”擂台赛-代码附加材料及说明

项目选题：SH-04 针对航天智能算法可靠性与漏洞对抗解决方案挑战赛

作品名称：**基于决策空间的人工智能视觉模型对抗脆弱性评估**

参赛院校：华中科技大学电子信息与通信学院

项目负责人：王子雄

指导老师：刘高扬，王琛


### 项目简介
航天领域视觉模型的运行环境苛刻，时常面临**强干扰**和**强延时**的极端条件，因此常用的对抗漏洞测试方法很难直接套用。综合考虑现有对抗技术和漏洞定位方法的**计算资源消耗大**且**效率低**、**依赖大量原始数据分布**、以及难以快速实定位和防御等缺陷，本项目设计了一种新型的对抗敏感性评估和修补技术，包含下列核心技术组件：

- 模型决策边界可视化
- 对抗敏感性区域定位
- 对抗漏洞定位及验证
- 对抗触发路径揭示
- 对抗边界修补

经过实验验证，本技术显著优化现有方法的**计算效率**、**数据依赖**、和**漏洞覆盖广度**。
<br>

### 准备工作
1. 首先借助[Anaconda](https://www.anaconda.com/download)环境管理软件安装虚拟环境：
```
conda create -n adv-yolo python==3.11
```

2. 检查主机CUDA版本，激活环境，并安装指定版本的Pytorch和Torchvision，建议安装GPU版本以加速运算，并且推荐从国内镜像直接安装:
  (本项目的运行环境为Windows-11系统下双卡4090服务器，CUDA版本12.6，内存为64G)
```
nvidia-smi
conda activate adv-yolo
pip install torch==2.6.0 torchvision==0.21.0 -f https://mirrors.aliyun.com/pytorch-wheels/cu126
```

3. 按照requirements.txt安装全部特定版本的依赖库：
```
pip install -r requirements.txt
```

4. 由于系统上传文件大小限制，此处提供**数据集**以及**模型预训练权重**的下载链接：

- COCO数据集：<https://cocodataset.org>
- DOTA v1.5数据集：<https://captain-whu.github.io/DOTA/index.html>
- iSAID数据集：<https://captain-whu.github.io/iSAID/dataset.html>
- 使用DOTA v1.5预训练的Yolo v3模型及配置文件：<https://drive.google.com/drive/folders/1Y-W2npeaqflfO8IUA7gx9PzmesaSl9rY?usp=sharing>
- DOTA v1.5-Yolo v3细节可参考：<https://github.com/postor/DOTA-yolov3/tree/master>
- DOTA v1.5-Yolo v5模型权重和细节可参考：<https://github.com/tecsai/YOLOv5_DOTA_OBB>.
- DOTA v1.5-Yolo v7可参考：<https://github.com/WongKinYiu/yolov7/tree/v0.1>.
- DOTA v1.5-Yolo v8可参考：<https://github.com/quantumxiaol/yolov8-small-target-detection>.
- DOTA v1.5-Yolo v9可参考：<https://github.com/davidgeorgewilliams/Aerial-YOLO-DOTA>.

### 代码结构
``` python
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

- `0_boundary_plot.py`: 此代码文件由于借助原始模型和它10%以下的原始训练数据完成决策边界可视化，观察和分析样本特征点的分布与决策边界的关系，即可**定位出高对抗敏感性的样本所在的部分区域**。
<br>

- `1_sensitive_location.py`: 此代码文件在决策边界可视化的基础上，借助邻域类别差异检测筛选出高对抗敏感性的样本。核心依据是：*非常靠近本类别决策边界或跨越决策边界的样本点更容易被噪声干扰，以至于被模型错误分类，因此具有更高的对抗敏感性*。此步骤会生成一个.txt文件，其中包含定位出的敏感样本的**绝对路径**，敏感patch的**横纵坐标**，以及距离其最近的**对抗目标类别**。
<br>

- `2_adv_patch_gen.py`: 本代码文件分批次并行处理上述筛选出的高对抗敏感性样本，并在每张图片的目标横纵坐标处添加和优化对抗噪声，使其被模型错误分类为目标类别。注意：此环节中，原始模型的**全部参数是冻结的**。
<br>

- `3_yolo_detection.py`: 此代码文件使用原始模型，对原始图片和对抗加噪图片同时**进行Yolo目标检测**，并将结果保存为.jpg格式，便于对比和分析对抗结果。经过模型正向预测，即可获知所定位的对抗敏感漏洞是否正确。
<br>

- `4_yolo_map_cal.py`: 此代码文件借助原始模型，计算模型在特定批次原始图片和对抗加噪图片上的**mAP指标**，根据对抗前后的ΔmAP指标可大致评估对抗漏洞挖掘的正确性。
<br>

- `5_adv_layer_location.py`: 此代码文件借助正常-加噪图片对，根据模型各层的激活值差别，筛选出对于对抗噪声奏效贡献最大的部分参数，据此**完成触发路径定位**。若实验验证中，对筛选出的触发参数部分进行剪枝后，对抗噪声的误导效果消除，则表明触发路径检测正确。此代码文件另可额外用于模拟参数漂移(如模型量化或参数噪声)引发的对抗漏洞。
<br>

- `6_boundary_repair.py`: 此代码文件借助Boundary Expanding技术(此技术为团队原创，相关论文发表于CVPR 2023和ACM SIGKDD 2025)，对于之前筛选出的高对抗敏感性进行边界扩张，**定向提升模型整体的对抗鲁棒性**。具体实现方法为：引入一个"影子"类别对敏感样本进行微调，重塑这部分样本的预测分布，**在10个训练step之内**即可提升该部分样本的对抗鲁棒性。
<br>

- `7_yolo_grad_cam.py`: 此代码文件借助Grad-Cam可视化技术，对于对抗攻击和防御前后的**模型分类损失梯度**以及**核心注意力**进行可视化，有助于分析模型在对抗攻击和防御下的预测行为。
<br>

- `8_input_noise.py`: 此代码文件模拟航天视觉检测领域的常见输入噪声，包括模糊处理、高斯噪声、高透明度处理、高对比度处理、高饱和度处理、随机旋转裁剪等。可用于识别**输入噪声引发的对抗漏洞**。
<br>

- `pytorchyolo/detect.py`: 此代码文件可帮助用户自行对任意图片进行Yolo目标检测测试，结果直接输出为.jpg文件。
<br>

- `pytorchyolo/test.py`: 此代码文件可帮助用户自行对任意模型进行推理测试，评估模型正常训练效果及实际可用性。
<br>

- `pytorchyolo/train.py`: 此代码文件可帮助用户自行对多种Yolo模型的loss进行改造，以对模型进行保护，或探索不同种类的Yolo对抗攻击。


### 运行结果概述

1. 对于使用DOTA v1.5数据集(2806张图片)训练的Yolo v3模型，使用其10%的训练数据进行决策边界绘制，并定位出高对抗敏感性样本，在动态处理批次设为8时，整体运行时间为74.3s，从运行结果中可以获知应当在哪些图片的哪个patch添加和优化对抗噪声补丁。

2. 向高对抗敏感性样本添加噪声仅需至多48个训练step，总体耗时为108.03s，且对抗补丁的攻击指标为：ASR=82.60%，ΔmAP=9.24%，此处我们的对抗补丁攻击仅仅影响被添加噪声的patch，而图片的其余部分预测不变。同时可以借助在正常图片和加噪图片上的各一次正向推理，定位出模型对抗效果的触发路径(可记录特定层的特定参数)，剪枝这部分参数即可使对抗补丁失效。

3. 本项目设计提出的决策边界修补技术，相比于传统对抗训练(如DP-SGD)显著提高了保护效率，每个待保护样本仅需额外8个训练step的微调，整体耗时仅为2.62s。

4. 通过以上代码可以模拟输入噪声、参数漂移和场景迁移等3种常见的场景降级。

5. 整体系统提供决策边界可视化和Grad-Cam损失梯度和注意力可视化，两种可解释性方法。

6. 本系统支持并行处理优化，当动态处理批次提高至64时，单次测试中整个系统所需的推理次数降低至63，远小于1000次。


### 项目团队相关论文参考
1. 此论文详细介绍了在模型决策空间中寻找单一样本的最近且分类错误的邻域样本的方法，此方法是本项目定位高敏感样本并引入对抗噪声补丁的基础。同时，此论文还引入了决策边界扩张修补技术，该技术构成本项目对于高对抗敏感度样本进行保护的核心理论。
    ``` bibtex
    @inproceedings{chen2023boundary,
      title={Boundary unlearning: Rapid forgetting of deep networks via shifting the decision boundary},
      author={Chen, Min and Gao, Weizhuo and Liu, Gaoyang and Peng, Kai and Wang, Chen},
      booktitle={Proceedings of the IEEE CVPR},
      pages={7766--7775},
      year={2023}
    }
    ```
    <br>

2. 此论文包含决策边界可视化的研究，以及样本特征点与决策边界关系的讨论。
    ``` bibtex
    @inproceedings{xu2024united,
      title={United We Stand, Divided We Fall: Fingerprinting Deep Neural Networks via Adversarial Trajectories}, 
      author={Xu, Tianlong and Wang, Chen and Liu, Gaoyang and Yang, Yang and Peng, Kai and Liu, Wei},
      year={2024},
      booktitle={Proceedings of NeurIPS}
    }
    ```
    <br>

3. 此论文继续推进了决策边界修补重塑的研究工作，为后续算法更新迭代奠定基础。
    ``` bibtex
    @inproceedings{chen2025from,
      title={From Expansion to Retraction: Long-tailed Machine Unlearning via Boundary Manipulation}, 
      author={Chen, Min and Gao, Weizhuo and Wang, Chen Liu, Gaoyang and Ahmed M. Abdelmoniem and Peng, Kai},
      year={2025},
      booktitle={Proceedings of ACM SIGKDD}
    }
    ```
