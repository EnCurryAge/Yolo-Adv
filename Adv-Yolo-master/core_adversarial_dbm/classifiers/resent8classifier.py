import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

device = "cuda" if T.cuda.is_available() else "cpu"
class ResNet18Classifier(nn.Module):
    def __init__(self,
                 n_classes: int = 10,
                 freeze_backbone: bool = True):  # 默认不冻结
        super().__init__()

        # 从头初始化 ResNet18，不加载预训练权重
        # self.backbone = resnet18(pretrained=True)
        # self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_classes)

        full_resnet = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(full_resnet.children())[:-1])  # 去掉fc
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        # return self.backbone(x)
        features = self.extract_features(x)
        return self.fc(features)

    def activations(self, x):
        return self.extract_features(x)

    def classify(self, x):
        return T.argmax(self.forward(x), dim=1)
    
    def classify_features(self, features):
        """
        使用特征提取器提取特征后进行分类
        """
        return T.argmax(self.fc(features), dim=1)

    def prob_best_class(self, x):
        return T.max(self.forward(x), dim=1)[0]

    def classification_entropy(self, x):
        probs = self.forward(x)
        entropy = T.where(probs > 0, -T.log(probs) * probs, 0.0)
        return (entropy / T.log(T.tensor(probs.size(1), device=probs.device))).sum(dim=1)

    def init_parameters(self):
        # 初始化所有参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def fit(self, dataset, epochs: int, optim_kwargs: dict = {}):
        train_dl = DataLoader(dataset, batch_size=128, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = T.optim.Adam(self.parameters(), lr=0.001, **optim_kwargs)  # 训练所有参数

        # loop = trange(epochs)
        for e in range(epochs):
            # loop = tqdm(train_dl, desc=f"Epoch {e+1}/{epochs}")
            epoch_loss = 0.0
            epoch_n = 0
            for batch in tqdm(train_dl, desc=f"Epoch {e+1}/{epochs}"):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                self.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets.long())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * targets.size(0)
                epoch_n += targets.size(0)
            # loop.set_description(f"Loss: {epoch_loss / epoch_n:.4f}")
            print(f"Epoch {e+1}/{epochs}, Loss: {epoch_loss / epoch_n:.4f}")
        # 🔍 模型评估（训练集准确率）
        self.eval()  # 关闭 Dropout/BatchNorm
        correct = 0
        total = 0
        with T.no_grad():
            for inputs, targets in DataLoader(dataset, batch_size=256):  # 批量评估
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.classify(inputs)
                correct += (outputs == targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        print(f"Training Accuracy: {acc:.4f}")
        
        return self
    
    def extract_features(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x
