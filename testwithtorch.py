<<<<<<< HEAD
import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg11
import torchvision.models as models

# 重新定义你的模型类
class ModifiedVGG(models.VGG):
    def __init__(self):
        super(ModifiedVGG, self).__init__(make_layers(cfgs['A'], batch_norm=True, in_channels=1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# 实例化模型

model = ModifiedVGG()  # 这里需要重新定义你的模型类
model.load_state_dict(torch.load('digit_recognizer.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 设置模型为评估模式

# 2. 预处理图片
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),  # 确保图片大小与训练时一致
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


image = Image.open('input.png')
image_tensor = transform(image).unsqueeze_(0)  # 添加批次维度

# 3. 将图片转换为模型输入
image_tensor = image_tensor.to(device)

# 4. 前向传播
with torch.no_grad():  # 不需要计算梯度
    output = model(image_tensor)

# 5. 解释输出
_, predicted = torch.max(output, 1)
=======
import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg11
import torchvision.models as models

# 重新定义你的模型类
class ModifiedVGG(models.VGG):
    def __init__(self):
        super(ModifiedVGG, self).__init__(make_layers(cfgs['A'], batch_norm=True, in_channels=1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# 实例化模型

model = ModifiedVGG()  # 这里需要重新定义你的模型类
model.load_state_dict(torch.load('digit_recognizer.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 设置模型为评估模式

# 2. 预处理图片
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),  # 确保图片大小与训练时一致
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


image = Image.open('input.png')
image_tensor = transform(image).unsqueeze_(0)  # 添加批次维度

# 3. 将图片转换为模型输入
image_tensor = image_tensor.to(device)

# 4. 前向传播
with torch.no_grad():  # 不需要计算梯度
    output = model(image_tensor)

# 5. 解释输出
_, predicted = torch.max(output, 1)
>>>>>>> 81c37fa34b06a5f06fc0dd78f7c45157432b6766
print(f"The model predicts the digit is: {predicted.item()}")