<<<<<<< HEAD


今年春假期间，我一位在南京同样是学习人工智能专业的朋友向我发来了求助。

任务的大致要求如下：

>  输入一张白底的图片，上面可能有各种颜色的印刷体阿拉伯数字，部分数字可能有一定旋转角度，请运用机器学习相关知识对图上的数字进行识别。输出一张图片，在原图的基础上用红色矩形框选出某一个数字所处的位置，然后在矩形附近打印出识别的结果。（如下图示）

![](https://files.mdnice.com/user/54972/b66a69a9-bb6b-4a6a-a1a8-dd452956099b.png)
![](https://files.mdnice.com/user/54972/432c9466-ef16-4e71-b795-725648375f18.png)

原本以为是一项非常简单的任务，但实际上自己做起来却发现效果却不尽人意，再加上这个学期课业压力有点大，自己在春假期间和五一假期期间都有所尝试，但是最后无疾而终。后来我请教了X老师，X老师明确指出这个任务其实并没有那么简单，建议我可以采用一些比较大的模型如VGG，再看看效果，并推荐了一位研究领域涉及图像的H老师给我。这几天也是开始请教H老师的一位博士学长，原本以为他能清楚地指导我该怎么做，但是事实上我们之间的对话更像是探讨。原因其实很简单，博士学长也并不是专门做这个方向的，所以很多时候我们AI领域的事还是得亲自尝试亲自复现，这也是我这个项目一个比较大的收获。

回到这个任务上来看，这个任务可以拆解为两个关键步骤：

1. 分离出待识别的数字的位置和图像

2. 采用模型对图像进行预测输出结果

![](https://files.mdnice.com/user/54972/ddc8d824-674c-4cfc-990c-6b40bf1ace85.png)
![](https://files.mdnice.com/user/54972/0f1d11fb-fc2b-4f08-9d16-d43aa0b9e77e.png)

对于任务2，一开始我自己尝试的时候，我自己爬了1000张训练数据（如上图示），每个数字100张128*128的黑白图片，同时发现网上的一些相关参考都是在tensorflow框架下实现的，于是一开始我采用keras的CNN来写脚本，效果非常糟糕。

我曾经还用过MINST手写数据集来训练一个CNN网络，结果效果也不尽人意。

暑招的时候酒店同房间的室友在完成一个科研任务，具体是用OCR整理Excel，我于是调用了TecentOCR的API写了一份脚本，效果很显著：

![](https://files.mdnice.com/user/54972/bc2c0758-9c06-48e6-b19b-fa59b6ccbec9.png)
![](https://files.mdnice.com/user/54972/0453fe5d-c342-4b89-9c48-a331878db5e3.png)

于是我以它的识别率为目标先用VGG11进行了一番尝试。同时我采用边缘探测的遍历所有的ROI（Region of Interest）以完成任务1。并在后续的尝试中基本保持这种方式。同时注意到分离出来的矩形可能是长方形，为了匹配我模型需要的128*128的尺寸，我们必须做一步transform，我采用的方式是把长方形的短边向两边扩增，使其成为一个正方形，再对新图像做一圈白色像素的padding后resize，这样能够保证样例与测试集数据的较为吻合。

```python
# 以下是train.py代码
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg11
import torchvision.models as models

# 1. 定义数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 将彩色图像转为灰度图像
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 2. 创建数据加载器
dataset = datasets.ImageFolder(root='trainpic', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

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


# 3. 定义模型
model = ModifiedVGG()

# model = vgg11(pretrained=False)
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, 10)])  # Add our layer with output size 10
model.classifier = nn.Sequential(*features)

# 4. 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练循环
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # 每100批打印一次
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    # 保存模型
    torch.save(model.state_dict(), 'digit_recognizer.pth')
```

```python
# 以下是部分的predict.ipynb代码
image = cv2.imread('input.png')
# image = cv2.imread('C.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用阈值处理以分离前景和背景
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 寻找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历每一个轮廓
for contour in contours:
    # 获取边界框
    x, y, w, h = cv2.boundingRect(contour)
    # 忽略太小的边界框，可能是噪声
    if w < 10 or h < 10:
        continue
    # 提取ROI
    roi = gray[y:y+h, x:x+w]

    # 将ROI转换为模型可以接受的格式
    roi = cv2.resize(roi, (32, 32))
    roi = transform(Image.fromarray(roi)).unsqueeze_(0).to(device)

    # 前向传播
    with torch.no_grad():
        output = model(roi)
        _, predicted = torch.max(output, 1)
        pred = predicted.item()

    # 绘制矩形和标签
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(image, str(pred), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 显示结果
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

而后对比效果，还是觉得浅层CNN的框架在处理这类问题可能更佳（当然除此以外还可以采用传统的一些机器学习模型如支持向量机等）

这里我自己用torch框架重写了CNN，发现识别效果骤增：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

from torchvision.transforms import functional as F
import random

class DynamicRotationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, angles=[-180, -90, -45, 0, 45, 90, 180], transform=None):
        self.dataset = datasets.ImageFolder(root_dir, transform=transforms.ToTensor())
        self.angles = angles
        self.transform = transform

    def __len__(self):
        return len(self.dataset) * len(self.angles)

    def __getitem__(self, idx):
        sample_idx = idx % len(self.dataset)
        angle_idx = idx // len(self.dataset)
        angle = self.angles[angle_idx]

        image, label = self.dataset[sample_idx]
        image = F.rotate(image, angle)

        if self.transform:
            image = self.transform(image)

        return image, label


train = datasets.ImageFolder(root='modified\trainpic', transform=data_transforms)
trainloader = DataLoader(train, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 32 * 32, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 120
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

print('Finished Training')

# 保存模型
torch.save(model.state_dict(), 'digit_recognition_model.pth')
Epoch 1, Loss: 1.9711331836879253
# Result:
# Epoch 2, Loss: 0.12846590229310095
# Epoch 3, Loss: 0.03395026904763654
# Epoch 4, Loss: 0.017787698379834183
# Epoch 5, Loss: 0.013690078725630883
# Epoch 6, Loss: 0.005368602876842488
# Epoch 7, Loss: 0.0020161027152880706
# Epoch 8, Loss: 0.0010407454797132232
# Epoch 9, Loss: 0.00045232775210024556
# Epoch 10, Loss: 0.00015856899085520126
# Epoch 11, Loss: 0.0001125274102378171
# Epoch 12, Loss: 9.382167240801209e-05
# Epoch 13, Loss: 9.087298360554996e-05
# Epoch 14, Loss: 7.821109579708718e-05
# Epoch 15, Loss: 7.269618754435214e-05
# Epoch 16, Loss: 6.695036722703662e-05
# Epoch 17, Loss: 6.51660575385904e-05
# Epoch 18, Loss: 6.0411965250750654e-05
# Epoch 19, Loss: 5.633815840155876e-05
# Epoch 20, Loss: 5.438113805666944e-05
# Epoch 21, Loss: 5.044702163559123e-05
# Epoch 22, Loss: 4.766013546486647e-05
# Epoch 23, Loss: 4.646430863886053e-05
# Epoch 24, Loss: 4.473361553891664e-05
# Epoch 25, Loss: 4.1317251714190206e-05
# ...
# Epoch 118, Loss: 3.844102263172999e-06
# Epoch 119, Loss: 3.7883660368720484e-06
# Epoch 120, Loss: 3.6269312460035508e-06
# Finished Training
```

![](https://files.mdnice.com/user/54972/dff4f763-8667-4905-8277-da12f36dbcf1.png)

可以发现，对没有偏转角的数字识别准确率已经非常可观了，于是我原本打算直接从数据集入手，对原本的图片进行随机角度的偏转后一起训练来学习成一个能学习偏转角的CNN模型（完成如下所示的类似功能），让其与之前的CNN搭配使用，结果代码误打误撞写错了，虽然我的全连接层选取了0~359来学习，但是在标注的时候还是以0~9来标注的，所以直接学习成了一个CNN替代了原有模型，由于偏转角度的随机，我主要是考虑增加训练轮数来学习。

![](https://files.mdnice.com/user/54972/09909f07-a378-4e1c-94ed-3c1444b5811a.png)

代码差异其实主要是添加了一个Lamada变换：

```python
# 定义一个新的Lambda变换，它使用你的模型来预测旋转角度，并旋转图像
def apply_rotation_correction(image):
    tensor_image = transforms.ToTensor()(image).unsqueeze_(0).to(device)
    
    # 使用模型预测旋转角度
    with torch.no_grad():
        output = pre_model(tensor_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_angle = torch.argmax(probabilities).item()
    # 旋转图像
    rotated_image = image.rotate(-predicted_angle, expand=True)  # 注意旋转方向
    return rotated_image
```

测试结果如下：

![](https://files.mdnice.com/user/54972/0cb8ab4c-3612-451a-b791-dd94eff96753.png)
![](https://files.mdnice.com/user/54972/4ae5ff7f-3744-4dfe-8b4f-577a3aeda0c2.png)

可以主要到准确率已经相当可观了，没比TecentOCR差多少。于是这个项目就顺利结束啦（虽然项目可能还有些小瑕疵）~

相关代码已经打包上传至Github了，就权当一个玩具吧~（毕竟真实世界中识别验证码真的不是一件很轻松的事情）

![](https://files.mdnice.com/user/54972/2c2121c0-431b-40f7-8a51-c24705488daa.png)

## 参考文献：

[1] Baluja, Shumeet. "Making Templates Rotationally Invariant. An Application to Rotated Digit Recognition." *Advances in Neural Information Processing Systems* 11 (1998).

[2] [在线文章] Correcting Image Orientation Using Convolutional Neural Networks - A hands-on introduction to deep learning applications. *Posted by Daniel Saez on January 12, 2017* https://d4nst.github.io/2017/01/12/image-orientation/
=======


今年春假期间，我在南京的一位同样是学习人工智能专业的朋友向我发来了求助。

任务的大致要求如下：

>  输入一张白底的图片，上面可能有各种颜色的印刷体阿拉伯数字，部分数字可能有一定旋转角度，请运用机器学习相关知识对图上的数字进行识别。输出一张图片，在原图的基础上用红色矩形框选出某一个数字所处的位置，然后在矩形附近打印出识别的结果。（如下图示）

![](https://files.mdnice.com/user/54972/b66a69a9-bb6b-4a6a-a1a8-dd452956099b.png)
![](https://files.mdnice.com/user/54972/432c9466-ef16-4e71-b795-725648375f18.png)

原本以为是一项非常简单的任务，但实际上自己做起来却发现效果却不尽人意，再加上这个学期课业压力有点大，自己在春假期间和五一假期期间都有所尝试，但是最后无疾而终。后来我请教了X老师，X老师明确指出这个任务其实并没有那么简单，建议我可以采用一些比较大的模型如VGG，再看看效果，并推荐了一位研究领域涉及图像的H老师给我。这几天也是开始请教H老师的一位博士学长，原本以为他能清楚地指导我该怎么做，但是事实上我们之间的对话更像是探讨。原因其实很简单，博士学长也并不是专门做这个方向的，所以很多时候我们AI领域的事还是得亲自尝试亲自复现，这也是我这个项目一个比较大的收获。

回到这个任务上来看，这个任务可以拆解为两个关键步骤：

1. 分离出待识别的数字的位置和图像

2. 采用模型对图像进行预测输出结果

![](https://files.mdnice.com/user/54972/ddc8d824-674c-4cfc-990c-6b40bf1ace85.png)
![](https://files.mdnice.com/user/54972/0f1d11fb-fc2b-4f08-9d16-d43aa0b9e77e.png)

对于任务2，一开始我自己尝试的时候，我自己爬了1000张训练数据（如上图示），每个数字100张128*128的黑白图片，同时发现网上的一些相关参考都是在tensorflow框架下实现的，于是一开始我采用keras的CNN来写脚本，效果非常糟糕。

我曾经还用过MINST手写数据集来训练一个CNN网络，结果效果也不尽人意。

暑招的时候酒店同房间的室友在完成一个科研任务，具体是用OCR整理Excel，我于是调用了TecentOCR的API写了一份脚本，效果很显著：

![](https://files.mdnice.com/user/54972/bc2c0758-9c06-48e6-b19b-fa59b6ccbec9.png)
![](https://files.mdnice.com/user/54972/0453fe5d-c342-4b89-9c48-a331878db5e3.png)

于是我以它的识别率为目标先用VGG11进行了一番尝试。同时我采用边缘探测的遍历所有的ROI（Region of Interest）以完成任务1。并在后续的尝试中基本保持这种方式。同时注意到分离出来的矩形可能是长方形，为了匹配我模型需要的128*128的尺寸，我们必须做一步transform，我采用的方式是把长方形的短边向两边扩增，使其成为一个正方形，再对新图像做一圈白色像素的padding后resize，这样能够保证样例与测试集数据的较为吻合。

```python
# 以下是train.py代码
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg11
import torchvision.models as models

# 1. 定义数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 将彩色图像转为灰度图像
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 2. 创建数据加载器
dataset = datasets.ImageFolder(root='trainpic', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

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


# 3. 定义模型
model = ModifiedVGG()

# model = vgg11(pretrained=False)
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, 10)])  # Add our layer with output size 10
model.classifier = nn.Sequential(*features)

# 4. 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练循环
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # 每100批打印一次
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    # 保存模型
    torch.save(model.state_dict(), 'digit_recognizer.pth')
```

```python
# 以下是部分的predict.ipynb代码
image = cv2.imread('input.png')
# image = cv2.imread('C.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用阈值处理以分离前景和背景
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 寻找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历每一个轮廓
for contour in contours:
    # 获取边界框
    x, y, w, h = cv2.boundingRect(contour)
    # 忽略太小的边界框，可能是噪声
    if w < 10 or h < 10:
        continue
    # 提取ROI
    roi = gray[y:y+h, x:x+w]

    # 将ROI转换为模型可以接受的格式
    roi = cv2.resize(roi, (32, 32))
    roi = transform(Image.fromarray(roi)).unsqueeze_(0).to(device)

    # 前向传播
    with torch.no_grad():
        output = model(roi)
        _, predicted = torch.max(output, 1)
        pred = predicted.item()

    # 绘制矩形和标签
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(image, str(pred), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 显示结果
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

而后对比效果，还是觉得浅层CNN的框架在处理这类问题可能更佳（当然除此以外还可以采用传统的一些机器学习模型如支持向量机等）

这里我自己用torch框架重写了CNN，发现识别效果骤增：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

from torchvision.transforms import functional as F
import random

class DynamicRotationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, angles=[-180, -90, -45, 0, 45, 90, 180], transform=None):
        self.dataset = datasets.ImageFolder(root_dir, transform=transforms.ToTensor())
        self.angles = angles
        self.transform = transform

    def __len__(self):
        return len(self.dataset) * len(self.angles)

    def __getitem__(self, idx):
        sample_idx = idx % len(self.dataset)
        angle_idx = idx // len(self.dataset)
        angle = self.angles[angle_idx]

        image, label = self.dataset[sample_idx]
        image = F.rotate(image, angle)

        if self.transform:
            image = self.transform(image)

        return image, label


train = datasets.ImageFolder(root='modified\trainpic', transform=data_transforms)
trainloader = DataLoader(train, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 32 * 32, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 120
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

print('Finished Training')

# 保存模型
torch.save(model.state_dict(), 'digit_recognition_model.pth')
Epoch 1, Loss: 1.9711331836879253
# Result:
# Epoch 2, Loss: 0.12846590229310095
# Epoch 3, Loss: 0.03395026904763654
# Epoch 4, Loss: 0.017787698379834183
# Epoch 5, Loss: 0.013690078725630883
# Epoch 6, Loss: 0.005368602876842488
# Epoch 7, Loss: 0.0020161027152880706
# Epoch 8, Loss: 0.0010407454797132232
# Epoch 9, Loss: 0.00045232775210024556
# Epoch 10, Loss: 0.00015856899085520126
# Epoch 11, Loss: 0.0001125274102378171
# Epoch 12, Loss: 9.382167240801209e-05
# Epoch 13, Loss: 9.087298360554996e-05
# Epoch 14, Loss: 7.821109579708718e-05
# Epoch 15, Loss: 7.269618754435214e-05
# Epoch 16, Loss: 6.695036722703662e-05
# Epoch 17, Loss: 6.51660575385904e-05
# Epoch 18, Loss: 6.0411965250750654e-05
# Epoch 19, Loss: 5.633815840155876e-05
# Epoch 20, Loss: 5.438113805666944e-05
# Epoch 21, Loss: 5.044702163559123e-05
# Epoch 22, Loss: 4.766013546486647e-05
# Epoch 23, Loss: 4.646430863886053e-05
# Epoch 24, Loss: 4.473361553891664e-05
# Epoch 25, Loss: 4.1317251714190206e-05
# ...
# Epoch 118, Loss: 3.844102263172999e-06
# Epoch 119, Loss: 3.7883660368720484e-06
# Epoch 120, Loss: 3.6269312460035508e-06
# Finished Training
```

![](https://files.mdnice.com/user/54972/dff4f763-8667-4905-8277-da12f36dbcf1.png)

可以发现，对没有偏转角的数字识别准确率已经非常可观了，于是我原本打算直接从数据集入手，对原本的图片进行随机角度的偏转后一起训练来学习成一个能学习偏转角的CNN模型（完成如下所示的类似功能），让其与之前的CNN搭配使用，结果代码误打误撞写错了，虽然我的全连接层选取了0~359来学习，但是在标注的时候还是以0~9来标注的，所以直接学习成了一个CNN替代了原有模型，由于偏转角度的随机，我主要是考虑增加训练轮数来学习。

![](https://files.mdnice.com/user/54972/09909f07-a378-4e1c-94ed-3c1444b5811a.png)

代码差异其实主要是添加了一个Lamada变换：

```python
# 定义一个新的Lambda变换，它使用你的模型来预测旋转角度，并旋转图像
def apply_rotation_correction(image):
    tensor_image = transforms.ToTensor()(image).unsqueeze_(0).to(device)
    
    # 使用模型预测旋转角度
    with torch.no_grad():
        output = pre_model(tensor_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_angle = torch.argmax(probabilities).item()
    # 旋转图像
    rotated_image = image.rotate(-predicted_angle, expand=True)  # 注意旋转方向
    return rotated_image
```

测试结果如下：

![](https://files.mdnice.com/user/54972/0cb8ab4c-3612-451a-b791-dd94eff96753.png)
![](https://files.mdnice.com/user/54972/4ae5ff7f-3744-4dfe-8b4f-577a3aeda0c2.png)

可以主要到准确率已经相当可观了，没比TecentOCR差多少。于是这个项目就顺利结束啦（虽然项目可能还有些小瑕疵）~

相关代码已经打包上传至Github了，就权当一个玩具吧~（毕竟真实世界中识别验证码真的不是一件很轻松的事情）

![](https://files.mdnice.com/user/54972/2c2121c0-431b-40f7-8a51-c24705488daa.png)

## 参考文献：

[1] Baluja, Shumeet. "Making Templates Rotationally Invariant. An Application to Rotated Digit Recognition." *Advances in Neural Information Processing Systems* 11 (1998).

[2] [在线文章] Correcting Image Orientation Using Convolutional Neural Networks - A hands-on introduction to deep learning applications. *Posted by Daniel Saez on January 12, 2017* https://d4nst.github.io/2017/01/12/image-orientation/

