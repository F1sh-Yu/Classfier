# 用Pytorch实现一个分类器
## 步骤
1. 选择数据集并读取，进行预处理
2. 建立神经网络class Net()
3. 确定损失函数Loss以及优化器optimizer
4. 梯度清零，然后运行神经网络
5. 反向传播计算梯度
6. 进行优化训练
7. 用测试集进行测试
## 需要的函数/类
1. `transforms.toTensor(),transforms.Normalize()`：对数据进行预处理
2. `torchvision.datasets.CIFAR10()`：数据集
3. `torch.utils.data.DataLoader()`：读取数据
4. `torch.nn.Conv2d(),torch.nn.MaxPool2d(),torch.nn.Linear()`：网络层
5. `torch.nn.Functional.relu()`：激活函数
6. `torch.nn.CrossEntropyLoss()`：损失函数
7. `torch.optim.SGD()`：随机梯度下降
可以看到主要用到的是两个库`torchvision,torch`，`torchvision`库的作用是生成数据集以及进行预处理，我们在这里用到`torchvision`下的`transform`和`datasets`两个部分。而`torch`库就是用于构建网络的库，`torch.nn`中有网络层的定义，如卷积、线性，也有各种损失函数，`torch.nn.Functional`中是各种激活函数，`torch.optim`里则是优化器。

## 具体实现
### 数据集的处理
我们使用的Pytorch提供的`CIFAR10()`数据集，预处理有两步，第一步是将数据集转换为Pytorch可处理的`tensor`格式，二是将数据Normalize
```
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
```
之后我们就可以开始读取数据了，并将`transform`用作`DataLoader`的参数。
```
trainset = torchvision.datasets.CIFAR10(root = ‘./data’, transform = transform, train = True, download = False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)
trainset = torchvision.datasets.CIFAR10(root = ‘./data’, transform = transform, train = False, download = False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)
```
`train`代表数据是否用于训练，`download`表示数据是否已下载，若为`True`则会自动下载并保存到`root`中，`shuffle`是自动打乱，`batch_size`是每次传入的图片数。

### 建立神经网络

```
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
该网络是由两层卷积层，两层Maxpool层，以及三层线性层构成的。总体比较简单，要注意的就是在两层卷积结束之后，要把特征量改变为一维，作为线性层的输入，以及最后`fc3`输出的时候是不需要`relu`函数的

### 确定损失函数与优化器

```
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.0001, momentum = 0.9)
```
损失函数使用了交叉熵（CrossEntropy），优化方法选择了随机梯度下降。
在这里我们可以看到网络的参数都传进了优化器中，因此我们只要使用优化器的`step()`方法就可以对整个网络进行优化。

### 开始训练

```
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader,0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 2000 == 1999:
                print('(%d,%d),the loss is %.3f’ %(epoch+1, i+1, running_loss/2000))
                running_loss = 0
    print(‘finish training’)
    path = './model.pth'
    torch.save(net.state_dict(),path)
```
训练两个epoch，并且每2000次训练打印一次loss。
最后将训练出的模型`state_dict()`保存起来，用于进行测试。

### 测试
```
    net = Net()
    net.load_state_dict(torch.load(‘./model.pth’))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs,labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs,1)
            correct += (predicted==labels).sum().item()
            total += labels.size(0)
    print(‘accuracy of the net is %f %%’ %(100 * correct/total))
```
使用`load_state_dict`读取模型，将测试集输入网络，用得到的`predicted`与数据集的标记`label`进行比较，计算出正确率即可。
#python笔记/pytorch