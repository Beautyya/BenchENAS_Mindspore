import torch
from torch import nn
from torchvision import transforms, datasets
import os
import utils
import torchvision.datasets as dset
# import moxing as mox


class LeNet(nn.Module):
    def __init__(self, num_class, init_weights=False):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_class))
        if init_weights:
            self.initial_weights()

    def initial_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


def train():
    device = torch.device(f"npu:{torch.npu.current_device()}" if torch.npu.is_available() else "cpu")
    torch.npu.set_device(device)
    print('using device {}'.format(device))
    #数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(32),   #随机缩放裁剪
            transforms.RandomHorizontalFlip(),   #随机水平翻转
            transforms.ToTensor(),               #转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   #归一化
        ]),
        "val": transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    #加载数据集
    batch_size = 32
    # mox.file.copy_parallel('s3://datasets/dogs/', '/cache/datasets/dogs')
    # data_path = '/cache/datasets/dogs'
    # assert os.path.exists(data_path), "{} does not exist".format(data_path)
    # train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=data_transform['train'])
    # val_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'val'), transform=data_transform['val'])
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=8)

    train_transform, valid_transform = utils._data_transforms_cifar10()
    train_data = dset.CIFAR10(root='/home/work/user-job-dir/201train/data', train=True, download=False,
                              transform=train_transform)
    valid_data = dset.CIFAR10(root='/home/work/user-job-dir/201train/data', train=False, download=False,
                              transform=valid_transform)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=2, drop_last=True)


    net = LeNet(num_class=2, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    epochs = 100
    save_path = 'lenet.pt'
    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                images, labels = val_data
                outputs = net(images.to(device))
                predict = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict, labels.to(device)).sum().item()
        val_acc = acc / len(val_data)
        print('epoch:{}, acc:{}'.format(epoch + 1, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
    # mox.file.copy("./lenet.pt", 's3://datasets/models/lenet/lenet.pt')
    print("finish train")


if __name__ == '__main__':
    train()