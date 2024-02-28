import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader




if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载预训练的VGG16模型
    resnet=models.resnet18(pretrained=True)
    # 修改分类器的输出层
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, 2)  # 2是您的输出类别数
    resnet.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    # 加载数据集
    data_dir = r"D:\容错\catsdogs\train"  # 替换为您的数据集路径
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # 训练模型
    num_epochs = 30
    test_count = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_count = test_count + 1
            print(f'Epoch : {epoch + 1} Round : {test_count}')

        test_count = 0
        epoch_loss = running_loss / len(dataloader)
        accuracy = correct / total * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

    print('Finished Training')

    # 保存模型
    torch.save(resnet, 'resnet_model.pth')
