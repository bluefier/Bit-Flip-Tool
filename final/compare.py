import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from userFile import ComplexModel

if __name__ == '__main__':


    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载验证数据集
    val_data = datasets.ImageFolder(root=r'D:\容错\catsdogs\train', transform=transform)
    # val_data = datasets.ImageFolder(root='C:/Users/86183/Desktop/22', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)

    # # 加载预训练的VGG16模型
    # vgg16 = models.vgg16(pretrained=False)  # 不加载预训练权重
    # num_features = vgg16.classifier[6].in_features
    # vgg16.classifier[6] = nn.Linear(num_features, 2)  # 2是两个输出类别
    # vgg16.to(device)

    # 加载预训练权重
    # model_path = 'C:/Users/ChenPanda/Desktop/pyqt/final/vgg16_model.pth'  # 原模型
    # model_path = 'C:/Users/ChenPanda/Desktop/fixed_models.pth'  # 增强模型
    # #
    model_path = r'D:\test_pth\2\fixed_inject_models.pth'  # 注入错误的增强模型
    # model_path = 'C:/Users/86183/Desktop/fixed_models.pth'  # 原模型
    model = torch.load(model_path, map_location=device)

    # 设置模型为评估模式
    model.eval()
    epoch=10
    for i in range(epoch):
        correct = 0
        total = 0

        count = 0
        with torch.no_grad():
            for inputs, labels in val_loader:

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                a=predicted==labels
                correct += (predicted == labels).sum().item()
                count += 1
                if count % 50 == 0:
                    print(f'process : {len(val_loader)} / {count}')

        accuracy = correct / total * 100
        print(f'Validation Accuracy: {accuracy:.2f}%')

