import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

# 定義數據轉換（轉換為 Tensor 並標準化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 將像素值標準化到 -1 到 1 之間
])

# 下載 MNIST 數據集
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# 分割數據集為訓練集和驗證集（90%訓練，10%驗證）
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 建立 DataLoader，訓練集進行shuffle，驗證集不shuffle
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

# 定義簡單的神經網路
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 將圖像展平成一維
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、損失函數和優化器
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練函數
def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()  # 設定為訓練模式
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()        # 清除先前的梯度
        output = model(data)         # 前向傳播
        loss = criterion(output, target)  # 計算損失
        loss.backward()              # 反向傳播
        optimizer.step()              # 更新權重
        total_loss += loss.item()     # 累積損失
    avg_loss = total_loss / len(train_loader)  # 計算平均損失
    print(f'Epoch {epoch}: 訓練損失 = {avg_loss:.4f}')

# 驗證函數
def validate_model(model, val_loader, criterion):
    model.eval()  # 設定為評估模式
    total_loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度計算
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()  # 累積損失
            pred = output.argmax(dim=1, keepdim=True)  # 找到最大值的索引作為預測
            correct += pred.eq(target.view_as(pred)).sum().item()  # 計算正確數量
    avg_loss = total_loss / len(val_loader)  # 計算平均損失
    accuracy = correct / len(val_loader.dataset)  # 計算準確率
    print(f'驗證損失 = {avg_loss:.4f}, 驗證準確率 = {accuracy * 100:.2f}%')

# 訓練和驗證流程
epochs = 10
for epoch in range(1, epochs + 1):
    train_model(model, train_loader, criterion, optimizer, epoch)  # 訓練模型
    validate_model(model, val_loader, criterion)  # 驗證模型