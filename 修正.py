import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import akshare as ak
import os
# 获取股票的日线数据，这里以中国银行为例，代码为 601988
start_date = '20230101'
end_date = '20240611'
stock_zh_a_daily_df = ak.stock_zh_a_daily(symbol="sh601988", start_date=start_date, end_date=end_date)
print("中国银行股票的日线数据：")
print(stock_zh_a_daily_df)
#丢弃'open', 'high', 'low'这三个特征
stock_zh_a_daily_df1=stock_zh_a_daily_df.drop(columns=['open', 'high', 'low'])
# 将中国银行股票的日线数据保存为 CSV 文件
folder_path="D:"
stock_zh_a_daily_df1.to_csv(os.path.join(folder_path, "中国银行股票日线数据.csv"), index=False, encoding="utf_8_sig")
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据
df = pd.read_csv("D:", index_col=0, parse_dates=True)

# 索引是日期，按索引升序排序
df = df.sort_index(ascending=True)
# 添加30日平均
df['Close_30_day_MA'] = df['close'].rolling(window=30).mean()
df = df[31:]

# print(df.tail())
# print(df.head())
# plt.figure(figsize=(14, 7))
# plt.plot(df.index, df['close'], label='Original Price')
# plt.plot(df.index[:], df['Close_30_day_MA'], label='30-day Moving Average', linestyle='--')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.title('Stock Price with 30-day Moving Average')
# plt.legend()
# plt.show()
# 特征工程
y = df['close']
X = df

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# 数据标准化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 序列数据准备
def create_sequences_scaled(data, target, seq_length):
    X, y = [], []
    for i in range(len(target) - seq_length - 1):
        seq_x = data[i:i + seq_length]
        seq_y = target.iloc[i + seq_length]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

seq_length = 3
X_train_seq, y_train_seq = create_sequences_scaled(X_train_scaled, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences_scaled(X_test_scaled, y_test, seq_length)

# 变为 tensors格式
X_train_seq_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_train_seq_tensor = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1).to(device)
X_test_seq_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
y_test_seq_tensor = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1).to(device)

# 定义 LSTM 模型
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出用于预测
        return out

# 模型实例化及参数设置
input_size = X_train_seq.shape[2]  # 基于训练数据特征数
hidden_size = 64  # 例如，隐藏层大小设置为64
num_layers = 2  # 假设使用两层LSTM
output_size = 1  # 回归任务，输出一个值

model = StockLSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 100
train_losses = []  # 存储训练损失
test_losses = []   # 存储测试损失

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_seq_tensor)
    loss = criterion(outputs, y_train_seq_tensor)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())  # 收集训练损失数据

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_seq_tensor)
            test_loss = criterion(test_outputs, y_test_seq_tensor)
            test_losses.append(test_loss.item())  # 收集测试损失数据
            print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss.item():.4f}')
# 继续使用之前的变量，绘制测试集上的预测值与实际值的对比图
plt.figure(figsize=(12, 6))
plt.plot(y_test_seq_tensor.cpu().numpy(), label='Actual Values', color='blue')
plt.plot(test_outputs.cpu().numpy(), label='Predicted Values', color='red', linestyle='--')

# 确保在绘图前将其变为一维数组
actual_values = y_test_seq_tensor.squeeze().cpu().numpy()
predicted_values = test_outputs.squeeze().cpu().numpy()

# # 添加图例、标题和标签
# plt.legend()
# plt.title('Prediction vs Actual Values on Test Set at Epoch {}'.format(epoch + 1))
# plt.xlabel('Sequence Index')
# plt.ylabel('Stock Price')
#
# # 显示图表
# plt.show()
# # 绘制损失曲线
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Test Loss Over Epochs')
# plt.legend()
# plt.show()

# 预测函数
def predict_and_inverse_transform(model, last_sequence, seq_length, prediction_days=1):
    model.eval()
    predictions_scaled = []
    last_sequence_tensor = torch.tensor(last_sequence.reshape(1, seq_length, -1), dtype=torch.float32).to(device)

    for _ in range(prediction_days):
        with torch.no_grad():
            prediction_scaled = model(last_sequence_tensor)
            predictions_scaled.append(prediction_scaled.cpu().numpy().flatten()[0])

    return predictions_scaled

# 获取最后一个已知数据点的序列
last_known_sequence = X_train_scaled[-seq_length:]

# 设置预测的天数为1
prediction_days = 1

# 获取预测结果
predictions_scaled = predict_and_inverse_transform(model, last_known_sequence,
                                 seq_length, prediction_days)
predictions_reshaped = np.full((1, 6), predictions_scaled[0])
predictions_original = scaler.inverse_transform(predictions_reshaped)
print("预测结果为",predictions_original[0][0])
