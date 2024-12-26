# Qlib 库
import qlib
from qlib.data import D
from qlib.config import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# 数据处理和计算库
import pandas as pd
import numpy as np

# PyTorch 深度学习库
import torch
from torch import nn, optim

# 数据预处理和评估指标
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 数据可视化
import matplotlib.pyplot as plt
import logging  # 新增导入日志模块

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Qlib 环境
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# 获取宁德时代日线数据
def get_daily_data(stock_code, start_date, end_date):
    """
    获取指定股票的日线数据
    """
    df = D.features(
        instruments=[stock_code],
        fields=["$close", "$open", "$high", "$low", "$volume"],
        start_time=start_date,
        end_time=end_date,
        freq="day"
    )
    return df

def calculate_rsi(data, window=14):
    """
    计算相对强弱指数 (RSI)
    """
    delta = data['$close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    计算 MACD 及其信号线和直方图
    """
    # 计算短期和长期 EMA
    ema_short = data['$close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['$close'].ewm(span=long_window, adjust=False).mean()
    
    # 计算 MACD 线
    macd_line = ema_short - ema_long
    
    # 计算信号线
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    
    # 计算直方图
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_kdj(data, window=14):
    """
    计算 KDJ 因子
    """
    low_min = data['$low'].rolling(window=window).min()
    high_max = data['$high'].rolling(window=window).max()
    
    rsv = (data['$close'] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

def calculate_chaikin_oscillator(data, short_window=3, long_window=10):
    """
    计算 Chaikin Oscillator
    """
    # 计算 A/D 值
    ad = ((data['$close'] - data['$low']) - (data['$high'] - data['$close'])) / (data['$high'] - data['$low']) * data['$volume']
    ad = ad.fillna(0).cumsum()  # 累积 A/D 值

    # 计算短期和长期 A/D EMA
    ad_short = ad.ewm(span=short_window, adjust=False).mean()
    ad_long = ad.ewm(span=long_window, adjust=False).mean()

    # 计算 Chaikin Oscillator
    chaikin_oscillator = ad_short - ad_long

    return chaikin_oscillator

def calculate_sma(data, window):
    """
    计算简单移动平均 (SMA)
    """
    return data['$close'].rolling(window=window).mean()

def calculate_ema(data, window):
    """
    计算指数移动平均 (EMA)
    """
    return data['$close'].ewm(span=window, adjust=False).mean()

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """
    计算布林带
    """
    sma = calculate_sma(data, window)
    rolling_std = data['$close'].rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)
    return upper_band, sma, lower_band

def calculate_atr(data, window=14):
    """
    计算平均真实波幅 (ATR)
    """
    high_low = data['$high'] - data['$low']
    high_close = (data['$high'] - data['$close'].shift()).abs()
    low_close = (data['$low'] - data['$close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)  # 真实波幅
    atr = tr.rolling(window=window).mean()  # ATR
    return atr

def calculate_momentum(data, window=5):
    """
    计算动量特征
    """
    return data['$close'].diff(window)

def calculate_williams_r(data, window=14):
    """
    计算 Williams %R 指标
    """
    high_max = data['$high'].rolling(window=window).max()
    low_min = data['$low'].rolling(window=window).min()
    williams_r = (high_max - data['$close']) / (high_max - low_min) * -100
    return williams_r

def get_daily_data_with_factors(stock_code, start_date, end_date):
    """
    获取指定股票的日线数据和 Alpha158 因子
    """
    # 获取基础数据
    df = D.features(
        instruments=[stock_code],
        fields=["$close", "$open", "$high", "$low", "$volume"],
        start_time=start_date,
        end_time=end_date,
        freq="day"
    )

    # 计算 RSI 并添加到数据框
    df['RSI'] = calculate_rsi(df)
    # 计算 MACD 并添加到数据框
    t1, df['signal_line'], df['macd_histogram'] = calculate_macd(df)
    # 计算 KDJ 并添加到数据框
    df['K'], t1, t2 = calculate_kdj(df)

    # 计算 Chaikin Oscillator 并添加到数据框
    df['Chaikin_Oscillator'] = calculate_chaikin_oscillator(df)
    
    # 计算 SMA 并添加到数据框
    df['SMA_5'] = calculate_sma(df, window=5)
    df['SMA_10'] = calculate_sma(df, window=10)
    df['SMA_20'] = calculate_sma(df, window=20)
    df['SMA_50'] = calculate_sma(df, window=50)
    df['SMA_100'] = calculate_sma(df, window=100)
    df['SMA_200'] = calculate_sma(df, window=200)

    # 计算 EMA 并添加到数据框
    df['EMA_5'] = calculate_ema(df, window=5)
    df['EMA_10'] = calculate_ema(df, window=10)
    df['EMA_20'] = calculate_ema(df, window=20)
    #df['EMA_50'] = calculate_ema(df, window=50)
    #df['EMA_100'] = calculate_ema(df, window=100)
  
    # 计算布林带并添加到数据框
    t1, t2, t3 = calculate_bollinger_bands(df)

    # 计算 ATR 并添加到数据框
    #df['ATR'] = calculate_atr(df)
    
    # 计算动量特征并添加到数据框
    #df['Momentum'] = calculate_momentum(df, window=5)

    # 计算 Williams %R 并添加到数据框
    df['Williams_R'] = calculate_williams_r(df)

    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # 预测 $close 价格
    return np.array(X), np.array(y)


if __name__ == "__main__":  # 确保 main 函数在脚本直接运行时执行
    # Step 1: 数据加载
    stock_code = "sh600519"
    start_date = "2018-06-11"  # 上市日期
    end_date = "2024-12-25"

    logging.info(f"Fetching daily data and Alpha158 factors for {stock_code} from {start_date} to {end_date}.")
    data = get_daily_data_with_factors(stock_code, start_date, end_date)
    print(f"Data shape after merging: {data.shape}")

    if data.empty:
        raise ValueError(f"No data found for {stock_code} from {start_date} to {end_date}.")

    # 数据检查
    logging.info("Data loaded successfully. Checking data head.")
    print(data.head())

    # Step 2: 数据预处理
    # 填充缺失值
    data = data.fillna(method="ffill").fillna(method="bfill")

    # 归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    SEQ_LENGTH = 30  # 使用过去30天的数据预测下一天的收盘价
    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    # 数据集划分
    split = int(len(X) * 0.8)  # 80% 训练，20% 测试
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 转换为 PyTorch 张量
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    # Step 3: 构建 LSTM 模型
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            _, (hn, _) = self.lstm(x)  # LSTM 的输出
            out = self.fc(hn[-1])     # 全连接层输出
            return out

    # 模型参数
    INPUT_SIZE = X_train.shape[2]  # 特征数量
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1

    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 4: 训练模型
    logging.info("Starting model training.")
    EPOCHS = 50
    BATCH_SIZE = 32

    def train_model(model, X_train, y_train, criterion, optimizer, epochs, batch_size):
        model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    train_model(model, X_train_torch, y_train_torch, criterion, optimizer, EPOCHS, BATCH_SIZE)

    # Step 5: 测试模型
    logging.info("Testing the model.")
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_torch).squeeze().numpy()
        true_values = y_test

    # 反归一化结果
    predictions_rescaled = scaler.inverse_transform(
        np.hstack((predictions.reshape(-1, 1), np.zeros((len(predictions), scaled_data.shape[1] - 1))))
    )[:, 0]
    true_values_rescaled = scaler.inverse_transform(
        np.hstack((true_values.reshape(-1, 1), np.zeros((len(true_values), scaled_data.shape[1] - 1))))
    )[:, 0]

    # Step 6: 专业评估

    # 计算指标
    mae = mean_absolute_error(true_values_rescaled, predictions_rescaled)
    mse = mean_squared_error(true_values_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values_rescaled, predictions_rescaled)

    print(f"Mean Absolute Error (MAE): {mae:.5f}")
    print(f"Mean Squared Error (MSE): {mse:.5f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.5f}")
    print(f"R2 Score: {r2:.5f}")

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(true_values_rescaled, label="True Values", color="blue", alpha=0.7)
    plt.plot(predictions_rescaled, label="Predictions", color="red", alpha=0.7)
    plt.title("The result of LSTM prediction vs true value")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()
    plt.show()

