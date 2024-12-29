# Qlib 库
import qlib
from qlib.data import D
from qlib.config import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import argparse  # 新增导入 argparse 模块
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
import os
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Qlib 环境
qlib.init(provider_uri="/Users/huiyu/Desktop/qlib_data/us_data", region=REG_CN)

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

def calculate_stoch_k(data, window=14, smooth_window=3):
    """
    计算 Stochastic K 指标
    """
    low_min = data['$low'].rolling(window=window).min()
    high_max = data['$high'].rolling(window=window).max()
    
    stoch_k = ((data['$close'] - low_min) / (high_max - low_min)) * 100
    stoch_k_smooth = stoch_k.rolling(window=smooth_window).mean()  # 平滑处理

    return stoch_k_smooth

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

def calculate_ppo(data, short_window=12, long_window=26):
    """
    计算 Percentage Price Oscillator (PPO)
    """
    ema_short = data['$close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['$close'].ewm(span=long_window, adjust=False).mean()
    
    ppo = (ema_short - ema_long) / ema_long * 100  # 计算 PPO
    return ppo

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
    
    # 计算 PPO 并添加到数据框
    df['PPO'] = calculate_ppo(df)  # 新增 PPO 因子
    
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
    df['EMA_50'] = calculate_ema(df, window=50)
    df['EMA_100'] = calculate_ema(df, window=100)

     # 计算 Stochastic K 并添加到数据框
    df['Stoch_K'] = calculate_stoch_k(df)  # 新增 Stochastic K 因子
  
    # 计算布林带并添加到数据框
    t1, t2, t3 = calculate_bollinger_bands(df)

    # 计算 ATR 并添加到数据框
    df['ATR'] = calculate_atr(df)
    
    # 计算动量特征并添加到数据框
    df['Momentum'] = calculate_momentum(df, window=5)

    # 计算 Williams %R 并添加到数据框
    df['Williams_R'] = calculate_williams_r(df)

    # 计算 Chaikin Oscillator 并添加到数据框
    df['Chaikin_Oscillator'] = calculate_chaikin_oscillator(df)

    return df

def create_sequences(data, seq_length):
    X, y_price, y_trend, future = [], [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y_price.append(data[i + seq_length, 0])  # Predict $close price
        # Predict trend: 1 if price goes up, 0 if price goes down or stays the same
        y_trend.append(1 if data[i + seq_length, 0] >= data[i + seq_length - 1, 0] else 0)
    future.append(data[i:i + seq_length])
    return np.array(X), np.array(y_price), np.array(y_trend), np.array(future)


if __name__ == "__main__":  # 确保 main 函数在脚本直接运行时执行

    # 新增命令行参数解析
    parser = argparse.ArgumentParser(description='Stock Code for LSTM Prediction')
    parser.add_argument('--stock_code', type=str, default='BABA', help='Stock code to analyze (default: BABA)')
    args = parser.parse_args()

    # Step 1: 数据加载
    stock_code = args.stock_code  # 使用传入的 stock_code 参数
    start_date = "2015-01-01"  # 上市日期
    end_date = "2024-12-31"

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
    X, y_price, y_trend, future = create_sequences(scaled_data, SEQ_LENGTH)

    # 数据集划分
    split = int(len(X) * 0.9)  # 80% 训练，20% 测试
    X_train, X_test = X[:split], X[split:]
    y_train_price, y_test_price = y_price[:split], y_price[split:]
    y_train_trend, y_test_trend = y_trend[:split], y_trend[split:]

    # 转换为 PyTorch 张量
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_price_torch = torch.tensor(y_train_price, dtype=torch.float32).unsqueeze(-1)
    y_train_trend_torch = torch.tensor(y_train_trend, dtype=torch.float32).unsqueeze(-1)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_price_torch = torch.tensor(y_test_price, dtype=torch.float32).unsqueeze(-1)
    y_test_trend_torch = torch.tensor(y_test_trend, dtype=torch.float32).unsqueeze(-1)

    future_torch = torch.tensor(future, dtype=torch.float32)

    # Step 3: 构建 LSTM 模型
    class MultiTaskLSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size_price, output_size_trend):
            super(MultiTaskLSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc_price = nn.Linear(hidden_size, output_size_price)
            self.fc_trend = nn.Linear(hidden_size, output_size_trend)
       

        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            out_price = self.fc_price(hn[-1])
            out_trend = self.fc_trend(hn[-1])
            return out_price, out_trend

    # 模型参数
    INPUT_SIZE = X_train.shape[2]  # 特征数量
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE_PRICE = 1
    OUTPUT_SIZE_TREND = 1

    model = MultiTaskLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE_PRICE, OUTPUT_SIZE_TREND)
    criterion_price = nn.MSELoss()
    criterion_trend = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 4: 训练模型
    logging.info("Starting model training.")
    EPOCHS = 100
    BATCH_SIZE = 32

    def train_model(model, X_train, y_train_price, y_train_trend, criterion_price, criterion_trend, optimizer, epochs, batch_size):
        model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch_price = y_train_price[i:i + batch_size]
                y_batch_trend = y_train_trend[i:i + batch_size]

                optimizer.zero_grad()
                output_price, output_trend = model(X_batch)
                loss_price = criterion_price(output_price, y_batch_price)
                loss_trend = criterion_trend(output_trend, y_batch_trend)
                loss = loss_price + 0.01*loss_trend
                #print(f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss_price, loss_trend}")
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    train_model(model, X_train_torch, y_train_price_torch, y_train_trend_torch, criterion_price, criterion_trend, optimizer, EPOCHS, BATCH_SIZE)

    # Step 5: 测试模型
    logging.info("Testing the model.")
    model.eval()
    with torch.no_grad():
        predictions_price, predictions_trend = model(X_test_torch)
        predictions_price = predictions_price.squeeze().numpy()
        predictions_trend = (predictions_trend.squeeze().numpy() > 0.5).astype(int)
        true_values_price = y_test_price
        true_values_trend = y_test_trend

        future_predictions_price, future_predictions_trend = model(future_torch)
        future_predictions_price = future_predictions_price.numpy()
        future_predictions_trend = (future_predictions_trend.numpy() > 0.5).astype(int)
        print(future_predictions_price, future_predictions_trend)
        #print(predictions_price)
    # Check the shape of future_predictions_price
    logging.info(f"Future predictions price after squeeze: {future_predictions_price.shape}")

    # 反归一化结果
    predictions_price_rescaled = scaler.inverse_transform(
        np.hstack((predictions_price.reshape(-1, 1), np.zeros((len(predictions_price), scaled_data.shape[1] - 1))))
    )[:, 0]
    true_values_price_rescaled = scaler.inverse_transform(
        np.hstack((true_values_price.reshape(-1, 1), np.zeros((len(true_values_price), scaled_data.shape[1] - 1))))
    )[:, 0]

    future_predictions_price_rescaled = scaler.inverse_transform(
        np.hstack((future_predictions_price.reshape(-1, 1), np.zeros((len(future_predictions_price), scaled_data.shape[1] - 1))))
    )[:, 0]
    # Step 6: 专业评估

    # 计算指标
    mae = mean_absolute_error(true_values_price_rescaled, predictions_price_rescaled)
    mse = mean_squared_error(true_values_price_rescaled, predictions_price_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values_price_rescaled, predictions_price_rescaled)

    # 计算 MAPE
    mape = np.mean(np.abs((true_values_price_rescaled - predictions_price_rescaled) / true_values_price_rescaled)) * 100
    # 计算额外的指标
    accuracy = np.mean(predictions_trend == true_values_trend)
    # 保存指标到文件
    today_date = datetime.now().strftime("%Y-%m-%d")
    folder_path = f"LSTM-{today_date}-daily/{stock_code}/"
    os.makedirs(folder_path, exist_ok=True)  # 创建文件夹

    metrics_file_path = os.path.join(folder_path, "metrics.txt")
    with open(metrics_file_path, "w") as f:
        f.write(f"Mean Absolute Error (MAE): {mae:.5f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.5f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.5f}\n")
        f.write(f"R2 Score: {r2:.5f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.5f}%\n")
        f.write(f"Trend Prediction Accuracy: {accuracy:.5f}\n")
        
        # 新增：打印 future_predictions 的结果
        f.write(f"Future Predictions Price: {future_predictions_price_rescaled.tolist()}\n")  # 将数组转换为列表
        f.write(f"Future Predictions Trend: {future_predictions_trend.tolist()}\n")  # 将数组转换为列表

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(true_values_price_rescaled, label="True Values", color="blue", alpha=0.7)
    plt.plot(predictions_price_rescaled, label="Predictions", color="red", alpha=0.7)
    plt.title("The result of LSTM prediction vs true value")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()

    # Annotate the plot with evaluation metrics
    metrics_text = (
        f"MAE: {mae:.5f}\n"
        f"MSE: {mse:.5f}\n"
        f"RMSE: {rmse:.5f}\n"
        f"R2: {r2:.5f}\n"
        f"MAPE: {mape:.5f}%\n"
        f"Trend Accuracy: {accuracy:.5f}"
    )

    # 保存图像
    plt.savefig(os.path.join(folder_path, "prediction_plot.png"))
    #plt.show()
    plt.close()

