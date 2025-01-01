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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# 数据可视化
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用非 GUI 后端
import logging  # 新增导入日志模块
import os
from datetime import datetime
import torch.nn.functional as F

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Qlib 环境
qlib.init(provider_uri="/Users/huiyu/Desktop/qlib_data/us_data", region=REG_CN)

STRONG_BUY_THRESHOLD = 0.95
STRONG_SELL_THRESHOLD = 0.95

HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 3  # 三个动作：买入、卖出、持有
EPOCHS = 150
BATCH_SIZE = 32

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

def calculate_future_returns(data, days_ahead=1):
    """
    Calculate future returns for a given number of days ahead.
    """
    future_returns = (data['$close'].shift(-days_ahead) - data['$close']) / data['$close']
    return future_returns

def calculate_golden_cross(data, short_window=50, long_window=200):
    """
    计算金叉和死叉
    """
    data['SMA_short'] = calculate_sma(data, window=short_window)
    data['SMA_long'] = calculate_sma(data, window=long_window)
    
    # 金叉：短期均线穿越长期均线
    data['golden_cross'] = ((data['SMA_short'] > data['SMA_long']) & (data['SMA_short'].shift(1) <= data['SMA_long'].shift(1))).astype(int)
    
    # 死叉：短期均线下穿长期均线
    data['death_cross'] = ((data['SMA_short'] < data['SMA_long']) & (data['SMA_short'].shift(1) >= data['SMA_long'].shift(1))).astype(int)

    return data[['golden_cross', 'death_cross']]

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
    #df['upper_band'], t2, df['lower_band'] = calculate_bollinger_bands(df)

    # 计算 ATR 并添加到数据框
    df['ATR'] = calculate_atr(df)
    
    # 计算动量特征并添加到数据框
    df['Momentum'] = calculate_momentum(df, window=5)

    # 计算 Williams %R 并添加到数据框
    df['Williams_R'] = calculate_williams_r(df)

    # 计算 Chaikin Oscillator 并添加到数据框
    df['Chaikin_Oscillator'] = calculate_chaikin_oscillator(df)

    # 计算金叉和死叉并添加到数据框
    df[['golden_cross', 'death_cross']] = calculate_golden_cross(df)

    return df

def create_sequences(data, seq_length, action_period):
    X, Y = [], [] 
    #for i in range(len(data) - seq_length - action_period + 1):  # 确保有足够的数据用于计算未来3天的收益
    for i in range(len(data) - seq_length ):  
        X.append(data[i:i + seq_length])  # 添加序列数据
        Y.append(data[i + seq_length, -1])  # 添加action label
    return np.array(X), np.array(Y)

# Function to calculate and save metrics
def calculate_and_save_metrics(y_true, y_pred, label, file):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = f"{label} - MAE: {mae}, MSE: {mse}, R2: {r2}\n"
    print(metrics.strip())
    
    # Save metrics to file
    with open(file, 'a') as f:
        f.write(metrics)
    
    return mae, mse, r2

# Function to plot predictions vs actual values and save the plot
def plot_and_save_predictions(y_true, y_pred, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual', color='b')
    plt.plot(y_pred, label='Predicted', color='r')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    plt.savefig(filename)  # Save the plot as an image
    plt.close()  # Close the plot to free memory

def middle_term_calculate_action(data):
    """
    根据未来价格变动计算动作。
    """
    actions = []
    action_period = 30
    for i in range(len(data) - action_period):
        current_close = data['$close'].iloc[i]
        future_max = data['$close'].iloc[i+1:i+action_period+1].max()
        future_min = data['$close'].iloc[i+1:i+action_period+1].min()

        if future_max > 1.1 * current_close and future_min >= 0.93 * current_close:
            actions.append(0)  # 买入
        elif future_min <= 0.92 * current_close or (future_min <= 0.95 * current_close and future_max <= 1.01 * current_close):
            actions.append(1)  # 卖出
        else:
            actions.append(2)  # 持有

    # 用持有（2）填充剩余的动作以匹配数据长度
    actions.extend([2] * action_period)
    return actions, action_period

def short_term_calculate_action(data):
    """
    根据未来价格变动计算动作。
    """
    actions = []
    action_period = 15
    for i in range(len(data) - action_period):
        current_close = data['$close'].iloc[i]
        future_max = data['$close'].iloc[i+1:i+action_period+1].max()
        future_min = data['$close'].iloc[i+1:i+action_period+1].min()

        if future_max > 1.06 * current_close and future_min >= 0.98 * current_close:
            actions.append(0)  # 买入
        elif future_min <= 0.95 * current_close or (future_min <= 0.97 * current_close and future_max <= 1.01 * current_close):
            actions.append(1)  # 卖出
        else:
            actions.append(2)  # 持有

    # 用持有（2）填充剩余的动作以匹配数据长度
    actions.extend([2] * action_period)
    return actions, action_period


# 修改绘制带有买卖标记的预测的函数
def plot_price_with_actions(data, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data['$close'].values, label='Price', color='blue', alpha=0.5)  # 绘制股价
    
    # Ensure the indices for y_pred match the length of data
    buy_indices = np.where(y_pred == 0)[0]  # Buy actions
    sell_indices = np.where(y_pred == 1)[0]  # Sell actions
    
    plt.scatter(buy_indices, data['$close'].values[buy_indices], color='red', label='Buy', marker='o')  # 买入点
    plt.scatter(sell_indices, data['$close'].values[sell_indices], color='green', label='Sell', marker='x')  # 卖出点
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    
    # 设置 x 和 y 轴的边界
    plt.xlim(0, len(data) - 1)  # x 轴范围
    plt.ylim(data['$close'].min(), data['$close'].max())  # y 轴范围

    plt.legend()
    plt.axis('equal')  # 确保 x 和 y 轴的比例相同
    plt.show()

def backtest_strategy(data, predictions, actions, strong_buy_points, strong_sell_points, stock_code, initial_capital=100000, transaction_fee=0.005, model_name='Model', folder_path='', production=False):
    capital = initial_capital
    position = 0  # 当前持有的股票数量
    buy_price = 0  # 记录买入价格
    buy_indices = []  # 记录买入点的索引
    sell_indices = []  # 记录卖出点的索引
    strong_buy_indices = np.where(strong_buy_points)[0]  # 强买点索引
    strong_sell_indices = np.where(strong_sell_points)[0]  # 强卖点索引

    for i in range(len(predictions)):
        action = predictions[i]
        current_price = data['$close'].iloc[i]

        if action == 0:  # 买入
            if capital > 0:  # 确保有足够的资金
                shares_to_buy = capital // current_price  # 计算可购买的股票数量
                capital -= shares_to_buy * current_price * (1 + transaction_fee)  # 扣除资金和交易费用
                position += shares_to_buy  # 更新持有的股票数量
                buy_price = current_price  # 更新买入价格
                buy_indices.append(i)  # 记录买入点的索引

        elif action == 1:  # 卖出
            if position > 0:  # 确保有持有的股票
                capital += position * current_price * (1 - transaction_fee)  # 卖出所有持有的股票并扣除交易费用
                position = 0  # 清空持仓
                sell_indices.append(i)  # 记录卖出点的索引

    # 计算最终的总资产
    final_value = capital + position * data['$close'].iloc[-1]  # 加上最后持有股票的价值

    # 创建图形和子图
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # 绘制回测结果
    axs[0].plot(data['$close'].values, label='Price', color='blue', alpha=0.5)  # 绘制股价
    axs[0].scatter(buy_indices, data['$close'].iloc[buy_indices], color='red', label='Buy', marker='o')  # 买入点
    axs[0].scatter(sell_indices, data['$close'].iloc[sell_indices], color='green', label='Sell', marker='x')  # 卖出点
    axs[0].set_title(f'Backtest Results: {model_name} - {stock_code} - Buy and Sell Points')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Price')
    axs[0].legend()

    # 绘制测试数据中的动作
    action_buy_indices = np.where(predictions == 0)[0]  # action 为 0 的索引
    action_sell_indices = np.where(predictions == 1)[0]  # action 为 1 的索引
    axs[1].plot(data['$close'].values, label='Price', color='blue', alpha=0.5)  # 绘制股价
    axs[1].scatter(action_buy_indices, data['$close'].iloc[action_buy_indices], color='red', label='Action Buy (0)', marker='o', alpha=0.5)  # 测试数据买入点
    axs[1].scatter(action_sell_indices, data['$close'].iloc[action_sell_indices], color='green', label='Action Sell (1)', marker='x', alpha=0.5)  # 测试数据卖出点
    
    # 在第二个坐标图上绘制强买点和强卖点
    axs[1].scatter(strong_buy_indices, data['$close'].iloc[strong_buy_indices], color='blue', label='Strong Buy', marker='^')  # 强买点
    axs[1].scatter(strong_sell_indices, data['$close'].iloc[strong_sell_indices], color='orange', label='Strong Sell', marker='v')  # 强卖点
    
    axs[1].set_title(f'Test Actions - {stock_code} - Buy (0) and Sell (1)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Price')
    axs[1].legend()

    plt.tight_layout()

    # 保存回测结果的图片到每个股票对应的文件夹
    if production == False:
        plot_filename = f"{folder_path}{stock_code}_{datetime.now().strftime('%Y-%m-%d')}.png"
    else:
        plot_filename = f"{folder_path}{stock_code}_{datetime.now().strftime('%Y-%m-%d')}_production.png"
    plt.savefig(plot_filename)  # 保存回测结果的图片
    plt.close()  # 关闭图形以释放内存

    return final_value

class TCTSModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TCTSModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=2, stride=1, padding=1)
                
        # Transformer layer with batch_first=True
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape to (batch_size, input_size, seq_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))      

        x = self.transformer_encoder(x)  # Pass through Transformer encoder
        x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)  # Global average pooling
        
        x = self.fc(x)
        return x

# 训练 TCTS 模型
def train_tcts_model(model, X_train, y_train, criterion, optimizer, epochs, batch_size):
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"TCTS Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 预测概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  # Focal Loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class EnsembleModel(nn.Module):
    def __init__(self, lstm_model, tcts_model, input_size, output_size):
        super(EnsembleModel, self).__init__()
        self.lstm_model = lstm_model
        self.tcts_model = tcts_model
        self.fc = nn.Linear(2 * output_size, output_size)  # 2个模型的输出

    def forward(self, x):
        lstm_output = self.lstm_model(x)
        tcts_output = self.tcts_model(x)
        combined_output = torch.cat((lstm_output, tcts_output), dim=1)  # 拼接两个模型的输出
        final_output = self.fc(combined_output)  # 通过MLP进行最终预测
        return final_output

# 在模型评估中识别强买点和强卖点
def evaluate_model(model, X_test, strong_buy_threshold, strong_sell_threshold):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        scores = torch.softmax(outputs, dim=1)  # 计算每个动作的概率
        _, predicted = torch.max(outputs, 1)

        buy_points = (predicted == 0)
        sell_points = (predicted == 1)
        strong_buy_points = (predicted == 0) & (scores[:, 0] > strong_buy_threshold)  # 强买点
        strong_sell_points = (predicted == 1) & (scores[:, 1] > strong_sell_threshold)  # 强卖点

    return predicted.numpy(), strong_buy_points.numpy(), strong_sell_points.numpy(), buy_points.numpy(), sell_points.numpy()

# 在回测后添加逻辑来记录潜力买卖股票
def record_signals(stock_code, strong_buy_points, strong_sell_points, recent_predictions, buy_points, sell_points):
    potential_buy_stocks = []  # 潜力买入股票
    potential_sell_stocks = []  # 潜力卖出股票
    strong_buy_stocks = []
    strong_sell_stocks = []

    # 检查最近一天的买点和卖点
    if buy_points[-1]:  # 最近一天有买点
        potential_buy_stocks.append(stock_code)
    if sell_points[-1]:  # 最近一天有卖点
        potential_sell_stocks.append(stock_code)

    # 检查最近5天的买点和卖点
    recent_buy_signals = sum(buy_points[-5:])  # 最近5天的买点数量
    recent_sell_signals = sum(sell_points[-5:])  # 最近5天的卖点数量

    if recent_buy_signals >= 3 and recent_sell_signals == 0:
        strong_buy_stocks.append(stock_code)
    if recent_sell_signals >= 3 and recent_buy_signals == 0:
        strong_sell_stocks.append(stock_code)

    return potential_buy_stocks, potential_sell_stocks, strong_buy_stocks, strong_sell_stocks

# 移动到外部的函数
def train_model(model, X_train, y_train, criterion, optimizer, epochs, batch_size):
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

if __name__ == "__main__":  # 确保 main 函数在脚本直接运行时执行
    # 1. Define start_date and end_date
    start_date = "2015-01-01"  # Set your desired start date
    end_date = datetime.now().strftime("%Y-%m-%d")  # Updated to use today's date

    # 3. 增加stock_list
    stock_list = ["UNG","TLT","BEKE","GOLD","RIOT","MARA","FXI","FUTU","COIN","CPNG","UVXY","XIACY","IQ","ASHR","XBI","LABU","FNGU","PLTR","TCEHY","MPNGY","OXY","SOXL","SOXS","KWEB","CWEB","YINN","YANG","TQQQ","SQQQ","TNA","AAPL","AMGN","AMZN","BABA","BIDU","BILI","DIS","GILD","GOOGL","JD","LI","META","MSFT","SBUX","NFLX","NIO","NTES","NVDA","PDD","PG","PYPL","PEP","COST","CSCO","ORCL","AMD","UPS","TME","TSLA","XPEV"]

    listing_dates = {
        "UNG": "2007-04-18",
        "TLT": "2002-07-22",
        "BEKE": "2020-08-13",
        "GOLD": "1983-05-02",
        "RIOT": "2003-10-24",
        "MARA": "2012-07-28",
        "FXI": "2004-10-05",
        "FUTU": "2019-03-08",
        "COIN": "2021-04-14",
        "CPNG": "2021-03-11",
        "UVXY": "2011-10-03",
        "XIACY": "2010-12-15",
        "IQ": "2018-03-29",
        "ASHR": "2013-11-06",
        "XBI": "2006-01-31",
        "LABU": "2015-05-28",
        "FNGU": "2018-01-22",
        "PLTR": "2020-09-30",
        "TCEHY": "2004-06-16",
        "MPNGY": "2010-12-15",
        "OXY": "1964-07-01",
        "SOXL": "2010-03-11",
        "SOXS": "2010-03-11",
        "KWEB": "2013-07-31",
        "CWEB": "2016-08-17",
        "YINN": "2009-11-05",
        "YANG": "2009-11-05",
        "TQQQ": "2010-02-09",
        "SQQQ": "2010-02-09",
        "TNA": "2008-11-05",
        "AAPL": "1980-12-12",
        "ALXN": "1996-02-28",
        "AMGN": "1983-06-17",
        "AMZN": "1997-05-15",
        "BABA": "2014-09-19",
        "BIDU": "2005-08-05",
        "BIIB": "1991-09-17",
        "BILI": "2018-03-28",
        "BRK.B": "1996-05-09",
        "DIS": "1957-11-12",
        "GILD": "1992-01-22",
        "GOOGL": "2004-08-19",
        "HD": "1981-09-22",
        "ILMN": "2000-07-28",
        "INCY": "1993-11-01",
        "INTC": "1971-10-13",
        "JD": "2014-05-22",
        "JNJ": "1944-09-24",
        "JPM": "1969-05-28",
        "LI": "2020-07-30",
        "META": "2012-05-18",
        "MSFT": "1986-03-13",
        "NBIX": "1996-05-23",
        "SBUX": "1992-06-26",
        "NFLX": "2002-05-23",
        "NIO": "2018-09-12",
        "NTES": "2000-06-30",
        "NVDA": "1999-01-22",
        "PDD": "2018-07-26",
        "PG": "1891-01-01",
        "PYPL": "2015-07-20",
        "REGN": "1991-03-27",
        "SGEN": "2001-03-01",
        "PEP": "1919-01-01",
        "KO": "1919-09-05",
        "COST": "1985-12-05",
        "T": "1984-07-19",
        "CMCSA": "1972-06-29",
        "MCD": "1966-04-21",
        "CVX": "1926-01-01",
        "XOM": "1920-03-01",
        "CSCO": "1990-02-16",
        "ORCL": "1986-03-12",
        "ABT": "1929-03-19",
        "ABBV": "2013-01-02",
        "DHR": "1981-09-24",
        "ACN": "2001-07-19",
        "ADBE": "1986-08-20",
        "CRM": "2004-06-23",
        "TXN": "1953-10-01",
        "AMD": "1972-09-27",
        "QCOM": "1991-12-13",
        "HON": "1925-01-01",
        "LIN": "1995-09-06",
        "PM": "2008-03-17",
        "NEE": "1950-06-27",
        "LOW": "1961-12-19",
        "MDT": "1977-09-20",
        "BKNG": "1999-03-30",
        "AMT": "1998-07-08",
        "UPS": "1999-11-10",
        "NKE": "1980-12-02",
        "C": "1968-01-02",
        "SCHW": "1987-09-23",
        "TME": "2018-12-12",
        "TSLA": "2010-06-29",
        "UNH": "1984-10-05",
        "V": "2008-03-19",
    "VRTX": "1991-06-18",
    "WMT": "1970-08-25",
    "XPEV": "2020-08-27"
}

    # 4. 创建文件夹
    today_date = datetime.now().strftime("%Y-%m-%d")
    base_folder_path = f"/Users/huiyu/Desktop/pytorch/data/TCTS-Transformer-{today_date}/"  # 基础文件夹路径
    os.makedirs(base_folder_path, exist_ok=True)  # 创建基础文件夹

    for stock_code in stock_list:
        # Create folder for saving the model if it doesn't exist
        model_production_folder_path = f"/Users/huiyu/Desktop/model/model_production/{stock_code}/"
        os.makedirs(model_production_folder_path, exist_ok=True)  # Create the folder

        model_folder_path = f"/Users/huiyu/Desktop/model/model/{stock_code}/"
        os.makedirs(model_folder_path, exist_ok=True)  # Create the folder


        folder_path = f"{base_folder_path}{stock_code}/"  # 每个股票的文件夹路径
        os.makedirs(folder_path, exist_ok=True)  # 创建文件夹
        # 使用股票的上市日期作为开始日期
        start_date = listing_dates.get(stock_code, "2015-01-01")  # 默认值为2015-01-01
        logging.info(f"Fetching daily data and Alpha158 factors for {stock_code} from {start_date} to {end_date}.")
        data = get_daily_data_with_factors(stock_code, start_date, end_date)
        print(f"Data shape after merging: {data.shape}")

        if data.empty:
            raise ValueError(f"No data found for {stock_code} from {start_date} to {end_date}.")

        # 数据检查
        logging.info("Data loaded successfully. Checking data head.")

        # Fill missing values
        data = data.ffill().bfill()  # Use ffill() and bfill() directly

        # 计算动作并添加到数据框
        data['action'], action_period = short_term_calculate_action(data)
        action_counts = data['action'].value_counts()
        logging.info(f"Action counts in the original data: Buy (0): {action_counts.get(0, 0)}, Sell (1): {action_counts.get(1, 0)}, Hold (2): {action_counts.get(2, 0)}")

        # 归一化特征（不包括动作列）
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(data.drop(columns=['action']))

        # 创建包含归一化数据和动作的数据框
        scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])  # 排除动作列
        scaled_data['action'] = data['action'].values  # 添加动作列

        SEQ_LENGTH = 30
        X, y = create_sequences(scaled_data.values, SEQ_LENGTH, action_period)

        # 划分数据集
        split = int(len(X) * 0.9)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # 转换为PyTorch张量
        X_train_torch = torch.tensor(X_train, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train, dtype=torch.long)  # 使用long进行分类
        X_test_torch = torch.tensor(X_test, dtype=torch.float32)
        y_test_torch = torch.tensor(y_test, dtype=torch.long)

        # 划分production
        X_train_production, X_test_production = X[:-SEQ_LENGTH], X[-SEQ_LENGTH:]
        y_train_production, y_test_production = y[:-SEQ_LENGTH], y[-SEQ_LENGTH:]

        # 转换为PyTorch张量
        X_train_production_torch = torch.tensor(X_train_production, dtype=torch.float32)
        y_train_production_torch = torch.tensor(y_train_production, dtype=torch.long)  # 使用long进行分类
        X_test_production_torch = torch.tensor(X_test_production, dtype=torch.float32)
        y_test_production_torch = torch.tensor(y_test_production, dtype=torch.long)

        INPUT_SIZE = X_train.shape[2]

        # 使用 Focal Loss 训练 TCTS 模型
        tcts_model = TCTSModel(INPUT_SIZE, OUTPUT_SIZE)
        #criterion_tcts = nn.CrossEntropyLoss()  # 修改为标准交叉熵损失
        criterion_tcts = FocalLoss(alpha=1.0, gamma=2.0)
        optimizer_tcts = optim.Adam(tcts_model.parameters(), lr=0.001)
        EPOCHS_TCTS = 150
        train_tcts_model(tcts_model, X_train_torch, y_train_torch, criterion_tcts, optimizer_tcts, EPOCHS_TCTS, BATCH_SIZE)

        # 使用 Focal Loss 训练 TCTS Production模型
        tcts_model_production = TCTSModel(INPUT_SIZE, OUTPUT_SIZE)
        criterion_tcts = FocalLoss(alpha=1.0, gamma=2.0)
        optimizer_tcts_production = optim.Adam(tcts_model_production.parameters(), lr=0.001)
        train_tcts_model(tcts_model_production, X_train_production_torch, y_train_production_torch, criterion_tcts, optimizer_tcts_production, EPOCHS_TCTS, BATCH_SIZE)

        # 测试 TCTS 模型
        tcts_model.eval()
        with torch.no_grad():
            outputs_tcts = tcts_model(X_test_torch)
            _, predicted_tcts = torch.max(outputs_tcts, 1)
            accuracy_tcts = accuracy_score(y_test, predicted_tcts.numpy())
            print(f"TCTS Test Accuracy for {stock_code}: {accuracy_tcts}")

        # 测试 TCTS Production模型
        tcts_model_production.eval()
        with torch.no_grad():
            outputs_tcts_production = tcts_model_production(X_test_production_torch)
            _, predicted_tcts_production = torch.max(outputs_tcts_production, 1)
            #accuracy_tcts = accuracy_score(y_test, predicted_tcts.numpy())
            #print(f"TCTS Test Accuracy for {stock_code}: {accuracy_tcts}")

        # 计算强买点和强卖点
        predicted, tcts_strong_buy_points, tcts_strong_sell_points, tcts_buy_points, tcts_sell_points = evaluate_model(tcts_model, X_test_torch, STRONG_BUY_THRESHOLD, STRONG_SELL_THRESHOLD)

        # 计算Production强买点和强卖点
        predicted_production, tcts_strong_buy_points_production, tcts_strong_sell_points_production, tcts_buy_points_production, tcts_sell_points_production = evaluate_model(tcts_model_production, X_test_production_torch, STRONG_BUY_THRESHOLD, STRONG_SELL_THRESHOLD)
        
        # 进行回测
        final_capital_tcts = backtest_strategy(data.iloc[len(data) - len(y_test):], predicted_tcts.numpy(), y_test, tcts_strong_buy_points, tcts_strong_sell_points, stock_code=stock_code, folder_path=folder_path, production=False)

        # 进行Production回测
        final_capital_tcts_production = backtest_strategy(data.iloc[len(data) - len(y_test_production):], predicted_tcts_production.numpy(), y_test_production, tcts_strong_buy_points_production, tcts_strong_sell_points_production, stock_code=stock_code, folder_path=folder_path, production=True)

        # 记录信号
        potential_buy_stocks, potential_sell_stocks, strong_buy_stocks, strong_sell_stocks = record_signals(stock_code, tcts_strong_buy_points, tcts_strong_sell_points, predicted_tcts, tcts_buy_points, tcts_sell_points)


        # 记录Production信号
        potential_buy_stocks_production, potential_sell_stocks_production, strong_buy_stocks_production, strong_sell_stocks_production = record_signals(stock_code, tcts_strong_buy_points_production, tcts_strong_sell_points_production, predicted_tcts_production, tcts_buy_points_production, tcts_sell_points_production)

        # 将信号写入到 signal.txt 文件
        with open(f"{base_folder_path}signal.txt", 'a') as signal_file:
            if potential_buy_stocks:
                signal_file.write(f"Potential Buy Stocks: {', '.join(potential_buy_stocks)}\n")
            if potential_sell_stocks:
                signal_file.write(f"Potential Sell Stocks: {', '.join(potential_sell_stocks)}\n")
            if strong_buy_stocks:
                signal_file.write(f"Strong Buy Stocks: {', '.join(strong_buy_stocks)}\n")
            if strong_sell_stocks:
                signal_file.write(f"Strong Sell Stocks: {', '.join(strong_sell_stocks)}\n")
            signal_file.flush()

        # 将信号写入到 signal_production.txt 文件
        with open(f"{base_folder_path}signal_production.txt", 'a') as signal_file:
            if potential_buy_stocks_production:
                signal_file.write(f"Potential Buy Stocks: {', '.join(potential_buy_stocks_production)}\n")
            if potential_sell_stocks_production:
                signal_file.write(f"Potential Sell Stocks: {', '.join(potential_sell_stocks_production)}\n")
            if strong_buy_stocks_production:
                signal_file.write(f"Strong Buy Stocks: {', '.join(strong_buy_stocks_production)}\n")
            if strong_sell_stocks_production:
                signal_file.write(f"Strong Sell Stocks: {', '.join(strong_sell_stocks_production)}\n")
            signal_file.flush()

        # Save the production model
        torch.save(tcts_model, f"{model_folder_path}tcts_model.pth") 
        torch.save(tcts_model_production, f"{model_production_folder_path}tcts_model_production.pth") 
        # 释放内存
        del X_train_torch, y_train_torch, X_test_torch, y_test_torch, predicted_tcts  # 删除不再需要的变量
        del X_train_production_torch, y_train_production_torch, X_test_production_torch, y_test_production_torch, predicted_tcts_production  # 删除不再需要的变量
        torch.cuda.empty_cache()  # 如果使用 GPU，释放未使用的 GPU 内存

        # 5. 回测结果(acc和收益值)输入到统一的文件
        backtest_file_path = f"{base_folder_path}backtest.txt"  # 统一的文件路径
        with open(backtest_file_path, 'a') as f:
            f.write(f"{stock_code} - Accuracy: {accuracy_tcts}, Final Capital: ${final_capital_tcts:.2f}\n")
            f.flush()

        
