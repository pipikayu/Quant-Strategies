import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import yfinance as yf
import logging
import time
import datetime  # Add this import statement
from tensorflow.keras.initializers import HeNormal  # 导入 HeNormal 初始化器
import os  # Add this import statement
from tensorflow.keras.regularizers import l1  # 添加L1正则化器的导入
import sys  # 添加导入语句
import matplotlib
import re
matplotlib.use('Agg')  # Use a non-interactive backend

def download_daily_data(ticker, start_date):
    logging.info(f"Fetching daily data for {ticker} from {start_date}.")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Get the latest date
    
    retries = 5  # 最大重试次数
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            if data.empty:
                raise ValueError("No data returned from yfinance.")
            return data
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:  # 如果不是最后一次尝试
                time.sleep(10)  # 等待 10 秒后重试
            else:
                logging.critical("All attempts to fetch data failed.")
                raise  # 重新抛出异常以便处理

def calculate_alpha_factors(data):
    logging.info("Calculating alpha factors...")
    window = 14
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window).mean()
    avg_loss = down.rolling(window).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    data['macd'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    
    # KDJ计算
    low_min = data['Low'].rolling(window=9).min()
    high_max = data['High'].rolling(window=9).max()
    rsv = (data['Close'] - low_min) / (high_max - low_min) * 100
    data['K'] = rsv.ewm(com=2, adjust=False).mean()
    data['D'] = data['K'].ewm(com=2, adjust=False).mean()
    data['J'] = 3 * data['K'] - 2 * data['D']

    # 金叉银叉
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['golden_cross'] = np.where(data['SMA_50'] > data['SMA_200'], 1, 0)  # 金叉
    data['death_cross'] = np.where(data['SMA_50'] < data['SMA_200'], 1, 0)  # 银叉

       # SMA (Simple Moving Averages)
    data['sma5'] = data['Close'].rolling(window=5).mean()
    data['sma20'] = data['Close'].rolling(window=20).mean()
    data['sma100'] = data['Close'].rolling(window=100).mean()

    # EMA (Exponential Moving Averages)
    data['ema5'] = data['Close'].ewm(span=5).mean()
    data['ema20'] = data['Close'].ewm(span=20).mean()
    data['ema50'] = data['Close'].ewm(span=50).mean()
    data['ema100'] = data['Close'].ewm(span=100).mean()
    data['ema200'] = data['Close'].ewm(span=200).mean()

    # 生成优化因子
    data['Rel_SMA_200'] = (data['Close'][ticker].iloc[0] - data['SMA_200']) / data['SMA_200']
    data['Rel_SMA_50'] = (data['Close'][ticker].iloc[0] - data['SMA_50']) / data['SMA_50']
    data['Rel_SMA_100'] = (data['Close'][ticker].iloc[0] - data['sma100']) / data['sma100']
    data['Rel_SMA_20'] = (data['Close'][ticker].iloc[0] - data['sma20']) / data['sma20']
    data['Rel_SMA_5'] = (data['Close'][ticker].iloc[0] - data['sma5']) / data['sma5']
    data['Rel_EMA_200'] = (data['Close'][ticker].iloc[0] - data['ema200']) / data['ema200']
    data['Rel_EMA_50'] = (data['Close'][ticker].iloc[0] - data['ema50']) / data['ema50']
    data['Rel_EMA_100'] = (data['Close'][ticker].iloc[0] - data['ema100']) / data['ema100']
    data['Rel_EMA_20'] = (data['Close'][ticker].iloc[0] - data['ema20']) / data['ema20']
    data['Rel_EMA_5'] = (data['Close'][ticker].iloc[0] - data['ema5']) / data['ema5']
    data['Momentum_20'] = data['Close'] - data['Close'].shift(20)
    data['Quantile_50'] = data['Close'].rolling(50).apply(lambda x: x.rank().iloc[-1] / len(x))

    # 添加其他优化特征
    data['Volatility'] = data['Close'].rolling(20).std()
    data['Adjusted_SMA_20'] = data['sma20'] / data['Volatility']

    data['golden_cross_short'] = np.where(data['sma5'] > data['sma20'], 1, 0)  # 金叉
    data['death_cross_short'] = np.where(data['sma5'] < data['sma20'], 1, 0)  # 银叉

    # PPO (Percentage Price Oscillator)
    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    data['ppo'] = (ema12 - ema26) / ema26 * 100

    # ROC (Rate of Change)
    data['roc'] = data['Close'].pct_change(periods=12) * 100

    # OBV (On-Balance Volume)
    data['obv'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

    # MFI (Money Flow Index)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)
    
    # Convert positive_flow and negative_flow to Pandas Series
    positive_flow_series = pd.Series(positive_flow.ravel(), index=data.index)  # Ensure positive_flow is 1D
    negative_flow_series = pd.Series(negative_flow.ravel(), index=data.index)  # Ensure negative_flow is 1D

    positive_mf = positive_flow_series.rolling(window=14).sum()  # Use the rolling method on the Series
    negative_mf = negative_flow_series.rolling(window=14).sum()  # Use the rolling method on the Series
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    data['mfi'] = mfi

    # Stochastic %K and %D
    low_min = data['Low'].rolling(window=14).min()
    high_max = data['High'].rolling(window=14).max()
    data['stoch_K'] = (data['Close'] - low_min) / (high_max - low_min) * 100
    data['stoch_D'] = data['stoch_K'].rolling(window=3).mean()

    # Chaikin Oscillator
    adl = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']) * data['Volume']
    data['chaikin_oscillator'] = adl.rolling(window=3).sum()  # 3-day sum of ADL

    # 神奇九转（示例实现）
    data['magic_nine'] = (data['Close'] - data['Close'].shift(9)) / data['Close'].shift(9) * 100  # 计算百分比变化

    # 布林带
    data['Bollinger_Upper'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2)
    data['Bollinger_Lower'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2)

    # 动量
    data['momentum'] = data['Close'].pct_change(periods=10)  # 过去10天的动量

    # 成交量变化率
    data['volume_change'] = data['Volume'].pct_change()

    # 平均真实范围（ATR）
    data['high_low'] = data['High'] - data['Low']
    data['high_close'] = abs(data['High'] - data['Close'].shift())
    data['low_close'] = abs(data['Low'] - data['Close'].shift())
    data['tr'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
    data['ATR'] = data['tr'].rolling(window=14).mean()

    # CCI计算
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    n = 20  # 计算CCI的周期
    sma_tp = typical_price.rolling(window=n).mean()  # 计算典型价格的SMA
    mean_deviation = (typical_price - sma_tp).abs().rolling(window=n).mean()  # 计算平均偏差
    data['cci'] = (typical_price - sma_tp) / (0.015 * mean_deviation)  # 计算CCI因子

    # Williams %R计算
    n = 14  # 计算周期
    high_max = data['High'].rolling(window=n).max()  # 过去n天的最高价
    low_min = data['Low'].rolling(window=n).min()    # 过去n天的最低价
    data['williams_r'] = (high_max - data['Close']) / (high_max - low_min) * -100  # 计算Williams %R

    data.fillna(0, inplace=True)
    logging.info("Alpha factors calculated.")

    # 保存因子到Excel
    data.to_excel("alpha_factors.xlsx", index=True)
    logging.info("Alpha factors saved to alpha_factors.xlsx.")
    
    return data

def prepare_data(data):
    logging.info("Preparing data...")
    """
    features = [
        'rsi', 'macd', 'macd_signal', 
        'K', 'D', 'J', 
        'SMA_50', 'SMA_200', 
        'golden_cross', 'death_cross', 
        'magic_nine', 
        'Bollinger_Upper', 'Bollinger_Lower', 
        'momentum', 'volume_change', 
        'ATR','high_low','high_close','low_close','tr'
    ]

    """
    features = [
        'rsi', 'macd', 'macd_signal', 'ppo', 'roc', 'obv','mfi',
        'K', 'D', 'stoch_K', 'stoch_D','chaikin_oscillator',
        'SMA_50', 'SMA_200', 'sma20', 'sma100','ema20','ema50','ema100','ema200','sma5','ema5',
        'golden_cross', 'death_cross', 'golden_cross_short', 'death_cross_short',
        'magic_nine', 'williams_r',
        'Bollinger_Upper', 'Bollinger_Lower', 
        'momentum', 'volume_change', 
        'ATR', 'high_low', 
        'tr','cci','J','low_close','high_close', 
        'Rel_SMA_200', 'Rel_SMA_50', 'Rel_SMA_100', 'Rel_SMA_20', 'Rel_SMA_5', 'Rel_EMA_200', 'Rel_EMA_50', 'Rel_EMA_100', 'Rel_EMA_20', 'Rel_EMA_5', 'Momentum_20', 'Quantile_50', 'Volatility', 'Adjusted_SMA_20'
    ]
    
    
    # Check if data is empty
    if data.empty:
        logging.error("No data available to prepare.")
        return data, features  # Return empty data and features if no data is available

    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    logging.info("Data preparation completed.")
    
    # 打印特征数
    logging.info(f"Number of features: {len(features)}")
    
    # 保存特征到本地文件
    features_df = data[features]
    features_df.to_csv("features.csv", index=False)  # 保存为 CSV 文件
    logging.info("Features saved to features.csv.")
    
    return data, features

class DQLAgent:
    def __init__(self, state_size, action_size, transaction_cost=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.transaction_cost = transaction_cost
        self.model = self._build_model()
        self.target_model = self._build_model()  # 添加目标网络
        self.update_target_model()  # 初始化时更新目标网络

    def update_target_model(self):
        """将主网络的权重复制到目标网络"""
        print("更新目标网络")
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu', kernel_initializer=HeNormal()),  # 添加L1正则化
            Dropout(0.1),
            Dense(64, activation='relu', kernel_initializer=HeNormal()),  # 添加L1正则化
            Dropout(0.1),
            Dense(self.action_size, activation='linear', kernel_initializer=HeNormal())  # 最后一层不需要正则化
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                      loss='mean_squared_error')
        
        # 打印模型结构
        model.summary()
        
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        predicted_action = np.argmax(q_values[0])
        if predicted_action == 0 and abs(np.max(q_values)) <= self.transaction_cost * 100:  # 买入动作预测收益过低
            return 2, q_values  # 持有
        elif predicted_action == 1 and abs(np.max(q_values)) <= self.transaction_cost * 100:
            return 2, q_values  # 持有
        return predicted_action, q_values

    def replay(self, batch_size):
        # 按时间顺序从 memory 中选择 minibatch
        # 采样时要保证时间上的连续性
        minibatch = []
        for i in range(batch_size):
            idx = random.randint(0, len(self.memory) - 2)  # 防止越界，确保 next_state 存在
            state, action, reward, next_state, done = self.memory[idx]
            minibatch.append((state, action, reward, next_state, done))

        # 提取所有状态和下一个状态
        states = np.array([x[0] for x in minibatch]).reshape(batch_size, -1)
        next_states = np.array([x[3] for x in minibatch]).reshape(batch_size, -1)
        

        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # 计算 next_states 对应的最大 Q 值
        next_q_values = self.target_model.predict(next_states, verbose=0)  # 使用目标网络预测
        next_q_values = np.amax(next_q_values, axis=1)

        # 计算 target
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        targets = np.clip(targets, -10, 10)  # 确保目标值不爆炸

        # 获取当前状态对应的 Q 值
        target_f = self.model.predict(states, verbose=0)  # 批量预测

        # 更新每个样本的 target_f，目标是根据动作索引来更新
        for i in range(batch_size):
            target_f[i][actions[i]] = targets[i]

        # 执行一次批量训练
        self.model.fit(states, target_f, epochs=1, verbose=0)
        # 更新 epsilon，逐步降低探索概率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    

    def save_model(self, model_filename):
        #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳
        #model_filename = f"dql_agent_model_{timestamp}.h5"  # 创建文件名
        self.model.save(model_filename)  # 保存模型
        logging.info(f"Model saved to {model_filename}.")  # 记录保存信息

    def load_agent_model(self, model_path):
        """Load a model from the specified path."""
        self.model = tf.keras.models.load_model(model_path, custom_objects={'mean_squared_error': tf.keras.losses.MeanSquaredError()})
        logging.info(f"Model loaded from {model_path}.")  # 记录加载信息


def calculate_reward_2(current_value, previous_value, action, last_action, holding_period, current_price, next_price, holdings, transaction_cost=0.001):
    """
    改进后的奖励函数，增加对未来趋势、持有时间、回撤的动态调整。
    """
    if previous_value == 0:
        previous_value = 1e-10  # 避免除零

    reward = 0  # 初始化奖励
    price_change = 0.0
    
    # 动作奖励逻辑
    if action == 0:  # 买入
        future_trend_bonus = (next_price - current_price) / current_price
        future_trend_bonus = future_trend_bonus.item()
        reward += future_trend_bonus
        reward -= transaction_cost  # 扣除交易成本

    elif action == 1:  # 卖出
        reward -= transaction_cost  # 扣除交易成本
        # 增加对未来价格下跌的奖励
        future_trend_penalty = (current_price - next_price) / current_price
        # Ensure future_trend_penalty is a scalar
        future_trend_penalty = future_trend_penalty.item()  # Convert to scalar if it's a Series
        if future_trend_penalty > 0:
            reward += future_trend_penalty
        else:
            reward -= abs(future_trend_penalty)

    elif action == 2:  # 持有
        price_change = (next_price - current_price) / current_price
        """
        if holdings > 0:  # 持股期持有
            reward += price_change * 0.5  # 股价变化的奖励
            # 持有时间奖励
            if price_change >= 0.01:
                reward += holding_period * 0.0001
        else:  # 非持股期持有
            if price_change <= 0:
                reward += abs(price_change) * 0.5  # 规避风险的奖励
            elif price_change > 0:
                reward -= price_change             # 错失机会的惩罚
        """
        if abs(price_change) <= 0.01:
            reward += 0.005
        else:
            reward -= abs(price_change)
        
    # 动态回撤惩罚
    drawdown = (next_price - current_price) / current_price
    if drawdown >= 0.05:
        reward -= drawdown

    logging.info(f"Action: {action}, Current Price: {current_price}, Next Price: {next_price}, Reward: {reward}, Drawdown: {drawdown}, Pct_change:{price_change}")  # New logging statement

    # 扩大奖励值范围
    reward *= 100.0
    return reward

def train_agent(data, features, initial_balance=100000.0, transaction_cost=0.001, train_split=0.8, model_save_path="dql_agent_model.h5"):
    """
    训练 DQN 智能体，基于给定的市场数据和特征。
    """
    train_size = int(len(data) * train_split)
    train_data = data.iloc[:train_size]
    
    if train_data.empty:
        logging.error("Training data is empty. Cannot train the agent.")
        return None, None, features, []

    test_data = data.iloc[train_size:]
    state_size = len(features)
    action_size = 3  # 动作空间：0=买入, 1=卖出, 2=持有
    agent = DQLAgent(state_size, action_size)
    batch_size = 64
    logging.info("Starting training...")
    epoch_num = 10
    epoch_rewards = []  # 记录每个 epoch 的总奖励
    q_values = []
    for episode in range(epoch_num):
        state = train_data[features].iloc[0].values.reshape(1, -1)
        cash, holdings = initial_balance, 0
        previous_value = initial_balance
        last_action = 2  # 初始为持有
        holding_period = 0  # 持有时间计数
        total_reward = 0.0  # Initialize as float instead of just 0
        cash = float(cash)
        if episode % 2 == 0:  # 每10个episode更新一次目标网络
            agent.update_target_model()
        for t in range(len(train_data) - 1):
            current_price = train_data['Close'].iloc[t].iloc[0]
            next_price = train_data['Close'].iloc[t + 1].iloc[0]
            action, q_values = agent.act(state)

            # 更新资金、持仓和持有期
            if action == 0:  # 买入逻辑
                max_shares = int(cash // (current_price * (1 + transaction_cost)).item())
                if max_shares > 0:
                    cash -= max_shares * current_price * (1 + transaction_cost)
                    holdings += max_shares
                    holding_period = 0  # 重置持有时间

            elif action == 1:  # 卖出逻辑
                cash += holdings * current_price * (1 - transaction_cost)  # 计算卖出后现金
                holdings = 0  # 清空持仓
                holding_period = 0  # 重置持有时间

            elif action == 2:  # 持有逻辑
                if holdings >= 0:
                    holding_period += 1

            # 确保 holdings 是标量
            if isinstance(holdings, (pd.Series, np.ndarray)):
                holdings_value = holdings.item()  # 转换为标量
            else:
                holdings_value = holdings

            # Ensure holdings_value is a scalar
            holdings_value = holdings_value.item() if isinstance(holdings_value, (pd.Series, np.ndarray)) else holdings_value
            
            # 计算当前资产价值
            current_value = cash + holdings_value * next_price  # 确保 current_value 是标量
            current_value = current_value.item()  # Convert to scalar if it's a Series

            # 计算奖励
            reward = 0.0
            reward = calculate_reward_2(current_value, previous_value, action, last_action, holding_period, current_price, next_price, holdings,transaction_cost)
            total_reward += reward.item() if isinstance(reward, pd.Series) else reward
            
            # 状态更新
            next_state = train_data[features].iloc[t + 1].values.reshape(1, -1)
            agent.remember(state, action, reward, next_state, t == len(train_data) - 1)
            state = next_state
            last_action = action 

            # 经验回放
            n_step = 3
            #if len(agent.memory) > batch_size + n_step:
            #    agent.replay_n2(batch_size, n_step)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            
            # 日志输出
            logging.info(f"Episode: {episode + 1},Action:{action},Step: {t + 1},Cash: {cash:.2f},Holdings: {holdings}, "
                         f"Current Price: {current_price:.2f}, Next_price:{next_price:.2f},Current Value:{current_value:.2f},Previous Value:{previous_value:.2f},Reward:{reward},Total_reward:{total_reward:.4f}")
            previous_value = current_value
        # 记录每个 epoch 的总奖励
        total_value = cash + holdings_value * next_price
        epoch_reward = total_value - initial_balance
        epoch_rewards.append(epoch_reward)  # 记录每个 epoch 的总奖励
        total_value = float(total_value)
        epoch_reward = float(epoch_reward)
        logging.info(f"Epoch: {episode + 1}, Total Value: {total_value:.2f}, Total Reward: {epoch_reward:.2f}")

    # 在训练结束后绘制总奖励图
    plt.figure(figsize=(14, 7))
    plt.plot(epoch_rewards, label="Total Reward per Epoch")
    plt.title("Total Reward Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.close()  # 关闭图像以释放资源

    # 保存模型
    agent.save_model(model_save_path)
    return agent, test_data, features, epoch_rewards

def backtest(agent, test_data, features, initial_balance=100000.0, transaction_cost=0.001, output_dir="./", model_path="dql_agent_model.h5"):
    """
    使用训练好的智能体进行回测，评估策略表现，并输出专业回测指标。
    """
    if model_path is not None:
        agent.load_agent_model(model_path)  # 加载训练好的模型
    
    cash, holdings = initial_balance, 0
    portfolio_values = []  # 保存每个时间步的资产价值
    daily_returns = []  # 保存每日收益率
    buy_signals, sell_signals = [], []
    strong_buy_signals, strong_sell_signals = [], []  # 新增强买入和强卖出信号列表
    recent_signals = []  # 新增：保存最近5天的信号
    signals = []

    for t in range(len(test_data)):
        state = test_data[features].iloc[t].values.reshape(1, -1)
        action,q_values = agent.act(state)  # Ensure action is a scalar
        current_price = test_data['Close'].iloc[t].iloc[0]
        # 动作逻辑
        if action == 0 and q_values[0][0] >= 1.2 * q_values[0][1] and cash >= current_price * (1 + transaction_cost):  # 买入
            max_shares = int(cash // (current_price * (1 + transaction_cost)))
            cash -= max_shares * current_price * (1 + transaction_cost)
            holdings += max_shares
            #buy_signals.append(test_data.index[t])

        elif action == 1 and q_values[0][1] >= 1.2 * q_values[0][0] and holdings > 0:  # 卖出
            cash += holdings * current_price * (1 - transaction_cost)
            holdings = 0
            #sell_signals.append(test_data.index[t])

        if action == 0:
            buy_signals.append(test_data.index[t])
            if q_values[0][0] >= 2 * q_values[0][1]:
                strong_buy_signals.append(test_data.index[t])
                signal_entry = f"{ticker} - Strong Buy Signal on {test_data.index[t]}: Action = {action}, Price = {current_price:.2f}"
            else:
                signal_entry = f"{ticker} - Buy Signal on {test_data.index[t]}: Action = {action}, Price = {current_price:.2f}"
            signals.append(signal_entry)
            logging.info(signal_entry)

        elif action == 1:
            sell_signals.append(test_data.index[t])
            if q_values[0][1] >= 2 * q_values[0][0]:
                strong_sell_signals.append(test_data.index[t])
                signal_entry = f"{ticker} - Strong Sell Signal on {test_data.index[t]}: Action = {action}, Price = {current_price:.2f}"
            else:
                signal_entry = f"{ticker} - Sell Signal on {test_data.index[t]}: Action = {action}, Price = {current_price:.2f}"
            signals.append(signal_entry)
            logging.info(signal_entry)


        # 计算当前组合的资产价值
        portfolio_value = cash + holdings * current_price
        portfolio_values.append(portfolio_value)

        # 计算日收益率（避免除零）
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)

        # 日志输出
        portfolio_value = float(portfolio_value)
        logging.info(f"Time: {test_data.index[t]}, Action: {action}, Cash: {cash:.2f}, Holdings: {holdings}, "
                     f"Portfolio Value: {portfolio_value:.2f},Q-values: {q_values}")

    # 计算回测指标
    portfolio_values = np.array(portfolio_values)
    cumulative_return = (portfolio_values[-1] - initial_balance) / initial_balance
    annualized_return = (1 + cumulative_return) ** (252 / len(test_data)) - 1
    max_drawdown = np.max((np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values))
    volatility = np.std(daily_returns) * np.sqrt(252)  # 年化波动率
    sharpe_ratio = (np.mean(daily_returns) * 252) / (volatility + 1e-10)  # 避免除零
    win_rate = np.sum(np.array(daily_returns) > 0) / len(daily_returns)

    # 输出回测指标到日志
    logging.info("\n=== Backtest Summary ===")
    logging.info(f"Initial Balance: {initial_balance:.2f}")
    logging.info(f"Final Portfolio Value: {portfolio_values[-1]:.2f}")
    logging.info(f"Cumulative Return: {cumulative_return * 100:.2f}%")
    logging.info(f"Annualized Return: {annualized_return * 100:.2f}%")
    logging.info(f"Max Drawdown: {max_drawdown * 100:.2f}%")
    logging.info(f"Annualized Volatility: {volatility * 100:.2f}%")
    logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logging.info(f"Win Rate: {win_rate * 100:.2f}%")

     # 获取最新日期和5天前的日期
    latest_date = test_data.index[-1]
    five_days_ago = latest_date - pd.Timedelta(days=5)

    # 遍历 signals 列表
    for signal in signals:
        # 使用正则表达式提取日期
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', signal)  # 匹配格式为 YYYY-MM-DD 的日期
        if date_match:
            signal_date_str = date_match.group(0)  # 获取匹配的日期字符串
            signal_date = pd.to_datetime(signal_date_str)  # 将字符串转换为日期时间对象

            # 检查信号日期是否在最近5天内
            if five_days_ago <= signal_date <= latest_date:
                recent_signals.append(signal)  # 如果在范围内，添加到 recent_signals 列表中
     # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存信号到指定文件夹
    signal_file_path = os.path.join(output_dir, "signal.txt")
    with open(signal_file_path, "w") as f:
        for signal in recent_signals:
            f.write(signal + "\n")

    logging.info(f"Signals written to {signal_file_path}")
    
    return portfolio_values, buy_signals, sell_signals, strong_buy_signals, strong_sell_signals

if __name__ == "__main__":
    # 获取 ticker 参数，默认为 "BABA"
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # 计算当前日期并转换为字符串
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BABA"  # 处理命令行参数

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间戳
    output_dir = f"./data/DQN-RL-Daily-new-features-{current_date}/{ticker}-DQN-RL-Daily-{current_date}"  # 创建文件夹名
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)  # 创建文件夹

    # Configure logging to save to a file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=os.path.join(output_dir, 'trading_log.txt'),  # Update the log file path
        filemode='w'  # 'w' to overwrite the file each time, 'a' to append
    )

    start_date = "2020-01-01"
    print(ticker)
    data = download_daily_data(ticker, start_date)
    data = calculate_alpha_factors(data)
    data, features = prepare_data(data)
    

    # 继续执行后续代码
    agent, test_data, features, epoch_rewards = train_agent(data, features, model_save_path=os.path.join(output_dir, "dql_agent_mlp_model_daily.h5"))
    
    state_size = len(features)
    action_size = 3  # 动作空间：0=买入, 1=卖出, 2=持有
    agent = DQLAgent(state_size, action_size)  # Reinitialize the agent

    # Check if agent and test_data are valid before backtesting
    if agent is not None and test_data is not None:
        portfolio_values, buy_signals, sell_signals, strong_buy_signals, strong_sell_signals = backtest(agent, test_data, features, 100000, 0.001, output_dir, model_path=os.path.join(output_dir, "dql_agent_mlp_model_daily.h5"))
    else:
        logging.error("Agent training failed or test data is empty. Backtesting cannot proceed.")
        exit()  # Prevents the code from trying to plot None values

    plt.figure(figsize=(14, 7))
    if test_data is not None:
        plt.plot(test_data['Close'], label='Close Price')
        buy_signal_indices = [test_data.index.get_loc(ts) for ts in buy_signals]
        plt.scatter(test_data.index[buy_signal_indices], test_data['Close'].iloc[buy_signal_indices], marker='^', color='g', label='Buy Signal', alpha=1)
        sell_signal_indices = [test_data.index.get_loc(ts) for ts in sell_signals]
        plt.scatter(test_data.index[sell_signal_indices], test_data['Close'].iloc[sell_signal_indices], marker='v', color='r', label='Sell Signal', alpha=1)
        strong_buy_signal_indices = [test_data.index.get_loc(ts) for ts in strong_buy_signals]
        plt.scatter(test_data.index[strong_buy_signal_indices], test_data['Close'].iloc[strong_buy_signal_indices], marker='^', color='yellow', label='Strong Buy Signal', alpha=1)
        strong_sell_signal_indices = [test_data.index.get_loc(ts) for ts in strong_sell_signals]
        plt.scatter(test_data.index[strong_sell_signal_indices], test_data['Close'].iloc[strong_sell_signal_indices], marker='v', color='orange', label='Strong Sell Signal', alpha=1)
        plt.title(f"Trading Strategy - Buy & Sell Signals for {ticker}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "trading_strategy_signals.png"))  # Save image
        plt.close()  # Close the figure to free resources
    else:
        logging.error("No valid test data available for plotting.")

    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.title(f"Portfolio Value Over Time for {ticker}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "portfolio_value_over_time.png"))  # Save image
    plt.close()  # Close the figure to free resources