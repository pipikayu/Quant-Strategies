import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_and_save_stock_data(stock_symbols, start_date, end_date):
    # 下载多支股票的数据
    stock_data = yf.download(stock_symbols, start=start_date, end=end_date)

    # 遍历每个股票符号，分开存储
    for symbol in stock_symbols:
        # 选择当前股票的数据
        symbol_data = stock_data.xs(symbol, level=1, axis=1)  # 使用 xs() 方法选择特定股票的数据
        
        # 添加 ticker 列
        symbol_data['symbol'] = symbol
        print(symbol_data.columns)  # 打印当前列名
        # 重命名列
        symbol_data.reset_index(inplace=True)
        symbol_data['dividends'] = 0.0
        symbol_data.columns = ['date','open', 'high', 'low', 'close', 'adjclose', 'volume', 'symbol','dividends']
        symbol_data = symbol_data[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adjclose', 'dividends']]
        # 保存为 CSV 文件
        individual_output_file = f"/Users/huiyu/Desktop/qlib_data/source/{symbol}.csv"  # 创建每个股票的文件名
        symbol_data.to_csv(individual_output_file, index=False)  # 指定日期格式
        
        print(f"Data for {symbol} saved to {individual_output_file}")


#symbols = ["IQ","ASHR", "LI", "KWEB", "MPNGY", "BIDU", "TSLA", "NVDA", "BABA", "TCEHY", "PDD", "JD", "NIO", "TQQQ", "SOXL", "OXY", "TAL", "FUTU","BILI"]
symbols = ["UNG","TLT","DIDIY","BEKE","GOLD","RIOT","MARA","FXI","FUTU","COIN","CPNG","UVXY","XIACY","IQ","ASHR","XBI","LABU","FNGU","PLTR","TCEHY","MPNGY","OXY","SOXL","SOXS","KWEB","CWEB","YINN","YANG","TQQQ","SQQQ","TNA","AAPL","ALXN","AMGN","AMZN","BABA","BIDU","BIIB","BILI","BRK.B","DIS","GILD","GOOGL","HD","ILMN","INCY","INTC","JD","JNJ","JPM","LI","META","MSFT","NBIX","SBUX","NFLX","NIO","NTES","NVDA","PDD","PG","PYPL","REGN","SGEN","PEP","KO","COST","T","CMCSA","MCD","CVX","XOM","CSCO","ORCL","ABT","ABBV","DHR","ACN","ADBE","CRM","TXN","AMD","QCOM","HON","LIN","PM","NEE","LOW","MDT","BKNG","AMT","UPS","NKE","C","SCHW","TME","TSLA","UNH","V","VRTX","WMT","XPEV"]
start_date = '2015-01-01'
end_date = datetime.now().strftime("%Y-%m-%d")  # Updated to use today's date

fetch_and_save_stock_data(symbols, start_date, end_date)
