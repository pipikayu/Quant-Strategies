import yfinance as yf
import pandas as pd

def fetch_and_save_stock_data(stock_symbols, start_date, end_date, output_file):
    all_data = []

    for symbol in stock_symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        stock_data.reset_index(inplace=True)
        
        stock_data['symbol'] = symbol
        stock_data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adjclose',
            'Volume': 'volume'
        }, inplace=True)
        stock_data['dividends'] = 0.0
        stock_data = stock_data[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adjclose', 'dividends']]
        
        all_data.append(stock_data)

        # Save each stock's data to a separate CSV file
        individual_output_file = f"/Users/huiyu/Desktop/qlib_data/source/{symbol}.csv"  # Create a filename for each stock
        #individual_output_file1 = f"/Users/huiyu/Desktop/qlib_data/source/{symbol}.csv"  
        stock_data.to_csv(individual_output_file, index=False)  # 指定日期格式
        # Load the raw and target data
        #data_aapl = pd.read_csv(individual_output_file, skiprows=[1])  # 解析日期
        #data_cleaned = data_aapl.copy()
        #data_cleaned.to_csv(individual_output_file1, index=False)  # 指定日期格式

        print(f"Data for {symbol} saved to {individual_output_file}")

    final_data = pd.concat(all_data, ignore_index=True)
    final_data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

# Example usage
output_file_path = 'stock_data.csv'  # Adjust path as needed
#symbols = ["IQ","ASHR", "PLTR", "YANG", "UVXY", "BIDU", "LI", "KWEB", "XBI", "LABU", "WB", "MPNGY", "YINN", "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "MRNA", "PFE", "AMGN", "GILD", "BNTX", "BABA", "TCEHY", "PDD", "JD", "AMD", "NIO", "TQQQ", "SOXL", "SOXS", "SQQQ", "OXY", "RIOT", "BILI", "MARA", "TAL", "FUTU"]  # Replace with desired stock symbols
symbols = ["IQ","ASHR", "LI", "KWEB", "MPNGY", "BIDU", "TSLA", "NVDA", "BABA", "TCEHY", "PDD", "JD", "NIO", "TQQQ", "SOXL", "OXY", "TAL", "FUTU","BILI"]
start_date = '2015-01-01'
end_date = '2024-12-31'

fetch_and_save_stock_data(symbols, start_date, end_date, output_file_path)