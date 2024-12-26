import qlib
from qlib.data import D
from qlib.config import REG_CN
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_daily_data(stock_code, start_date, end_date):
    """
    使用 Qlib 获取指定股票的日线数据
    """
    df = D.features(
        instruments=[stock_code],
        fields=["$close", "$open", "$high", "$low", "$volume"],
        freq="day",
        start_time=start_date,
        end_time=end_date,
    )
    return df

def main():
    # Step 1: 初始化 Qlib 并获取宁德时代的日线数据
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    
    stock_code = "sz300750"
    start_date = "2020-01-01"
    end_date = "2020-01-30"
    
    logging.info(f"Fetching daily data for {stock_code} from {start_date} to {end_date}.")
    df = get_daily_data(stock_code, start_date, end_date)

    if df.empty:
        logging.error(f"No data found for {stock_code} from {start_date} to {end_date}.")
        return

    logging.info("Data fetched successfully.")
    print(df)

    data_handler_config={
        'start_time':"2020-01-01",
        'end_time':"2020-01-30",
        'fit_start_time':"2020-01-01",
        'fit_end_time':"2020-01-30",
        'instruments':[stock_code],
    }
    h = Alpha158(**data_handler_config)
    print(h.get_cols())

    # 检查所有可用字段
    all_fields = D.list_columns(freq="day")
    print(all_fields)

if __name__ == "__main__":
    main()
