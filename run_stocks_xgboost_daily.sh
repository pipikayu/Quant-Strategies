tickers=("IQ" "ASHR" "PLTR" "YANG" "UVXY"  "BIDU" "LI" "KWEB" "XBI" "LABU" "WB" "MPNGY" "YINN" "AAPL" "MSFT" "GOOG" "AMZN" "META" "TSLA" "NVDA" "MRNA" "PFE" "AMGN" "GILD" "BNTX" "BABA" "TCEHY" "PDD" "JD" "AMD" "NIO" "TQQQ" "SOXL" "SOXS" "SQQQ" "OXY" "RIOT" "BILI" "MARA" "TAL" "FUTU")

# 定义邮件列表
email_list=("winneraa@msn.com" "18600841013@163.com" )
subject="Daily Signal Report - $today"

# 获取今天的日期
today=$(date +%Y-%m-%d)  # 格式化为 YYYY-MM-DD

# 循环遍历每个股票代码并执行 Python 脚本
for ticker in "${tickers[@]}"; do
    echo "Running script for ticker: $ticker"
    python alpha-factor-Xgboost-daily.py "$ticker"
    #python alpha-factor-Xgboost-daily-fs.py "$ticker"
    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing ticker: $ticker. Exiting."
        exit 1  # 如果发生错误，退出脚本
    fi
done


# 合并今天的所有 signal.txt 文件
signal_directory="data/Xgboost-Daily-$today"
if [ -d "$signal_directory" ]; then
    echo "Combining signal.txt files from $signal_directory"
    cat "$signal_directory"/*/signal.txt > "$signal_directory"/"combined_signals_$today.txt"
    echo "Combined signal file created: combined_signals_$today.txt"
else
    echo "Directory $signal_directory does not exist. No signal files to combine."
fi

# 发送邮件功能
echo "Sending signal report to email list..."

for email in "${email_list[@]}"; do
    # 发送邮件
    mail -s "$subject" "$email" < "$signal_directory"/"combined_signals_$today.txt"
    if [ $? -eq 0 ]; then
        echo "Successfully sent signal report to $email"
    else
        echo "Failed to send signal report to $email"
    fi
done
