#python qlib_data_generation.py

# dump 1d cn
cp -r /Users/huiyu/Desktop/qlib_data/us_data/ /Users/huiyu/Desktop/qlib_data/us_data.bak
rm -rf /Users/huiyu/Desktop/qlib_data/us_data/
mkdir /Users/huiyu/Desktop/qlib_data/us_data/
python /Users/huiyu/qlib/scripts/dump_bin.py dump_all --csv_path /Users/huiyu/Desktop/qlib_data/source/ --qlib_dir /Users/huiyu/Desktop/qlib_data/us_data/ --freq day --exclude_fields date,symbol
