rm -rf /Users/huiyu/Desktop/qlib_data/source.bak
mv /Users/huiyu/Desktop/qlib_data/source /Users/huiyu/Desktop/qlib_data/source.bak
mkdir /Users/huiyu/Desktop/qlib_data/source
python qlib_data_generation_new.py

# dump 1d cn
rm -rf /Users/huiyu/Desktop/qlib_data/us_data.bak
mv /Users/huiyu/Desktop/qlib_data/us_data/ /Users/huiyu/Desktop/qlib_data/us_data.bak
mkdir /Users/huiyu/Desktop/qlib_data/us_data/
python /Users/huiyu/qlib/scripts/dump_bin.py dump_all --csv_path /Users/huiyu/Desktop/qlib_data/source/ --qlib_dir /Users/huiyu/Desktop/qlib_data/us_data/ --freq day --exclude_fields date,symbol
