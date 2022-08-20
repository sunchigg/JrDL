# preprocess
python preprocess.py -c ./data/context.json
python preprocess.py -q data/valid.json -o valid -c ./data/context.json
python preprocess.py -q data/test.json -o test -c ./data/context.json

python preprocess.py -t qa -o trainqa -c ./data/context.json
python preprocess.py -q data/valid.json -t qa -o validqa -c ./data/context.json

# train
python3.8 train_slctn.py --saved_dir ./SLC

# predict (inference Context selection model)
python3.8 test_slctn.py --raw_test_file data/test.json --out_file ./slctn_results.json --target_dir ./SLC

# train
python3.8 train_qa.py --saved_dir ./QA

# after test_slctn.py
#python preprocess.py -q slctn_results.json -t qa -o testqa -c ./data/context.json

# predict (inference Span selection model)
python3.8 test_qa.py --test_file task_data/testqa_0.json --out_file ./prediction.csv --target_dir ./QA