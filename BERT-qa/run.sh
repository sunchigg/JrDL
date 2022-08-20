# bash run.sh ./data/context.json ./data/test.json ./prediction.csv

# preprocess
#python preprocess.py -c ./data/context.json
#python preprocess.py -c "${1}"
#python preprocess.py -q data/valid.json -o valid -c ./data/context.json
#python preprocess.py -q data/valid.json -o valid -c "${1}"
#python preprocess.py -q data/test.json -o test -c ./data/context.json
python preprocess.py -q "${2}" -o test -c "${1}"

#python preprocess.py -t qa -o trainqa -c ./data/context.json
#python preprocess.py -t qa -o trainqa -c "${1}"
#python preprocess.py -q data/valid.json -t qa -o validqa -c ./data/context.json
#python preprocess.py -q data/valid.json -t qa -o validqa -c "${1}"

# predict (inference Context selection model)
#python3.8 test_slctn.py --raw_test_file data/test.json --out_file ./slctn_results.json --target_dir ./SLC
python3.8 test_slctn.py --raw_test_file "${2}" --out_file ./slctn_results.json --target_dir ./SLC

# after test_slctn.py
#python preprocess.py -q slctn_results.json -t qa -o testqa -c ./data/context.json
python preprocess.py -q slctn_results.json -t qa -o testqa -c "${1}"

# predict (inference Span selection model)
#python3.8 test_qa.py --test_file task_data/testqa_0.json --out_file ./prediction.csv --target_dir ./QA
python3.8 test_qa.py --test_file task_data/testqa_0.json --out_file "${3}" --target_dir ./QA

# TA will predict testing data as follow:
# bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv