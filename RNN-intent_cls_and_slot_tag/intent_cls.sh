# python3.8 test_intent.py --test_file "data/intent/test.json" --ckpt_path "ckpt/intent/model.ckpt" --pred_file "pred.intent.csv"
python3.8 test_intent.py --test_file "${1}" --ckpt_path "ckpt/intent/model.ckpt" --pred_file "${2}"

# TA will predict testing data as follow:
# bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
# bash intent_cls.sh data/intent/test.json pred.intent.csv