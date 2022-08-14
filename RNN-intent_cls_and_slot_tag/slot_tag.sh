# python3.8 test_slot.py --test_file "data/slot/test.json" --ckpt_path "ckpt/slot/model.ckpt" --pred_file "pred.slot.csv"
python3.8 test_slot.py --test_file "${1}" --ckpt_path ckpt/slot/model.ckpt --pred_file "${2}"

# TA will predict testing data as follow:
# bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
# bash slot_tag.sh data/slot/test.json pred.slot.csv