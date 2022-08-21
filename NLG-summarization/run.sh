# predict
#python3.8 test.py --target_dir ./saved/0510-0106_rlFrom0509-0059_512 --test_file data/public.jsonl --out_file ./submission.jsonl
python3.8 test.py --target_dir ./0510-0106_rlFrom0509-0059_512 --test_file "${1}" --out_file "${2}"


# eval
#python3.8 eval.py -r data/public.jsonl -s submission.jsonl
#python3.8 eval.py -r "${1}" -s "${2}"

# TA will predict testing data as follow:
# bash ./run.sh data/public.jsonl submission.jsonl