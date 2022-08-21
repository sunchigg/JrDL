pip install -e tw_rouge
pip install -r requirements.txt
# https://pytorch.org/
#conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
#conda install -c nvidia cudnn

# Assumed that the data required for training the model has been stored in the "./data" directory
# preprocess
python3.8 preprocess.py -a ./data/train.jsonl -o ./task_data
# train
python3.8 train.py
python3.8 train_rl.py --model_name ./saved/{mmdd-hhmm}  # 這沒參數化，要填入train.py的產出資料夾