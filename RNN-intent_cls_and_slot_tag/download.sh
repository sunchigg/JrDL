mkdir -p ckpt/intent
mkdir -p ckpt/slot
wget --output-document="ckpt/intent/model.ckpt" https://www.dropbox.com/s/z76spzzeiv1ca81/model_90044.ckpt
wget --output-document="ckpt/slot/model.ckpt" https://www.dropbox.com/s/5a5pqy55afl2k99/model_80428.ckpt
# bash download.sh