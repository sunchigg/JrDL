# download "zh_core_web_md"
python3 -m spacy download zh_core_web_md
# download my Context selection model
mkdir -p model
wget --output-document="model/SLC.zip" https://www.dropbox.com/s/pjtikqgbuw68q7j/SLC.zip
unzip model/SLC.zip
# download my Span selection model
wget --output-document="model/QA.zip" https://www.dropbox.com/s/ruw1ok64inhfrv2/QA.zip
unzip model/QA.zip