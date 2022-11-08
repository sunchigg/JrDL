# ADL-BERT-qa

## Context selection model
(based on scripts from Hugging Face)
1. Integrate the information in train.json, valid.json and context.json, organize them into the data format required for subsequent training, and keep the "id", "question", "paragraphs", "relevant" fields. In addition, there is a function of dividing into two packages according to the specified ratio, but after several attempts, the training effect of merging train.json and valid.json and then splitting has no significant difference. In the future, the training data will be maintained from train.json, and the validation data will be from valid.json. [preprocess.py]
2. According to the structure required by Hugging Face datasets.load_metric, and use Accuray for this task's metrics and set the function to compute the prediction and reference.[metric_slctn.py]
3. Initialize Accelerator() and set its required information, and then import config and specified pre-trained model and tokenizer  through Hugging Face `AutoConfig.from_pretrained()`. In this report, "bert-base-chinese" is used. [train_slctn.py]
4. Create DataLoader respectively, and prepare corresponding features respectively.[train_slctn.py, utils.py]
5. Set an optimizer(AdamW),scheduler and  parameters of them. Additionally take CrossEntropyLoss() as criterion and accuracy as metrics. [train_slctn.py]
6. Start iterating over train dataloader and set accuracy and loss = 0. Additionally, train model through back propagation. [train_slctn.py]
7. In evaluation loop, calculate accuracy and save it and model weights. If validation accuracy in this epoch reach historical high, save the model. [train_slctn.py]
8. Finally, we get the best model after all epochs are done.

## Span selection model
(based on scripts from Hugging Face)
1. Integrate the information in train.json, valid.json and context.json, organize them into the data format required for subsequent training, and keep the "id", "question", "paragraphs", "answer" fields. In addition, there is a function of dividing into two packages according to the specified ratio, but after several attempts, the training effect of merging train.json and valid.json and then splitting has no significant difference. In the future, the training data will be maintained from train.json, and the validation data will be from valid.json. [preprocess.py]
2. According to the structure required by Hugging Face datasets.load_metric, and use EM and F1 for this task's metrics and set the function to compute the prediction and reference.[metric_qa.py]
3. Initialize Accelerator() and set its required information, and then import config and specified pre-trained model and tokenizer  through Hugging Face `AutoConfig.from_pretrained()`. In this report, "hfl/chinese-roberta-wwm-ext" is used. [train_qa.py]
4. Create DataLoader respectively, and prepare corresponding features respectively.[train_qa.py, utils_qa.py]
5. Set an optimizer(AdamW),scheduler and  parameters of them. Additionally take CrossEntropyLoss() as criterion, and EM and F1 as metrics. In addition, a score is calculated by adding up EM and F1, which is used as the basis for judging the version of the retention model. [train_qa.py]
6. Start iterating over train dataloader and set score and loss = 0. Additionally, train model through back propagation. [train_qa.py]
7. In evaluation loop, calculate score and save it and model weights. If validation score in this epoch reach historical high, save the model. [train_qa.py]
8. Finally, we get the best model after all epochs are done.


## Reproduce
### Environment
`conda create -n hw2 python=3.8` . 
`conda activate hw2` . 
`pip install -r requirements.txt`

### Preprocessing and Training
`bash train.sh`

### Download model
`bash download.sh`
### Predict
`bash run.sh`