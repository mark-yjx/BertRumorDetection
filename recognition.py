# -*- coding: utf-8 -*-

import os

import torch
from transformers import BertTokenizer

from Bert_main.bert_train_lstm import BertLSTMClassifier

bert_name = 'C:/Users/M4RKY/PycharmProjects/RumorDetection/Bert_main/chinese-lert-small'
tokenizer = BertTokenizer.from_pretrained(bert_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_path = 'C:/Users/M4RKY/PycharmProjects/RumorDetection/Bert_main/bert_lstm_checkpoint'
model = BertLSTMClassifier()
model.load_state_dict(torch.load(os.path.join(save_path, 'best.pt')))
model = model.to(device)
model.eval()

real_labels = ['谣言', '非谣言', '不确定']


def rec(text):
    bert_input = tokenizer(text, padding='max_length',
                           max_length=35,
                           truncation=True,
                           return_tensors="pt")
    input_ids = bert_input['input_ids'].to(device)
    masks = bert_input['attention_mask'].unsqueeze(1).to(device)
    output = model(input_ids, masks)
    pred = output.argmax(dim=1)
    return real_labels[pred]
