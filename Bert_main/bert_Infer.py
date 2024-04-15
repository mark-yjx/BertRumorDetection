# -*- coding: utf-8 -*-

import os

import torch
from transformers import BertTokenizer

from Bert_main.bert_train_lstm import BertLSTMClassifier

bert_name = 'C:/Users/M4RKY/PycharmProjects/RumorDetection/Bert_main/chinese-lert-small'
tokenizer = BertTokenizer.from_pretrained(bert_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_path = 'C:/Users/M4RKY/PycharmProjects/RumorDetection/Bert_main/bert_lstm_checkpoint'


def infer():
    model = BertLSTMClassifier()
    model.load_state_dict(torch.load(os.path.join(save_path, 'best.pt'), map_location=device))
    model = model.to(device)
    model.eval()

    real_labels = ['非谣言', '谣言', '不确定']  # 更新标签列表，移除“不确定”

    while True:
        text = input('请输入文本：')
        bert_input = tokenizer(text, padding='max_length',
                               max_length=35,
                               truncation=True,
                               return_tensors="pt")
        input_ids = bert_input['input_ids'].to(device)
        masks = bert_input['attention_mask'].unsqueeze(1).to(device)

        with torch.no_grad():  # 推理时不计算梯度
            output = model(input_ids, masks)
            probabilities = torch.softmax(output, dim=1)  # 应用softmax获取概率分布
            pred = output.argmax(dim=1)
            print(f"预测类别：{real_labels[pred.item()]}")
            rumor_confidence = probabilities[0][1].item()  # 获取谣言类别的置信度
            print(f"谣言置信度：{rumor_confidence * 100:.4f}%")


if __name__ == '__main__':
    infer()
