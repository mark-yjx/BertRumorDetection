# -*- coding: utf-8 -*-


import os
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertModel
from transformers import BertTokenizer
from Bert_main.bert_get_data import generate_data

bert_name = 'C:/Users/M4RKY/PycharmProjects/RumorDetection/Bert_main/chinese-lert-small'
tokenizer = BertTokenizer.from_pretrained(bert_name)


class BertLSTMClassifier(nn.Module):
    def __init__(self):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(256, 3)
        # self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        lstm_output, _ = self.lstm(pooled_output.unsqueeze(0))
        lstm_output = lstm_output.squeeze(0)
        dropout_output = self.dropout(lstm_output)
        linear_output = self.linear(dropout_output)
        # final_layer = self.relu(linear_output)
        #
        return linear_output


# class BertLSTMClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 定义BERT模型
#         self.bert = BertModel.from_pretrained(bert_name)
#         # 定义分类器
#         self.classifier = nn.Linear(256, 3)
#
#     def forward(self, input_ids, attention_mask):
#         # BERT的输出
#         bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         # 取[CLS]位置的pooled output
#         # pooled = bert_output.logits[:,-1,:]
#         pooled = bert_output.last_hidden_state[:, -1, :]
#         # 分类
#         # pooled = self.linear(pooled)
#         logits = self.classifier(pooled)
#         # 返回softmax后结果
#         return logits


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


# 训练超参数
epoch = 100
batch_size = 64
lr = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 20240121
save_path = './bert_lstm_checkpoint'
setup_seed(random_seed)

# 定义模型
model = BertLSTMClassifier()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
model = model.to(device)
criterion = criterion.to(device)


def train_model():
    # 构建数据集
    train_dataset = generate_data(mode='train')
    dev_dataset = generate_data(mode='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    writer = SummaryWriter('./logs/withLSTM')
    # 训练
    train_dataset_size = len(train_loader.dataset)
    print(f"训练集样本总数: {train_dataset_size}")
    best_dev_acc = 0
    for epoch_num in range(epoch):
        total_acc_train = 0
        total_loss_train = 0
        for inputs, labels in tqdm(train_loader):
            input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device)  # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)

            batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += batch_loss.item()
            writer.add_scalar('Loss/train', total_loss_train / len(train_loader), epoch_num)
            writer.add_scalar('Accuracy/train', total_acc_train / len(train_dataset), epoch_num)

        # ----------- 验证模型 -----------
        model.eval()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for inputs, labels in dev_loader:
                input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
                masks = inputs['attention_mask'].to(device)  # torch.Size([32, 1, 35])
                labels = labels.to(device)
                output = model(input_ids, masks)

                batch_loss = criterion(output, labels)
                acc = (output.argmax(dim=1) == labels).sum().item()
                total_acc_val += acc
                total_loss_val += batch_loss.item()
                writer.add_scalar('Loss/val', total_loss_val / len(dev_loader), epoch_num)
                writer.add_scalar('Accuracy/val', total_acc_val / len(dev_dataset), epoch_num)

            print(f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_dataset): .3f} 
              | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
              | Val Loss: {total_loss_val / len(dev_dataset): .3f} 
              | Val Accuracy: {total_acc_val / len(dev_dataset): .3f}''')
            # 更新学习率
            scheduler.step()

            # 现在获取并打印更新后的学习率
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch_num + 1}, Current Learning Rate: {current_lr:.4e}")
            writer.add_scalar('Learning Rate', current_lr, epoch_num)

            # 保存最优的模型
            if total_acc_val / len(dev_dataset) > best_dev_acc:
                best_dev_acc = total_acc_val / len(dev_dataset)
                save_model('best.pt')

        model.train()

    # 保存最后的模型
    save_model('last.pt')
    writer.close()


if __name__ == "__main__":
    train_model()
