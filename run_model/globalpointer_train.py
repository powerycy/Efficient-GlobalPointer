import re
import json
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW,get_linear_schedule_with_warmup
from data_process import yeild_data
from model import EfficientGlobalPointerNet as GlobalPointerNet
from loss_fun import global_pointer_crossentropy
from metrics import global_pointer_f1_score
import sys
import argparse
import torch.distributed as dist
from tools import reduce_tensor
import logging
from tools import setup_seed
from inference import NER
from data_process import load_data
setup_seed(1234)
# torch.cuda.manual_seed_all(seed)
# from inference import NamedEntityRecognizer
# NER = NamedEntityRecognizer()
# from torch.nn.parallel import DistributedDataParallel as DDP
#DDP
# from torch.utils.data.distributed import DistributedSampler
# 1) 初始化
# parser = argparse.ArgumentParser()
# parser.add_argument('--local_rank', default=-1, type=int,
#                     help='node rank for distributed training')
# args = parser.parse_args()
# local_rank = torch.distributed.get_rank()
# dist.init_process_group(backend='nccl')
# torch.cuda.set_device(args.local_rank)
# dist.init_process_group(backend='nccl')
# device = torch.device(f'cuda:{args.local_rank}')
from tqdm import tqdm
#DP
# gpus = [0,1,2,3]
# torch.cuda.set_device('cuda:{}'.format(gpus[0]))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print("Using {} device".format(device))
import configparser
con = configparser.ConfigParser()
file = './train_config/config.ini'
con.read(file,encoding='utf8')
items = con.items('path')
path = dict(items)
items = con.items('model_superparameter')
model_sp = dict(items)
model_path = path['model_path']
train_file_data = path['train_file_data']
val_file_data = path['val_file_data']
model_save_path = path['model_save_path']
head_size = eval(model_sp['head_size'])
hidden_size = eval(model_sp['hidden_size'])
learning_rate = eval(model_sp['learning_rate'])
clip_norm = eval(model_sp['clip_norm'])
re_maxlen = eval(model_sp['re_maxlen'])
train_dataloader,categories_size,categories2id,id2categories = yeild_data(train_file_data,is_train=True,DDP=False)
val_data = load_data(train_file_data,is_train=False)
val_dataloader = yeild_data(val_file_data,categories_size=categories_size,categories2id=categories2id,is_train=False,DDP=False)
model = GlobalPointerNet(model_path,categories_size,head_size,hidden_size).to(device)
# model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])
# model = DDP(model,device_ids=[args.local_rank],find_unused_parameters=True)
epochs = eval(model_sp['epochs'])
warmup_steps = eval(model_sp['warmup_steps'])
total_steps = len(train_dataloader) * epochs
param_optimizer = list(model.named_parameters())
# train_epoch_loss = 0
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
def train(dataloader, model, loss_func, optimizer):
    model.train()
    size = len(dataloader.dataset)
    total_n, total_d = 0, 0
    for batch, (data,label) in enumerate(dataloader):
        input_ids = data['input_ids'].squeeze().to(device)
        attention_mask = data['attention_mask'].squeeze().to(device)
        token_type_ids = data['token_type_ids'].squeeze().to(device)
        label = label.to(device)
        pred = model(input_ids,attention_mask,token_type_ids)
        loss = loss_func(label,pred)
        num,den = global_pointer_f1_score(label,pred)
        total_n += num
        total_d += den
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(input_ids)
            print(f"loss: {loss:>5f}  [{current:>5d}/{size:>5d}]")
    print(f"Train F1: {(2*total_n/total_d):>4f}%")

def evaluate(dataloader,loss_func, model):
    size = len(dataloader.dataset)
    model.eval()
    val_loss = 0
    total_n, total_d = 0, 0
    with torch.no_grad():
        for data,label in dataloader:
            input_ids = data['input_ids'].squeeze().to(device)
            attention_mask = data['attention_mask'].squeeze().to(device)
            token_type_ids = data['token_type_ids'].squeeze().to(device)
            label = label.to(device)
            pred = model(input_ids,attention_mask,token_type_ids)
            val_loss += loss_func(label,pred).item()
            num,den = global_pointer_f1_score(label,pred)
            total_n += num
            total_d += den
    val_loss /= size
    print(f"F1:{(2*total_n/total_d):>4f},Avg loss: {val_loss:>5f} \n")
    return val_f1
def evaluate_val(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0],id2categories,model))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T) 
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall
class Evaluator(object):
    """评估与保存
    """
    def __init__(self,best_val_f1):
        self.best_val_f1 = best_val_f1
    def on_epoch_end(self, epoch, logs=None):
        f1 = evaluate(val_dataloader,global_pointer_crossentropy, model)
        # f1, precision, recall = evaluate_val(val_data)
        # 保存最优
        # print(f1,precision,recall)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            torch.save(model.state_dict(), f=model_save_path)
        # if args.local_rank == 0:
        print(
            'valid:  f1: %.5f,  best f1: %.5f\n' %
            (f1,self.best_val_f1)
            # 'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            # (f1, precision, recall, self.best_val_f1)
        )
        return self.best_val_f1
def run_model(optimizer):
    best_val_f1 = 0
    for epoch in range(epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        print(f"Epoch {epoch + 1}")
        train(train_dataloader, model, global_pointer_crossentropy, optimizer)
        best_val_f1 = Evaluator(best_val_f1).on_epoch_end(epoch)
    print('end')
if __name__ == '__main__':
    run_model(optimizer)
