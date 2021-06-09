# module import
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score


import albumentations as albu
import numpy as np
import pandas as pd

import os
from datetime import datetime
from tqdm import tqdm
from time import time
import random
import argparse
import configparser
import logging

from data import *
from model import PreTrainedEfficientNet, ResizingNetwork
from loss import FocalLoss, F1Loss


# Logging
logging.basicConfig(filename='train.log',level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

logger.addHandler(ch)


# Arg_parser
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--batch', type=int, default=32)

args = arg_parser.parse_args()


# Random Seed
def seed_everything(seed):
    """
    동일한 조건으로 학습을 할 때, 동일한 결과를 얻기 위해 seed를 고정시킵니다.
    
    Args:
        seed: seed 정수값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed_everything(0)


# Evaluation
def eval_func(model, data_iter, device):
    model.eval()
    with torch.no_grad():
        count, total_count = 0,0
        for batch_in, batch_label in tqdm(data_iter):
            x = batch_in.to(device)
            y = batch_label.to(device)
            y_pred = model.forward(x)
            y_ = torch.argmax(y_pred, dim=-1)
            count += (y==y_).sum().item()
            total_count += batch_in.size(0)
    model.train()
    return count/total_count

def eval_func_bce(model, data_iter, device):
    model.eval()
    with torch.no_grad():
        count, total_count = 0,0
        for batch_in, batch_label in tqdm(data_iter):
            x = batch_in.to(device)
            y = batch_label.unsqueeze(1).to(device)
            y_pred = model.forward(x)
            y_ = y_pred>0.5
            # y_ = torch.argmax(y_pred, dim=-1)
            k = y == y_
            count += (y==y_).sum().item()

            total_count += batch_in.size(0)
    model.train()
    return count/total_count


# PreProcessing
image_root = '../data/train/images'
mask_label_num, gender_label_num, age_label_num = 3,2,3
csv_path = '../data/train/train.csv'
k_list = range(5) # 0,1,2,3,4 중에 어디부분을 자를까?

for k in k_list:
    dir_list = glob.glob(image_root + '/*')
    jpg_list = glob.glob(image_root + '/**/*')
    df = pd.read_csv(csv_path)

    train_dir_list, validation_dir_list = get_train_validation_dir_list(df, k, image_root) # 2163, 537
    train_jpg_list, validation_jpg_list = get_train_validation_jpg_list(train_dir_list, validation_dir_list) # 14220, 3759
    tag2label = make_tag2label(mask_label_num, gender_label_num, age_label_num)
    tag2label_mask = make_tag2label_mask(mask_label_num)
    tag2label_gender = make_tag2label_gender(gender_label_num)
    tag2label_age = make_tag2label_age(age_label_num)

    train_dataset = make_dataset(train_jpg_list, tag2label)
    train_dataset_mask = make_dataset_mask(train_jpg_list, tag2label_mask)
    train_dataset_gender = make_dataset_gender(train_jpg_list, tag2label_gender)
    train_dataset_age = make_dataset_age_58(train_jpg_list, tag2label_age)

    validation_dataset = make_dataset(validation_jpg_list, tag2label)
    validation_dataset_mask = make_dataset_mask(validation_jpg_list, tag2label_mask)
    validation_dataset_gender = make_dataset_gender(validation_jpg_list, tag2label_gender)
    validation_dataset_age = make_dataset_age(validation_jpg_list, tag2label_age)

    dataset = make_dataset_age_58(jpg_list, tag2label_age) # img_path, label


    # 데이터 불러오기
    batch_size = args.batch

    transform = albu.Compose([
        albu.Resize(height=256*2,width=192*2),
        albu.OneOf([albu.Rotate(5),albu.Rotate(10)]),
        albu.HorizontalFlip(p=0.5),
        albu.Normalize(),
        ])

    train_data = MyDataset(dataset, transform)
    train_iter = DataLoader(train_data, sampler=ImbalancedDatasetSampler(train_data),batch_size=batch_size, num_workers=4)
    validation_data = MyDataset(validation_dataset_age, transform)
    validation_iter = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=4)


    # Train Initialization (Divice, Model, Loss, Optimizer)
    num_classes = 3
    lr_decay_step = 3

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PreTrainedEfficientNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),lr=1e-4)

    # training
    version = 'v4.0 age_total_59__'
    PATH = 'save_model'
    ext = '.pt'

    logger.info(f'Start Training! v:{version} k: {k}')
    EPOCH = 13
    print_num = 4
    max_acc = 0
    model.train()

    total_loss = 0
    # model.set_free()
    for epoch in range(EPOCH):
        numer = 0
        for batch_in, batch_label in tqdm(train_iter):
            x = batch_in.to(device)
            y = batch_label.to(device)
            y_pred = model.forward(x)
            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step() 
            total_loss += loss.item()
            numer += 1

        total_loss /= numer
        if epoch == 2:
            # optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
            model.set_free()
        elif epoch == 4:
            optimizer = optim.SGD(model.parameters(),lr=0.005, momentum=0.9)
        val_acc = eval_func(model,validation_iter,device)
        
        # logger.info(f'EPOCH : {epoch+1} Loss : {total_loss/numer} VAL_ACC : {val_acc}')
        logger.info(f'EPOCH : {epoch+1} Loss : {total_loss}')
        if max_acc - 0.01 < val_acc:
            max_acc = val_acc
        torch.save(model.state_dict(), os.path.join(PATH, (f'v: {version}' + f'-k: {k}' + f'-e: {epoch+1}' + ext)))
        
        # if epoch%print_num == 0 and epoch != 0:
        #     train_acc = eval_func(model,train_iter,device)
        #     logger.info(f'Train acc : {train_acc}')
        

        

