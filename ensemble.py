import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import argparse
import albumentations as albu

from model import PreTrainedEfficientNet, ResizingNetwork
from data import TestDataset


if __name__ == '__main__':
    # meta 데이터와 이미지 경로를 불러옵니다.
    

    submission_list = []

    test_root = '../data/train/eval'
    submission = pd.read_csv(os.path.join(test_root, 'info.csv'))
    image_root = os.path.join(test_root, 'images')


    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    batch_size = 16

    image_paths = [os.path.join(image_root, img_id) for img_id in submission.ImageID]
    transform = albu.Compose([albu.Resize(height=512, width=384),
        albu.Normalize(),])
    dataset = TestDataset(image_paths, transform)
    loader = DataLoader(dataset,shuffle=False,batch_size=batch_size ,num_workers=4)


    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    model_list = [
        'v: v3.4 f1-k: 0-e: 7**',
        'v: v3.4 f1-k: 1-e: 6**'
        ]
    num_classes = 18
    pt_root = 'save_model'
    ext = '.pt'

    total_answer = None
    device = torch.device('cuda')
    model = PreTrainedEfficientNet(num_classes).to(device)
    for pt_file_name in model_list:
    
        # inference
        model.load_state_dict(torch.load(os.path.join(pt_root, pt_file_name + ext), map_location='cuda:0'))

        
        # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
        all_predictions = []
        pred_group = None
        model.eval()
        for images in tqdm(loader):
            with torch.no_grad():
                images = images.to(device)
                pred = model.forward(images) # batch x 18
                # pred = F.softmax(pred, dim=-1)
                if pred_group == None:
                    pred_group = pred
                else:
                    pred_group = torch.cat((pred_group, pred))
        if total_answer == None:
            total_answer = pred_group
        else:
            total_answer += pred_group
    total_answer = total_answer.argmax(dim=-1)
    all_predictions.extend(total_answer.cpu().numpy())
    submission['ans'] = all_predictions


    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join('submission', pt_file_name + '.csv'), index=False)
    print('test inference is done!')