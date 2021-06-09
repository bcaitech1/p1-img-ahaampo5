import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import argparse
import albumentations as albu

from data import TestDataset
from model import PreTrainedEfficientNet, ResizingNetwork


def infer_test(loader, model, device, submission):
    all_predictions = []
    model.eval()
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model.forward(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions
    return submission


if __name__ == '__main__':
    # meta 데이터와 이미지 경로를 불러옵니다.
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-n',type=int, default=1)

    args = arg_parser.parse_args()

    test_dir = '../data/train/eval'
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')


    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    batch_size = 16

    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = albu.Compose([albu.Resize(height=512, width=384),
        albu.Normalize(),])
    dataset = TestDataset(image_paths, transform)
    loader = DataLoader(dataset,shuffle=False,batch_size=batch_size ,num_workers=4)



    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    epoch_num = args.n
    num_classes = 18
    pt_root = 'save_model'
    pt_file_name = 'v: v3.7 f1,ce-k: 3-e: 9'
    ext = '.pt'
    
    device = torch.device('cuda')
    model = PreTrainedEfficientNet(num_classes).to(device)
    
    # inference
    model.load_state_dict(torch.load(os.path.join(pt_root,pt_file_name + ext), map_location='cuda:0'))


    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    submission = infer_test(loader, model, device, submission)


    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join('submission', pt_file_name + '.csv'), index=False)
    print('test inference is done!')