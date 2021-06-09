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

from data import TestDataset, make_tag2label
from model import PreTrainedEfficientNet


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
    batch_size = 32

    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = albu.Compose([albu.Resize(height=512, width=384),
        albu.Normalize(),])
    dataset = TestDataset(image_paths, transform)
    loader = DataLoader(dataset,shuffle=False,batch_size=batch_size ,num_workers=4)



    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    epoch_num = args.n
    num_classes = 18
    pt_root = 'save_model'
    file_name = '996_'
    pt_file_name_list = ['v: v4.2 mask-k: 0-e: 9','v: v4.2 gender_total-k: 0-e: 9','v: v4.0 age_total_59__-k: 0-e: 6']
    ext = '.pt'
    
    device = torch.device('cuda')
    

    tag2label = make_tag2label(3,2,3)
    predictions_mask = []
    predictions_gender = []
    predictions_age = []
    num_list = [3,1,3]
    idx = 0
    for i, pt_file_name in enumerate(pt_file_name_list):
        # inference

        model = PreTrainedEfficientNet(num_list[idx]).to(device)
        model.load_state_dict(torch.load(os.path.join(pt_root,pt_file_name + ext), map_location='cuda:0'))
        idx += 1
        model.eval()
        for images in tqdm(loader):
            with torch.no_grad():
                images = images.to(device)
                pred = model.forward(images)
                if i == 0:
                    pred = pred.argmax(dim=-1)
                    predictions_mask.extend(pred.cpu().numpy())
                elif i == 1:
                    y_ = pred >= 0.5
                    predictions_gender.extend(y_.cpu().numpy())
                else:
                    pred = pred.argmax(dim=-1)
                    predictions_age.extend(pred.cpu().numpy())
                    
    all_predictions = []
    for i in range(len(predictions_mask)):
        tag = str(int(predictions_mask[i])) + str(int(predictions_gender[i])) + str(int(predictions_age[i]))
        all_predictions.append(tag2label[tag])
    submission['ans'] = all_predictions


    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    # submission = infer_test(loader, model, device, submission)


    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join('submission', file_name + '.csv'), index=False)
    print('test inference is done!')