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
    parser = argparse.ArgumentParser()

    parser.add_argument('-n',type=int, default=1)

    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-seperate', type=bool, default=False)

    args = parser.parse_args()

    test_dir = '../data/train/eval'
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.

    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = albu.Compose([albu.Resize(height=512, width=384),
        albu.Normalize(),])
    dataset = TestDataset(image_paths, transform)
    loader = DataLoader(dataset, shuffle=False,batch_size=args.batch_size ,num_workers=4)


    # 모델을 정의합니다.
    pt_root = './save_model'
    output_file_name = '996_.csv'
    device = torch.device('cuda')

    if args.seperate == True:
        mask_model = 'v: v4.2 mask-k: 0-e: 9.pt'
        gender_model = 'v: v4.2 gender_total-k: 0-e: 9.pt'
        age_model = 'v: v4.0 age_total_59__-k: 0-e: 6.pt'
        pt_file_name_list = [mask_model, gender_model, age_model]

        tag2label = make_tag2label(3,2,3)
        predictions_mask = []
        predictions_gender = []
        predictions_age = []
        class_list = [3,1,3]

        for num, pt_file_name in enumerate(pt_file_name_list):
            # inference
            model = PreTrainedEfficientNet(class_list[num]).to(device)
            model.load_state_dict(torch.load(os.path.join(pt_root, pt_file_name), map_location='cuda:0'))
            model.eval()
            for images in tqdm(loader):
                with torch.no_grad():
                    images = images.to(device)
                    pred = model.forward(images)
                    if num == 0:
                        pred = pred.argmax(dim=-1)
                        predictions_mask.extend(pred.cpu().numpy())
                    elif num == 1:
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
    else:
        num_classes = 18
        pt_file_name = 'v: v3.7 f1,ce-k: 3-e: 9.pt'

        model = PreTrainedEfficientNet(num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(pt_root,pt_file_name), map_location='cuda:0'))
        submission = infer_test(loader, model, device, submission)

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join('submission', output_file_name), index=False)
    print('test inference is done!')