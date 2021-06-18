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
import albumentations as A

from model import PreTrainedEfficientNet, ResizingNetwork
from data import TestDataset


if __name__ == '__main__':
    # meta 데이터와 이미지 경로를 불러옵니다.
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch_size', type=int, default=16)

    args = parser.parse_args()

    submission_list = []

    test_root = '../data/train/eval'
    submission = pd.read_csv(os.path.join(test_root, 'info.csv'))
    image_root = os.path.join(test_root, 'images')

    image_paths = [os.path.join(image_root, img_id) for img_id in submission.ImageID]
    transform = A.Compose([A.Resize(height=512, width=384),
        A.Normalize(),])
    dataset = TestDataset(image_paths, transform)
    loader = DataLoader(dataset,shuffle=False,batch_size=args.batch_size ,num_workers=4)

    pt_file_list = [
        'v: v3.4 f1-k: 0-e: 7**.pt',
        'v: v3.4 f1-k: 1-e: 6**.pt'
        ]
    num_classes = 18
    pt_root = './save_model'

    total_answer = None
    device = torch.device('cuda')
    model = PreTrainedEfficientNet(num_classes).to(device)
    for pt_file_name in pt_file_list:
    
        model.load_state_dict(torch.load(os.path.join(pt_root, pt_file_name), map_location='cuda:0'))

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

    submission.to_csv(os.path.join('submission', pt_file_name + '.csv'), index=False)
    print('Ensemble is done!')