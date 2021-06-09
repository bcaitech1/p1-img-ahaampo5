import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from collections import defaultdict
import glob
from time import time
import cv2
import numpy as np
import os
from PIL import Image


# Dataset 클래스 생성
class MyDataset(Dataset):
    def __init__(self, root_list, transform):
        super(MyDataset, self).__init__()
        self.root_list = root_list
        self.image = []
        self.label = []
        self.transform = transform
        for img, lab in self.root_list:
            self.image.append(img)
            self.label.append(lab)
        

    def __getitem__(self, idx):
        img_path = self.image[idx]
        label = self.label[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        image = transforms.ToTensor()(image)
        return image, label
    

    def __len__(self):
        return len(self.label)


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform


    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        image = transforms.ToTensor()(image)
        return image


    def __len__(self):
        return len(self.img_paths)
        

def split_img_label(root_list):
    img_list = []
    label_list = []
    for i in root_list:
        img_list.append(i[0])
        label_list.append(i[1])
    return img_list, label_list

# 
def make_tag2label(mask_label_num, gender_label_num, age_label_num):
    mask_list = list(map(str,range(mask_label_num)))
    gender_list = list(map(str,range(gender_label_num)))
    age_list = list(map(str, range(age_label_num)))
    tag2label = defaultdict()
    label_num = 0
    for i in mask_list:
        for j in gender_list:
            for k in age_list:
                tag2label[i+j+k] = label_num
                label_num += 1
    return tag2label       


def make_tag2label_mask(mask_label_num):
    tag2label = defaultdict()
    label_num = 0
    mask_list = list(map(str,range(mask_label_num)))
    for i in mask_list:
        tag2label[i] = label_num
        label_num += 1
    return tag2label

def make_tag2label_gender(gender_label_num):
    tag2label = defaultdict()
    label_num = 0
    gender_list = list(map(str,range(gender_label_num)))
    for i in gender_list:
        tag2label[i] = label_num
        label_num += 1
    return tag2label


def make_tag2label_age(age_label_num):
    tag2label = defaultdict()
    label_num = 0
    age_list = list(map(str,range(age_label_num)))
    for i in age_list:
        tag2label[i] = label_num
        label_num += 1
    return tag2label


def make_dataset(jpg_list, tag2label):
    dataset = []
    for filepath in jpg_list:
        tag = ''
        gender = filepath.split('/')[-2].split('_')[-3]
        age = int(filepath.split('/')[-2].split('_')[-1])
        mask = filepath.split('/')[-1]
        if 'normal' in mask: # mask
            tag += '2'
        elif 'incorrect' in mask:
            tag += '1'
        else:
            tag += '0'
        if gender == 'male': # gender
            tag += '0'
        else:
            tag += '1'
        if age < 30:         # age
            tag += '0'
        elif age >= 60:
            tag += '2'
        else:
            tag += '1'
        label = tag2label[tag]
        dataset.append([filepath, label])
    return dataset

def make_dataset_mask(jpg_list, tag2label):
    dataset = []
    for filepath in jpg_list:
        tag = ''
        mask = filepath.split('/')[-1]
        if 'normal' in mask: # mask
            tag += '2'
        elif 'incorrect' in mask:
            tag += '1'
        else:
            tag += '0'
        label = tag2label[tag]
        dataset.append([filepath, label])
    return dataset

def make_dataset_gender(jpg_list, tag2label):
    dataset = []
    for filepath in jpg_list:
        tag = ''
        gender = filepath.split('/')[-2].split('_')[-3]

        if gender == 'male': # gender
            tag += '0'
        else:
            tag += '1'

        label = tag2label[tag]
        dataset.append([filepath, label])
    return dataset

def make_dataset_age(jpg_list, tag2label):
    dataset = []
    for filepath in jpg_list:
        tag = ''
        age = int(filepath.split('/')[-2].split('_')[-1])
        if age < 30:         # age
            tag += '0'
        elif age >= 60:
            tag += '2'
        else:
            tag += '1'
        label = tag2label[tag]
        dataset.append([filepath, label])
    return dataset


def make_dataset_age_58(jpg_list, tag2label):
    dataset = []
    for filepath in jpg_list:
        tag = ''
        age = int(filepath.split('/')[-2].split('_')[-1])
        if age < 30:         # age
            tag += '0'
        elif age >= 58:
            tag += '2'
        else:
            tag += '1'
        label = tag2label[tag]
        dataset.append([filepath, label])
    return dataset

def get_train_validation_dir_list(df, n, image_dir_path):
        
    group = []
    group.append(df[np.logical_and(df.age<30,df.gender=='male')])
    group.append(df[np.logical_and(df.age>=30,df.age<60)][df.gender=='male'])
    group.append(df[np.logical_and(df.age>=60,df.gender=='male')])
    group.append(df[np.logical_and(df.age<30,df.gender=='female')])
    group.append(df[np.logical_and(df.age>=30,df.age<60)][df.gender=='female'])
    group.append(df[np.logical_and(df.age>=60,df.gender=='female')])

    validation_dir_list, train_dir_list = [], []

    for name in group:
        cut_left, cut_right = len(name)*n//5, len(name)*(n+1)//5
        validation_dir_list += [os.path.join(image_dir_path, path) for path in name[cut_left:cut_right].path]
        train_dir_list += [os.path.join(image_dir_path, path) for path in name[:cut_left].path]
        train_dir_list += [os.path.join(image_dir_path, path) for path in name[cut_right:].path]
    return train_dir_list, validation_dir_list


def get_train_validation_jpg_list(train_dir_list, validation_dir_list):
    train_jpg_list, validation_jpg_list = [], []
    
    for i in train_dir_list:
        train_jpg_list += glob.glob(os.path.join(i, '*'))
    for j in validation_dir_list:
        validation_jpg_list += glob.glob(os.path.join(j, '*'))
    return train_jpg_list, validation_jpg_list


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif isinstance(dataset, MyDataset):
            return dataset.label[idx]

        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples