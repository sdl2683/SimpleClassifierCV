# 首先导入包
import torch, json
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns
from d2l import torch as d2l
import joblib
from sklearn.utils import shuffle
from os.path import join as opj
from os.path import exists as ope
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import cv2
# from cutmix.cutmix import CutMix
# from cutmix.utils import CutMixCrossEntropyLoss
from sklearn.model_selection import StratifiedKFold
import datetime


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

with open('class2num.json', 'r') as f:
    class2num = json.loads(f.read())

with open('num2class.json', 'r') as f:
    num2class = json.loads(f.read())

class2num = {k:v-1 for k,v in class2num.items()}


def get_today_time():
    today = datetime.date.today()
    year, month, day = str(today.year), str(today.month), str(today.day)
    if len(month) < 2:
        month = '0' + month
    if len(day) < 2:
        day = '0' + day
    
    return year + month + day

# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


def resnet50_model(num_classes, feature_extract = False, use_pretrained=True):

    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256), 
                                nn.ReLU(),
                                nn.Linear(256, 128), 
                                nn.ReLU(),
                                nn.Linear(128,num_classes))

    return model_ft

def resnet34_model(num_classes, feature_extract = False, use_pretrained=True):
    
    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256), 
                                nn.ReLU(),
                                nn.Linear(256, 128), 
                                nn.ReLU(),
                                nn.Linear(128,num_classes))

    return model_ft

def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(512, 512),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.7),
            albumentations.RandomBrightnessContrast(),
            albumentations.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            ),
            albumentations.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(512, 512),
            albumentations.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0)
        ]
    )

# 继承pytorch的dataset，创建自己的
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, class2num, num2class, train_transform, valid_transform, mode='train', valid_ratio=0.1, resize_height=512, resize_width=512):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """
        
        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode
        self.class2num = class2num
        self.num2class = num2class
        self.train_transform = train_transform()
        self.valid_transform = valid_transform()

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  #header=None是去掉表头部分
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))
        
        if mode == 'train':
            # 第一列包含图像文件的名称
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0])  #self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image 
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])  
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image
            
        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # # 读取图像文件 PIL
        # img_as_img = Image.open(self.file_path + single_image_name)
        
        # 读取图像文件 cv2
        img_as_img = cv2.imread(self.file_path + single_image_name)
        img_as_img = cv2.cvtColor(img_as_img, cv2.COLOR_BGR2RGB)
        
        #设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = self.train_transform
        else:
            # valid和test不做数据增强
            transform = self.valid_transform
        
        img_as_img = transform(image=img_as_img)['image']
        
        if self.mode == 'test':
            return single_image_name, img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]

            return img_as_img, class2num[label]  #返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len

def test(model, test_loader, device):
    assert device is not None
    
    img_paths, y_preds = [], []
    model = model.to(device)
    model.eval()
    for img_path, x in tqdm(test_loader):
        x = x.to(device)
        y_hat = model(x)
        
        y_hat = y_hat.argmax(dim=1)
        
        img_paths.extend(list(img_path))
        y_preds.extend(y_hat.detach().cpu().numpy().tolist())
    
    df = pd.DataFrame(data={'uuid':img_paths, 'label':y_preds})
    return df
        

li = os.listdir(r'G:\讯飞比赛\苹果病害图像识别挑战赛\苹果病害图像识别挑战赛公开数据\main\data\test')
test_data = pd.DataFrame(data={'img_path':li})
test_data.to_csv('test_data.csv',index=False)

batch_size = 4
test_path = 'test.csv'
img_path = 'data/test/'

test_dataset = LeavesData(test_path, img_path, class2num, num2class, get_train_transforms, get_valid_transforms, 'test')


test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size, 
        shuffle=False
    )


# model = resnet50_model(len(class2num.keys()))
# model.load_state_dict(torch.load('/home/sunduolin/competitions/apple_disease/main/checkpoints/model_100.pth', map_location=torch.device('cpu')))

resnet34 = resnet34_model(len(class2num.keys()))
resnet34.load_state_dict(torch.load(r'G:\讯飞比赛\苹果病害图像识别挑战赛\苹果病害图像识别挑战赛公开数据\main\checkpoints\20230513\ResNet114_20230513_epoch10_fold6.pth'))


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

df = test(resnet34, test_loader, device)

df['label'] = df['label'].apply(lambda x:'d'+str(x+1))

df.to_csv(f'submission_resnet34.csv', index=False)

