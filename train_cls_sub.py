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
import random, copy
# from cutmix.cutmix import CutMix
# from cutmix.utils import CutMixCrossEntropyLoss
import timm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

with open('class2num.json', 'r') as f:
    class2num = json.loads(f.read())

with open('num2class.json', 'r') as f:
    num2class = json.loads(f.read())

class2num = {k:v-1 for k,v in class2num.items()}

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)

def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        
    cmp = astype(y_hat, (y.dtype)) == y
    return float(astype(cmp, (y.dtype)).sum())

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
    

def get_model_num_conv(model):
    return len([each[0] for each in model.named_parameters()])

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

def resnet101_model(num_classes, feature_extract = False, use_pretrained=True):

    model_ft = models.resnet101(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256), 
                                nn.ReLU(),
                                nn.Linear(256, 128), 
                                nn.ReLU(),
                                nn.Linear(128,num_classes))
    return model_ft

def resnext50_model(num_classes, feature_extract = False, use_pretrained=True):
    
    model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256), 
                                nn.ReLU(),
                                nn.Linear(256, num_classes))

    return model_ft

def efficientnet_v2s_model(num_classes, feature_extract = False, use_pretrained=True):
    
    model_ft = models.efficientnet_b0(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[-1].out_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256), 
                                nn.ReLU(),
                                nn.Linear(256,num_classes))

    return model_ft

def densenet121_model(num_classes, feature_extract = False, use_pretrained=True):
    
    model_ft = models.densenet121(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.out_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                                nn.ReLU(),
                                nn.Linear(256, num_classes))

    return model_ft

def densenet161_model(num_classes, feature_extract = False, use_pretrained=True):
    
    model_ft = models.densenet161(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.out_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                                nn.ReLU(),
                                nn.Linear(256, num_classes))

    return model_ft

def train_model(model, fold_idx, train_iter, val_iter, num_epoches, loss, optimizer, scheduler, device):
    model = model.to(device)
    print('training on', device)

    model_name = model.default_cfg['architecture']

    for epoch in range(1, num_epoches+1):
        metric = Accumulator(3)
        for X, y in tqdm(train_iter):
            
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y).mean()
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

        scheduler.step()
        
        test_l, val_acc = evaluate_accuracy_gpu(model, val_iter, loss)
        print('epoch:{}/{}, train_loss:{:.4f}, train_acc:{:.4f}, val_acc:{:.4f}, learning_rate:{}'.format(epoch, num_epoches, train_l, train_acc, val_acc, scheduler.get_last_lr()))
        
        today = get_today_time()
        if not ope(f'checkpoints/{today}/'):
            os.makedirs(f'checkpoints/{today}/')
            
        if epoch % 5 == 0:            
            torch.save(model.state_dict(), 'checkpoints/{}/{}_{}_epoch{}_fold{}_accuracy{:.4f}.pth'.format(today, model_name, today, epoch, fold_idx, val_acc))
        
        if not ope(f'train_logs/{today}/'):
            os.makedirs(f'train_logs/{today}/')
            
        with open(f'train_logs/{today}/train_logs_{model_name}_{today}_fold{fold_idx}.json', 'a') as f:
            f.write(str({'epoch':epoch, 'train_loss':train_l, 'train_acc':train_acc, 'val_acc':val_acc})+'\n')
        


# 继承pytorch的dataset，创建自己的
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, class2num, num2class, train_transform, valid_transform, mode='train', valid_ratio=0.1, resize_height=512, resize_width=512, cutout_prob=0.4):
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
        self.cutout_prob = cutout_prob
        
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
        
        if random.random() < self.cutout_prob:
            img_as_img = self.cutout(img_as_img)

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
    
    
    def cutout(self, im, ratio=0.3):
        h, w, c = im.shape
    
        img = copy.deepcopy(im)
        
        mask_x1, mask_y1 = random.randint(0, w-1), random.randint(0, h)
        mask_x2, mask_y2 = int(min(w, mask_x1 + w*ratio)), int(min(h, mask_y1 + h*ratio))
        
        img[mask_x1:mask_x2, mask_y1:mask_y2, :] *= 0
        
        return img



def evaluate_accuracy_gpu(net, data_iter, loss, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(3)

    print(f'Validation on {device}')
    with torch.no_grad():
        for X, y in tqdm(data_iter):
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            metric.add(accuracy(y_hat, y), loss(y_hat, y), len(y))
    return metric[1] / metric[2], metric[0] / metric[2]


def test_model(model, test_loader, device):
    assert device is not None
    
    img_paths, y_preds = [], []
    model = model.to(device)
    model.eval()
    print(f'test on {device}')
    for img_path, x in tqdm(test_loader):
        x = x.to(device)
        y_hat = model(x)
        
        y_hat = y_hat.argmax(dim=1)
        
        img_paths.extend(list(img_path))
        y_preds.extend(y_hat.detach().cpu().numpy().tolist())
    
    df = pd.DataFrame(data={'uuid':img_paths, 'label':y_preds})
    return df

class CFG:
    train_path = 'train_data.csv'
    test_path = 'test_data.csv'
    train_img_path = 'data/train/'
    test_img_path = 'data/test/'
    learning_rate = 4e-3
    weight_decay = 1e-3
    num_epoch = 10
    k = 5
    seed = 990511

seed_torch(seed=CFG.seed)

def train_model_kfold(model, bs, optimizer, scheduler, loss, skf, train_data, device):
    model_name = model.default_cfg['architecture']

    num_convs = get_model_num_conv(model)
    for fold_idx, (train_ids, val_ids) in enumerate(skf.split(df['img_path'], df['label'])):
        print(f'K-Fold:\t{fold_idx}')
        train_subset, val_subset = df.iloc[train_ids], df.iloc[val_ids]
    
        today = get_today_time()
        if not ope(today):
            os.makedirs(today)
        
        train_data = pd.concat([train_subset, val_subset], axis=0)
        train_data.to_csv(f'{today}/train_data_{model_name}_fold{fold_idx}.csv', index=False)
        
        val_ratio = 1 / skf.n_splits
        train_dataset = LeavesData(CFG.train_path, CFG.train_img_path, class2num, num2class, get_train_transforms, get_valid_transforms, mode='train', valid_ratio=val_ratio)
        val_dataset = LeavesData(CFG.train_path, CFG.train_img_path, class2num, num2class, get_train_transforms, get_valid_transforms, mode='valid', valid_ratio=val_ratio)
        test_dataset = LeavesData(CFG.test_path, CFG.test_img_path, class2num, num2class, get_train_transforms, get_valid_transforms, mode='test')
        
        # train_dataset = CutMix(train_dataset, num_class=len(class2num.keys()), beta=1.0, prob=0.5, num_mix=2)
        # val_dataset = CutMix(val_dataset, num_class=len(class2num.keys()), beta=1.0, prob=0.5, num_mix=2)
        
        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=bs, 
                shuffle=False
            )

        val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=bs, 
                shuffle=False
            )
        
        test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=bs, 
                shuffle=False
            )
        
        train_model(model, fold_idx, train_loader, val_loader, CFG.num_epoch, loss, optimizer, scheduler, device=device)

        submission = test_model(model, test_loader, device)
        
        today = get_today_time()
        if not ope(f'submission/{today}'):
            os.makedirs(f'submission/{today}')
        
        submission['label'] = submission['label'].apply(lambda x:'d'+str(x+1))
        submission.to_csv('submission/{}/submission_{}_fold{}.csv'.format(today, model_name, fold_idx), index=False)
        
        

def train_models_kfold(models_list, skf, train_data):
    for model, bs, device, lr in models_list:
        
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=0.02)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.num_epoch)
        loss = nn.CrossEntropyLoss()
        # loss = CutMixCrossEntropyLoss(True)
        
        train_model_kfold(model, bs, optimizer, scheduler, loss, skf, train_data, device)


class Trainer():
    def __init__(self, model_name, num_classes, batch_size=32, lr=3e-4, num_epoches=10, skf=StratifiedKFold(n_splits=CFG.k), optimizer='AdamW', loss=nn.CrossEntropyLoss(), device='cuda:0', checkpoints_path=None):
        if checkpoints_path is not None:
            assert ope(checkpoints_path)
            self.trainer = timm.create_model(model_name, num_classes=num_classes)
            self.trainer.load_state_dict(torch.load(checkpoints_path))
        else:
            self.trainer = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        self.batch_size = batch_size
        self.lr = lr
        self.skf = skf
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
    
    def train(self):
        pass
    


skf = StratifiedKFold(n_splits=CFG.k)
df = pd.read_csv(CFG.train_path)


resnext = resnext50_model(len(class2num.keys()))
# resnext.load_state_dict(torch.load('checkpoints/20230512/ResNet_20230512_epoch10_fold6.pth'))

efficientnet = efficientnet_v2s_model(len(class2num.keys()))
# efficientnet.load_state_dict(torch.load('checkpoints/20230512/EfficientNet_20230512_epoch5_fold3.pth'))


densnet121 = densenet121_model(len(class2num.keys()))

densnet161 = densenet161_model(len(class2num.keys()))

# resnet34 = resnet34_model(len(class2num.keys()))
# resnet34.load_state_dict(torch.load(r'G:\讯飞比赛\苹果病害图像识别挑战赛\苹果病害图像识别挑战赛公开数据\main\checkpoints\20230513\ResNet114_20230513_epoch10_fold8.pth'))



resnet18 = timm.create_model('resnet18', pretrained=True, num_classes=len(class2num.keys()))

resnext101 = timm.create_model('resnext101_32x4d', num_classes=len(class2num.keys()))
resnext101.load_state_dict(torch.load('checkpoints/20230517/ResNet314_20230517_epoch10_fold4_accuracy0.9611351017890192.pth'))

resnext50 = timm.create_model('resnext50_32x4d',pretrained=True,num_classes=len(class2num.keys()))

efficientnet_b0 = timm.create_model('efficientnet_b0',pretrained=True,num_classes=len(class2num.keys()))
efficientnet_b1 = timm.create_model('efficientnet_b1',pretrained=True,num_classes=len(class2num.keys()))
efficientnet_b2 = timm.create_model('efficientnet_b2',pretrained=True,num_classes=len(class2num.keys()))
efficientnet_b3 = timm.create_model('efficientnet_b3',pretrained=True,num_classes=len(class2num.keys()))
efficientnet_b4 = timm.create_model('efficientnet_b4',pretrained=True,num_classes=len(class2num.keys()))

densnet121d = timm.create_model('densenet169',pretrained=True,num_classes=len(class2num.keys()))

mobilenetv3_large_075 = timm.create_model('mobilenetv3_large_075',pretrained=True,num_classes=len(class2num.keys()))
mobilenetv3_large_100 = timm.create_model('mobilenetv3_large_100',pretrained=True,num_classes=len(class2num.keys()))

seresnet101 = timm.create_model('seresnet101',pretrained=True,num_classes=len(class2num.keys()))

seresnext50 = timm.create_model('seresnext50_32x4d',pretrained=True,num_classes=len(class2num.keys()))
seresnext101 = timm.create_model('seresnext101_32x4d',pretrained=True,num_classes=len(class2num.keys()))
seresnext101.load_state_dict(torch.load('checkpoints/20230518/seresnext101_32x4d_20230518_epoch10_fold0_accuracy0.6644.pth'))
# model_list = [[densnet121, 16], [densnet161, 8], [efficientnet, 16], [resnet34, 64]]




# model_list = [[efficientnet_b0, 32, 'cuda:1']]
# model_list = [[efficientnet_b1, 32, 'cuda:2']]
# model_list = [[efficientnet_b2, 32, 'cuda:3']]
# model_list = [[densnet121d, 24, 'cuda:4']]
# model_list = [[efficientnet_b3, 24, 'cuda:4']]
# model_list = [[efficientnet_b4, 16, 'cuda:1']]

# model_list = [[mobilenetv3_large_075, 64, 'cuda:5']]
# model_list = [[mobilenetv3_large_100, 24, 'cuda:2']]


# model_list = [[resnext101, 16, 'cuda:0', 7e-5]]
# model_list = [[seresnet101, 32, 'cuda:1', 3e-4]]
# model_list = [[seresnext50, 32, 'cuda:2', 3e-5]]
model_list = [[seresnext101, 16, 'cuda:3', 3e-4]]

train_models_kfold(model_list, skf, df)