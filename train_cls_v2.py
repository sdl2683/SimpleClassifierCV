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
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
import timm
import glob


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 990511
seed_torch(seed=seed)

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

# def accuracy(y_hat, y):
#     acc = (y_hat.argmax(dim=-1) == y).float().mean()
#     return acc

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
    

# 继承pytorch的dataset，创建自己的
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, class2num, num2class, train_transform, valid_transform, mode='train', valid_ratio=0.1, resize_height=512, resize_width=512, cutout_prob=0):
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




def createFold(file_fold):
    if not ope(file_fold):
        os.makedirs(file_fold)





def getDataset(MyDataset, train_path, train_img_path, test_path, test_img_path, class2num, num2class, get_train_transforms, get_valid_transforms, val_ratio):

    train_dataset = MyDataset(train_path, train_img_path, class2num, num2class, get_train_transforms, get_valid_transforms, mode='train', valid_ratio=val_ratio)
    val_dataset = MyDataset(train_path, train_img_path, class2num, num2class, get_train_transforms, get_valid_transforms, mode='valid', valid_ratio=val_ratio)
    test_dataset = MyDataset(test_path, test_img_path, class2num, num2class, get_train_transforms, get_valid_transforms, mode='test')
    
    return train_dataset, val_dataset, test_dataset

def getDataloader(MyDataloader, train_dataset, val_dataset, test_dataset, batch_size, shuffle):
    
    if type(batch_size) != list:
        batch_size = [batch_size]*3

    if type(shuffle) != list:
        shuffle = [shuffle]*3

    train_loader = MyDataloader(
            dataset=train_dataset,
            batch_size=batch_size[0], 
            shuffle=shuffle[0]
        )

    val_loader = MyDataloader(
            dataset=val_dataset,
            batch_size=batch_size[1], 
            shuffle=shuffle[1]
        )
    
    test_loader = MyDataloader(
            dataset=test_dataset,
            batch_size=batch_size[2], 
            shuffle=shuffle[2]
        )

    return train_loader, val_loader, test_loader


class Trainer():
    def __init__(self, model_name, CFG, checkpoints_path=None, **kwargs):
        if checkpoints_path is not None:
            assert ope(checkpoints_path)
            self.trainer = timm.create_model(model_name, num_classes=len(CFG.class2num.keys()))
            self.trainer.load_state_dict(torch.load(checkpoints_path))
        else:
            self.trainer = timm.create_model(model_name, pretrained=True, num_classes=len(CFG.class2num.keys()))
        
        self.CFG = CFG
        
        #data path
        self.train_path = CFG.train_path
        self.test_path = CFG.test_path
        self.train_img_path = CFG.train_img_path
        self.test_img_path = CFG.test_img_path

        #dataset, dataloader
        self.MyDataset = CFG.MyDataset
        self.MyDataloader = CFG.MyDataloader

        #data info cfg
        self.num2class = CFG.num2class
        self.class2num = CFG.class2num

        #training cfg
        self.num_epoches = CFG.num_epoches if 'num_epoches' not in kwargs else kwargs['num_epoches']
        self.batch_size = CFG.batch_size if 'batch_size' not in kwargs else kwargs['batch_size']
        self.lr = CFG.learning_rate if 'learning_rate' not in kwargs else kwargs['learning_rate']
        self.weight_decay = CFG.weight_decay
        self.momentum = getattr(CFG, 'momentum', 1)
        self.checkpoint_period = CFG.checkpoint_period

        #training trick cfg
        self.get_train_transforms = CFG.get_train_transforms
        self.get_valid_transforms = CFG.get_valid_transforms
        self._k = CFG.k
        self.skf = CFG.skf(n_splits=self._k)
        self.use_cutmix = CFG.use_cutmix

        #optim cfg
        self.optimizer = CFG.optimizer
        self.scheduler = CFG.scheduler
        self.loss = CFG.loss
        self.device = CFG.device if 'device' not in kwargs else kwargs['device']

        self.date = getattr(CFG, 'date', get_today_time())
        self.data_shuffle = CFG.data_shuffle

        if self.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.trainer.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.trainer.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.trainer.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.scheduler == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epoches//2)
        
        if self.use_cutmix:
            self.train_loss = CutMixCrossEntropyLoss(True)
            self.val_loss = self.loss
        else:
            self.train_loss = self.loss
            self.val_loss = self.loss


        #create file fold named given date, used to save training data, training logs, ckeckpoints and submissions
        createFold(str(self.date))


    def train_model(self, model, fold_idx, train_iter, val_iter, num_epoches, optimizer, scheduler, device):
        model = model.to(device)
        print('training on', device)

        model_name = model.default_cfg['architecture']

        for epoch in range(1, num_epoches+1):
            metric = Accumulator(3)
            for X, y in tqdm(train_iter):              
                
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                l = self.train_loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], 0.9, X.shape[0])
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                continue

            scheduler.step()
            
            val_l, val_acc = evaluate_accuracy_gpu(model, val_iter, self.val_loss)
            print('model: {}, epoch:{}/{}, train_loss:{:.4f}, val_loss:{:.4f}, train_acc:{:.4f}, val_acc:{:.4f}, learning_rate:{}'.format(model_name, epoch, num_epoches, train_l, val_l, train_acc, val_acc, scheduler.get_last_lr()))
            
            today = str(self.date)
            
            #save path of checkpoints and train logs
            checkpoints_path = f'{today}/{model_name}/checkpoints'
            train_logs_path = f'{today}/{model_name}/train_logs'
            createFold(checkpoints_path)
            createFold(train_logs_path)

            if epoch % self.checkpoint_period == 0:            
                torch.save(model.state_dict(), '{}/{}_{}_epoch{}_fold{}_accuracy{:.4f}.pth'.format(checkpoints_path, model_name, today, epoch, fold_idx, val_acc))
            

            best_models = glob.glob(f'{checkpoints_path}/best_model*_fold{fold_idx}.pth')
            if best_models == []:
                torch.save(model.state_dict(), '{}/best_model_accuracy{:.4f}_fold{}.pth'.format(checkpoints_path, val_acc, fold_idx))
            else:
                best_model_path = best_models[0]
                best_acc = float(os.path.split(best_model_path)[1].split('_')[2][8:])
                if val_acc > best_acc:
                    os.remove(f'{best_model_path}')
                    torch.save(model.state_dict(), '{}/best_model_accuracy{:.4f}_fold{}.pth'.format(checkpoints_path, val_acc, fold_idx))

            with open(f'{train_logs_path}/train_logs_{model_name}_{today}_fold{fold_idx}.json', 'a') as f:
                f.write(str({'epoch':epoch, 'train_loss':train_l, 'val_loss':val_l, 'train_acc':train_acc, 'val_acc':val_acc})+'\n')
    
    def test_model(self, model, test_loader, device):
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

    def train_kfold(self):
        #get model name, the model should be from timm
        model_name = self.trainer.default_cfg['architecture']

        df = pd.read_csv(self.train_path)

        for fold_idx, (train_ids, val_ids) in enumerate(self.skf.split(df['img_path'], df['label'])):
            
            print(f'current Fold:\t{fold_idx+1} / {self.skf.n_splits}')
            
            train_subset, val_subset = df.iloc[train_ids], df.iloc[val_ids]

            today = str(self.date)

            train_data_info_path = f'{today}/{model_name}/train_data_info'
            createFold(train_data_info_path)
            
            train_data = pd.concat([train_subset, val_subset], axis=0)
            train_data.to_csv(f'{train_data_info_path}/train_data_{model_name}_fold{fold_idx}.csv', index=False)
            
            val_ratio = 1 / self.skf.n_splits
            train_dataset, val_dataset, test_dataset = getDataset(MyDataset=self.MyDataset, 
                                                                  train_path=self.train_path, 
                                                                  train_img_path=self.train_img_path, 
                                                                  test_path=self.test_path, 
                                                                  test_img_path=self.test_img_path, 
                                                                  class2num=self.class2num, 
                                                                  num2class=self.num2class, 
                                                                  get_train_transforms=self.get_train_transforms, 
                                                                  get_valid_transforms=self.get_valid_transforms,
                                                                  val_ratio=val_ratio)
            
            train_dataset = CutMix(train_dataset, num_class=len(class2num.keys()), beta=1.0, prob=0.5, num_mix=2)
            
            train_loader, val_loader, test_loader = getDataloader(MyDataloader=self.MyDataloader, 
                                                                  train_dataset=train_dataset,
                                                                  val_dataset=val_dataset,
                                                                  test_dataset=test_dataset,
                                                                  batch_size=self.batch_size,
                                                                  shuffle=self.data_shuffle)
            
            self.train_model(self.trainer, fold_idx, train_loader, val_loader, self.num_epoches, self.optimizer, self.scheduler, self.device)

            submission = self.test_model(self.trainer, test_loader, self.device)
            
            #save predictions submission
            submission_path = f'{today}/{model_name}/submission'
            createFold(submission_path)
            
            submission['label'] = submission['label'].apply(lambda x:'d'+str(x+1))
            submission.to_csv('{}/submission_{}_fold{}.csv'.format(submission_path, model_name, fold_idx), index=False)

class CFG:
    #data path
    train_path = 'train_data.csv'
    test_path = 'test_data.csv'
    train_img_path = 'data/train/'
    test_img_path = 'data/test/'

    #data info cfg
    num2class = num2class
    class2num = class2num

    #dataset and dataloader
    MyDataset = LeavesData
    MyDataloader = DataLoader
    data_shuffle = False

    #training cfg
    num_epoches = 30
    batch_size = 12
    learning_rate = 3e-5
    weight_decay = 1e-3
    momentum = 0.95
    checkpoint_period = 10

    #training trick cfg
    get_train_transforms = get_train_transforms
    get_valid_transforms = get_valid_transforms
    k = 5
    skf = StratifiedKFold
    use_cutmix = True

    #optim cfg
    optimizer = 'adamw'
    scheduler = 'cos'
    loss = nn.CrossEntropyLoss()
    device = 'cuda:0'

    date = 20230522

def trainModels(models):
    for model in models:
        model.train_kfold()





resnet101 = Trainer('resnet101', CFG)
resnet101.device = 'cuda:4'

resnest50d = Trainer('resnest50d', CFG, '20230522/resnest50d/checkpoints/best_model_accuracy0.9846_fold0.pth')
resnest50d.device = 'cuda:4'

seresnet50 = Trainer('seresnet50', CFG)
seresnet50.device = 'cuda:4'



seresnext50 = Trainer('seresnext50_32x4d', CFG, 'checkpoints/20230519/seresnext50_32x4d_20230519_epoch10_fold4_accuracy0.9895.pth')
seresnext50.device = 'cuda:3'
seresnext50.learning_rate = 8e-7

resnext50 = Trainer('resnext50_32x4d', CFG)
resnext50.device = 'cuda:3'

resnext101 = Trainer('resnext101_32x4d', CFG, 'checkpoints/20230519/resnext101_32x4d_20230519_epoch10_fold4_accuracy0.9698.pth')
resnext101.device = 'cuda:3'

resnet50 = Trainer('resnet50', CFG)
resnet50.device = 'cuda:3'



densenet121 = Trainer('densenet121', CFG, 'checkpoints/20230520/densenet121_20230520_epoch10_fold2_accuracy0.9907.pth')
densenet121.device = 'cuda:5'
densenet121.learning_rate = 8e-7

densenet161 = Trainer('densenet161', CFG, 'checkpoints/20230520/densenet161_20230520_epoch10_fold2_accuracy0.9922.pth')
densenet161.device = 'cuda:5'
densenet161.learning_rate = 8e-7


# resnet18 = Trainer('resnet18', CFG, 'checkpoints/20230520/resnet18_20230520_epoch10_fold0_accuracy0.9679.pth')
# resnet18.device = 'cuda:1'
# resnet18.batch_size = 100
# resnet18.train_kfold()

# trainModels([resnest50d, resnet101, seresnet50])

# trainModels([resnet50, resnext50, resnext101, seresnext50])

# trainModels([densenet121, densenet161])


resnest50d.device = 'cuda:1'

resnet101.device = 'cuda:1'
seresnet50.device = 'cuda:1'
resnet50.device = 'cuda:1'
resnext50.device = 'cuda:1'
resnext101.device = 'cuda:1'
seresnext50.device = 'cuda:1'
densenet121.device = 'cuda:1'
densenet161.device = 'cuda:1'

trainModels([resnest50d, resnet101, seresnet50, resnet50, resnext50, resnext101, seresnext50, densenet121, densenet161])