import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.multimodal.utils.misc import shopee_dataset
# download_dir = 'ag_automm_tutorial_imgcls'
# train_data_path, test_data_path = shopee_dataset(download_dir)
# print(train_data_path)


data = pd.read_csv(r'G:\讯飞比赛\苹果病害图像识别挑战赛\苹果病害图像识别挑战赛公开数据\main\train_data.csv')
train_data, val_data = train_test_split(data, test_size=0.2)


from autogluon.multimodal import MultiModalPredictor
import uuid
model_path = f"tmp/{uuid.uuid4().hex}-automm_shopee"
predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(
    train_data=train_data, # you can use train_data_byte as well
    time_limit=30, # seconds
) # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model
