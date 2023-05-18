import os
import pandas as pd


# with open(r'G:\讯飞比赛\苹果病害图像识别挑战赛\苹果病害图像识别挑战赛公开数据\main\data\train\labletoclass.txt', 'r') as f:
#     content = f.readlines()

# content = [each.replace('\n', '') for each in content]
# content = [each.split(' ', 1) for each in content]
# content = {each[0]:each[1] for each in content}
# content = {int(k[1])-1:v for k,v in content.items()}

# df = pd.read_csv('submission.csv')
# df['preds'] = df['preds'].apply(lambda x:'d'+str(x+1))
# df.columns = ['uuid', 'label']
# df.to_csv('submission_v1_res50.csv',index=False)


src = r'G:\讯飞比赛\苹果病害图像识别挑战赛\苹果病害图像识别挑战赛公开数据\main\data\train'
li = os.listdir(src)
li.remove('labletoclass.txt')

tmp = []
for each in li:
    imgs = os.listdir(os.path.join(src, each))
    for img in imgs:
       tmp.append({'image_path':os.path.join(src, each, img), 'label':each})
df = pd.DataFrame(data=tmp)
df.to_csv('train_data.csv',index=False)