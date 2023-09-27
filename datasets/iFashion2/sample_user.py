# ---------------------- Sample User iFashion ----------------------
import random
import os.path as osp
import pandas as pd
import shutil
import numpy as np

sample_size = 20000

cur_dir = ""
dir = "../iFashion"
col_names = ['user', 'bundle']

user_bundle_train = pd.read_csv(osp.join(dir, 'user_bundle_train.txt'), sep='\t', names=col_names)
user_bundle_test = pd.read_csv(osp.join(dir, 'user_bundle_test.txt'), sep='\t', names=col_names)
user_bundle_tune = pd.read_csv(osp.join(dir, 'user_bundle_tune.txt'), sep='\t', names=col_names)
user_item = pd.read_csv(osp.join(dir, 'user_item.txt'), sep='\t', names=col_names)

# ids = user_bundle_train['user'].unique().reshape(-1)
# random.shuffle(ids)
sample_ids = np.arange(0, sample_size)

user_bundle_train = user_bundle_train[user_bundle_train['user'].isin(sample_ids)]
user_bundle_test = user_bundle_test[user_bundle_test['user'].isin(sample_ids)]
user_bundle_tune = user_bundle_tune[user_bundle_tune['user'].isin(sample_ids)]
user_item = user_item[user_item['user'].isin(sample_ids)]

user_bundle_train.to_csv(osp.join(cur_dir, 'user_bundle_train.txt'), index=False, header=False, sep='\t')
user_bundle_test.to_csv(osp.join(cur_dir, 'user_bundle_test.txt'), index=False, header=False, sep='\t')
user_bundle_tune.to_csv(osp.join(cur_dir, 'user_bundle_tune.txt'), index=False, header=False, sep='\t')
user_item.to_csv(osp.join(cur_dir, 'user_item.txt'), index=False, header=False, sep='\t')

shutil.copy(osp.join(dir, 'bundle_item.txt'), osp.join(cur_dir, 'bundle_item.txt'))
shutil.copy(osp.join(dir, 'iFashion_data_size.txt'), osp.join(cur_dir, 'iFashion2_data_size.txt'))
