import pandas as pd

colname = ['user', 'bundle']
uid = 6

ub_train = pd.read_csv('user_bundle_train.txt', sep='\t', names=colname)
ub_test = pd.read_csv('user_bundle_test.txt', sep='\t', names=colname)
b_i = pd.read_csv('bundle_item.txt', sep='\t', names = ['bundle', 'item'])

uid_train = ub_train[ub_train['user'] == uid]
uid_test = ub_test[ub_test['user'] == uid]

bundles_train_uid = uid_train['bundle'].tolist()
bundles_test_uid = uid_test['bundle'].tolist()

train_dict = dict()
for i in bundles_train_uid:
    temp = b_i[b_i['bundle'] == i]
    train_dict[i] = temp['item'].tolist()

test_dict = dict()
for i in bundles_test_uid:
    temp = b_i[b_i['bundle'] == i]
    test_dict[i] = temp['item'].tolist()



print('train',train_dict)
print('test',test_dict)