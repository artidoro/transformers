#%%
import json

#%%
f = open('VisualNews_UMD_gt_partition_pp_fix.json')
dataset_all = json.loads(f.read())

#%%

train_data = [elt for elt in dataset_all if elt['SRI_partition'] == 'train']
with open('training_data.txt', 'w') as wfile:
    for elt in train_data:
        wfile.write(elt['article_text'] + '\n')

dev_data = [elt for elt in dataset_all if elt['SRI_partition'] == 'dev']
with open('dev_data.txt', 'w') as wfile:
    for elt in dev_data:
        wfile.write(elt['article_text'] + '\n')

test_data = [elt for elt in dataset_all if elt['SRI_partition'] == 'test']
with open('test_data.txt', 'w') as wfile:
    for elt in test_data:
        wfile.write(elt['article_text'] + '\n')
# %%
dev_data = [elt for elt in dataset_all if elt['SRI_partition'] == 'dev']
with open('dev_data_trunc.txt', 'w') as wfile:
    new_len = int(len(dev_data)/32)*32
    for i, elt in enumerate(dev_data[:new_len]):
        wfile.write(elt['article_text'] + '\n')

# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
train_data = [len(elt['article_text'].split()) for elt in dataset_all if elt['SRI_partition'] == 'train']
np.average(train_data)
# %%
plt.hist(train_data, bins=50, range=(0, 2500))
# %%
np.average([elt for elt in train_data if elt < 1024])
# %%
