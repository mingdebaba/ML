# %%
import os 
import tarfile
import urllib.request

import tensorflow as tarfile
from tensorflow import keras

import numpy as np
import re 
import string
from random import randint

# %%
url="http://ai.sranford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="data/aclImdb_v1.tar.gz"

# %%
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.isfile(filepath):
    print('downloading...')
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
else:
    print(filepath,'is existed!')

# %%
#モデル
model=keras.models.Sequential()


