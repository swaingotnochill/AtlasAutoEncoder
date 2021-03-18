import csv
from csv import reader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from fastai import basic_train, basic_data, train as Train
from fastai.basic_data import DatasetType

list = []
with open('/content/monojet_Zp2000.0_DM_50.0_chan3.csv') as read_obj:
  csv_reader = reader(read_obj, delimiter = ';', skipinitialspace = True, quotechar = ',')
  for row in csv_reader:
    list.append(row)

data = []
for i in range(len(list)):
  for rows in list[i]:
    test = rows.split(',')
    if(len(test) == 5):
      data.append(test)

column_name = ['obj','E','pt','eta','phi']
dataframe = pd.DataFrame(data, columns = column_name)
dataframe_copy = dataframe.copy(deep = True)
jetData = dataframe_copy.loc[dataframe_copy['obj'] == 'j'].reset_index(drop=True)

float_column = ['E','pt','eta','phi']
for i in float_column:
  jetData[i] = jetData[i].astype(float)

dataset = jetData.drop(['obj'], axis = 1)

 train, test = train_test_split(dataset, test_size = 0.2, random_state = 42)


#NORMALIZATION
  train_mean = train.mean()
  train_std = train.std()

  train_data = (train - train_mean) / train_std
  test_data = (test - train_mean) / train_std

  train_x = train_data
  test_x = test_data

# this is to check whether how close encoder data is compared to decoder data
  train_y = train_x
  test_y = test_x

#SVD 
  svd = TruncatedSVD(n_components=3)

  def add_singular_values(df1):

    df_svd= df1.copy() # make a copy of the data
    sing_vals = []

    # compute and append the singular values into list sing_vals for each data entry
    for i in range(len(df_svd)):
        a = np.diag(df_svd.iloc[i])
        svd.fit(a)
        sing_vals.append(svd.singular_values_)

    # add the singular values into DataFrame
    for i in range(3):
        df_svd.insert(len(df_svd.columns), f'sv_{i}', np.array(sing_vals)[:,i])

    return df_svd

  train_df = add_singular_values(train_x)
  test_df = add_singular_values(test_x)


  train_ds = TensorDataset(torch.tensor(train_df.values).float(), torch.tensor(train_y.values).float())
  valid_ds = TensorDataset(torch.tensor(test_df.values).float(), torch.tensor(test_y.values).float())
  bs = 256 # batch size

  train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=3, pin_memory=True)
  valid_dl = DataLoader(valid_ds, batch_size=bs * 2, num_workers=3, pin_memory = True)


  databunch = basic_data.DataBunch(train_dl, valid_dl)
