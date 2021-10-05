# USAGE
# python create_dataset.py --image_type GramianAngularField --dimensions 128 --wavelengths 500

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from pyts.image import MarkovTransitionField
from pyts.image import GramianAngularField
from pyts.image import RecurrencePlot
from tempfile import TemporaryFile
from os.path import isfile, join
from utilities import config
from utilities import utils
from pathlib import Path
from os import listdir
import cv2
import os
import argparse
import shutil
import gdown
import zipfile
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMAGE_TYPE = "GramianAngularField"
#RESIZED_DIMS = 128
#FIRST_X_COLUMS = 500

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_type", required=True,
    default="GramianAngularField",
	help="Type of timeseries image representation, MarkovTransitionField, GramianAngularField & RecurrencePlot")
ap.add_argument("-d", "--dimensions", required=True,
    default=128,
	help="Type of timeseries image representation, MarkovTransitionField, GramianAngularField & RecurrencePlot")
ap.add_argument("-w", "--wavelengths", required=True,
    default=500,
	help="Type of timeseries image representation, MarkovTransitionField, GramianAngularField & RecurrencePlot")

args = vars(ap.parse_args())


if not os.path.isfile('data5'):
    url = 'https://drive.google.com/uc?id=14FHTR8Ss7L3wwHlfLF3TyyZlYJfiu2HP'
    output = '24_samples.zip'
    gdown.download(url, output, quiet = False)


with zipfile.ZipFile('24_samples.zip', 'r') as zip_ref:
    zip_ref.extractall('')

shutil.rmtree('__MACOSX')

config.IMAGE_TYPE = "GramianAngularField"
#config.RESIZED_DIMS = config.IMG_SHAPE[0]
#first_x_colums = 500

first_col = pd.DataFrame(range(0, int(args['wavelengths'])))


def resize(data, dim = 256):
    im_data = []
    for (i,im) in enumerate(data):
        new_arr = ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')
        resized_image = cv2.resize(new_arr, (dim, dim)) 
        im_data.append(resized_image)
    return np.array(im_data)

def preprocess(filter_x_cols, path, differentiate = False, train=True):
    print("[INFO] Loading and pre-processing data...")
    filenames = listdir(path)
    onlyfiles = [filename for filename in filenames if filename.endswith(".csv")]
    onlyfiles.sort()
    all_data = []
    standardise = False

    for (i,file) in enumerate(onlyfiles):
        df_orig = pd.read_csv(path+'/'+file)
        df_orig.insert(0, 'Index', first_col)

        df = df_orig.copy()
        labels = df.iloc[:,0]
        df = df.T

        new_header = labels #grab the first row for the header
        df = df[1:] #take the data less the header row
        df.columns = new_header #set the header row as the df header

        df['labels'] = df.index
        df['orignal_labels'] = df['labels']
        if train:
            ind = 0
            try:
                df['labels'] = df['labels'].str.split(pat="_", expand=True).iloc[:,ind]
            except:
                df['labels'] = df['labels'].str.split(pat="-", expand=True).iloc[:,ind]

        else:
            ind = 0
            df['labels'] = df['labels'].str.split(pat="-", expand=True).iloc[:,ind]

        all_data.append(df)
        print(f'[INFO] Processed {file}, {i+1} out of {len(onlyfiles)}')

    
    all_data = pd.concat(all_data)
    unique_list = list(all_data['labels'].unique())

    key_hash = dict(zip(unique_list, range(len(unique_list))))
    all_data = all_data.replace({"labels": key_hash})
    
    cols = all_data.columns.tolist()
    cols.insert(0, cols.pop(cols.index('labels')))
    all_data = all_data.reindex(columns= cols)
    
    all_data_labels = all_data['labels']
    orig_data_labels = all_data['orignal_labels']
    data = all_data[all_data.columns[1:len(all_data.columns)]]
    data.reset_index(drop=True, inplace=True)
    all_data_labels.reset_index(drop=True, inplace=True)

    if filter_x_cols is not None:
        data = data.iloc[:, : filter_x_cols]

    print('Normalising...')
    x = data.values #returns a numpy array

    x = pd.DataFrame(x)
    x = x.iloc[:, :-1]
    x = x.apply(pd.to_numeric)
    x = x.diff(axis=1)
    x = x.clip(lower = 0)
    x = np.array(x)

    #min_max_scaler = MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x)

    return data, all_data_labels, orig_data_labels, key_hash

train_df, trainY, orig_data_labels_train, key_hash_train = preprocess(int(args['wavelengths']), path = '24_samples/train', differentiate = True)
#test_df, testY, orig_data_labels_test, key_hash_test = preprocess(first_x_colums, path = 'data3/test', differentiate = True, train=True)


# Splitting Data
train_df, test_df, trainY, testY = train_test_split(train_df, trainY, test_size=0.333, shuffle=False)
testY = list(testY)
trainY = list(trainY)

print('[INFO] Converting to images...')
if args['image_type'] == "RecurrencePlot":
    transformer = RecurrencePlot()
if args['image_type'] == "MarkovTransitionField":
    transformer = MarkovTransitionField()
if args['image_type']  == "GramianAngularField":
    transformer = GramianAngularField()

del train_df[0]
del test_df[0]

trainX = resize(transformer.transform(train_df), int(args['dimensions']))
testX = resize(transformer.transform(test_df), int(args['dimensions']))

np.save('24_samples_trainX.npy', trainX)
np.save('24_samples_testX.npy', testX)
np.save('24_samples_trainY.npy', trainY)
np.save('24_samples_testY.npy', testY)


print('[INFO] Complete conversion to image matix')
print(f'Train size {len(train_df)}')
print(f'Test size {len(test_df)}')
print(f'Unique classes in Training Data - {np.unique(np.array(trainY))}')
print(f'Unique classes in Test Data - {np.unique(np.array(testY))}') 

# Plot a single sample as a sanity check
#row = train_df.iloc[0]
#row.plot(kind='line')
#plt.show()

print('[INFO] Creating image jpegs')
# Convert to images
try:
    os.remove("images/test/")
    os.remove("images/train/")
except:
    pass

os.makedirs("images/train", exist_ok=True) 
os.makedirs("images/test", exist_ok=True) 

transformer = GramianAngularField()
X_new = transformer.transform(test_df)
X_new.shape

im_data = []
for (i,im) in enumerate(X_new):
    new_arr = ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')
    file_name = f'images/test/{testY[i]}_{i}.jpg'
    resized_image = cv2.resize(new_arr, (int(args['dimensions']), int(args['dimensions']))) 
    im_data.append(resized_image)
    cv2.imwrite(file_name, resized_image)

transformer = GramianAngularField()
X_new = transformer.transform(train_df)
X_new.shape

im_data = []
for (i,im) in enumerate(X_new):
    new_arr = ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')
    file_name = f'images/train/{trainY[i]}_{i}.jpg'
    resized_image = cv2.resize(new_arr, (int(args['dimensions']), int(args['dimensions']))) 
    im_data.append(resized_image)
    cv2.imwrite(file_name, resized_image)
    
lab = list(orig_data_labels_train)
lab = lab[720:]

os.makedirs("images/test", exist_ok=True) 
transformer = GramianAngularField()
X_new = transformer.transform(test_df)
X_new.shape

im_data = []
orig_file = []

for (i,im) in enumerate(X_new):
    new_arr = ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')
    file_name = f'images/test/{testY[i]}_{i}.jpg'
    orig_file.append(f'{testY[i]}_{i}')
    resized_image = cv2.resize(new_arr, (int(args['dimensions']), int(args['dimensions']))) 
    im_data.append(resized_image)
    #cv2.imwrite(file_name, resized_image)

original_mapping_dict = dict(zip(orig_file, lab))


print("[INFO] Done")

