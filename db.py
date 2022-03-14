import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from pathlib import Path
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from pyts.image import GramianAngularField
import cv2
import random
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.models import load_model
from tensorflow import keras
from imutils.paths import list_images
from sklearn import preprocessing
from pathlib import Path


first_x_colums = 360
first_col = pd.DataFrame(range(first_x_colums))

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    A prettier confusion matrix
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    img = BytesIO()
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
    plt.savefig(img, format='png')

    #plt.savefig('new_plot.png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

def processResults(df_summary_a):
    
    df_summary_a['Reading'] = df_summary_a['Ground Truth'].apply(lambda x: x.split('_')[0].split('/')[2])
    print(df_summary_a)
    df_agg = df_summary_a.groupby(['Ground Truth', 'Compared With']).agg({'Similarity':sum})
    print('df_agg')
    print(df_agg)
    g = df_agg['Similarity'].groupby('Ground Truth', group_keys=False)
    res = g.apply(lambda x: x.sort_values().head(3))
    print('res')
    print(res)
    res = res.reset_index()
    res['Sample'] = res['Ground Truth'].apply(lambda x: x.split('/')[2].split('.')[0])
    res['Prediction'] = res['Compared With'].apply(lambda x: x.split('/')[1].split('.')[0])
    print('res2')
    print(res)
    #res = res[res['Similarity'] <= 0.3]
    res = res.groupby(['Sample', 'Prediction']).agg({'Similarity':'mean'}).reset_index().sort_values(by='Similarity')
    res['Sample'] = res['Sample'].apply(lambda x: x.split('_')[0])
    res['Prediction'] = res['Prediction'].apply(lambda x: x.split('_')[0])
    res = res.groupby(['Sample','Prediction']).mean().reset_index().sort_values(by = ['Sample','Similarity'])
    return res

def processResultsC(df_summary_a):
    df_summary_a['Reading'] = df_summary_a['Sample'].apply(lambda x: x.split('_')[0])
    #df_summary_a = df_summary_a[df_summary_a['Confidence'] >= 0.3]
    df_agg = df_summary_a.groupby(['Reading', 'Prediction']).agg({'Confidence':'mean'}).reset_index()
    df_agg = df_agg.sort_values(by = ['Reading','Confidence'], ascending=False)
    df_agg['Confidence'] = df_agg['Confidence'] * 100
    df_agg['Confidence'] = df_agg['Confidence'].map('{:,.3f}%'.format)
    return df_agg

def getPredictions():
    args = {
        "input": "examples",
        "test": "image/train",
        "train": "images/train",
        "avg": "image_averaged"
    }

    print("[INFO] loading siamese model...")
    model = load_model('model/contrastive_siamese_model/', compile=False)

    ImagePathsTest = list(list_images(args["test"]))
    ImagePathsAvg = list(list_images(args["avg"]))

    GTsa = []
    CWa = []
    PRa = []

    print(ImagePathsTest)
    print(ImagePathsAvg)
    # loop over all image pairs
    for (i, pathA) in enumerate(ImagePathsTest):
        # load both the images and convert them to grayscale
        for (j, pathB) in enumerate(ImagePathsAvg):
            imageA = cv2.imread(pathA, 0)
            imageB = cv2.imread(pathB, 0)

            # add channel a dimension to both the images
            imageA = np.expand_dims(imageA, axis=-1)
            imageB = np.expand_dims(imageB, axis=-1)

            # add a batch dimension to both images
            imageA = np.expand_dims(imageA, axis=0)
            imageB = np.expand_dims(imageB, axis=0)

            # scale the pixel values to the range of [0, 1]
            imageA = imageA / 255.0
            imageB = imageB / 255.0

            # use our siamese model to make predictions on the image pair,
            # indicating whether or not the images belong to the same class
            preds = model.predict([imageA, imageB])
            proba = preds[0][0]
            ground_truth = pathA.split('_')[0].split('/')[2]
            #compared_with = int(pathB.split('_')[0].split('/')[1])
            compared_with = pathB.split('/')[1]
            #if j % 200:   
            print(f'{j}-{i} - {pathA}, {pathB}, {ground_truth}, {compared_with}, {proba}')  
            GTsa.append(pathA)

            CWa.append(pathB)
            PRa.append(proba)

    return pd.DataFrame(
    {'Ground Truth': GTsa,
     'Compared With': CWa,
     'Similarity': PRa
    })

def getPredictionsConfusionMatrix():
    args = {
        "input": "examples",
        "test": "image/train",
        "train": "images/train",
        "avg": "image_averaged_db/train"
    }
    
    label_dict = {  0: 'A',
                    1: 'B',
                    2: 'C',
                    3: 'D',
                    4: 'E',
                    5: 'F',
                    6: 'G',
                    7: 'H',
                    8: 'I',
                    9: 'J',
                    10: 'K',
                    11: 'L',
                    12: 'M',
                    13: 'N',
                    14: 'O',
                    15: 'P',
                    16: 'Q',
                    17: 'R',
                    18: 'S',
                    19: 'T',
                    20: 'U',
                    21: 'V',
                    22: 'W',
                    23: 'X'}

    print("[INFO] loading model...")
    model = keras.models.load_model('model/classifier_model/CNN_20220310_190832.h5')
    #model = keras.models.load_model('model/classifier_model/data_aug_1_25.h5')

    ImagePathsTest = "image/train/"

    filenames = listdir(ImagePathsTest)
    image_file_names = [filename for filename in filenames if filename.endswith(".jpg")]
    image_file_names.sort()
    
    preds = []
    for f in image_file_names:
        image = cv2.imread(ImagePathsTest+f, 0)
        #image = cv2.imread(ImagePathsTest+f)
        image = image / 255.0

        # We need to add a 4th dimension to the first axis
        input_im = image.reshape(1,256,256,1)
        #input_im = image.reshape(3,256,256,1)  

        # We now get the predictions for that single image
        #pred = np.argmax(model.predict(input_im), axis=-1)[0]

        probs = model.predict(input_im)
        prediction = np.argmax(probs)
        print(prediction)
        preds.append(np.argmax(probs))

    return preds


def getClassifications():
    args = {
        "input": "examples",
        "test": "image/train",
        "train": "images/train",
        "avg": "image_averaged_db/train"
    }
    
    label_dict = {  0: 'A',
                    1: 'B',
                    2: 'C',
                    3: 'D',
                    4: 'E',
                    5: 'F',
                    6: 'G',
                    7: 'H',
                    8: 'I',
                    9: 'J',
                    10: 'K',
                    11: 'L',
                    12: 'M',
                    13: 'N',
                    14: 'O',
                    15: 'P',
                    16: 'Q',
                    17: 'R',
                    18: 'S',
                    19: 'T',
                    20: 'U',
                    21: 'V',
                    22: 'W',
                    23: 'X',
                    24: 'AB',
                    25: 'CB',
                    26: 'DB',
                    27: 'EB',
                    28: 'PB',
                    29: 'RB',
                    30: 'SB',
                    31: 'UB',
                    32: 'VB',
                    33: 'WB',
                    34: 'XB'}

    print("[INFO] loading siamese model...")
    #model = keras.models.load_model('model/classifier_model/Feb_28.h5')
    #model = keras.models.load_model('model/classifier_model/CNN_20220308_172743.h5')
    model = keras.models.load_model('model/classifier_model/CNN_20220310_190832.h5')
    #model = keras.models.load_model('model/classifier_model/data_aug_1_25.h5')

    ImagePathsTest = "image/train/"

    filenames = listdir(ImagePathsTest)
    image_file_names = [filename for filename in filenames if filename.endswith(".jpg")]
    image_file_names.sort()
    
    data_frames = []
    for f in image_file_names:
        image = cv2.imread(ImagePathsTest+f, 0)
        #image = cv2.imread(ImagePathsTest+f)
        image = image / 255.0

        # We need to add a 4th dimension to the first axis
        input_im = image.reshape(1,256,256,1)
        #input_im = image.reshape(3,256,256,1)  

        # We now get the predictions for that single image
        #pred = np.argmax(model.predict(input_im), axis=-1)[0]

        n = 5
        probs = model.predict(input_im)
        y_preds = np.flip(np.argsort(probs, axis=1)[:,-n:])

        label_names = []
        for y in list(y_preds[0]):
            label_names.append(label_dict[int(y)])

        top_3_prob = [probs[0][int(y_preds[0][0])],
                    probs[0][int(y_preds[0][1])],
                    probs[0][int(y_preds[0][2])],
                    probs[0][int(y_preds[0][3])],
                    probs[0][int(y_preds[0][4])]]

        df = pd.DataFrame(data=zip(label_names,top_3_prob),columns=['Prediction','Confidence'])
        print(f)
        df['Sample'] = f.split('.')[0]
        new_order = [2,1,0]
        df = df[df.columns[new_order]]
        #df['Confidence'] = df['Confidence'] * 100
        #df['Confidence'] = df['Confidence'].map('{:,.3f}%'.format)
        data_frames.append(df)

    return pd.concat(data_frames)

def preprocess_excel_files(filterno,
                           path, 
                           start=0,
                            end=360,
                            data_sets = '1,2,3',
                            differentiate = True,
                            differentiate_2 = False,
                            normalize_a = False,
                            normalize_b = True,
                            train=True):
    
    filenames = listdir(path)
    onlyfiles = [filename for filename in filenames if filename.endswith(".xlsx")]
    onlyfiles.sort()

    all_data = []
    # training_sets = data_sets.split(',')
    # onlyfiles = ["Set " + f + '.csv' for f in training_sets]
    print(onlyfiles)
    for (i,file) in enumerate(onlyfiles):
        df_orig = pd.read_excel(path+'/'+file)
        df_orig = df_orig.iloc[start:end, : ]
        first_col = pd.DataFrame(range(start,end))
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
        print(f'[INFO] Processed {file}, {i+1} out of {len(onlyfiles)} {len(df)}')


    all_data = pd.concat(all_data)
    #print('ALL DATA')
    #print(all_data)
    #all_data.to_csv('data.csv')

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

    #data = data.iloc[start:end, : ]
    data = data.drop(columns=data.columns[-1])
    print('DATA')
    print(data)

    if normalize_b:
      data = preprocessing.normalize(data.values, axis=1, norm='max')

      # min_max_scaler_a = MinMaxScaler()
      # x = data.values
      # data = min_max_scaler_a.fit_transform(x)
      data = pd.DataFrame(data)

    if differentiate:
      print('Differentiating...')
      x = data.values #returns a numpy array

      x = pd.DataFrame(x)
      x = x.iloc[:, :-1]
      x = x.apply(pd.to_numeric)
      x = x.diff(axis=1)
      x = x.clip(lower = 0)
      x = np.array(x)
      data = pd.DataFrame(x)
      del data[0]

      if differentiate_2:
        x = pd.DataFrame(x)
        x = x.iloc[:, :-1]
        x = x.apply(pd.to_numeric)
        x = x.diff(axis=1)
        x = x.clip(lower = 0)
        x = np.array(x)
        data = pd.DataFrame(x)
        del data[0]
    else:
      #del data["orignal_labels"]
      data = pd.DataFrame(data)

    if normalize_a:
      min_max_scaler = MinMaxScaler()
      x = data.values
      data = min_max_scaler.fit_transform(x)
      data = pd.DataFrame(data)

    print('fINAL DATA')
    print(data)    
    return data, all_data_labels, orig_data_labels, key_hash

# def preprocess_excel_files(filter_x_cols, path, train=True):
#     print("[INFO] Loading and pre-processing data...")
#     filenames = listdir(path)
#     onlyfiles = [filename for filename in filenames if filename.endswith(".xlsx")]
#     onlyfiles.sort()
#     all_data = []

#     for (i,file) in enumerate(onlyfiles):
#         df_orig = pd.read_excel(path+'/'+file)
#         #df_orig = df_orig.iloc[48:549,]
#         print(df_orig)
        
#         #del df_orig['Unnamed: 0']
#         #df_orig = df_orig.drop(df_orig.columns[[0]], axis=1)
#         df_orig.insert(0, 'Index', first_col)

#         df = df_orig.copy()
#         labels = df.iloc[:,0]
#         df = df.T

#         print(df)
#         new_header = labels #grab the first row for the header
#         #df = df[1:] #take the data less the header row
#         df.columns = new_header #set the header row as the df header
        
#         df['labels'] = df.index
#         print(df['labels'])
#         df['orignal_labels'] = df['labels']
#         ind = 0
#         if train:          
#             try:
#                 df['labels'] = df['labels'].str.split(pat="_", expand=True).iloc[:,ind]
#             except:
#                 df['labels'] = df['labels'].str.split(pat="-", expand=True).iloc[:,ind]

#         else:
#             df['labels'] = df['labels'].str.split(pat="-", expand=True).iloc[:,ind]

#         all_data.append(df)
#         print(f'[INFO] Processed {file}, {i+1} out of {len(onlyfiles)}')

    
#     all_data = pd.concat(all_data)
#     # all_data.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
#     # all_data = all_data.sort_index()
#     print(all_data)
#     unique_list = list(all_data['labels'].unique())

#     key_hash = dict(zip(unique_list, range(len(unique_list))))
#     all_data = all_data.replace({"labels": key_hash})
    
#     cols = all_data.columns.tolist()
#     cols.insert(0, cols.pop(cols.index('labels')))
#     all_data = all_data.reindex(columns= cols)
    
#     all_data_labels = all_data['labels']
#     orig_data_labels = all_data['orignal_labels']
#     data = all_data[all_data.columns[1:len(all_data.columns)]]
#     data.reset_index(drop=True, inplace=True)
#     all_data_labels.reset_index(drop=True, inplace=True)
#     print(data)
#     print(f'Data shape - {data.shape}')
#     filter_x_cols = 360
#     if filter_x_cols is not None:
#         data = data.iloc[:, : filter_x_cols]
#     print(f'filter_x_cols - {filter_x_cols}')
#     print(f'Data shape - {data.shape}')
    
#     print('[INFO] Normalising and Differentiating data...')
#     #x = data.values #returns a numpy array

#     min_max_scaler_a = MinMaxScaler()
#     x = data.values
#     data = min_max_scaler_a.fit_transform(x)
#     data = pd.DataFrame(data)

#     x = pd.DataFrame(data)
#     x = x.iloc[:, :-1]
#     x = x.apply(pd.to_numeric)
#     x = x.diff(axis=1)
#     x = x.clip(lower = 0)
#     x = np.array(x)
    
#     # x = pd.DataFrame(x)
#     # x = x.iloc[:, :-1]
#     # x = x.apply(pd.to_numeric)
#     # x = x.diff(axis=1)
#     # x = x.clip(lower = 0)
#     # x = np.array(x)

#     # min_max_scaler = MinMaxScaler()
#     # x = min_max_scaler.fit_transform(x)
#     # print(f'Data shape 260 - {data}')
#     data = pd.DataFrame(x)
#     # print(f'Data shape 262 - {data}')
#     print('[INFO] Pre-processing completed.')
#     del data[0]
#     #print(data)

#     return data, all_data_labels, orig_data_labels, key_hash


from pathlib import Path
from shutil import copyfile
import json

class Build_Database():
    def __init__(self, dims = 499, k = 30):
        self.data = None
        self.dims = dims
        self.k = k
        self.index = None
        self.df = None 
        self.labels = None
        self.orig_data_labels_train = None
        self.key_hash_train = None
        self.uploads = "uploads"
        self.tests = "tests"
        self.makeDir('tests')
        
    def resize(self, data, dim = 256):
        im_data = []
        for im in data:
            new_arr = ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')
            resized_image = cv2.resize(new_arr, (256, 256)) 
            im_data.append(resized_image)
        return np.array(im_data)
        
    def convertToImages(self,
                        train_df,
                        trainY,
                        orig_data_labels_train,
                        resize_dims = 256,
                        image_type = "GramianAngularField"):
        #train_df, test_df, trainY, testY = train_test_split(train_df, trainY, test_size=0.333, shuffle=False)
        trainY = list(trainY)

        print('[INFO] Converting to images...')
        if image_type == "GramianAngularField":
            transformer = GramianAngularField()

        print(train_df)
        #del train_df[1]
        trainX = self.resize(transformer.transform(train_df), (256, 256, 1))

        print('[INFO] Complete conversion to image matix')
        print(f'Train size {len(train_df)}')
        print(f'Unique classes - {np.unique(np.array(trainY))}')

        print('[INFO] Creating image jpegs')
        # Convert to images
        #!rm -rf images/train/
        self.makeDir("image/train")
        files = glob.glob('image/train/*')
        for f in files:
            os.remove(f)

        try:
            os.makedirs("images/train", exist_ok=True) 
        except:
            print('Directory already exists')

        transformer = GramianAngularField()

        X_new = transformer.transform(train_df)

        im_data = []
        orig_file = []

        for (i,im) in enumerate(X_new):
            new_arr = ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')
            file_name = f'image/train/{trainY[i]}{i}.jpg'
            orig_file.append(f'{trainY[i]}_{i}')
            resized_image = cv2.resize(new_arr, (resize_dims, resize_dims)) 
            im_data.append(resized_image)
            cv2.imwrite(file_name, resized_image)
       
       
    def convertToImagesAdd(self,
                        train_df,
                        trainY,
                        orig_data_labels_train,
                        resize_dims = 256,
                        image_type = "GramianAngularField"):
        #train_df, test_df, trainY, testY = train_test_split(train_df, trainY, test_size=0.333, shuffle=False)
        trainY = list(trainY)

        print('[INFO] Converting to images...')
        if image_type == "GramianAngularField":
            transformer = GramianAngularField()

        print(train_df)
        #del train_df[1]
        trainX = self.resize(transformer.transform(train_df), (256, 256, 1))

        print('[INFO] Complete conversion to image matix')
        print(f'Train size {len(train_df)}')
        print(f'Unique classes - {np.unique(np.array(trainY))}')

        print('[INFO] Creating image jpegs')
        # Convert to images
        #!rm -rf images/train/
        self.makeDir("image/add")
        files = glob.glob('image/add/*')
        for f in files:
            os.remove(f)

        try:
            os.makedirs("images/add", exist_ok=True) 
        except:
            print('Directory already exists')

        transformer = GramianAngularField()

        X_new = transformer.transform(train_df)

        im_data = []
        orig_file = []

        for (i,im) in enumerate(X_new):
            new_arr = ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')
            file_name = f'image/add/{trainY[i]}{i}.jpg'
            orig_file.append(f'{trainY[i]}_{i}')
            resized_image = cv2.resize(new_arr, (resize_dims, resize_dims)) 
            im_data.append(resized_image)
            cv2.imwrite(file_name, resized_image)
            
        # lab = list(orig_data_labels_train)
        # lab = lab[720:]

        # original_mapping_dict = dict(zip(orig_file, lab))

        # print(f'Train shape - {trainX.shape}')

        # # Convert to Float
        # trainX = trainX / 255.0

        # # add a channel dimension to the images
        # trainX = np.expand_dims(trainX, axis=-1)

        # # prepare the positive and negative pairs
        # print("[INFO] preparing positive and negative pairs...")

        # print(trainX.shape)

        # IMG_SHAPE = trainX[0].shape
        # print(f'Image size - {IMG_SHAPE}')
        # print("[INFO] Done")
       
        
    def makeDir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
    def initializeDB(self, excel_path, path = 'database'):           
        self.df, self.labels, self.orig_data_labels_train, self.key_hash_train = preprocess_excel_files(self.dims+1, excel_path)
        #self.saveData(path)
        return self.df, self.labels, self.orig_data_labels_train, self.key_hash_train
      
    # def saveData(self, path):
    #     self.makeDir(path)
    #     np.savetxt('database/database.txt', np.array(self.df))
    #     np.savetxt('database/labels.txt', np.array(self.labels), fmt='%d')
    #     with open('database/key_hash_train.json', 'w') as file:
    #         file.write(json.dumps(self.key_hash_train))
    
    def loadQuery(self, excel_path):
        df, labels, orig_data_labels_train, key_hash_train = preprocess_excel_files(self.dims+1, excel_path)
        return df, labels, orig_data_labels_train, key_hash_train
    
    def addData(self, new_file, addorread = 'add'):
        self.makeDir('uploads')
        [f.unlink() for f in Path("uploads").glob("*") if f.is_file()] 
        # files = glob.glob('uploads')
        # for f in files:
        #     os.remove(f)
        if addorread == 'add':
            filenames = listdir(self.uploads)
            onlyfiles = [filename for filename in filenames if filename.endswith(".xlsx")]
            onlyfiles.sort()
            new_file_name = f'{len(onlyfiles)+1}.xlsx'
            new_file_name

            copyfile(new_file, f'{self.uploads}/{new_file_name}')
            print(f'[INFO] Added {new_file} to database')
        else:
            copyfile(new_file, f'{self.tests}/samples.xlsx')
            print(f'[INFO] Uploaded {new_file}.')        
        
#         # Load new file
#         df, labels, orig_data_labels_train, key_hash_train = preprocess_excel(self.dims+1, excel_path)
#         print(len(df))
#         # Load databse
#         db = np.loadtxt('database/database.txt')
#         db_labels = np.loadtxt('database/labels.txt', dtype=int)
#         with open('database/key_hash_train.json') as f:
#             data = json.load(f)
        
#         # Update database by merging files
#         updated_db = np.concatenate((db, np.array(df)), axis=0)
#         updated_labels = np.concatenate((db_labels, np.array(labels)), axis=0)
#         temp = data.copy()
#         temp.update(key_hash_train)
        
#         np.savetxt('database/database.txt', updated_db)
#         np.savetxt('database/labels.txt', updated_labels, fmt='%d')
#         with open('database/key_hash_train.json', 'w') as file:
#             file.write(json.dumps(temp))
        
#         print('[INFO] Updating DB...')
#         self.createIndex()
#         return updated_db

    def createIndex(self, data):
        # build the index
        #data = np.loadtxt('database/database.txt', dtype='float32')
        print(data.shape)
        self.index = faiss.IndexFlatL2(self.dims)   
        if self.index.is_trained:
            print('[INFO] Training Completed.')
        data = np.ascontiguousarray(np.array(data).astype('float32'))

        # add vectors to the index
        self.index.add(data)                  

        print(f'[INFO] {self.index.ntotal} Records added to Database.')
        #return self.index
        
    def movetoAve(self):
        # build the index
        mypath = 'image/add/'
        added_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        sep_files = [f.split('_')[0] for f in added_files]
        unique = list(set(sep_files))
        all = []
        for u in unique:
            sub = [f for f in added_files if u == f.split('_')[0]]
            all.append(sub)

        final = []
        for a in all:
            random.shuffle(a)
        final.append(a[0])
        print(final)
        
        for f in final:
            copyfile(mypath+f, f'image_averaged/{f}')

    
    def search(self, df, ind = -1):
        print('[INFO] Searching Database...')
        #df, labels, orig_data_labels_train, key_hash_train = preprocess_excel(self.dims+1, query_path)
        data = np.ascontiguousarray(np.array(df).astype('float32'))
        if ind > 0:
            print('searching individual')
            D, I = self.index.search(data[ind].reshape(1, self.dims), self.k)
        else:
            print('searching all')
            D, I = self.index.search(data, self.k)
        return D, I
    
    # def getLabel(self, D, I):
    #     l = list(self.labels)
    #     #print(l)
    #     #print(len(l))
    #     inv_map = {v: k for k, v in self.key_hash_train.items()}
    #     #print(inv_map)
    #     results = []
    #     int_results = []

    #     for row in I:
    #         values = []
    #         for r in row:
    #             ind = int(r)
    #             #print(ind)
    #             values.append(l[ind])

    #         int_results.append(max(set(values), key=values.count))
    #         results.append(inv_map[max(set(values), key=values.count)])

    #     return int_results
    
        
    def createDataFrame(self, names, scores, orig_data_labels_train, i):
        data_res = pd.DataFrame([names,scores]).T
        data_res = data_res.groupby([0]).mean().reset_index().sort_values(by=1)
        data_res = data_res.head(3).T
        try:
            data_res.columns = ['1st','2nd','3rd']
        except:
            try:
                data_res.columns = ['1st','2nd']
            except:
                data_res.columns = ['1st']
        data_res['label'] = ["Prediction", "Score"]
        labels_test = list(orig_data_labels_train)
        data_res['sample_number'] = str(labels_test[i])
        try:
            data_res = data_res[["sample_number","label", "1st", "2nd", "3rd"]]
        except:
            try:
                data_res = data_res[["sample_number","label", "1st", "2nd"]]
            except:
                data_res = data_res[["sample_number","label", "1st"]]
        return data_res
    
    def getLabel(self, D, I, orig_data_labels_test):
        l = list(self.labels)
        #print(l)
        #print(len(l))
        inv_map = {v: k for k, v in self.key_hash_train.items()}
        #print(inv_map)
        results = []
        int_results = []
        data_frame = []

        for (i,row) in enumerate(I):
            values = []
            scores = []
            names = []
            for (j,r) in enumerate(row):
                ind = int(r)
                #print(i,j)
                #print(D[i][j])
                #print(ind)
                values.append(l[ind])
                names.append(inv_map[l[ind]])
                scores.append(D[i][j])
            
            #print(names, scores, i)
            df = self.createDataFrame(names,scores,list(orig_data_labels_test), i)
            data_frame.append(df)
            int_results.append(max(set(values), key=values.count))
            results.append(inv_map[max(set(values), key=values.count)])

        #return int_results, results 
        return pd.concat(data_frame)