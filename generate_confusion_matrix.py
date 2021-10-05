# USAGE
# python test_contrastive_siamese_network.py --input examples

# import the necessary packages
from utilities import config
from utilities import utils
from tensorflow.keras.models import load_model
from imutils.paths import list_images
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import time
from os import listdir
from os.path import isfile, join
import random

args = {
    "input": "examples",
    "test": "images/test",
    "train": "images/train",
    "val": "images/val"
}

print("[INFO] loading siamese model...")
model = load_model(config.MODEL_PATH, compile=False)

GTs = []
CW = []
PR = []

ImagePathsTrain = list(list_images(args["train"]))
ImagePathsTest = list(list_images(args["test"]))
ImagePathsVal = list(list_images(args["val"]))

start = time.time()

# loop over all image pairs
for (i, pathA) in enumerate(ImagePathsTest):
  # load both the images and convert them to grayscale
  for (j, pathB) in enumerate(ImagePathsTrain):
    imageA = cv2.imread(pathA, 0)
    imageB = cv2.imread(pathB, 0)

    # create a copy of both the images for visualization purpose
    origA = imageA.copy()
    origB = imageB.copy()

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
    ground_truth = int(pathA.split('_')[0].split('/')[2])
    compared_with = int(pathB.split('_')[0].split('/')[2])    
    print(f'{i} - {pathA}, {pathB}, {ground_truth}, {compared_with}, {proba}')  
    GTs.append(pathA)
    
    CW.append(pathB)
    PR.append(proba)
  
end = time.time()
print(f'[INFO] Execution time = {end - start} seconds')


df_summary = pd.DataFrame(
    {'Ground Truth': GTs,
     'Compared With': CW,
     'Similarity': PR
    })
df_summary    
    
top_k = 10

df_group = df_summary.groupby(['Ground Truth', 'Compared With']).mean().sort_values(by='Similarity')
df_agg = df_summary.groupby(['Ground Truth', 'Compared With']).agg({'Similarity':sum})
g = df_agg['Similarity'].groupby('Ground Truth', group_keys=False)
res = g.apply(lambda x: x.sort_values().head(10))
res = res.reset_index()

res['ID'] = res['Ground Truth'].apply(lambda x: int(x.split('_')[0].split('/')[2]))
res['Predicted'] = res['Compared With'].apply(lambda x: int(x.split('_')[0].split('/')[2]))

final_results = res.groupby(['Ground Truth']).agg(lambda x:x.value_counts().index[0]).reset_index()

target_names = list(range(0,10))
conf_mat = confusion_matrix(final_results['ID'], final_results['Predicted'])
utils.plot_confusion_matrix(conf_mat, config.PLOT_PATH, target_names)


df_agg = df_summary.groupby(['Ground Truth', 'Compared With']).agg({'Similarity':sum})
g = df_agg['Similarity'].groupby('Ground Truth', group_keys=False)
res_2 = g.apply(lambda x: x.sort_values())
res_2 = res_2.reset_index()

res_2['ID'] = res_2['Ground Truth'].apply(lambda x: x.split('/')[2].split('.')[0])
res_2['Predicted'] = res_2['Compared With'].apply(lambda x: x.split('_')[0].split('/')[2])
t = res_2.groupby(['ID', 'Predicted']).mean().reset_index()
out = t.sort_values('Similarity',ascending = False).groupby('ID').tail(1).sort_values(by='ID')
out['GT'] = out['ID'].apply(lambda x: int(x.split('_')[0])).astype(str)
conf_mat = confusion_matrix(out['GT'], out['Predicted'])
utils.plot_confusion_matrix(conf_mat, config.PLOT_PATH, target_names)