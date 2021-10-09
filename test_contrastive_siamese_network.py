# USAGE
# python test_contrastive_siamese_network.py

# import the necessary packages
from utilities import config
from utilities import utils
from tensorflow.keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pre_trained", required=False,
    default='no',
	help="Specifiy whether you want to use the pretrained model or not")
args_path = vars(ap.parse_args())

args = {
    "input": "examples",
    "test": "images/test",
    "train": "images/train",
    "val": "images/val"
}
from os import listdir
from os.path import isfile, join
import random

onlyfiles = [f for f in listdir(args["test"]) if isfile(join(args["test"], f))]

test_examples = []
for i in range(0,20):
    test_examples.append((args["test"]) + '/' + random.choice(onlyfiles))

# grab the test dataset image paths and then randomly generate a
# total of 20 image pairs
print("[INFO] loading test dataset...")

np.random.seed(42)
pairs = np.random.choice(test_examples, size=(20, 2))

# load the model from disk
print("[INFO] loading siamese model...")

if args_path['pre_trained'] == 'yes':
  model = load_model('pretrained_siamese_model', compile=False)
else:
  model = load_model(config.MODEL_PATH, compile=False)

ImagePathsTrain = list(list_images(args["train"]))
ImagePathsTest = list(list_images(args["test"]))
ImagePathsVal = list(list_images(args["val"]))
os.makedirs("comparison_plots", exist_ok=True) 
# loop over all image pairs
for (i, (pathA, pathB)) in enumerate(pairs):
    # load both the images and convert them to grayscale
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

    # initialize the figure
    fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 2))
    nameA = pathA.split('/')[2].split('.')[0] #pathA.split('_')[0].split('/')[2]
    nameB = pathB.split('/')[2].split('.')[0] #pathB.split('_')[0].split('/')[2]
    
    plt.suptitle(f'{nameA} compared to {nameB}. Similarity: {proba*100:.2f}%')

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(origA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(origB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the plot
    fig.savefig(f'comparison_plots/comparison_{nameA}_{nameB}_{i}.png', dpi=fig.dpi)


