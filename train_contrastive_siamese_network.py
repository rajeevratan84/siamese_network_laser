# USAGE
# python train_contrastive_siamese_network.py --epochs 100 --batch_size 4 --image_dims 128 --embedding_vector 196

# import the necessary packages
from utilities.siamese_network import build_siamese_model
from utilities import metrics
from utilities import config
from utilities import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", required=False,
    default=50,
	help="Training epochs.")
ap.add_argument("-b", "--batch_size", required=False,
    default=4,
	help="Training epochs.")
ap.add_argument("-i", "--image_dims", required=False,
    default=128,
	help="Training epochs.")
ap.add_argument("-v", "--embedding_vector", required=False,
    default=196,
	help="Training epochs.")
args = vars(ap.parse_args())


print()
# load dataset and scale the pixel values to the range of [0, 1]
try:
	trainX =np.load('24_samples_trainX.npy')
	testX = np.load('24_samples_testX.npy')
	trainY =np.load('24_samples_trainY.npy')
	testY = np.load('24_samples_testY.npy')
except ValueError:
	print('[ERROR] Run create_dataset.py first.')

print(f'Train shape - {trainX.shape}')
print(f'Train shape - {testX.shape}')

# Convert to Float
trainX = trainX / 255.0
testX = testX / 255.0

# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(trainX, np.array(trainY))
(pairTest, labelTest) = utils.make_pairs(testX, np.array(testY))

print(trainX.shape)
print(testX.shape)

IMG_SHAPE = trainX[0].shape
#print(f'Image size - {}')

# configure the siamese network
print("[INFO] building siamese network...")

IMG_SHAPE = (int(args['image_dims']), int(args['image_dims']), 1)

#dims = int(args['image_dims'])
imgA = Input(IMG_SHAPE)
imgB = Input(IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE, int(args['embedding_vector']))
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
model = Model(inputs=[imgA, imgB], outputs=distance)

# compile the model
print("[INFO] compiling model...")
model.compile(loss=metrics.contrastive_loss, optimizer="adam")

# train the model
print("[INFO] training model...")
history = model.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=int(args['batch_size']),
	epochs=int(args['epochs']))

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)

# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)