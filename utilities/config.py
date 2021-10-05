# import the necessary packages
import os

# time series to imgage paramters
IMAGE_TYPE = "GramianAngularField"
RESIZED_DIMS = 128
FIRST_X_COLUMS = 500

# specify the shape of the inputs for our network
IMG_SHAPE = (128, 128, 1)

# specify the batch size and number of epochs
BATCH_SIZE = 8
EPOCHS = 80

# define the path to the base output directory
BASE_OUTPUT = "output"

# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT,
	"contrastive_siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT,
	"contrastive_plot.png"])

PLOT_PATH_1 = os.path.sep.join([BASE_OUTPUT,
	"confusion_matrix_1.png"])

PLOT_PATH_2 = os.path.sep.join([BASE_OUTPUT,
	"confusion_matrix_2.png"])