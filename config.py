## Config
PATH = 'C:\\Users\\user\\Desktop\\deep-learning\\cervical\\rsna-2022-cervical-spine-fracture-detection\\'
TRAIN_IMAGE_PATH = 'C:\\Users\\user\\Desktop\\deep-learning\\cervical\\rsna-2022-cervical-spine-fracture-detection\\train_images\\'
TEST_IMAGE_PATH = 'C:\\Users\\user\\Desktop\\deep-learning\\cervical\\rsna-2022-cervical-spine-fracture-detection\\test_images\\'

CORES = 16
IMAGE_SIZE = (384, 384)

# Hyperparameters
N_FOLDS = 5
N_EPOCHS = 5
PATIENCE = 3
BATCH_SIZE = 64
MODEL_NAME = 'efficientnet_b0'
NUM_WORKERS = 8
LEARNING_RATE = 0.001
FRAC_LOSS_WEIGHT = 2.