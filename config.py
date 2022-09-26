from glob import glob

## Config
CORES = 16
VOLUMN_SIZE = (384, 384, 64)
SAVE_PATH = 'C:\\Users\\user\\Desktop\\deep-learning\\cervical\\rsna-2022-cervical-spine-fracture-detection\\train_numpy_data\\'
TRAIN_PATH = 'C:\\Users\\user\\Desktop\\deep-learning\\cervical\\rsna-2022-cervical-spine-fracture-detection\\train_images\\'
TEST_PATH = 'C:\\Users\\user\\Desktop\\deep-learning\\cervical\\rsna-2022-cervical-spine-fracture-detection\\test_images\\'
PATIENTS_LISTS = glob('C:\\Users\\user\\Desktop\\deep-learning\\cervical\\rsna-2022-cervical-spine-fracture-detection\\train_images\\*')