import data_preprocessing
import numpy as np

# files

DATASET_DIR             = '../dataset/EurosportAll/'

BASE_FILE               = DATASET_DIR + 'girosmallveryslow2'
VIDEO_FILE              = BASE_FILE + '.mp4'

COMPLETE_SKELS_FILE     = DATASET_DIR + 'esqueletosveryslow_complete.mat' 
INCOMPLETE_SKELS_FILE   = COMPLETE_SKELS_FILE.replace('_complete', '')

TO_FRAME                = 5906
FROM_FRAME              = 5895

LABELS                  = np.arange(FROM_FRAME, TO_FRAME + 1)

IMAGES_DIR = '../images/'
SAVE_IMAGES = False

SINGULAR_VAL_THRESHOLD = 90

# frames
#girosmallveryslow2.mp4_features.mat

FEATURES_FILE   = VIDEO_FILE + '_features.mat'
FEATURES_MAT    = data_preprocessing.load_data(FEATURES_FILE)
FEATURES, FEATURES_DEV = data_preprocessing.get_features(FEATURES_MAT, 'features', FROM_FRAME, TO_FRAME)

# skeletons

COMPLETE_SKELS_MAT      = data_preprocessing.load_data(COMPLETE_SKELS_FILE)
INCOMPLETE_SKELS_MAT    = data_preprocessing.load_data(INCOMPLETE_SKELS_FILE)

COMPLETE_SKELS_FRAME, COMPLETE_SKELS_FRAME_DEV          = data_preprocessing.get_features(COMPLETE_SKELS_MAT, 'skeldata', FROM_FRAME, TO_FRAME)
INCOMPLETE_SKELS_FRAME, INCOMPLETE_SKELS_FRAME_DEV      = data_preprocessing.get_features(INCOMPLETE_SKELS_MAT, 'skeldata', FROM_FRAME, TO_FRAME)

COMPLETE_SKELS          = COMPLETE_SKELS_FRAME[1:]
COMPLETE_SKELS_DEV      = COMPLETE_SKELS_FRAME_DEV[1:]

INCOMPELTE_SKELS        = INCOMPLETE_SKELS_FRAME[1:]
INCOMPLETE_SKELS_DEV    = INCOMPLETE_SKELS_FRAME_DEV[1:]

SKELS_JOINTS_CONNECTION = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),  # (2, 16), (5, 17)
]

JOINT_PERCENT_TOLERANCE = 0.4
NUM_JOINTS = 18
NUM_JOINTS_TOLERANCE = 1/18
NUM_JOINTS_CUTOFF = np.ceil(NUM_JOINTS_TOLERANCE * NUM_JOINTS).astype(int)

# missing data completion

MISSING_DATA_MAT_APPROX  = 0.99                 # stop when approx is this close
MISSING_DATA_MAX_ITERATIONS = 10000             # stop when reached this limit
MISSING_DATA_ITER_TOLERANCE = 1e-4              # stop when from one iteration to the next, it changes less than this
