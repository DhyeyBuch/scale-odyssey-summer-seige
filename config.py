from pathlib import Path

# PATHS

DATA_ROOT = Path('data')

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

OUTPUT_DIR      = Path('outputs')
MODEL_DIR       = OUTPUT_DIR / 'models'
BEST_MODEL_PATH = MODEL_DIR / 'best_model.pth'

# CLASSES

CLASS_NAMES = ['elliptical', 'nebula', 'planetary', 'spiral', 'star_cluster']
NUM_CLASSES = len(CLASS_NAMES)

# Model 

BACKBONE   = 'resnet50'
HIDDEN_DIM = 512
DROPOUT    = 0.4

# Hyperparameters

BATCH_SIZE = 32
NUM_EPOCHS = 20
PATIENCE   = 7

# Different learning rates for each layer

LEARNING_RATES = {
    'layer1': 1e-6,
    'layer2': 1e-6,
    'layer3': 5e-5,
    'layer4': 1e-4,
    'fc'    : 5e-4,
}
WEIGHT_DECAY   = 1e-4
GRAD_CLIP_NORM = 1.0
LR_ETA_MIN     = 1e-7   

# Transforms for Image

IMAGE_SIZE    = 224   # ResNet-50 input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Data Augmentation

COLORJITTER_BRIGHTNESS = 0.4
COLORJITTER_CONTRAST   = 0.4
COLORJITTER_SATURATION = 0.4
COLORJITTER_HUE        = 0.1
RANDOM_GRAYSCALE_P     = 0.2

# Data Loaders

NUM_WORKERS        = 4
PIN_MEMORY         = True
PERSISTENT_WORKERS = True

# GradCAM

GRADCAM_LAYER = 'layer4'
GRADCAM_ALPHA = 0.4

SEED = 42

# CREATE OUTPUT DIRECTORIES

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
