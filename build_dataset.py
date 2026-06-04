import os
import shutil
from pathlib import Path
from config import DATA_ROOT, SEED

# Note that these data source paths are the ones I used in the Kaggle Notebook. To Reproduce
# the results, update the path links with the local links of where you have stored the data.

GALAXY_ZOO_DIR = Path('/kaggle/input/datasets/anjosut/galaxy-zoo-classification/Train_images/Train_images')
SPACENET_DIR   = Path('/kaggle/input/datasets/dhyeybuch/astronomy-final-dataset/final_dataset')

GALAXY_MAP = {
    'Cigar-shaped smooth'    : 'elliptical',
    'In between smooth'      : 'elliptical',
    'completely round smooth': 'elliptical',
    'spiral'                 : 'spiral',
}

SPACENET_CLASSES = ['nebula', 'planetary', 'star_cluster']

FINAL_DIR = DATA_ROOT / 'final'


def build():

    for cls in ['elliptical', 'spiral'] + SPACENET_CLASSES:
        (FINAL_DIR / cls).mkdir(parents=True, exist_ok=True)

    # Copying the necessary files from Galaxy Zoo Dataset

    copied = 0
    for folder_name, class_name in GALAXY_MAP.items():
        src_folder = GALAXY_ZOO_DIR / folder_name

        for image in src_folder.iterdir():
            new_name = f'{class_name}_{image.name}'
            shutil.copy2(image, FINAL_DIR / class_name / new_name)
            copied += 1

            if copied % 5000 == 0:
                print(f'  Copied {copied} images so far')

    print('Galaxy Zoo done.')

    # Copying the necessary files from SpaceNet Dataset


    copied = 0
    for class_name in SPACENET_CLASSES:
        src_folder = SPACENET_DIR / class_name
  
        for image in src_folder.iterdir():
            new_name = f'{class_name}_{image.name}'
            shutil.copy2(image, FINAL_DIR / class_name / new_name)
            copied += 1

            if copied % 1000 == 0:
                print(f'  Copied {copied} images so far...')

    print(f'SpaceNet done.')


if __name__ == '__main__':
    build()