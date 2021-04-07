import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = 'OULAD'
DATASET_DIR = os.path.join(WORKING_DIR, 'data/' + DATASET + '/')
# Path of zip file of the raw dataset:
ZIP_PATH = os.path.join(WORKING_DIR, 'data/archive.zip')
