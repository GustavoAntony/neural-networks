import os

MEAN = 'mean'
STD = 'std'
COV = 'cov'
CURR_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = project_root = os.path.dirname(os.path.dirname(CURR_FILE_PATH))
IMAGES_OUTPUTS_FILE_PATH = os.path.join(PROJECT_ROOT, 'docs', 'assets', 'images') 
DATA_OUTPUTS_FILE_PATH = os.path.join(PROJECT_ROOT, 'data')