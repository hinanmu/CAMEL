import os
import shutil
import numpy as np

def path_exists(path, overwrite=False):
    if not os.path.isdir(path):
        os.mkdir(path)
    elif overwrite == True :
        shutil.rmtree(path)
    return path

def remove_dir(path):
    os.rmdir(path)
    return True
