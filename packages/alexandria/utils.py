import os

def makedir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)