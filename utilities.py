import shutil
import os


def format_string(string):
    string = ' __CLINE__ '.join(string.split('\n'))
    return string


def remove_folder(folder):
    shutil.rmtree(folder)
    os.mkdir(folder)
