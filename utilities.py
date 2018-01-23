import os


def format_string(string):
    string = ' __CLINE__ '.join(string.split('\n'))
    return string


def remove_folder(folder):
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))
        print('remove file {}'.format(os.path.join(folder, file)))
