try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')

import os


def download(url_base, dataset_dir, file_name):
    file_path = dataset_dir + "/" + file_name
    print(file_path)

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " from " + (url_base + file_name) + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")