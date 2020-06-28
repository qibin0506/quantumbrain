import numpy as np
import os
from quantumbrain.datasets.downloader import download


def load_data(test_split=0.2, seed=113):
  assert 0 <= test_split < 1
  base_url = 'https://github.com/yuxiwang66/boston_housing/raw/master/'
  dataset_dir = os.path.dirname(os.path.abspath(__file__))
  file_name = "boston_housing.npz"
  save_file = dataset_dir + "/{}".format(file_name)

  download(base_url, dataset_dir, file_name)

  with np.load(save_file, allow_pickle=True) as f:
    x = f['x']
    y = f['y']

  np.random.seed(seed)
  indices = np.arange(len(x))
  np.random.shuffle(indices)
  x = x[indices]
  y = y[indices]

  x_train = np.array(x[:int(len(x) * (1 - test_split))])
  y_train = np.array(y[:int(len(x) * (1 - test_split))])
  x_test = np.array(x[int(len(x) * (1 - test_split)):])
  y_test = np.array(y[int(len(x) * (1 - test_split)):])

  y_train = y_train.reshape(y_train.size, 1)
  y_test = y_test.reshape(y_test.size, 1)

  return (x_train, y_train), (x_test, y_test)
