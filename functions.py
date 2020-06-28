import numpy as np


def one_hot(x):
    return (np.arange(np.max(x) + 1) == x[:, None]).astype(np.integer)


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy_error(y_true, y_pred):
    delta = 1e-7
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, y_true.size)
        y_pred = y_pred.reshape(1, y_pred.size)

    batch_size = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + delta)) / batch_size


def im2col(img, filter_h, filter_w, stride, pad):
    N, C, H, W = img.shape

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(img, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose((0, 4, 5, 1, 2, 3))
    col = col.reshape((N*out_h*out_w, -1))

    return col


def col2im(col, input_shape, filter_h, filter_w, stride, pad):
    N, C, H, W = input_shape

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    col = col.reshape((N, out_h, out_w, C, filter_h, filter_w))
    col = col.transpose((0, 3, 4, 5, 1, 2))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]

    img = img[:, :, pad:H+pad, pad:W+pad]
    return img