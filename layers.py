from quantumbrain.graph import graph
from quantumbrain.exceptions import DuplicateNameException
from quantumbrain.initializers import get
from quantumbrain.functions import sigmoid, im2col, col2im
import numpy as np


class Layer:
    def __init__(self, channels, name=None):
        self.__build_layer_name(name)
        self.channels = channels

        self.training = True
        self.shape = None
        self.x = None

    def __call__(self, *args, **kwargs):
        graph.layers[self.name] = self
        if len(args) != 0:
            self.call(args[0])

        return self

    def call(self, inputs):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass

    def _get_param(self, name):
        return graph.params[self.__wrap_arg_name(name)]

    def _set_param(self, name, value):
        graph.params[self.__wrap_arg_name(name)] = value

    def _set_grads(self, name, grads):
        graph.grads[self.__wrap_arg_name(name)] = grads

    def _get_grads(self, name):
        return graph.grads[self.__wrap_arg_name(name)]

    def __wrap_arg_name(self, arg_name):
        return self.name + ":" + arg_name

    def __build_layer_name(self, name):
        if name is None:
            self.name = self.__class__.__name__ + "_" + str(len(graph.layers))
        else:
            if graph.layers.get(name) is not None:
                raise DuplicateNameException(name)

            self.name = name


class Input(Layer):
    def __init__(self, input_shape, name=None):
        Layer.__init__(self, input_shape[1], name)
        self.shape = input_shape
        self.__call__()

    def forward(self, x):
        return x

    def backward(self, dout):
        return dout


class Flatten(Layer):
    def __init__(self, name=None):
        Layer.__init__(self, None, name)

    def call(self, inputs):
        # input: (N, C, H, W)
        self.channels = 1
        for item in inputs.shape[1:]:
            self.channels *= item

        self.shape = [inputs.shape[0], self.channels]

    def forward(self, x):
        self.x = x
        # (N, C*H*W)
        x = x.reshape((x.shape[0], -1))
        return x

    def backward(self, dout):
        dx = dout.reshape(self.x.shape)
        return dx


class Dense(Layer):
    def __init__(self, units, kernel_initializer="normal", name=None):
        Layer.__init__(self, units, name)
        self.kernel_initializer = get(kernel_initializer)

    def call(self, inputs):
        # (N, units)
        self.shape = [inputs.shape[0], self.channels]

        # (previous_units, units)
        self._set_param("W", self.kernel_initializer(inputs.channels, self.channels))
        self._set_param("b", np.zeros(self.channels))

    def forward(self, x):
        self.x = x
        W = self._get_param("W")
        b = self._get_param("b")

        # x: (N, previous_units), W: (previous_units, units)
        return np.dot(x, W) + b

    def backward(self, dout):
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self._set_grads("W", dW)
        self._set_grads("b", db)

        W = self._get_param("W")

        dx = np.dot(dout, W.T)
        return dx


class Conv(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 stride=1,
                 pad='same',
                 kernel_initializer="normal",
                 name=None):
        Layer.__init__(self, filters, name)

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = 1 if pad is 'same' else 0
        self.kernel_initializer = get(kernel_initializer)

        self.col = None
        self.col_w = None

    def call(self, inputs):
        # (N, units)
        N, C, H, W = inputs.shape

        out_h = (H + 2 * self.pad - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.kernel_size) // self.stride + 1

        # (N, FN, out_h, out_W)
        self.shape = [inputs.shape[0], self.filters, out_h, out_w]

        # W: (FN, C, FH, FW)
        self._set_param("W", self.kernel_initializer(self.filters, C, self.kernel_size, self.kernel_size))
        self._set_param("b", np.zeros(self.filters))

    def forward(self, x):
        self.x = x
        pW = self._get_param("W")
        pb = self._get_param("b")

        N, C, H, W = x.shape
        FN, C, FH, FW = pW.shape

        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        # (N*out_h*out_w, C*filter_h*filter_w)
        col = im2col(x, FH, FW, self.stride, self.pad)
        # (FN, C*filter_h*filter_w)
        col_w = pW.reshape((FN, -1))
        # (C*filter_h*filter_w, FN)
        col_w = col_w.T

        # (N*out_h*out_w, FN)
        out = np.dot(col, col_w) + pb
        # (N, out_h, out_w, FN)
        out = out.reshape((N, out_h, out_w, -1))
        # (N, FN, out_h, out_w)
        out = out.transpose((0, 3, 1, 2))

        self.x = x
        self.col = col
        self.col_w = col_w

        return out

    def backward(self, dout):
        # dout: (N, FN, out_h, out_w)

        pW = self._get_param("W")

        N, C, H, W = self.x.shape
        FN, C, FH, FW = pW.shape

        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        # (N, out_h, out_w, FN)
        dout = dout.transpose((0, 2, 3, 1))
        # (N*out_h*out_w, FN)
        dout = dout.reshape((N * out_h * out_w, -1))

        db = np.sum(dout, axis=0)

        # dout: (N*out_h*out_w, FN), col: (N*out_h*out_w, C
        # *filter_h*filter_w)
        # (C*filter_h*filter_w, FN)
        dW = np.dot(self.col.T, dout)
        # (FN, C*filter_h*filter_w)
        dW = dW.T
        # (FN, C, FH, FW)
        dW = dW.reshape((FN, C, FH, FW))

        # dout: (N*out_h*out_w, FN), col_w:(C*filter_h*filter_w, FN)
        # (N*out_h*out_w, C*filter_h*filter_w)
        dcol = np.dot(dout, self.col_w.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        self._set_grads("W", dW)
        self._set_grads("b", db)

        return dx


class Dropout(Layer):
    def __init__(self, drop_rate=0, name=None):
        Layer.__init__(self, None, name)
        self.drop_rate = drop_rate
        self.mask = None

    def call(self, inputs):
        self.shape = inputs.shape
        self.channels = inputs.channels

    def forward(self, x):
        if self.training:
            self.mask = np.random.rand(*x.shape) > self.drop_rate
            return x * self.mask
        else:
            return x

    def backward(self, dout):
        if self.training:
            return dout * self.mask
        return dout


class Relu(Layer):
    def __init__(self, name=None):
        Layer.__init__(self, None, name)

    def call(self, inputs):
        self.shape = inputs.shape
        self.channels = inputs.channels

    def forward(self, x):
        self.x = x
        return (abs(x) + x) / 2

    def backward(self, dout):
        dout[abs(self.x) + self.x == 0] = 0
        return dout


class Sigmoid(Layer):
    def __init__(self, name=None):
        Layer.__init__(self, None, name)
        self.sigmoid_x = None

    def call(self, inputs):
        self.shape = inputs.shape
        self.channels = inputs.channels

    def forward(self, x):
        self.x = x
        self.sigmoid_x = sigmoid(x)
        return self.sigmoid_x

    def backward(self, dout):
        return dout * self.sigmoid_x * (1.0 - self.sigmoid_x)
