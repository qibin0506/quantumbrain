from quantumbrain.graph import graph
from quantumbrain.serialize import serialize, unserialize


class Model:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, *args, **kwargs):
        return self.call(args[0])

    def call(self, x):
        for layer in graph.layers.values():
            x = layer.forward(x)

        return x

    def save(self, path):
        serialize(path, graph.params, graph.grads)

    def restore(self, path):
        graph.params, graph.grads = unserialize(path)
