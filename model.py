from quantumbrain.graph import graph
from quantumbrain.serialize import serialize, unserialize
from quantumbrain import debug


class Model:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        self.trainable = False

    def __call__(self, *args, **kwargs):
        return self.call(args[0])

    def call(self, x):
        layer = self.inputs
        layer.trainable = self.trainable

        layer.run_forward(x)
        layer = layer.next[0]

        while layer is not None:
            layer.trainable = self.trainable
            previous = layer.previous

            if len(previous) == 1:
                layer.run_forward(previous[0].out)

                if debug.debug_mode:
                    debug.dump("{}.forward()".format(layer.name))
            else:
                not_forward_root = self.__find_not_forward_layer_root(layer)
                if not_forward_root != layer:
                    layer = not_forward_root
                    continue

                next_input = []
                for item in previous:
                    next_input.append(item.out)

                layer.run_forward(next_input)

                if debug.debug_mode:
                    debug.dump("{}.forward()".format(layer.name))

            if layer is self.outputs:
                break

            layer = layer.next[0]

        for layer in graph.layers.values():
            layer.forwarded = False

        return self.outputs.out

    def __find_not_forward_layer_root(self, layer):
        for previous in layer.previous:
            if not previous.forwarded:
                return self.__find_not_forward_layer_root(previous)

        return layer

    def save(self, path):
        serialize(path, graph.params, graph.grads)

    def restore(self, path):
        graph.params, graph.grads = unserialize(path)
