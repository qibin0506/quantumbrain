from quantumbrain.graph import graph
from quantumbrain.serialize import serialize, unserialize
from quantumbrain import debug


class Model:
    def __init__(self, inputs, outputs, name=None):
        self.inputs = inputs
        self.outputs = outputs

        self.trainable = False
        self.name = "model" if name is None else name

    def __call__(self, *args, **kwargs):
        return self.call(args[0])

    def call(self, x):
        for layer in graph.layers.values():
            layer.forward_degree = layer.forward_degree_origin
            layer.trainable = self.trainable

        removed = [self.inputs]

        while len(removed) > 0:
            layer = removed.pop()
            self.__run_forward(layer, x)

            for next_layer in layer.next:
                next_layer.forward_degree -= 1
                if next_layer.forward_degree == 0:
                    removed.append(next_layer)

        return self.outputs.out

    def __run_forward(self, layer, x):
        previous = layer.previous

        if len(previous) == 0:
            layer.run_forward(x)
        elif len(previous) == 1:
            layer.run_forward(previous[0].out)

            if debug.debug_mode:
                debug.dump("{}.forward()".format(layer.name))
        else:
            next_input = []
            for item in previous:
                next_input.append(item.out)

            layer.run_forward(next_input)

            if debug.debug_mode:
                debug.dump("{}.forward()".format(layer.name))

    def summary(self):
        print("Model: \"{}\"".format(self.name))
        print("----------------------------------------------------------")
        print("{:30}\t\t{:30}".format("Layer(type)", "Output Shape"))
        print("==========================================================")

        layers = list(graph.layers.values())
        for layer in layers[:len(layers) - 1]:
            layer_name_col = "{}({})".format(layer.name, layer.__class__.__name__)
            print("{:30}\t\t{}".format(layer_name_col, str(layer.shape)))
            print("---------------------------------------------------------")

        last_layer = layers[-1]
        last_layer_name_col = "{}({})".format(last_layer.name, last_layer.__class__.__name__)
        print("{:30}\t\t{}".format(last_layer_name_col, str(last_layer.shape)))

        print("==========================================================")

    def save(self, path):
        serialize(path, graph.params, graph.grads)

    def restore(self, path):
        graph.params, graph.grads = unserialize(path)
