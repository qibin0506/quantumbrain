from quantumbrain.graph import graph
from quantumbrain import debug


class Trainer:
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def train(self, x, y):
        self.model.trainable = True

        pred = self.model(x)
        loss_value = self.loss(y, pred)
        dout = self.loss.backward()

        for layer in graph.layers.values():
            layer.backward_degree = layer.backward_degree_origin

        removed = [self.model.outputs]

        while len(removed) > 0:
            layer = removed.pop()
            self.__run_backward(layer, dout)

            for pre_layer in layer.previous:
                pre_layer.backward_degree -= 1
                if pre_layer.backward_degree == 0:
                    removed.append(pre_layer)

        self.optimizer.apply_gradients()
        return pred, loss_value

    def __run_backward(self, layer, dout):
        if len(layer.next) == 0:
            layer.run_backward(dout)
        else:
            for next_layer in layer.next:
                if len(next_layer.previous) == 1:
                    layer.run_backward(next_layer.grads)
                else:
                    idx = next_layer.previous.index(layer)
                    layer.run_backward(next_layer.grads[idx])

        if debug.debug_mode:
            debug.dump("{}.backward()".format(layer.name))
