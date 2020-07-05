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

        layers = [self.model.outputs]
        douts = [dout]

        while len(layers) > 0:
            layer = layers.pop()
            dout = douts.pop()
            dout = layer.execute_backward(dout)

            if debug.debug_mode:
                debug.dump("{}.backward()".format(layer.name))

            if layer == self.model.inputs:
                continue

            if len(layer.previous) == 1:
                layers.append(layer.previous[0])
                douts.append(dout)
            else:
                layers.extend(layer.previous)
                douts.extend(dout)

        self.optimizer.apply_gradients()

        return pred, loss_value
