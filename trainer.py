from quantumbrain.graph import graph


class Trainer:
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def train(self, x, y):
        pred = self.model(x)
        loss_value = self.loss(y, pred)

        layers = list(graph.layers.values())
        layers.reverse()

        dout = self.loss.backward()
        for layer in layers:
            dout = layer.backward(dout)

        self.optimizer.apply_gradients()

        return pred, loss_value

