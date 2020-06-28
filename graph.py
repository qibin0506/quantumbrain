from collections import OrderedDict


class Graph:
    def __init__(self):
        self.layers = OrderedDict()
        self.params = OrderedDict()
        self.grads = OrderedDict()


graph = Graph()
