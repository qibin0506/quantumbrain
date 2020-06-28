class DuplicateNameException(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "duplicate name {}".format(self.name)