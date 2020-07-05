debug_mode = False


def dump(msg):
    if debug_mode:
        print("quantumbrain:{}".format(msg))