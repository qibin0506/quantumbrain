import codecs
import json
import os
import numpy as np
from collections import OrderedDict


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def serialize(path, params, grads):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_content = {"params": params, "grads": grads}
    json.dump(save_content, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'),
              sort_keys=False, cls=NumpyEncoder)


def unserialize(path):
    with open(path, 'r') as f:
        content = json.load(f)
        params = OrderedDict()
        grads = OrderedDict()

        for key in content["params"].keys():
            params[key] = np.array(content["params"][key])

        for key in content["grads"].keys():
            grads[key] = np.array(content["grads"][key])

        return params, grads
