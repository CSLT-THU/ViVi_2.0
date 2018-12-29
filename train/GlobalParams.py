"""Load arrays or pickled objects from .npy, .npz or pickled files."""

import os
import numpy as np

params = {}
init_token = ''


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in pp.iteritems():
        params[kk] = pp[kk]

    return params


load_params(os.path.abspath('../resource/initialvalue_resource/randomvalue.npz'), params)
