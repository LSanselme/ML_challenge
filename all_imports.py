# python imports
import time
import os
from argparse import ArgumentParser

# pandas/mpl imports
import pandas as pd
import matplotlib.pyplot as mpl

# local imports
from reader.read_data import mydata, select_for_test
from myprediction.parallel import pool_result
import myprediction.make_predictions as pr

parser = ArgumentParser()
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--use_cache_data', action='store_true', default=False)
parser.add_argument('--use_cache_trainingset', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--plot', action='store_true', default=False)
parser.add_argument('--compute_reality', action='store_true', default=False)
parser.add_argument('--jobs', type=int, default=-1)
# to avoid bug when launching ipython --pylab
parser.add_argument('--pylab', action='store_true', default=False)

args = parser.parse_args()

def in_ipython():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True
