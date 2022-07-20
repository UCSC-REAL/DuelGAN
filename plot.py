import argparse
import os
import copy
import pprint
from os import path

import torch
import numpy as np
from torch import nn

import json
import pandas as pd

import matplotlib.pyplot as plt 


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Path to config file.')
args = parser.parse_args()


json_path = os.path.join(args.path,'fid_results.json')
with open(json_path) as json_file:
    data = json.load(json_file)

plt_x = []
plt_y = []
for x in data:
  plt_x.append(int(x)*0.001)
  plt_y.append(data[x])
new_x, new_y = zip(*sorted(zip(plt_x, plt_y)))
plt.plot(new_x, new_y)

plt.savefig(os.path.join(args.path,'fid_plot.png'))



