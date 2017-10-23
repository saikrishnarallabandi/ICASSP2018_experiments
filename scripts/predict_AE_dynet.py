import sys, os
sys.path.append('/home/srallaba/hacks/repos/clustergen_steroids')
from dynet_modules.AutoEncoders import *
import dynet as dy
import numpy as np
from math import sqrt
from sklearn import preprocessing
import random

src = sys.argv[1]
#src_name = sys.argv[1].split('/')[-1]
enc = sys.argv[2]
bot = sys.argv[3]

arch = str(enc) + 'T' + str(enc) + 'T' + str(enc) + 'T' + str(enc) + 'T'


# Hyperparameters for the AE
units_input = 50
units_hidden_1 = int(enc)
units_hidden_2 = int(bot)
units_hidden_3 = int(enc)
units_output = 50


# Instantiate AE and define the loss
m = dy.Model()
ae = AutoEncoder(m, units_input, [units_hidden_1, units_hidden_2, units_hidden_3], units_output, dy.rectify)

# Load the model
ae.load(m, 'models/model_' + arch)
