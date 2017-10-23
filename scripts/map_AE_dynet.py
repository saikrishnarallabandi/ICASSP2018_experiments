import sys
sys.path.append('/home/srallaba/hacks/repos/clustergen_steroids')
from dynet_modules.AutoEncoders import *
import dynet as dy
import numpy as np
from math import sqrt
from sklearn import preprocessing
import random

src = sys.argv[1]
src_name = sys.argv[1].split('/')[-1]
enc = sys.argv[2]
bot = sys.argv[3]

arch = str(enc) + 'T' + str(enc) + 'T' + str(enc) + 'T' + str(enc) + 'T'
train_input = np.loadtxt(src)

# Preprocess the data
input_scaler = preprocessing.StandardScaler().fit(train_input)
a = input_scaler.transform(train_input)

num_examples = len(a)
num_toprint = 0.1 * num_examples

# Hyperparameters for the AE
units_input = 50
units_hidden_1 = int(enc)
units_hidden_2 = int(bot)
units_hidden_3 = int(enc)
units_output = 50

#arch = units_input + 'T' + units_hidden_1 + 'T' + units_hidden_2 + 'T' + units_hidden_3 + 'T' + units_output + 'T'

# Instantiate AE and define the loss
m = dy.Model()
ae = AutoEncoder(m, units_input, [units_hidden_1, units_hidden_2, units_hidden_3], units_output, dy.rectify)
trainer = dy.AdamTrainer(m)


# Loop over the training instances and call the mlp
for epoch in range(30):
  train_loss = 0
  count = 1
  random.shuffle(a)
  for (i,o) in zip(a,a):
     dy.renew_cg()
     count += 1
     I = dy.inputTensor(i)
     O = dy.inputTensor(o)
     loss = ae(I,O, 0)
     train_loss += loss.value()
     if count % num_toprint == 1:
     #    print "  Loss at epoch ", epoch, " after " , count, " examples is ", float(train_loss/count)
         ae.save('models/' +  arch) 
     loss.backward()
     if count % 100 == 1:
        trainer.update() 
  print "Train Loss after epoch ", epoch , " : ", float(train_loss/count)
  ae.save('models/' + arch + '_latestepoch')
  print '\n'
