import keras
from keras.layers import Input, Dense
from keras.constraints import maxnorm
from keras.layers.core import Dropout
from keras.optimizers import SGD
from keras.models import Model, Sequential
import numpy as np
from keras.models import load_model
import os,sys
from sklearn import preprocessing
import pickle, logging
from keras.callbacks import *


src = sys.argv[1]
src_name = sys.argv[1].split('/')[-1]
#tgt = sys.argv[2]
enc = sys.argv[2]

arch = enc + 'T' + enc + 'T' + enc + 'T' + enc + 'T'

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn="print"):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
 
        # If  first epoch, remove the log file
        if epoch == 0:
            g = open('logs/logs_' + arch + '_scaled.txt','w')
            g.close()

        # Log the progress
        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)
        with open('logs/logs_inputscaled_ouptutscaled_inputdropout_onlymlsamceps_' + src_name.split('.')[0] + '_' + arch + '.txt','a') as g:
            g.write(msg + '\n')
        
        # Save the model every 5 epochs
        #if epoch % 5 == 1:
        #     print self.model
        #     self.model.save('models/feature_mapper_' + arch + '.h5')            



def source_encoder():
   inp_dim = 50
   encoding_dim=int(enc)
   print "Building an NN encoder with encoding dimesions ", enc, " and input dimensions ", inp_dim
   A = np.loadtxt( src, usecols=range(inp_dim))
   #B = np.loadtxt( src, usecols=range(inp_dim,inp_dim*2))
   
   # Preprocessing
   input_scaler = preprocessing.StandardScaler().fit(A)
   a = input_scaler.transform(A)
   #output_scaler = preprocessing.StandardScaler().fit(B)
   #b = output_scaler.transform(B)
   pickle.dump(input_scaler, open('models/frame_mapper_' + src_name +   '_input_scaler', 'wb'))
   #pickle.dump(output_scaler, open('models/frame_mapper_' + src_name +  '_output_scaler', 'wb'))
   
   # Create the model	
   model = Sequential()
   model.add(Dropout(0.0, input_shape=(inp_dim,)))
   model.add(Dense(inp_dim,kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3)))
   model.add(Dense(encoding_dim, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3)))  
   model.add(Dense(encoding_dim, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3)))
   model.add(Dense(encoding_dim, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3)))
   model.add(Dense(encoding_dim, kernel_initializer='normal', activation='tanh', kernel_constraint=maxnorm(3)))
   model.add(Dense(inp_dim, kernel_initializer='normal', activation='tanh'))

   # Compile the model
   sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False)
   model.compile(optimizer=sgd, loss='mse')
   model.fit(a,a,epochs=40, batch_size=32, shuffle=True, callbacks=[LoggingCallback(logging.info)])
   model.save('models/frame_mapper_' + src_name + '_' + arch + '.h5')
   pickle.dump(input_scaler, open('models/frame_mapper_' + src_name +   '_input_scaler', 'wb'))
   pickle.dump(output_scaler, open('models/frame_mapper_' + src_name +  '_output_scaler', 'wb'))
   return model, input_scaler, output_scaler



q,w,e = source_encoder()
