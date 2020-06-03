# Code from Oliveira et al. (2020)
# Historical demography and climate driven distributional changes in a  
# widespread Neotropical freshwater species with high economic importance.

#Script by Manolo Perez
#Loads SNP data as images and uses to train a Convolution Neural Network

#Script modified from Flagel et al. (2019) 
#Lex Flagel, Yaniv Brandvain, Daniel R Schrider, The Unreasonable Effectiveness of 
#Convolutional Neural Networks in Population Genetic Inference, Molecular Biology and 
#Evolution, Volume 36, Issue 2, February 2019, Pages 220–238, https://doi.org/10.1093/molbev/msy224

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras import backend as K
#from sklearn.preprocessing import StandardScaler
from random import shuffle, choice

batch_size = 250
epochs = 20
num_classes = 3

u1 = np.load("trainingSims/simModel1.npz")
u3 = np.load("trainingSims/simModel2.npz")
u4 = np.load("trainingSims/simModel3.npz")
x=np.concatenate((u1['simModel1'],u3['simModel2'],u4['simModel3']),axis=0)

y=[0 for i in xrange(20000)]
y.extend([1 for i in xrange(20000)])
y.extend([2 for i in xrange(20000)])
y = np.array(y)

print len(x), len(y)
shf = range(len(x))
shuffle(shf)

y = y[shf]
x = x[shf]

xtrain, xtest = x[2000:], x[:2000]
ytrain, ytest = y[2000:], y[:2000]

ytest = keras.utils.to_categorical(ytest, num_classes)
ytrain = keras.utils.to_categorical(ytrain, num_classes)

model = Sequential()
model.add(Conv1D(250, kernel_size=2,
                 activation='relu',
                 input_shape=(xtest.shape[1], xtest.shape[2])))
model.add(Conv1D(125, kernel_size=2, activation='relu'))
model.add(AveragePooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(125, kernel_size=2, activation='relu'))
model.add(AveragePooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(125, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(125, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),# lr=0.001 -> loss 0.7170 and acc 0.6840 after 20 epochs
              metrics=['accuracy'])
print(model.summary())
model.fit(xtrain, ytrain, batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xtest, ytest))


model.save(filepath='big.data.89.2.acc.mod')
