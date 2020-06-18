'''
This function  function used for training and cross-validating model using. The database is not 
included in this repo, please download the CinC Challenge database and truncate/pad data into a 
NxM matrix array, being N the number of recordings and M the window accepted by the network (i.e. 
30 seconds).


For more information visit: https://github.com/fernandoandreotti/cinc-challenge2017
 
 Referencing this work
   Andreotti, F., Carr, O., Pimentel, M.A.F., Mahdi, A., & De Vos, M. (2017). Comparing Feature Based 
   Classifiers and Convolutional Neural Networks to Detect Arrhythmia from Short Segments of ECG. In 
   Computing in Cardiology. Rennes (France).
--
 cinc-challenge2017, version 1.0, Sept 2017
 Last updated : 27-09-2017
 Released under the GNU General Public License
 Copyright (C) 2017  Fernando Andreotti, Oliver Carr, Marco A.F. Pimentel, Adam Mahdi, Maarten De Vos
 University of Oxford, Department of Engineering Science, Institute of Biomedical Engineering
 fernando.andreotti@eng.ox.ac.uk
   
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''


import numpy as np
import sys
sys.path.insert(0, './preparation')

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.callbacks import Callback

import warnings

from model import ResNet_model


class AdvancedLearnignRateScheduler(Callback):
    '''
   # Arguments
       monitor: quantity to be monitored.
       patience: number of epochs with no improvement
           after which training will be stopped.
       verbose: verbosity mode.
       mode: one of {auto, min, max}. In 'min' mode,
           training will stop when the quantity
           monitored has stopped decreasing; in 'max'
           mode it will stop when the quantity
           monitored has stopped increasing.
   '''

    def __init__(self, monitor='val_loss', patience=0,verbose=0, mode='auto', decayRatio=0.1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decayRatio = decayRatio

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Mode %s is unknown, '
                          'fallback to auto mode.'
                          % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current_lr = K.get_value(self.model.optimizer.lr)
        print("\nLearning rate:", current_lr)
        if current is None:
            warnings.warn('AdvancedLearnignRateScheduler'
                          ' requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: reducing learning rate' % (epoch))
                    assert hasattr(self.model.optimizer, 'lr'), \
                        'Optimizer must have a "lr" attribute.'
                    current_lr = K.get_value(self.model.optimizer.lr)
                    new_lr = current_lr * self.decayRatio
                    K.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0
            self.wait += 1


def model_train(X, y, window_size, test_ratio, epochs, batch, best_ckpt_filename):

    # Callbacks definition
    callbacks = [
        # Early stopping definition
        EarlyStopping(monitor='val_loss', patience=3, verbose=1),
        # Decrease learning rate by 0.1 factor
        AdvancedLearnignRateScheduler(monitor='val_loss', patience=1, verbose=1, mode='auto', decayRatio=0.1),
        # Saving best model
        ModelCheckpoint(best_ckpt_filename, monitor='val_loss', save_best_only=True, verbose=1),
    ]
    # Load model
    model = ResNet_model(window_size)
    model.summary()

    # split train and validation sets
    # alternative way of using shuffle
    idxval = np.random.choice(X.shape[0], int(X.shape[0] * test_ratio), replace=False)
    idxtrain = np.invert(np.in1d(range(X.shape[0]), idxval))

    ytrain = y[np.asarray(idxtrain), :]
    Xtrain = X[np.asarray(idxtrain), :, :]
    Xval = X[np.asarray(idxval), :, :]
    yval = y[np.asarray(idxval), :]

    # Train model
    model.fit(Xtrain, ytrain,
              validation_data=(Xval, yval),
              epochs=epochs, batch_size=batch, callbacks=callbacks)

    return model, Xval, yval
