#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Convert multiple files from Physionet/Computing in Cardiology challenge into 
file single matrix. As input argument 

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


import scipy.io
import numpy as np
import os
import glob
from tqdm import tqdm
import csv

# Parameters
input_data_dir = './some_path/'  # <---- change!!
output_data_dir = './some_path/'

FS = 300
WINDOW_SIZE = 60*FS


def load_mats_from_dir(data_dir):
    files = sorted(glob.glob(data_dir + "*.mat"))
    signals = []
    for i in tqdm(range(len(files))):
        try:
            # record = f[:-4]
            # record = record[-6:]

            # Loading
            mat_data = scipy.io.loadmat(files[i][:-4] + ".mat")
            # print('Loading record {}'.format(record))

            signal = mat_data['val'].squeeze()

            # Preprocessing
            # print('Preprocessing record {}'.format(record))
            signals.append(np.nan_to_num(signal))  # removing NaNs and Infs

        except Exception:
            print(files[i])

    return signals


def array_from_signals(signals, window_size, normalize=True):
    trainset = np.zeros((len(signals), window_size))

    for i in tqdm(range(len(signals))):
        data = signals[i]
        if normalize:
            data = data - np.mean(data)
            data = data / np.std(data)

        trainset[i, :min(window_size, len(data))] = data[:min(window_size, len(data))].T  # padding sequence

    return trainset


def load_array_from_dir(data_dir, window_size):
    """
    preprocess while loading, can save some RAM

    :param data_dir:
    :param window_size:
    :return:
    """

    # Loading time serie signals
    files = sorted(glob.glob(data_dir + "*.mat"))
    trainset = np.zeros((len(files), window_size))

    count = 0
    for f in tqdm(files):
        try:
            # record = f[:-4]
            # record = record[-6:]

            # Loading
            mat_data = scipy.io.loadmat(f[:-4] + ".mat")
            # print('Loading record {}'.format(record))

            data = mat_data['val'].squeeze()

            # Preprocessing
            # print('Preprocessing record {}'.format(record))
            data = np.nan_to_num(data)  # removing NaNs and Infs

            # zero mean unit variance
            data = data - np.mean(data)
            data = data/np.std(data)

            trainset[count, :min(window_size, len(data))] = data[:min(window_size, len(data))].T  # padding sequence
            count += 1

        except Exception:
            print(f)

    return trainset


# Loading labels
def load_ann_from_dir(data_dir):
    csvfile = list(csv.reader(open(data_dir+'REFERENCE.csv')))

    traintarget = np.zeros((len(csvfile), 4))
    classes = ['A', 'N', 'O', '~']
    for row in range(len(csvfile)):
        traintarget[row, classes.index(csvfile[row][1])] = 1
    return traintarget


if __name__ == "__main__":
    trainset = load_array_from_dir(input_data_dir, WINDOW_SIZE)
    traintarget = load_ann_from_dir(input_data_dir)

    assert len(trainset) == len(traintarget)

    # Saving both
    scipy.io.savemat(os.path.join(output_data_dir, 'trainingset.mat'),
                     mdict={'trainset': trainset, 'traintarget': traintarget})
