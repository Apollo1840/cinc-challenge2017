import numpy as np
import gc
import tensorflow as tf
from keras import backend as K
import scipy
from sklearn.metrics import confusion_matrix

from .train_model import model_train


def model_eval(X, y, window_size):
    batch = 64
    epochs = 20
    rep = 1         # K fold procedure can be repeated multiple times
    Kfold = 5
    Ntrain = 8528 # number of recordings on training set
    Nsamp = int(Ntrain/Kfold) # number of recordings to take as validation

    # Need to add dimension for training
    X = np.expand_dims(X, axis=2)
    classes = ['A', 'N', 'O', '~']
    Nclass = len(classes)
    cvconfusion = np.zeros((Nclass, Nclass, Kfold*rep))
    cvscores = []
    counter = 0
    # repetitions of cross validation
    for r in range(rep):
        print("Rep %d" % (r+1))
        # cross validation loop
        for k in range(Kfold):
            print("Cross-validation run %d"%(k+1))

            best_ckpt_filename = 'weights-best_k{}_r{}.hdf5'.format(k, r)

            model, Xval, yval = model_train(X, y, window_size, Ntrain, Nsamp, epochs, batch, best_ckpt_filename)

            # Evaluate best trained model
            model.load_weights(best_ckpt_filename)
            ypred = model.predict(Xval)
            ypred = np.argmax(ypred, axis=1)
            ytrue = np.argmax(yval, axis=1)

            cvconfusion[:, :, counter] = confusion_matrix(ytrue, ypred)

            F1 = np.zeros((4, 1))
            for i in range(4):
                F1[i] = 2*cvconfusion[i, i, counter]/(np.sum(cvconfusion[i, :, counter])+np.sum(cvconfusion[:, i, counter]))
                print("F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
            cvscores.append(np.mean(F1) * 100)
            print("Overall F1 measure: {:1.4f}".format(np.mean(F1)))
            K.clear_session()
            gc.collect()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)
            counter += 1

    # Saving cross validation results
    scipy.io.savemat('xval_results.mat', mdict={'cvconfusion': cvconfusion.tolist()})