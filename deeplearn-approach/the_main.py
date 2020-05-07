import matplotlib.pyplot as plt
import itertools
import scipy
import numpy as np

import tensorflow as tf

from cross_validation import model_eval


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm = np.around(cm, decimals=3)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion.eps', format='eps', dpi=1000)


def loaddata(window_size):
    '''
        Load training/test data into workspace

        This function assumes you have downloaded and padded/truncated the
        training set into a local file named "trainingset.mat". This file should
        contain the following structures:
            - trainset: NxM matrix of N ECG segments with length M
            - traintarget: Nx4 matrix of coded labels where each column contains
            one in case it matches ['A', 'N', 'O', '~'].

    '''
    print("Loading data training set")
    matfile = scipy.io.loadmat('trainingset.mat')
    X = matfile['trainset']
    y = matfile['traintarget']

    # Merging datasets
    # Case other sets are available, load them then concatenate
    # y = np.concatenate((traintarget, augtarget),axis=0)
    # X = np.concatenate((trainset, augset),axis=0)

    # kind of duplicate work, but it can make sure that dimension matches
    X = X[:, 0:window_size]
    return (X, y)


if __name__ == "__main__":
    classes = ['A', 'N', 'O', '~']

    # traing environment setting
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    seed = 7
    np.random.seed(seed)

    # Parameters
    FS = 300
    WINDOW_SIZE = 30 * FS  # padding window for CNN

    # Loading data
    (X_train, y_train) = loaddata(WINDOW_SIZE)

    # Training model
    model_eval(X_train, y_train, WINDOW_SIZE)

    # show results
    # Outputing results of cross validation
    matfile = scipy.io.loadmat('xval_results.mat')
    cv = matfile['cvconfusion']
    F1mean = np.zeros(cv.shape[2])
    for j in range(cv.shape[2]):
        classes = ['A', 'N', 'O', '~']
        F1 = np.zeros((4, 1))
        for i in range(4):
            F1[i] = 2 * cv[i, i, j] / (np.sum(cv[i, :, j]) + np.sum(cv[:, i, j]))
            print("F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
        F1mean[j] = np.mean(F1)
        print("mean F1 measure for: {:1.4f}".format(F1mean[j]))
    print("Overall F1 : {:1.4f}".format(np.mean(F1mean)))

    # Plotting confusion matrix
    cvsum = np.sum(cv, axis=2)
    F1 = np.zeros((4, 1))
    for i in range(4):
        F1[i] = 2 * cvsum[i, i] / (np.sum(cvsum[i, :]) + np.sum(cvsum[:, i]))
        print("F1 measure for {} rhythm: {:1.4f}".format(classes[i], F1[i, 0]))
    F1mean = np.mean(F1)
    print("mean F1 measure for: {:1.4f}".format(F1mean))
    plot_confusion_matrix(cvsum, classes, normalize=True, title='Confusion matrix')


