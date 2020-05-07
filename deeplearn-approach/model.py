import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Flatten, Dropout,MaxPooling1D, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K


#####################################
## Model definition              ##
## ResNet based on Rajpurkar    ##
##################################
def ResNet_model(WINDOW_SIZE):
    # Add CNN layers left branch (higher frequencies)
    # Parameters from paper
    INPUT_FEAT = 1
    OUTPUT_CLASS = 4    # output classes

    k = 1    # increment every 4th residual block
    p = True # pool toggle every other residual block (end with 2^8)
    convfilt = 64
    convstr = 1
    ksize = 16
    poolsize = 2
    poolstr  = 2
    drop = 0.5

    # Modelling with Functional API
    #input1 = Input(shape=(None,1), name='input')
    input1 = Input(shape=(WINDOW_SIZE,INPUT_FEAT), name='input')

    ## First convolutional block (conv,BN, relu)
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(input1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 =  Conv1D(filters=convfilt,
                 kernel_size=ksize,
                 padding='same',
                 strides=convstr,
                 kernel_initializer='he_normal')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(drop)(x1)
    x1 =  Conv1D(filters=convfilt,
                 kernel_size=ksize,
                 padding='same',
                 strides=convstr,
                 kernel_initializer='he_normal')(x1)
    x1 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr)(x1)
    # Right branch, shortcut branch pooling
    x2 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr)(x)
    # Merge both branches
    x = keras.layers.add([x1, x2])
    del x1,x2

    ## Main loop
    p = not p
    for l in range(15):

        if (l%4 == 0) and (l>0): # increment k on every fourth residual block
            k += 1
            # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=convfilt*k,kernel_size=1)(x)
        else:
            xshort = x
            # Left branch (convolutions)
        # notice the ordering of the operations has changed
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
                     kernel_size=ksize,
                     padding='same',
                     strides=convstr,
                     kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
                     kernel_size=ksize,
                     padding='same',
                     strides=convstr,
                     kernel_initializer='he_normal')(x1)
        if p:
            x1 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(x1)

            # Right branch: shortcut connection
        if p:
            x2 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(xshort)
        else:
            x2 = xshort  # pool or identity
        # Merging branches
        x = keras.layers.add([x1, x2])
        # change parameters
        p = not p # toggle pooling


    # Final bit
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    #x = Dense(1000)(x)
    #x = Dense(1000)(x)
    out = Dense(OUTPUT_CLASS, activation='softmax')(x)
    model = Model(inputs=input1, outputs=out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #model.summary()
    #sequential_model_to_ascii_printout(model)
    plot_model(model, to_file='model.png')
    return model
