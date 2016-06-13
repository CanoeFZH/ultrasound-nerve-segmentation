from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Flatten,Dense,Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.cross_validation import train_test_split
from data import load_train_data, load_test_data
seed = 1024
np.random.seed(seed)

img_rows = 227
img_cols = 227

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((1, img_rows, img_cols))
    #55
    conv1 = Convolution2D(96, 11, 11,subsample=(4,4), activation='relu', border_mode='valid',name='conv1')(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)  
    #27
    pool1 = MaxPooling2D(pool_size=(3, 3),strides=(2,2),name='pool1')(conv1)
    
    conv1a = Convolution2D(256, 4, 4,subsample=(4,4), activation='relu', border_mode='valid',name='conv1a')(pool1)
    
    bn1 = BatchNormalization(name='bn1')(pool1)
    
    conv2 = Convolution2D(256, 5, 5, activation='relu', border_mode='same',name='conv2')(bn1)
    conv2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D(pool_size=(3, 3),strides=(2,2),name='pool2')(conv2)
    bn2 = BatchNormalization(name='bn2')(pool2)

    conv3 = Convolution2D(384, 3, 3, activation='relu', border_mode='same',name='conv3')(bn2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3a = Convolution2D(256, 2, 2,subsample=(2,2), activation='relu', border_mode='valid',name='conv3a')(conv3)

    conv4 = Convolution2D(384, 3, 3, activation='relu', border_mode='same',name='conv4')(conv3)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',name='conv5')(conv4)
    conv5 = BatchNormalization(axis=1)(conv5)
    pool5 = MaxPooling2D(pool_size=(3, 3),strides=(2,2),name='pool5')(conv5)


    # conv_merge = merge([conv1a,conv3a,pool5],mode='concat', concat_axis=1,name='conv_merge')
    # conv_merge = BatchNormalization(axis=1)(conv_merge)

    # conv_all = Convolution2D(192, 1, 1, activation='relu', border_mode='same',name='conv_all')(conv_merge)
    # conv_all = BatchNormalization(axis=1)(conv_all)
    
    # flatten = Flatten(name='flatten')(conv_all)
    flatten = Flatten(name='flatten')(pool5)

    fc1 = Dense(1024,activation='relu',name='fc1')(flatten)
    
    # fc2 = Dense(1024,activation='relu')(fc1)
    # bin_output = Dense(1,activation='sigmoid',name='bin_output')(fc2)
    dp1 = Dropout(0.5)(fc1)
    fc2 = Dense(1024,activation='relu',name='fc2')(dp1)
    dp2 = Dropout(0.5)(fc2)
    location_output = Dense(img_rows*img_cols,activation='sigmoid',name='location_output')(dp2)
    
    '''
    output
    '''
    
    # output = [location_output,bin_output]
    output = location_output
    model = Model(input=inputs, output=output)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=1e-5)
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)
    imgs_mask_train = imgs_mask_train.reshape(imgs_mask_train.shape[0],img_rows*img_cols)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    imgs_train,imgs_valid,imgs_mask_train,imgs_mask_valid = train_test_split(imgs_train,imgs_mask_train,test_size=0.2,random_state=seed)
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('E:\\UltrasoundNerve\\hyperface.hdf5', monitor='loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=15, verbose=1, shuffle=True,
              callbacks=[model_checkpoint],validation_data=[imgs_valid,imgs_mask_valid]
              )
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('E:\\UltrasoundNerve\\hyperface.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
