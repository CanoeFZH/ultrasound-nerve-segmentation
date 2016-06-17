from __future__ import print_function


'''
seed = 1024,10 epoch
seed = 1024+1,40 epoch

datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.0, # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)


'''
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras.layers.noise import GaussianNoise
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.visualize_util import plot
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold,StratifiedKFold
from data import load_train_data, load_test_data

seed = 1024
np.random.seed(seed)

img_rows = 128#*2
img_cols = 128#*2

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def build_res_block(x,a,b,c):

    shortcut_y = x
    conv_in = Convolution2D(a, b, c, border_mode='same')(x)
    conv_in = LeakyReLU(0.01)(conv_in)
    conv_in = BatchNormalization(axis=1)(conv_in)


    conv_out = Convolution2D(a, b, c, border_mode='same')(conv_in)
    conv_out = LeakyReLU(0.01)(conv_out)
    conv_out = BatchNormalization(axis=1)(conv_out)

    y = merge([shortcut_y,conv_out],mode='sum')

    res_block = Model(input=x, output=y)
    return res_block



def mask_not_blank(mask):
    return sum(mask.flatten()) > 0




def get_fcn():
    inputs = Input((1, img_rows, img_cols))

    conv1 = GaussianNoise(1.0)(inputs)
    conv1 = Convolution2D(32, 5, 5, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Convolution2D(32, 5, 5, border_mode='same')(inputs)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv1 = Convolution2D(32, 5, 5, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Convolution2D(32, 5, 5, border_mode='same')(inputs)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv1 = Convolution2D(32, 5, 5, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Convolution2D(32, 5, 5, border_mode='same')(inputs)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)


    conv1 = Convolution2D(32, 3, 3, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)


    conv1 = Convolution2D(128, 1, 1, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)

    conv1 = Convolution2D(128, 1, 1, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv1 = UpSampling2D(size=(2, 2))(conv1)
    conv1 = Convolution2D(2, 3, 3, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)

    conv1 = UpSampling2D(size=(2, 2))(conv1)
    conv1 = Convolution2D(2, 3, 3, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)


    conv1 = UpSampling2D(size=(2, 2))(conv1)
    conv1 = Convolution2D(2, 3, 3, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)




    # conv2 = Convolution2D(64, 3, 3, border_mode='same')(bn1)
    # conv2 = LeakyReLU(0.01)(conv2)
    # bn2 = BatchNormalization(axis=1)(conv2)

    # conv3 = Convolution2D(64, 3, 3, border_mode='same')(bn2)
    # conv3 = LeakyReLU(0.01)(conv3)
    # bn3 = BatchNormalization(axis=1)(conv3)

    # conv4 = Convolution2D(64, 3, 3, border_mode='same')(bn3)
    # conv4 = LeakyReLU(0.01)(conv4)
    # bn4 = BatchNormalization(axis=1)(conv4)

    # conv5 = Convolution2D(64, 3, 3, border_mode='same')(bn4)
    # conv5 = LeakyReLU(0.01)(conv5)
    # bn5 = BatchNormalization(axis=1)(conv5)

    # conv6 = Convolution2D(64, 3, 3, border_mode='same')(bn5)
    # conv6 = LeakyReLU(0.01)(conv6)
    # bn6 = BatchNormalization(axis=1)(conv6)





    
    # This outputs (nSample, 1, 60, 80), so in effects it seems to merge the channels of the previous layers, but does it make sense?

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='binary_crossentropy', optimizer=sgd)
    
    '''
    output
    '''
    conv5 = Convolution2D(1, 1, 1,activation='sigmoid', border_mode='same')(conv1)

    model = Model(input=inputs, output=conv5)

    model.compile(optimizer="rmsprop", loss=dice_coef_loss, metrics=[dice_coef])
    # sgd =SGD(lr=0.01,momentum=0.9,nesterov=True,decay=0.0005)
    # model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])

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
    
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    
    imgs_train -= mean
    imgs_train /= std
    
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    y_bin = np.array([mask_not_blank(mask) for mask in imgs_mask_train ])

    X_train,X_test,y_train,y_test = train_test_split(imgs_train,imgs_mask_train,test_size=0.2,random_state=seed)
    
    skf = StratifiedKFold(y_bin, n_folds=5, shuffle=True, random_state=seed)
    for ind_tr, ind_te in skf:
        X_train = imgs_train[ind_tr]
        X_test = imgs_train[ind_te]
        
        y_train = imgs_mask_train[ind_tr]
        y_test = imgs_mask_train[ind_te]
        y_bin_train = y_bin[ind_tr]
        y_bin_test = y_bin[ind_tr]
        break
    imgs_train = X_train
    imgs_valid = X_test
    imgs_mask_train = y_train
    imgs_mask_valid = y_test

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_fcn()
    
    # plot(model, to_file='E:\\UltrasoundNerve\\fcn.png',show_shapes=True)
    model_name = 'fcn_s_seed_1024_epoch_30_no_aug_64_80.hdf5'
    model_checkpoint = ModelCheckpoint('E:\\UltrasoundNerve\\'+model_name, monitor='loss', save_best_only=True)
    plot(model, to_file='E:\\UltrasoundNerve\\%s.png'%model_name.replace('.hdf5',''),show_shapes=True)
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    augmentation=False
    batch_size=32
    nb_epoch=30
    load_model=True
    use_all_data = False
    random_flip_fold = 3


    if use_all_data:
        imgs_train = np.concatenate((imgs_train,imgs_valid),axis=0)
        imgs_mask_train = np.concatenate((imgs_mask_train,imgs_mask_valid),axis=0)
        

    if random_flip_fold>0:
        skf_flip = StratifiedKFold(y_bin_train, n_folds=random_flip_fold, shuffle=True, random_state=seed)
        count = 0
        X_flips = []
        y_flips = []
        for ind_tr, ind_te in skf_flip:
            X_flip_train = imgs_train[ind_tr]
            y_flip_train = imgs_mask_train[ind_tr]
            if count==0:
                X_flip_train = X_flip_train[:,:,::-1,:]
            if count==1:
                X_flip_train = X_flip_train[:,:,:,::-1]
            if count==2:
                X_flip_train = X_flip_train[:,:,::-1,::-1]
                
            print(X_flip_train.shape)
            X_flips.append(X_flip_train)
            y_flips.append(y_flip_train)
            count+=1

        X_flips = np.concatenate(X_flips)
        y_flips = np.concatenate(y_flips)
        print(X_flips.shape)
        imgs_train = np.concatenate((imgs_train,X_flips),axis=0)
        imgs_mask_train = np.concatenate((imgs_mask_train,y_flips),axis=0)
        

    if load_model:
        model.load_weights('E:\\UltrasoundNerve\\'+model_name)
        
    if not augmentation:
        model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint],validation_data=[imgs_valid,imgs_mask_valid]
                  )
    else:
        
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.0, # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(imgs_train)
        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(imgs_train, imgs_mask_train,
                            batch_size=batch_size),
                            samples_per_epoch=imgs_train.shape[0],
                            nb_epoch=nb_epoch,
                            callbacks=[model_checkpoint],
                            validation_data=(imgs_valid,imgs_mask_valid))    
    
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
    model.load_weights('E:\\UltrasoundNerve\\'+model_name)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
