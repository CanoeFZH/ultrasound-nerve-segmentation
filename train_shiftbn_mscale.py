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
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.visualize_util import plot
from sklearn.cross_validation import train_test_split
from data import load_train_data, load_test_data
seed = 1024+3
np.random.seed(seed)

img_rows = 64#*2
img_cols = 80#*2

img_rows = 64#*2
img_cols = 64#*2

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


def get_unet():
    inputs1 = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, border_mode='same')(inputs1)
    conv1 = SReLU()(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    
    conv2 = Convolution2D(64, 3, 3, border_mode='same')(pool1)
    conv2 = SReLU()(conv2)
    conv2= Convolution2D(64, 3, 3, border_mode='same')(conv2)
    conv2 = SReLU()(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)   
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same')(pool2)
    conv3 = SReLU()(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same')(conv3)
    conv3 = SReLU()(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Convolution2D(256, 3, 3, border_mode='same')(pool3)
    conv4 = SReLU()(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same')(conv4)
    conv4 = SReLU()(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Convolution2D(512, 3, 3, border_mode='same')(pool4)
    conv5 = SReLU()(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same')(conv5)
    conv5 = SReLU()(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)   

    
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, border_mode='same')(up6) 
    conv6 = SReLU()(conv6)
    conv6 = Convolution2D(256, 3, 3, border_mode='same')(conv6)
    conv6 = SReLU()(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)  
    
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, border_mode='same')(up7) 
    conv7 = SReLU()(conv7)
    conv7 = Convolution2D(128, 3, 3, border_mode='same')(conv7)
    conv7 = SReLU()(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)   
    
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, border_mode='same')(up8)
    conv8 = SReLU()(conv8)
    conv8 = Convolution2D(64, 3, 3, border_mode='same')(conv8)
    conv8 = SReLU()(conv8)
    conv8 = BatchNormalization(axis=1)(conv8) 
    
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, border_mode='same')(up9)
    conv9 = SReLU()(conv9)
    conv9 = Convolution2D(32, 3, 3, border_mode='same')(conv9)
    conv9 = SReLU()(conv9)
    conv9 = BatchNormalization(axis=1)(conv9) 
    

    inputs2 = Input((1, img_rows/2, img_cols/2))
    conv1 = Convolution2D(32, 3, 3, border_mode='same')(inputs2)
    conv1 = SReLU()(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same')(conv1)
    conv1 = SReLU()(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    
    conv2 = Convolution2D(64, 3, 3, border_mode='same')(pool1)
    conv2 = SReLU()(conv2)
    conv2= Convolution2D(64, 3, 3, border_mode='same')(conv2)
    conv2 = SReLU()(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)   
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same')(pool2)
    conv3 = SReLU()(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same')(conv3)
    conv3 = SReLU()(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Convolution2D(256, 3, 3, border_mode='same')(pool3)
    conv4 = SReLU()(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same')(conv4)
    conv4 = SReLU()(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Convolution2D(512, 3, 3, border_mode='same')(pool4)
    conv5 = SReLU()(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same')(conv5)
    conv5 = SReLU()(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)   

    
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, border_mode='same')(up6) 
    conv6 = SReLU()(conv6)
    conv6 = Convolution2D(256, 3, 3, border_mode='same')(conv6)
    conv6 = SReLU()(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)  
    
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, border_mode='same')(up7) 
    conv7 = SReLU()(conv7)
    conv7 = Convolution2D(128, 3, 3, border_mode='same')(conv7)
    conv7 = SReLU()(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)   
    
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, border_mode='same')(up8)
    conv8 = SReLU()(conv8)
    conv8 = Convolution2D(64, 3, 3, border_mode='same')(conv8)
    conv8 = SReLU()(conv8)
    conv8 = BatchNormalization(axis=1)(conv8) 
    
    up99 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv99 = Convolution2D(32, 3, 3, border_mode='same')(up99)
    conv99 = SReLU()(conv99)
    conv99 = Convolution2D(32, 3, 3, border_mode='same')(conv99)
    conv99 = SReLU()(conv99)
    conv99 = BatchNormalization(axis=1)(conv99) 
    pool99 = UpSampling2D(size=(2, 2))(conv99)
    conv99 = Convolution2D(32, 3, 3, border_mode='same')(pool99)

    conv9 = merge([conv9,conv99],mode='sum')

    '''
    output
    '''
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
    
    inputs = [inputs1,inputs2]
    model = Model(input=inputs, output=conv10)
    
    model.compile(optimizer="rmsprop", loss=dice_coef_loss, metrics=[dice_coef])
    
    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p



def preprocess_twice(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows/2, img_cols/2), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols/2, img_rows/2), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()
    
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train2, imgs_mask_train2 = load_train_data()
    imgs_train2 = preprocess_twice(imgs_train2)
    imgs_mask_train2 = preprocess_twice(imgs_mask_train2)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    
    imgs_train -= mean
    imgs_train /= std

    imgs_train2 = imgs_train2.astype('float32')
    mean2 = np.mean(imgs_train2)  # mean for data centering
    std2 = np.std(imgs_train2)  # std for data normalization
    
    imgs_train2 -= mean2
    imgs_train2 /= std2
    
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    imgs_train,imgs_valid,imgs_mask_train,imgs_mask_valid,imgs_train2,imgs_valid2 = train_test_split(
        imgs_train,
        imgs_mask_train,
        imgs_train2,
        test_size=0.2,random_state=seed)
    
    print(
        imgs_train.shape,
        imgs_valid.shape,
        imgs_train2.shape,
        imgs_valid2.shape
    )
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_name = 'unet_seed_1024_epoch_30_no_aug_64_80_shiftbn_mscale_plus_10_3.hdf5'
    model_checkpoint = ModelCheckpoint('E:\\UltrasoundNerve\\'+model_name, monitor='loss', save_best_only=True)
    plot(model, to_file='E:\\UltrasoundNerve\\%s.png'%model_name.replace('.hdf5',''),show_shapes=True)
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    augmentation=False
    batch_size=32
    nb_epoch=10
    load_model=True
    use_all_data = True
    
    if use_all_data:
        imgs_train = np.concatenate((imgs_train,imgs_valid),axis=0)
        imgs_mask_train = np.concatenate((imgs_mask_train,imgs_mask_valid),axis=0)
        imgs_train2 = np.concatenate((imgs_train2,imgs_valid2),axis=0)
        
    if load_model:
        # model.load_weights('E:\\UltrasoundNerve\\'+'unet_seed_1024_epoch_30_no_aug_64_80_shiftbn_mscale.hdf5')
        model.load_weights('E:\\UltrasoundNerve\\'+model_name)
    if not augmentation:
        # model.fit([imgs_train,imgs_train2], imgs_mask_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
        #           callbacks=[model_checkpoint],
        #           validation_data=([imgs_valid,imgs_valid2],imgs_mask_valid)
        #           )
        pass
    else:
         
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
            vertical_flip=False)  # randomly flip images
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


    imgs_test2, imgs_id_test2 = load_test_data()
    imgs_test2 = preprocess_twice(imgs_test2)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std


    imgs_test2 = imgs_test2.astype('float32')
    imgs_test2 -= mean2
    imgs_test2 /= std2



    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('E:\\UltrasoundNerve\\'+model_name)
    
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict([imgs_test,imgs_test2], verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)
    
    
if __name__ == '__main__':
    train_and_predict()
