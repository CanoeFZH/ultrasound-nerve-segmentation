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
import pylab as plt
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, AtrousConvolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.optimizers import Adam,SGD#,Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.visualize_util import plot
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold,StratifiedKFold
from data import load_train_data, load_test_data
seed = 1024
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
def mask_not_blank(mask):
    return sum(mask.flatten()) > 0

def clahe(img):
    img = img[0]
    # img = np.concatenate([img,img,img],axis=0).transpose(1,2,0)
    # print(img.shape)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    cl1 = clahe.apply(img)
    # cl1 = np.dstack([cl1,cl1,cl1])
    # plt.imshow(cl1,plt.cm.gray)
    # plt.show()
    cl1 = np.expand_dims(cl1,0)
    # img = img[0,:,:]
    return cl1

def build_block(input,filters,r,c, border_mode='same'):
    x = input
    
    conv1 = BatchNormalization(axis=1)(x)
    conv1 = AtrousConvolution2D(filters/2, 1, 1, border_mode=border_mode)(x)
    conv1 = SReLU()(conv1)

    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = AtrousConvolution2D(filters/2, 3, 3, border_mode=border_mode)(x)
    conv1 = SReLU()(conv1)
    
    conv1 = BatchNormalization(axis=1)(conv1) 
    conv1 = AtrousConvolution2D(filters, r, c, border_mode=border_mode)(conv1)
    conv1 = SReLU()(conv1)
    
    y = conv1

    y = merge([x,y],mode="sum")
    return y


def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = AtrousConvolution2D(32, 3, 3, border_mode='same')(inputs)
    conv1 = SReLU()(conv1)
    # conv1 = build_block(inputs,32,3,3)
    conv1 = build_block(conv1,32,3,3)

    # conv1 = AtrousConvolution2D(32, 3, 3, border_mode='same')(conv1)
    # conv1 = SReLU()(conv1)
    # conv1 = BatchNormalization(axis=1)(conv1)  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    
    conv2 = AtrousConvolution2D(64, 3, 3, border_mode='same')(pool1)
    conv2 = SReLU()(conv2)
    # conv2 = build_block(conv1,64,3,3)
    conv2 = build_block(conv2,64,3,3)

    # conv2= AtrousConvolution2D(64, 3, 3, border_mode='same')(conv2)
    # conv2 = SReLU()(conv2)
    # conv2 = BatchNormalization(axis=1)(conv2)   
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = AtrousConvolution2D(128, 3, 3, border_mode='same')(pool2)
    conv3 = SReLU()(conv3)
    # conv3 = build_block(pool2,128,3,3)
    conv3 = build_block(conv3,128,3,3)

    # conv3 = AtrousConvolution2D(128, 3, 3, border_mode='same')(conv3)
    # conv3 = SReLU()(conv3)
    # conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = AtrousConvolution2D(256, 3, 3, border_mode='same')(pool3)
    conv4 = SReLU()(conv4)
    # conv4 = build_block(pool3,256,3,3)
    conv4 = build_block(conv4,256,3,3)

    # conv4 = AtrousConvolution2D(256, 3, 3, border_mode='same')(conv4)
    # conv4 = SReLU()(conv4)
    # conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = AtrousConvolution2D(512, 3, 3, border_mode='same')(pool4)
    conv5 = SReLU()(conv5)
    # conv5 = build_block(pool4,512,3,3)
    conv5 = build_block(conv5,512,3,3)

    # conv5 = AtrousConvolution2D(512, 3, 3, border_mode='same')(conv5)
    # conv5 = SReLU()(conv5)
    # conv5 = BatchNormalization(axis=1)(conv5)   

    
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = AtrousConvolution2D(256, 3, 3, border_mode='same')(up6) 
    conv6 = SReLU()(conv6)
    # conv6 = build_block(up6,256,3,3)
    conv6 = build_block(conv6,256,3,3)

    # conv6 = AtrousConvolution2D(256, 3, 3, border_mode='same')(conv6)
    # conv6 = SReLU()(conv6)
    # conv6 = BatchNormalization(axis=1)(conv6)  
    
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = AtrousConvolution2D(128, 3, 3, border_mode='same')(up7) 
    conv7 = SReLU()(conv7)
    # conv7 = build_block(up7,128,3,3)
    conv7 = build_block(conv7,128,3,3)

    # conv7 = AtrousConvolution2D(128, 3, 3, border_mode='same')(conv7)
    # conv7 = SReLU()(conv7)
    # conv7 = BatchNormalization(axis=1)(conv7)   
    
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = AtrousConvolution2D(64, 3, 3, border_mode='same')(up8)
    conv8 = SReLU()(conv8)
    # conv8 = build_block(up8,64,3,3)
    conv8 = build_block(conv8,64,3,3)

    # conv8 = AtrousConvolution2D(64, 3, 3, border_mode='same')(conv8)
    # conv8 = SReLU()(conv8)
    # conv8 = BatchNormalization(axis=1)(conv8) 
    
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = AtrousConvolution2D(32, 3, 3, border_mode='same')(up9)
    conv9 = SReLU()(conv9)
    # conv9 = build_block(up9,32,3,3)
    conv9 = build_block(conv9,32,3,3)

    # conv9 = AtrousConvolution2D(32, 3, 3, border_mode='same')(conv9)
    # conv9 = SReLU()(conv9)
    # conv9 = BatchNormalization(axis=1)(conv9) 
    
    '''
    output
    '''
    conv10 = AtrousConvolution2D(1, 1, 1, activation='sigmoid')(conv9)
    
    model = Model(input=inputs, output=conv10)
    # sgd = sgd()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])
    
    return model
    
def histeq(image_array,image_bins=256):
    image_array2,bins = np.histogram(image_array.flatten(),image_bins)
    cdf = image_array2.cumsum()
    cdf = (255.0/cdf[-1])*cdf
    image2_array = np.interp(image_array.flatten(),bins[:-1],cdf)
    return image2_array.reshape(image_array.shape),cdf

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def get_rotation(X,degree=45):
    new_X = []
    center = (img_cols/2,img_rows/2)
    M = cv2.getRotationMatrix2D(center,45,1.0)
    for image in X:
        image = np.dstack([image[0,:,:],image[0,:,:],image[0,:,:]])
        # print(image.shape)
        rotated = cv2.warpAffine(image,M,(img_cols,img_rows))
        # print(rotated.shape)
        # print('origin')
        # plt.imshow(image,plt.cm.gray)
        # plt.show()
        # print('rotate')
        # plt.imshow(rotated,plt.cm.gray)
        # plt.show()

        rotated = rotated[:,:,0]
        new_X.append(rotated)

    new_X = np.expand_dims(np.array(new_X),1)
    return new_X


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = np.array([ clahe(img) for img in imgs_train])

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
    
    # X_train,X_test,y_train,y_test = train_test_split(imgs_train,imgs_mask_train,test_size=0.2,random_state=seed)
    
    skf = StratifiedKFold(y_bin, n_folds=10, shuffle=True, random_state=seed)
    for ind_tr, ind_te in skf:
        X_train = imgs_train[ind_tr]
        X_test = imgs_train[ind_te]
        y_train = imgs_mask_train[ind_tr]
        y_test = imgs_mask_train[ind_te]
        break
    
    
    # X_train_rotate = get_rotation(X_train)
    # y_train_rotate  = get_rotation(y_train)
    # X_train = np.concatenate((X_train,X_train_rotate),axis=0)
    # y_train = np.concatenate((y_train,y_train_rotate),axis=0)
    # print(X_train.shape,y_train.shape)
    
    # X_train_rotate = get_rotation(X_train,degree=22.5)
    # y_train_rotate  = get_rotation(y_train,degree=22.5)
    # X_train = np.concatenate((X_train,X_train_rotate),axis=0)
    # y_train = np.concatenate((y_train,y_train_rotate),axis=0)
    # print(X_train.shape,y_train.shape)


    # X_train_rotate = get_rotation(X_train,degree=11.25)
    # y_train_rotate  = get_rotation(y_train,degree=11.25)
    # X_train = np.concatenate((X_train,X_train_rotate),axis=0)
    # y_train = np.concatenate((y_train,y_train_rotate),axis=0)
    # print(X_train.shape,y_train.shape)

    # X_train_flip = X_train[:,:,:,::-1]
    # y_train_flip = y_train[:,:,:,::-1]
    # X_train = np.concatenate((X_train,X_train_flip),axis=0)
    # y_train = np.concatenate((y_train,y_train_flip),axis=0)
    
    
    # X_train_flip = X_train[:,:,::-1,:]
    # y_train_flip = y_train[:,:,::-1,:]
    # X_train = np.concatenate((X_train,X_train_flip),axis=0)
    # y_train = np.concatenate((y_train,y_train_flip),axis=0)
    
    
    
    imgs_train = X_train
    imgs_valid = X_test
    imgs_mask_train = y_train
    imgs_mask_valid = y_test
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_name = 'unet_seed_1024_epoch_30_batch_8_aug_rotate_64_64_shiftbn_sgd_srelu_res_atrous.hdf5'
    model_checkpoint = ModelCheckpoint('E:\\UltrasoundNerve\\'+model_name, monitor='loss', save_best_only=True)
    plot(model, to_file='E:\\UltrasoundNerve\\%s.png'%model_name.replace('.hdf5',''),show_shapes=True)
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    augmentation=False
    batch_size=128
    nb_epoch=20
    load_model=False
    use_all_data = False
    
    if use_all_data:
        imgs_train = np.concatenate((imgs_train,imgs_valid),axis=0)
        imgs_mask_train = np.concatenate((imgs_mask_train,imgs_mask_valid),axis=0)
    
    # model.load_weights('E:\\UltrasoundNerve\\'+"unet_seed_1024_epoch_20_aug_rotate_64_80_shiftbn_sgd_srelu_plus5.hdf5")
    # model.save_weights('E:\\UltrasoundNerve\\'+'unet_seed_1024_epoch_20_aug_rotate_64_80_shiftbn_sgd_srelu_plus10.npy')
    # return 
    
    if load_model:
        model.load_weights('E:\\UltrasoundNerve\\'+model_name)
    if not augmentation:
        model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint],
                  validation_data=[imgs_valid,imgs_mask_valid]
                  )
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
    imgs_test = np.array([ clahe(img) for img in imgs_test])

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
