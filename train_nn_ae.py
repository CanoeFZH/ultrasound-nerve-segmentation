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
import h5py
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Flatten,Dense,Dropout
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
from scipy.signal import convolve2d
seed = 1024
np.random.seed(seed)

img_rows = 64#*2
img_cols = 80#*2


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

def conv_filter(x):
    x = x.reshape(img_rows,img_cols)
    # x = x[0,:,:]
    scharr = np.array([
                            [ -1, -1, -1],
                            [ -1,  9, -1],
                            [ -1, -1, -1]
                          ]) # Gx + j*Gy
    grad = convolve2d(x.astype('float32'), scharr, boundary='symm', mode='same').flatten()
    # plt.imshow(grad,plt.cm.gray)
    # plt.show()
    # print(grad.shape)
    return grad


def get_unet(X):
    inputs = Input(shape = (X.shape[1],))

    encoder1 = Dense(2048)(inputs)
    encoder1 = SReLU()(encoder1)
    encoder1= BatchNormalization()(encoder1)
    encoder1 = Dropout(0.2)(encoder1)

    # encoder1 = Dense(2048)(encoder1)
    # encoder1 = SReLU()(encoder1)
    # encoder1= BatchNormalization()(encoder1)
    # encoder1 = Dropout(0.2)(encoder1)

    encoder2 = Dense(1024)(encoder1)
    encoder2 = SReLU()(encoder2)
    encoder2 = BatchNormalization()(encoder2)
    encoder2 = Dropout(0.2)(encoder2)

    # encoder2 = Dense(1024)(encoder2)
    # encoder2 = SReLU()(encoder2)
    # encoder2 = BatchNormalization()(encoder2)
    # encoder2 = Dropout(0.2)(encoder2)

    encoder3 = Dense(512)(encoder2)
    encoder3 = SReLU()(encoder3)
    encoder3 = BatchNormalization()(encoder3)
    encoder3 = Dropout(0.2)(encoder3)

    # encoder3 = Dense(512)(encoder3)
    # encoder3 = SReLU()(encoder3)
    # encoder3 = BatchNormalization()(encoder3)
    # encoder3 = Dropout(0.2)(encoder3)


    encoder4 = Dense(1024)(encoder3)
    encoder4 = SReLU()(encoder4)
    encoder4 = BatchNormalization()(encoder4)
    encoder4 = Dropout(0.2)(encoder4)

    # encoder4 = Dense(1024)(encoder4)
    # encoder4 = SReLU()(encoder4)
    # encoder4 = BatchNormalization()(encoder4)
    # encoder4 = Dropout(0.2)(encoder4)

    # encoder4 = merge([encoder2,encoder4],mode='concat')
    # encoder4 = Dense(2048)(encoder4)
    # encoder4 = SReLU()(encoder4)
    # encoder4 = BatchNormalization()(encoder4)
    # encoder4 = Dropout(0.2)(encoder4)


    encoder5 = Dense(2048)(encoder4)
    encoder5 = SReLU()(encoder5)
    encoder5 = BatchNormalization()(encoder5)
    encoder5 = Dropout(0.2)(encoder5)

    # encoder5 = Dense(2048)(encoder5)
    # encoder5 = SReLU()(encoder5)
    # encoder5 = BatchNormalization()(encoder5)
    # encoder5 = Dropout(0.2)(encoder5)

    # encoder5 = merge([encoder1,encoder5],mode='concat')
    # encoder5 = Dense(2048)(encoder5)
    # encoder5 = SReLU()(encoder5)
    # encoder5 = BatchNormalization()(encoder5)
    # encoder5 = Dropout(0.2)(encoder5)

    outputs = Dense(img_rows*img_cols, activation='sigmoid')(encoder5)

    
    model = Model(input=inputs, output=outputs)
    # sgd = sgd()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def histeq(image_array,image_bins=256):
    image_array2,bins = np.histogram(image_array.flatten(),image_bins)
    cdf = image_array2.cumsum()
    cdf = (255.0/cdf[-1])*cdf
    image2_array = np.interp(image_array.flatten(),bins[:-1],cdf)
    return image2_array.reshape(image_array.shape),cdf


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


def load_weights(model, filepath,my_names,freeze=False):
        '''Load all layer weights from a HDF5 save file.
        '''
        import h5py
        f = h5py.File(filepath, mode='r')

        if hasattr(model, 'flattened_layers'):
            # support for legacy Sequential/Merge behavior
            flattened_layers = model.flattened_layers
        else:
            flattened_layers = model.layers

        if 'nb_layers' in f.attrs:
            # legacy format
            nb_layers = f.attrs['nb_layers']
            if nb_layers != len(flattened_layers):
                raise Exception('You are trying to load a weight file '
                                'containing ' + str(nb_layers) +
                                ' layers into a model with ' +
                                str(len(flattened_layers)) + '.')

            for k in range(nb_layers):
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                flattened_layers[k].set_weights(weights)
        else:
            # new file format
            layer_names = [n.decode('utf8') for n in my_names]
            # if len(layer_names) != len(flattened_layers):
            #     raise Exception('You are trying to load a weight file '
            #                     'containing ' + str(len(layer_names)) +
            #                     ' layers into a model with ' +
            #                     str(len(flattened_layers)) + ' layers.')
            
            # we batch weight value assignments in a single backend call
            # which provides a speedup in TensorFlow.
            weight_value_tuples = []
            for k, name in enumerate(layer_names):
                g = f[name]
                weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
                if len(weight_names):
                    weight_values = [g[weight_name] for weight_name in weight_names]
                    layer = flattened_layers[k]
                    if freeze:
                        layer.trainable = False
                    symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                    print('load',name,layer.name)
                    if len(weight_values) != len(symbolic_weights):
                        raise Exception('Layer #' + str(k) +
                                        ' (named "' + layer.name +
                                        '" in the current model) was found to '
                                        'correspond to layer ' + name +
                                        ' in the save file. '
                                        'However the new layer ' + layer.name +
                                        ' expects ' + str(len(symbolic_weights)) +
                                        ' weights, but the saved weights have ' +
                                        str(len(weight_values)) +
                                        ' elements.')

                    weight_value_tuples += zip(symbolic_weights, weight_values)
            K.batch_set_value(weight_value_tuples)
        f.close()
        return model

def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()
    
    
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)
    # imgs_train = np.array([ histeq(img)[0] for img in imgs_train])
    print(imgs_train.shape)
    

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    
    imgs_train -= mean
    imgs_train /= std
    
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    y_bin = np.array([mask_not_blank(mask) for mask in imgs_mask_train ])

    # X_train,X_test,y_train,y_test = train_test_split(imgs_train,imgs_mask_train,test_size=0.2,random_state=seed)
    
    skf = StratifiedKFold(y_bin, n_folds=5, shuffle=True, random_state=seed)
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
    
    X_train_flip = X_train[:,:,:,::-1]
    y_train_flip = y_train[:,:,:,::-1]
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    y_train = np.concatenate((y_train,y_train_flip),axis=0)


    X_train_flip = X_train[:,:,::-1,:]
    y_train_flip = y_train[:,:,::-1,:]
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    y_train = np.concatenate((y_train,y_train_flip),axis=0)
    

    X_train= X_train.reshape(X_train.shape[0],img_rows*img_cols)
    X_test= X_test.reshape(X_test.shape[0],img_rows*img_cols)
    y_train= y_train.reshape(y_train.shape[0],img_rows*img_cols)
    y_test= y_test.reshape(y_test.shape[0],img_rows*img_cols)
    
    
    imgs_train = X_train
    imgs_valid = X_test
    imgs_mask_train = y_train
    imgs_mask_valid = y_test
    imgs_train_conv = np.array([ conv_filter(x) for x in imgs_train])
    imgs_valid_conv = np.array([ conv_filter(x) for x in imgs_valid])
    # imgs_train = np.hstack([imgs_train,imgs_train_conv])
    # imgs_valid = np.hstack([imgs_valid,imgs_valid_conv])

    imgs_train = imgs_train_conv
    imgs_valid = imgs_valid_conv
    print ("imgs_train: %s,imgs_valid:%s"%(imgs_train.shape,imgs_valid.shape))
    print ("imgs_mask_train: %s,imgs_mask_valid:%s"%(imgs_mask_train.shape,imgs_mask_valid.shape))


    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet(imgs_train)
    model_name = 'nn_ae_seed_1024_epoch_50_aug_rotate_64_80_shiftbn_sgd_srelu.hdf5'
    model_checkpoint = ModelCheckpoint('E:\\UltrasoundNerve\\'+model_name, monitor='loss', save_best_only=True)
    plot(model, to_file='E:\\UltrasoundNerve\\%s.png'%model_name.replace('.hdf5',''),show_shapes=True)
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    augmentation=False
    batch_size=128
    nb_epoch=50
    load_model=False
    use_all_data = False
    
    if use_all_data:
        imgs_train = np.concatenate((imgs_train,imgs_valid),axis=0)
        imgs_mask_train = np.concatenate((imgs_mask_train,imgs_mask_valid),axis=0)
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='rmsprop', loss=dice_coef_loss, metrics=[dice_coef])
    
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
    imgs_test = preprocess(imgs_test)
    imgs_test= imgs_test.reshape(imgs_test.shape[0],img_rows*img_cols)
    imgs_test_conv = np.array([ conv_filter(x) for x in imgs_test])
    # imgs_test = np.hstack([imgs_test,imgs_test_conv])
    
    imgs_test = imgs_test_conv
    # imgs_test = np.array([ histeq(img)[0] for img in imgs_test])
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
    np.save('imgs_mask_test_nn.npy', imgs_mask_test)
    
    y_preds = model.predict(imgs_valid)
    
    for x,y,y_p in zip(imgs_valid,imgs_mask_valid,y_preds):
        y = y.reshape(img_rows,img_cols)
        y_p = y_p.reshape(img_rows,img_cols)
        mask = y
        y_ps = y_p
        print('mask')
        plt.imshow(mask*255,plt.cm.gray)
        plt.show()
        print('pred')
        plt.imshow(y_ps*255,plt.cm.gray)
        plt.show()
        
if __name__ == '__main__':
    train_and_predict()
