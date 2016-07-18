from __future__ import print_function
import pylab as plt
import cv2
import time
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
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
import lasagne
from theano import tensor as T
import theano
seed = 1024
np.random.seed(seed)


img_rows = 64#*2
img_cols = 64#*2

smooth = 1.



def mask_not_blank(mask):
    return sum(mask.flatten()) > 0

    

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
        rotated = cv2.warpAffine(image,M,(img_cols,img_rows))
        rotated = rotated[:,:,0]
        new_X.append(rotated)

    new_X = np.expand_dims(np.array(new_X),1)
    return new_X


def residual_block(input, num_filters, filter_size=(3, 3),pad='same'):


    block = lasagne.layers.BatchNormLayer(input)

    block = lasagne.layers.Conv2DLayer(
            block, num_filters=num_filters/2, filter_size=(1,1),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())
    
    block = lasagne.layers.BatchNormLayer(block)

    block = lasagne.layers.Conv2DLayer(
            block, num_filters=num_filters, filter_size=(3,3),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())

    block = lasagne.layers.BatchNormLayer(block)

    block = lasagne.layers.Conv2DLayer(
            block, num_filters=num_filters, filter_size=(1,1),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())
    
    block = lasagne.layers.ElemwiseSumLayer(incomings = [input,block])

    return block


def lasagne_unet(input_var=None):
    
    # input layer
    input = lasagne.layers.InputLayer(shape=(None, 1, img_rows, img_cols),
                                        input_var=input_var)

    # conv1
    conv1 = lasagne.layers.Conv2DLayer(
            input, num_filters=32, filter_size=(3, 3),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())

    # conv1 = lasagne.layers.Conv2DLayer(
    #         conv1, num_filters=32, filter_size=(3, 3),pad='same',
    #         nonlinearity=lasagne.nonlinearities.LeakyRectify(),
    #         W=lasagne.init.GlorotUniform())

    conv1 = residual_block(conv1, num_filters=32, filter_size=(3, 3))

    bn1 = lasagne.layers.BatchNormLayer(conv1)
    pool1 = lasagne.layers.MaxPool2DLayer(bn1,pool_size=(2,2))

    # conv2
    conv2 = lasagne.layers.Conv2DLayer(
            pool1, num_filters=64, filter_size=(3, 3),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())

    # conv2 = lasagne.layers.Conv2DLayer(
    #         conv2, num_filters=64, filter_size=(3, 3),pad='same',
    #         nonlinearity=lasagne.nonlinearities.LeakyRectify(),
    #         W=lasagne.init.GlorotUniform())
    
    conv2 = residual_block(conv2, num_filters=64, filter_size=(3, 3))
    bn2 = lasagne.layers.BatchNormLayer(conv2)
    pool2 = lasagne.layers.MaxPool2DLayer(bn2,pool_size=(2,2))


    # conv3
    conv3 = lasagne.layers.Conv2DLayer(
            pool2, num_filters=128, filter_size=(3, 3),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())

    # conv3 = lasagne.layers.Conv2DLayer(
    #         conv3, num_filters=128, filter_size=(3, 3),pad='same',
    #         nonlinearity=lasagne.nonlinearities.LeakyRectify(),
    #         W=lasagne.init.GlorotUniform())

    conv3 = residual_block(conv3, num_filters=128, filter_size=(3, 3))
    bn3 = lasagne.layers.BatchNormLayer(conv3)
    pool3 = lasagne.layers.MaxPool2DLayer(bn3,pool_size=(2,2))


    # conv4
    conv4 = lasagne.layers.Conv2DLayer(
            pool3, num_filters=256, filter_size=(3, 3),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())
    
    # conv4 = lasagne.layers.Conv2DLayer(
    #         conv4, num_filters=256, filter_size=(3, 3),pad='same',
    #         nonlinearity=lasagne.nonlinearities.LeakyRectify(),
    #         W=lasagne.init.GlorotUniform())
    conv4 = residual_block(conv4, num_filters=256, filter_size=(3, 3))
    bn4 = lasagne.layers.BatchNormLayer(conv4)
    pool4 = lasagne.layers.MaxPool2DLayer(bn4,pool_size=(2,2))

    # conv5
    conv5 = lasagne.layers.Conv2DLayer(
            pool4, num_filters=512, filter_size=(3, 3),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())
    
    # conv5 = lasagne.layers.Conv2DLayer(
    #         conv5, num_filters=512, filter_size=(3, 3),pad='same',
    #         nonlinearity=lasagne.nonlinearities.LeakyRectify(),
    #         W=lasagne.init.GlorotUniform())
    conv5 = residual_block(conv5, num_filters=512, filter_size=(3, 3))
    bn5 = lasagne.layers.BatchNormLayer(conv5)


    # conv6 and merge
    upscale6 = lasagne.layers.Upscale2DLayer(conv5,scale_factor=(2,2))
    merge6 = lasagne.layers.ConcatLayer([upscale6,conv4])
    conv6 = lasagne.layers.Conv2DLayer(
            merge6, num_filters=256, filter_size=(3, 3),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())
    # conv6 = lasagne.layers.Conv2DLayer(
    #         conv6, num_filters=256, filter_size=(3, 3),pad='same',
    #         nonlinearity=lasagne.nonlinearities.LeakyRectify(),
    #         W=lasagne.init.GlorotUniform())
    conv6 = residual_block(conv6, num_filters=256, filter_size=(3, 3))

    bn6 = lasagne.layers.BatchNormLayer(conv6)

    # conv7 and merge
    upscale7 = lasagne.layers.Upscale2DLayer(bn6,scale_factor=(2,2))
    merge7 = lasagne.layers.ConcatLayer([upscale7,conv3])
    conv7 = lasagne.layers.Conv2DLayer(
            merge7, num_filters=128, filter_size=(3, 3),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())
    # conv7 = lasagne.layers.Conv2DLayer(
    #         conv7, num_filters=128, filter_size=(3, 3),pad='same',
    #         nonlinearity=lasagne.nonlinearities.LeakyRectify(),
    #         W=lasagne.init.GlorotUniform())
    conv7 = residual_block(conv7, num_filters=128, filter_size=(3, 3))
    bn7 = lasagne.layers.BatchNormLayer(conv7)


    # conv8 and merge
    upscale8 = lasagne.layers.Upscale2DLayer(bn7,scale_factor=(2,2))
    merge8 = lasagne.layers.ConcatLayer([upscale8,conv2])
    conv8 = lasagne.layers.Conv2DLayer(
            merge8, num_filters=64, filter_size=(3, 3),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())
    # conv8 = lasagne.layers.Conv2DLayer(
    #         conv8, num_filters=64, filter_size=(3, 3),pad='same',
    #         nonlinearity=lasagne.nonlinearities.LeakyRectify(),
    #         W=lasagne.init.GlorotUniform())
    conv8 = residual_block(conv8, num_filters=64, filter_size=(3, 3))
    bn8 = lasagne.layers.BatchNormLayer(conv8)

    # conv9 and merge
    upscale9 = lasagne.layers.Upscale2DLayer(bn8,scale_factor=(2,2))
    merge9 = lasagne.layers.ConcatLayer([upscale9,conv1])
    conv9 = lasagne.layers.Conv2DLayer(
            merge9, num_filters=32, filter_size=(3, 3),pad='same',
            nonlinearity=lasagne.nonlinearities.LeakyRectify(),
            W=lasagne.init.GlorotUniform())
    # conv9 = lasagne.layers.Conv2DLayer(
    #         conv9, num_filters=32, filter_size=(3, 3),pad='same',
    #         nonlinearity=lasagne.nonlinearities.LeakyRectify(),
    #         W=lasagne.init.GlorotUniform())
    conv9 = residual_block(conv9, num_filters=32, filter_size=(3, 3))
    bn9 = lasagne.layers.BatchNormLayer(conv9)

    # conv10
    conv10 = lasagne.layers.Conv2DLayer(
            conv9, num_filters=1, filter_size=(1, 1),pad='same',
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform())

    return conv10



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def process_data():
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
    

    skf = StratifiedKFold(y_bin, n_folds=10, shuffle=True, random_state=seed)
    for ind_tr, ind_te in skf:
        X_train = imgs_train[ind_tr]
        X_test = imgs_train[ind_te]
        y_train = imgs_mask_train[ind_tr]
        y_test = imgs_mask_train[ind_te]
        break
    
    
    X_train_rotate = get_rotation(X_train)
    y_train_rotate  = get_rotation(y_train)
    X_train = np.concatenate((X_train,X_train_rotate),axis=0)
    y_train = np.concatenate((y_train,y_train_rotate),axis=0)
    print(X_train.shape,y_train.shape)
    
    X_train_rotate = get_rotation(X_train,degree=22.5)
    y_train_rotate  = get_rotation(y_train,degree=22.5)
    X_train = np.concatenate((X_train,X_train_rotate),axis=0)
    y_train = np.concatenate((y_train,y_train_rotate),axis=0)
    print(X_train.shape,y_train.shape)
    
    
    # X_train_rotate = get_rotation(X_train,degree=11.25)
    # y_train_rotate  = get_rotation(y_train,degree=11.25)
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
    
    print(X_train.shape,y_train.shape)

    imgs_train = X_train
    imgs_valid = X_test
    imgs_mask_train = y_train
    imgs_mask_valid = y_test
    
    imgs_train = lasagne.utils.floatX(imgs_train)
    imgs_valid = lasagne.utils.floatX(imgs_valid)

    imgs_mask_train = lasagne.utils.floatX(imgs_mask_train)
    imgs_mask_valid = lasagne.utils.floatX(imgs_mask_valid)


    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)
    
    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    imgs_test = lasagne.utils.floatX(imgs_test)


    return imgs_train,imgs_valid,imgs_mask_train,imgs_mask_valid,imgs_test


def lasagne_dice(prediction, target_var):
    y_true_f = T.flatten(prediction)
    y_pred_f = T.flatten(target_var)
    intersection = T.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (T.sum(y_true_f) + T.sum(y_pred_f))

def train_lasagne():

    model_name = 'unet_lasagne_res.npz'
    num_epochs = 20
    batch_size = 128
    load_model = True
    
    imgs_train,imgs_valid,imgs_mask_train,imgs_mask_valid,imgs_test = process_data()
    print('imgs_test shape',imgs_test.shape)
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')

    network = lasagne_unet(input_var)

    prediction = lasagne.layers.get_output(network)
    # loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    # loss = loss.mean()
    
    loss = -lasagne_dice(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
    
    
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
    #                                                   target_var)
    # test_loss = test_loss.mean()

    test_loss = -lasagne_dice(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_dice = lasagne_dice(test_prediction,target_var)
    test_dice = test_dice.mean()
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_dice])


    if load_model:
        with np.load('E:\\UltrasoundNerve\\'+model_name) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)


    
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(imgs_train, imgs_mask_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_dice = 0
        val_batches = 0
        for batch in iterate_minibatches(imgs_valid, imgs_mask_valid, batch_size, shuffle=False):
            inputs, targets = batch
            err, dice = val_fn(inputs, targets)
            val_err += err
            val_dice += dice
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation dice:\t\t{:.6f} %".format(
            val_dice / val_batches))
        np.savez('E:\\UltrasoundNerve\\'+model_name, *lasagne.layers.get_all_param_values(network))
        
    # np.savez('E:\\UltrasoundNerve\\'+model_name, *lasagne.layers.get_all_param_values(network))

    pred_fn = theano.function([input_var],test_prediction)
    imgs_mask_test = []
    for img in imgs_test:
        img = np.array([img])
        imgs_mask_test.append(pred_fn(img))
    imgs_mask_test = np.concatenate(imgs_mask_test,axis=0)
    
    print("prediction shape",imgs_mask_test.shape)
    np.save('imgs_mask_test.npy', imgs_mask_test)
    print("prediction saved")
    
    
if __name__ == '__main__':
    train_lasagne()
