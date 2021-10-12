# Final Project - Medical Segmentation Decathlon; U-net CNN with Generalized Dice Coefficient #
# Maor Oz -308540608 # BGU #

# %% Import libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel
import imageio
from skimage.transform import resize
from skimage.io import imsave
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb
from cv2 import bitwise_and, addWeighted
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from skimage.exposure import rescale_intensity

# %% Create, preprocess, save and load train data

data_path = 'data/'
# downsample by helf the resolution of the training 2D images later (for memory saving and runtime improvment)
image_rows = int(512/2)
image_cols = int(512/2)
# The number of classes that each mask (label) has 
Num_Classes = 2 # {0 -> background, 1 -> spleen}

def create_train_data():
    print('-'*30)
    print('Creating training data...')
    print('-'*30)
    TrainImages_data_path = os.path.join(data_path, 'Task09_Spleen/imagesTr')
    TrainImages = os.listdir(TrainImages_data_path)
    # training images file names (nii.gz - zipped)
    training_images = TrainImages[41::]
    # training images (unzipped)
    imgs_train = []

    TrainMasks_data_path = os.path.join(data_path, 'Task09_Spleen/labelsTr')
    TrainMasks = os.listdir(TrainMasks_data_path)
    # training masks/labels (organ+tumor) file names (nii.gz - zipped)
    training_masks = TrainMasks[41::]
    # training masks/labels (organ+tumor) (unzipped) as list of Numpy arrays
    masks_train = []
    
    # zip(training_masks, training_images)=(('mask0name','image0name'),('mask1name','image1name'),...)
    for mask, img in zip(training_masks, training_images):
        # load the 3D training mask (shape=[512,512,n], n --> 2D slice number) as Nifti1Image object
        training_mask = nibabel.load(os.path.join(TrainMasks_data_path, mask))
        # load the 3D training image (shape=[512,512,n], n --> 2D slice number) as Nifti1Image object
        training_image = nibabel.load(os.path.join(TrainImages_data_path, img)) 
        
        for k in range(training_mask.shape[2]):
            # take the axial cut at z=k plane (xy plane @ k=0,1,...,N) as numpy ndarray
            # downsample each image (slice) by helf, i.e. take every second pixel lengthwise and widthwise of the image
            # resize each image to size: (image_rows, image_cols)
            mask_2d = training_mask.get_fdata()[::2, ::2, k]
            mask_2d = resize(mask_2d, (image_rows, image_cols), preserve_range=True)
            image_2d = training_image.get_fdata()[::2, ::2, k]
            image_2d = resize(image_2d, (image_rows, image_cols), preserve_range=True)
            # if mask_2d contains only one gray level (only '0' values, i.e. black image), it means that there is no mask (organ+tumor)
            if len(np.unique(mask_2d)) != 1:
                masks_train.append(mask_2d)
                imgs_train.append(image_2d)
                    
    # imgs --> all the slices of all 3D original train images concatenated together to 3D numpy array
    imgs = np.ndarray(
            (len(imgs_train), image_rows, image_cols), dtype=np.uint8
            )
    # imgs_mask --> all the slices of all 3D original train masks/labels concatenated together to a 3D numpy array
    imgs_mask = np.ndarray(
            (len(masks_train), image_rows, image_cols), dtype=np.uint8
            )
    
    # flip between 'x' and 'z' axis 
    for index, img in enumerate(imgs_train):
        imgs[index, :, :] = img
        
    for index, mask in enumerate(masks_train):
        imgs_mask[index, :, :] = mask

    np.save('SplnImgs_train.npy', imgs)
    np.save('SplnLbls_train.npy', imgs_mask)
    print('Saving to .npy files done.')

def load_train_data():
    imgs_train = np.load('SplnImgs_train.npy')
    masks_train = np.load('SplnLbls_train.npy')
    return imgs_train, masks_train

# %% Create, preprocess, save and load test data

def create_test_data():
    print('-'*30)
    print('Creating test data...')
    print('-'*30)
    TestImages_data_path = os.path.join(data_path, 'Task09_Spleen/imagesTs')
    TestImages = os.listdir(TestImages_data_path)
    # test images file names (nii.gz - zipped)
    test_images = TestImages[20::]
    # test images (unzipped)
    imgs_test = []
    
    for image_name in test_images:
        # load the 3D test image (shape=[512,512,n], n --> 2D slice number) as Nifti1Image object
        img = nibabel.load(os.path.join(TestImages_data_path, image_name))
        
        for k in range(img.shape[2]):
            # take the axial cut at z=k plane (xy plane @ k=0,1,...,N) as numpy ndarray
            # downsample each image (slice) by helf, i.e. take every second pixel lengthwise and widthwise of the image
            # resize each image to size: (image_rows, image_cols)
            img_2d = np.array(img.get_fdata()[::2, ::2, k])
            img_2d = resize(img_2d, (image_rows, image_cols), preserve_range=True)
            imgs_test.append(img_2d)
    
    # imgst --> all the slices of all 3D original test images concatenated together to a 3D numpy array
    imgst = np.ndarray(
            (len(imgs_test), image_rows, image_cols), dtype=np.uint8
            )

    # flip between 'x' and 'z' axis
    for index, imge in enumerate(imgs_test):
        imgst[index, :, :] = imge

    np.save('SplnImgs_test.npy', imgst)
    print('Saving to .npy files done.')
    

def load_test_data():
    imgs_test = np.load('SplnImgs_test.npy')
    return imgs_test

# %% Build U-net model, loss function and metric

# TF dimension ordering in this code
K.set_image_data_format('channels_last')

# generalized dice coefficient as metric
def gen_dice_coef(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for num_classes labels (classes). Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes = Num_Classes)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

# generalized dice coefficient as loss function
def gen_dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - gen_dice_coef(y_true, y_pred)

def Unet():
    inputs = Input((image_rows, image_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    PredictedMask = Conv2D(Num_Classes, (1, 1), activation='sigmoid')(conv9)
    # last layer is the predicted mask/label image (organ+tumor), each pixel in the set {0,1}
    
    model = Model(inputs=[inputs], outputs=[PredictedMask])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss=gen_dice_coef_loss, metrics=[gen_dice_coef])

    return model

# %% Train the model and get predictions

def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    # load train data
    imgs_train, imgs_mask_train = load_train_data()
    
    # adapt the train images and masks dimensions so that we can feed it to the network (by inserting a new axis)
    imgs_train = imgs_train[..., np.newaxis]
    imgs_mask_train = imgs_mask_train[..., np.newaxis]

    # train data normalization: x --> z=(x-mean_x)/std_x --> mean_z=0 , std_z=1
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    imgs_train -= mean
    imgs_train /= std

    # each pixel in the range [0,2] (pixels values belong to {0,1}), no need to normalized
    imgs_mask_train = imgs_mask_train.astype('float32')

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = Unet()
    model_checkpoint = ModelCheckpoint('weights_Spleen.h5', monitor='val_loss', save_best_only=True)
    # saving the weights and the loss of the best predictions we obtained

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    history=model.fit(imgs_train, imgs_mask_train, batch_size=5, epochs=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    # load test data
    imgs_test = load_test_data()
    
    # adapt the test images dimensions so that we can feed it to the network (by inserting a new axis)
    imgs_test = imgs_test[..., np.newaxis]

    # test data normalization: x --> z=(x-mean_x)/std_x --> mean_z=0 , std_z=1
    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)
    std = np.std(imgs_test)
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights_Spleen.h5')
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('Predicted_SplnMasks.npy', imgs_mask_test)
    print('Saving to .npy files done.')
    
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    
    pred_dir = 'Predicted_Spleen_Masks'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    
    for k in range(len(imgs_test)):
        pred_mask = model.predict(imgs_test[k,:,:,:][np.newaxis,...])   # shape: (1, 256, 256, 3)
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        pred_mask = pred_mask[0]
        pred_mask = tf.keras.preprocessing.image.array_to_img(pred_mask)
        pred_mask = tf.keras.preprocessing.image.img_to_array(pred_mask)[:,:,0]
        mask = pred_mask.astype('uint8')

        testImg = rescale_intensity(imgs_test[k,:,:,0], out_range=(0,255))
        testImg = testImg.astype('uint8')
        testImgRGB = gray2rgb(testImg)

        blueImg = np.zeros(testImgRGB.shape, testImgRGB.dtype)
        blueImg[:,:] = (0, 0, 255)
        blueMask = bitwise_and(blueImg, blueImg, mask = mask)
        blendos = addWeighted(blueMask, 1, testImgRGB, 1, 0)
        sgmntdImg = mark_boundaries(blendos, mask, color = (0.8, 0.5, 0.38))
        imsave(os.path.join(pred_dir, str(k) + '_pred.png'), sgmntdImg)
    
    # animated images (GIF file)
    ImgsNames = sorted(os.listdir(pred_dir), key=len)
    ImgsNames = ImgsNames[837:923:]
    Imgs = []
    for image_name in ImgsNames:
        Imgs.append(imageio.imread(os.path.join(pred_dir,image_name)))

    imageio.mimwrite('animated_spleen.gif', Imgs)        
    
    # plotting the dice coeff results (accuracy) as a function of the number of epochs
    plt.figure()
    plt.plot(history.history['gen_dice_coef'])
    plt.plot(history.history['val_gen_dice_coef'])
    plt.title('Model: Generalized Dice Coefficient')
    plt.ylabel('Dice Coeff')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # plotting the dice coeff results (loss function) as a function of the number of epochs
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model: Generalized Dice Coefficient')
    plt.ylabel('Dice Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()
    
    
if __name__ == '__main__':
    create_train_data()
    create_test_data()
    train_and_predict()