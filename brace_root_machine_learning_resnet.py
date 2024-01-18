# import tensorflow as tf
# from tensorflow import keras
from simple_multi_unet_model import multi_unet_model  # Uses softmax
import tensorflow as tf
# from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

from typing import Tuple
# import focal_loss
# from scipy import ndimage
# from skimage import measure, color, io
import segmentation_models as sm

# import keras
# from keras.metrics import MeanIoU


def resize_with_pad(image: np.array,
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def resize_mask_with_pad(image: np.array, 
                    new_shape: Tuple[int, int]) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return image



os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# TF_GPU_ALLOCATOR=cuda_malloc_async

n_classes = 4  # Number of classes for segmentation

# Capture training image info as a list
train_images = []
train_masks = []

for directory_path in glob.glob(f"./2019_final_annotation/*/"):
    print(directory_path)
    train_images_path_list = []
    train_masks_path_list = []
    train_image_path = directory_path + "patches_images_resize_400/"
    train_masks_path = directory_path + "patches_masks_resize_400/"
    train_image_flip_path = directory_path + "patches_images_resize_400_flip/"
    train_masks_flip_path = directory_path + "patches_masks_resize_400_flip/"

    # print(train_image_path)
    # print(train_masks_path)

    for img_path in glob.glob(os.path.join(train_image_path, "*.tif")):
        train_images_path_list.append(img_path)


    for mask_path in glob.glob(os.path.join(train_masks_path, "*.tif")):
        train_masks_path_list.append(mask_path)

    for flip_img_path in glob.glob(os.path.join(train_image_flip_path, "*.tif")):
        train_images_path_list.append(flip_img_path)

    for flip_mask_path in glob.glob(os.path.join(train_masks_flip_path, "*.tif")):
        train_masks_path_list.append(flip_mask_path)

    train_images_path_list.sort()
    train_masks_path_list.sort()

    # print(train_masks_path_list)
    # print(train_images_path_list)

    for image_path in train_images_path_list:
        # print(image_path)
        img = cv2.imread(image_path, 1)
        #img = cv2.resize(img, (800,800))
        img = resize_with_pad(img,(416,416),(255,255,255))
        train_images.append(img)
        # print(train_images.shape)
    for mask_path in train_masks_path_list:
        mask = cv2.imread(mask_path, 0)
        mask = resize_mask_with_pad(mask,(416,416))
        stalk_index = np.where(mask == 1)
        mask[stalk_index] = 0
        #mask = cv2.resize(mask, (800,800),interpolation=cv2.INTER_NEAREST)
        train_masks.append(mask)
        # print(train_masks.shape)

#Convert list to array for machine learning processing
train_images = np.array(train_images)
train_masks = np.array(train_masks)

print("check the shape of mask list")
print(train_images.shape)
print(train_masks.shape)
print(np.unique(train_images))
print(np.unique(train_masks))


#########################################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder

print("before labelencoder.fit_transform")

labelencoder = LabelEncoder()
n, h, w = train_masks.shape
print(n)
train_masks_reshaped = train_masks.reshape(-1, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)

type(train_masks_reshaped)
print("check the shape of reshaped mask list")
print(train_masks_reshaped)


train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

train_masks_reshaped_encoded.reshape(n, h, w)

#train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

print("before train dataset split")


#train_images = train_images/255
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)


#Create a subset of data for quick testing
#Picking 20% for testing and remaining for training

from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(train_images, train_masks_input, test_size=0.20, random_state=0)

#np.unique(y_train)
#y_train.shape

print("before to categorical")

train_masks_cat = to_categorical(y_train,num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

print("after to categorical")

print(y_train.shape)
print(X_train.shape)

from sklearn.utils import class_weight

print("check the shape")
print(train_masks_reshaped_encoded.shape)

class_weights = class_weight.compute_class_weight(class_weight = "balanced",y = train_masks_reshaped_encoded,classes = np.unique(train_masks_reshaped_encoded))
print("Class weights are...:", class_weights)
class_weights_dict = dict(enumerate(class_weights))

#Reused parameters in all models

n_classes=4
activation='softmax'

initial_learning_rate = 0.0005

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=216,
    decay_rate=0.95,
    staircase=True)

optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)

Jaccard_loss = sm.losses.JaccardLoss()
dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss) + Jaccard_loss


#class_weights
#total_loss
#actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
#total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

####################################################################

#from tensorflow.python.keras.applications import resnet

BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train = X_train / np.float32(255)
X_test = X_test / np.float32(255)
X_train1 = preprocess_input1(X_train)
X_test1 = preprocess_input1(X_test)

#add this magic code to solve module 'keras.utils' has no attribute 'get_file' issue
sm.set_framework('tf.keras')
sm.framework()

train_data = tf.data.Dataset.from_tensor_slices((X_train1,y_train_cat))
val_data = tf.data.Dataset.from_tensor_slices((X_test1,y_test_cat))

batchsize=32
train_data = train_data.batch(batchsize,drop_remainder=True)
val_data = val_data.batch(batchsize,drop_remainder=True)

from segmentation_models import Unet
#define model

epoch=250
BACKBONE1 = 'resnet34'
Date_name = 'Sep_27th'

checkpoint_filepath = str(BACKBONE1) + "_backbone_model_" + str(epoch) + "_epoch_" + "best_save_with_sample_weight_" + str(Date_name) + "_" + str(batchsize) + "_batch_size_" + str(initial_learning_rate) + "_LR_Decay_gpu.hdf5"
 

print(checkpoint_filepath)

#checkpoint_filepath = "res18_backbone_model_100_epoch_best_save_with_sample_weight_April_21th_added_dataset_64_batch_size_0_0_1_LR_decay_gpu.hdf5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)

print("start training")
print("Try multiple thread on cpu")
from keras import backend as K
from datetime import datetime

from keras import backend as K
K.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')

#tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
print(gpus)
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  model1 = Unet(BACKBONE1, classes=n_classes, activation=activation, encoder_weights='imagenet')
  model1.compile(optimizer=optim, loss=total_loss, metrics=metrics)

print("compile completed")
print(model1.summary())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# print out the beginning time to see if multithread cpu help lower the time or not
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

#from keras import backend as K
#K.clear_session()

epoch=250
BACKBONE1 = 'resnet34'
Date_name = 'Sep_27th'

history = model1.fit(train_data,
                    verbose=1,
                    epochs=epoch,
                    validation_data=val_data,
                    callbacks=model_checkpoint_callback,
                     )


print("Current Time =", current_time)

#model_save_name = str(BACKBONE1) + "_backbone_model_" + str(epoch) + "_epoch_" + str(Date_name) + "_" + str(batchsize) + "_batch_size_" + str(initial_learning_rate) + "_LR_Deca$

#model1.save('res18_backbone_model_100_epoch_April_21th_added_dataset_64_batch_size_0_0_1_LR_Decay_gpu.hdf5')
#model1.save("model_save_name")


print("before loss picture")
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
loss_picuture_save_name = str(BACKBONE1) + "_backbone_model_" + str(epoch) + "_epoch_" + str(Date_name) + "_" + str(batchsize) + "_batch_size_" + str(initial_learning_rate) + "_LR_Decay_loss.png"
plt.savefig(loss_picuture_save_name)
plt.clf()

acc = history.history['iou_score']
val_acc = history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()
IOU_picuture_save_name = str(BACKBONE1) + "_backbone_model_" + str(epoch) + "_epoch_" + str(Date_name) + "_" + str(batchsize) + "_batch_size_" + str(initial_learning_rate) + "_LR_Decay_IOU.png"
plt.savefig(IOU_picuture_save_name)
plt.clf()

print("before class prediction")

y_pred=model1.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

from keras.metrics import MeanIoU

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:, :, :, 0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# To calculate I0U for each class...
print(n_classes)
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)

class1_IoU = values[0, 0] / (
            values[0, 0] + values[0, 1] + values[0, 2] + values[0, 3] + values[1, 0] + values[2, 0] + values[3, 0])

class2_IoU = values[1, 1] / (
            values[1, 1] + values[1, 0] + values[1, 2] + values[1, 3] + values[0, 1] + values[2, 1] + values[3, 1])

class3_IoU = values[2, 2] / (
            values[2, 2] + values[2, 0] + values[2, 1] + values[2, 3] + values[0, 2] + values[1, 2] + values[3, 2])

class4_IoU = values[3, 3] / (
        values[3, 3] + values[3, 0] + values[3, 1] + values[3, 2]  + values[0, 3] + values[1, 3] + values[2, 3])


print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)
print("This is the result for model resenet 34 with LR = " + str(initial_learning_rate) + " batch size = " + str(batchsize) + " epoch = " + str(epoch))



