from simple_multi_unet_model import multi_unet_model
from Unet3plus import unet3plus
import tensorflow as tf
import os
import glob
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import segmentation_models as sm

#set the num of gpu available for model in the environment
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="4"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Number of classes for segmentation
#input shape of picutures is 400 X 400
Input_X = 400
input_y = 400
n_classes = 4

#Capture training image info as a list
train_images = []
train_masks = []

# record the path for each image and mask in my data directory
for directory_path in glob.glob(f"/home/hl46161/brace_root/2019_new_annotation/2019_new_annotation_mask/test/*/"):
    print(directory_path)
    train_images_path_list = []
    train_masks_path_list = []
    train_image_path=directory_path + "patches_images_resize_400/"
    train_masks_path = directory_path + "patches_masks_resize_400/"
    train_image_flip_path = directory_path + "patches_images_resize_400_flip/"
    train_masks_flip_path = directory_path + "patches_masks_resize_400_flip/"

    # append the image path to path_list
    for img_path in glob.glob(os.path.join(train_image_path,"*.tif")):
      train_images_path_list.append(img_path)

    # append the mask path to mask_list
    for mask_path in glob.glob(os.path.join(train_masks_path, "*.tif")):
        train_masks_path_list.append(mask_path)

    # append the image path to path_list
    for flip_img_path in glob.glob(os.path.join(train_image_flip_path, "*.tif")):
        train_images_path_list.append(flip_img_path)

    # append the mask path to mask_list
    for flip_mask_path in glob.glob(os.path.join(train_masks_flip_path, "*.tif")):
        train_masks_path_list.append(flip_mask_path)

    #the image and mask path has to be sorted to match each other.
    train_images_path_list.sort()
    train_masks_path_list.sort()

    # print(train_masks_path_list)
    # print(train_images_path_list)

    # read in image and mask using opencv
    for image_path in train_images_path_list:
        #print(image_path)
        #read in color images
        img = cv2.imread(image_path, 1)
        train_images.append(img)

    for mask_path in train_masks_path_list:
        mask = cv2.imread(mask_path, 0)
        #following code here is to remove brace stalk annotation
        stalk_index = np.where(mask == 1)
        mask[stalk_index] = 0
        train_masks.append(mask)


# Convert list to array for machine learning processing
train_images = np.array(train_images)
train_masks = np.array(train_masks)

#sanity check for image readin process
print("check the shape of mask list")
print(train_masks.shape)
print(np.unique(train_images))
print(np.unique(train_masks))

###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder

print("before labelencoder.fit_transform")

labelencoder = LabelEncoder()
n, h, w = train_masks.shape
print(n)
train_masks_reshaped = train_masks.reshape(-1, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
type(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

#sanity check of mask reshape process
print("check the shape of reshaped mask list")
print(train_masks_reshaped)

train_masks_reshaped_encoded.reshape(n, h, w)

##################################################################
#since images are colored three channel images, normalize pixel value to 0-1 range by dividing 255
#train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)
train_images = train_images/255
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)



#Picking 20% for testing and remaining for training
from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(train_images, train_masks_input, test_size=0.20, random_state=0)

#np.unique(y_train)
#y_train.shape

#
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

print("after to categorical")

#np.unique(y_train)
#np.unique(train_masks_cat)
#np.unique(y_train_cat)
#np.unique(y_train)
#y_train.shape
#train_masks_cat

print(y_train.shape)
print(X_train.shape)
#######################################################

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

#Since the annotation for classes are not balanced(background has much more proportion in the picutures )
#need to calculate a class weight to penalize on easy annotation and make model focus on classes that are hard to identify
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                  classes = np.unique(train_masks_reshaped_encoded),
                                                  y = train_masks_reshaped_encoded)
print("Class weights are...:", class_weights)


class_weights_dict = dict(enumerate(class_weights))
type(class_weights_dict)
print(class_weights_dict)


#Reused parameters in all models
n_classes=4
activation='softmax'

initial_learning_rate = 0.001

#set a exponential decay schedule for better model converge
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True)

#use adam optimizer
optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

#define model
#set training parameters
batchsize=32
epoch=100
Date_name = 'May_22th'

checkpoint_filepath =  "Simple_Unet_model_" + str(epoch) + "_epoch_" + "best_save_with_sample_weight_" + str(Date_name) + "_" + str(batchsize) + "_batch_size_" + str(initial_learning_rate) + "_LR_Decay_gpu.hdf5"


#set a accuracy call back to monitor model performace. keep model weights when val accuracy is highest
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)

print("start training")
print("Try multiple thread on gpu")
from keras import backend as K
from datetime import datetime

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

print(IMG_HEIGHT)
print(IMG_WIDTH)
print(IMG_CHANNELS)
print(n_classes)

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

def get_unet3plus_model():
    return unet3plus(input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), output_channels = n_classes)


gpus = tf.config.list_physical_devices('GPU')


#use following compile code if mutiple gpus are available
#strategy = tf.distribute.MultiWorkerMirroredStrategy()

#with strategy.scope():
#model = get_model()
#model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

model = get_unet3plus_model()
model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

#set seed for image augmentation
seed = 24

#use keras image augmentation utilis
from keras.preprocessing.image import ImageDataGenerator

#add random rotation, brightness  adjustment, geograohic augmentation
img_data_gen_args = dict(rotation_range=90,
                         brightness_range=[0.2,1.0],
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         vertical_flip=True,
                         fill_mode='reflect')

#add random rotation, brightness  adjustment, geograohic augmentation
mask_data_gen_args = dict(rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          vertical_flip=True,
                          fill_mode='reflect',
                          )  # Binarize the output again.

#apply adjustment to both training and validation images and masks
image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_data_generator.fit(X_train, augment=True, seed=seed)

image_generator = image_data_generator.flow(X_train, seed=seed)
valid_img_generator = image_data_generator.flow(X_test, seed=seed)

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_data_generator.fit(y_train, augment=True, seed=seed)
mask_generator = mask_data_generator.flow(y_train, seed=seed)
valid_mask_generator = mask_data_generator.flow(y_test, seed=seed)


def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


my_generator = my_image_mask_generator(image_generator, mask_generator)

validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)

print("compile completed")
print(model.summary())

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#print training start time
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

history = model.fit(my_generator,
                    batch_size=batchsize,
                    verbose=1,
                    epochs=epoch,
                    validation_data=validation_datagen,
                    callbacks=model_checkpoint_callback,
                    shuffle=False)




#history = model.fit(X_train, y_train_cat,
                    #batch_size=batchsize,
                    #verbose=1,
                    #epochs=epoch,
                    #validation_data=(X_test, y_test_cat),
                    #callbacks=model_checkpoint_callback,
                    # class_weight=class_weights,
                    #shuffle=False)




#history = model.fit(train_data,
                    #verbose=1,
                    #epochs=epoch,
                    #validation_data=val_data,
                    #callbacks=model_checkpoint_callback,
                    #shuffle=False)

#print training end time 
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

#save model weight after training 
save_name = "Simple_Unet_model_" + str(epoch) + "_epoch_" + "final_weight_" + str(Date_name) + "_" + str(batchsize) + "_batch_size_" + str(initial_learning_rate) + "_LR_Decay_gpu.hdf5"
model.save(save_name)

#model.load_weights('/home/hl46161/PycharmProjects/machine_learning_image_recongization/Simple_Unet_model_100_epoch_best_save_with_sample_weight_May_22th_32_batch_size_0.001_LR_Decay_gpu.hdf5')
#model.load_weights('/home/hl46161/PycharmProjects/machine_learning_image_recongization/Simple_Unet_model_200_epoch_best_save_with_sample_weight_May_22th_64_batch_size_0.005_LR_Decay_gpu.hdf5')
#model.load_weights('/home/hl46161/PycharmProjects/machine_learning_image_recongization/Simple_Unet_model_200_epoch_best_save_with_sample_weight_May_22th_64_batch_size_0.0005_LR_Decay_gpu.hdf5')

#load previsou optimized weight for prediction 
model.load_weights('/home/hl46161/PycharmProjects/machine_learning_image_recongization/Simple_Unet_model_200_epoch_best_save_with_sample_weight_June_6th_32_batch_size_0.0001_LR_Decay_gpu.hdf5')

#use model to predict validation dataset
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)

##################################################

# Using built in keras function
from keras.metrics import MeanIoU

n_classes = 5
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:, :, :, 0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)

# To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)

class1_IoU = values[0, 0] / (
            values[0, 0] + values[0, 1] + values[0, 2] + values[1, 0] + values[2, 0])

class2_IoU = values[1, 1] / (
            values[1, 1] + values[1, 0] + values[1, 2] + values[0, 1] + values[2, 1])

class3_IoU = values[2, 2] / (
            values[2, 2] + values[2, 0] + values[2, 1] + values[0, 2] + values[1, 2])

class4_IoU = values[3, 3] / (
        values[3, 3] + values[3, 0] + values[3, 1] + values[3, 2] + values[3, 4] + values[0, 3] + values[1, 3] + values[2, 3] + values[4, 3])

class5_IoU = values[4, 4] / (
        values[4, 4] + values[4, 0] + values[4, 1] + values[4, 2] + values[4, 3] + values[0, 4] + values[1, 4] + values[2, 4] + values[3, 4])


print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)
print("IoU for class5 is: ", class5_IoU)

######################################################################

#choose random picture from validation images and see model prediction on it
import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_norm = test_img[:, :,:][:, :,:]
test_img_input = np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, 0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:, :, 0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()
