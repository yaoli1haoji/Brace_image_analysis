from simple_multi_unet_model import multi_unet_model 
import tensorflow as tf
from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import segmentation_models as sm
from keras.metrics import MeanIoU
from Unet3plus import unet3plus
from Unet3plus_deepsup_cgm import unet3plus_deepsup_cgm
import albumentations as A
import random
from Unet3plus_RL import unet3plusmore
from A_Unet3plus import A_unet3plus

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="3"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

n_classes = 4

#Number of classes for segmentation

#train_images_path_list = []
#train_masks_path_list = []

#Capture training image info as a list
train_images = []
train_masks = []


# Declare an augmentation pipeline
#transform = A.Compose([
    #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4),
    #A.GaussNoise(),
    #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2,p=0.5),
    #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    #A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1)
#])

random.seed(7)

for directory_path in glob.glob(f"./2019_final_annotation/*/"):
    print(directory_path)
    train_images_path_list = []
    train_masks_path_list = []
    train_image_path=directory_path + "patches_images_resize_400/"
    train_masks_path = directory_path + "patches_masks_resize_400/"
    train_image_flip_path = directory_path + "patches_images_resize_400_flip/"
    train_masks_flip_path = directory_path + "patches_masks_resize_400_flip/"
    #print(train_image_path)
    #print(train_masks_path)
    
    for img_path in glob.glob(os.path.join(train_image_path,"*.tif")):
      #img = cv2.imread(img_path, 0)
      #print(img)
      train_images_path_list.append(img_path)
      #img = cv2.resize(img, (SIZE_Y, SIZE_X))
      #type(train_images)
      #train_images.append(img)
    

    for mask_path in glob.glob(os.path.join(train_masks_path,"*.tif")):
         #print(mask_path)
         #mask = cv2.imread(mask_path, 0)
         #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
         train_masks_path_list.append(mask_path)
         #train_masks.append(mask)
        
    for flip_img_path in glob.glob(os.path.join(train_image_flip_path, "*.tif")):

        train_images_path_list.append(flip_img_path)

    for flip_mask_path in glob.glob(os.path.join(train_masks_flip_path, "*.tif")):
      
        train_masks_path_list.append(flip_mask_path)
     
    train_images_path_list.sort()
    train_masks_path_list.sort()

    #print(train_masks_path_list)
    #print(train_images_path_list)

    for index in range(len(train_images_path_list)):
        #print(image_path)
        #img = cv2.imread(image_path, 1)
        #add more information on img 
        #hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #IG = img[:,:,1]
        #IB = img[:,:,0]
        #IR = img[:,:,2]

        #ExG = (2*IG-IR-IB)
        #ExG = np.expand_dims(ExG, axis=2)

        #ExR = (1.4*IR-IG)
        #ExR = np.expand_dims(ExR, axis=2)

        #CIVE = (0.881*IG-0.441*IR-0.385*IB-18.78745)
        #CIVE = np.expand_dims(CIVE, axis=2)
       
        #laplacian = cv2.Laplacian(ExG,cv2.CV_64F)
        #laplacian = np.expand_dims(laplacian, axis=2)
        #sobelx = cv2.Sobel(ExG,cv2.CV_64F,1,0,ksize=5)  # x
        #sobelx = np.expand_dims(sobelx, axis=2)
        #sobely = cv2.Sobel(ExG,cv2.CV_64F,0,1,ksize=5)  # y
        #sobely = np.expand_dims(sobely, axis=2)

        #INDI =  (IG-IR)/(IG+IR)
        #INDI = np.expand_dims(INDI, axis=2)
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray_img = np.expand_dims(gray_img, axis=2)

        #revised_img = img.copy()
        #revised_img = np.concatenate([img,hsv_img], axis=2)
        #revised_img = np.concatenate([revised_img,ExG], axis=2)
        #revised_img = np.concatenate([revised_img,ExR], axis=2)
        #revised_img = np.concatenate([revised_img,CIVE], axis=2)
        #revised_img = np.concatenate([revised_img,laplacian], axis=2)
        #revised_img = np.concatenate([revised_img,sobelx], axis=2)
        #revised_img = np.concatenate([revised_img,sobely], axis=2)
        
        img = cv2.imread(train_images_path_list[index], 1)
        mask = cv2.imread(train_masks_path_list[index], 0)
        stalk_index = np.where(mask == 1)
        mask[stalk_index] = 0
        #transformed = transform(image=img, mask=mask)
        #transformed_image = transformed['image']
        #transformed_mask = transformed['mask']
        #train_images.append(transformed_image)
        #train_masks.append(transformed_mask)
        train_images.append(img)
        train_masks.append(mask)


        #revised_img = np.concatenate([img,gray_img], axis=2)
        #train_images.append(revised_img)

    #for mask_path in train_masks_path_list:
        #mask = cv2.imread(mask_path, 0)
        #stalk_index = np.where(mask == 1)
        #mask[stalk_index] = 0
        #train_masks.append(mask)


# Convert list to array for machine learning processing
train_images = np.array(train_images)
train_masks = np.array(train_masks)

print("check the shape of mask list")
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
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

print("check the shape of reshaped mask list")
print(train_masks_reshaped)

train_masks_reshaped_encoded.reshape(n, h, w)

#np.unique(train_masks_encoded_original_shape)

#train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)
train_images = train_images/255
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)



#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(train_images, train_masks_input, test_size=0.20, random_state=0)

#np.unique(y_train)
#y_train.shape

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


from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                  classes = np.unique(train_masks_reshaped_encoded),
                                                  y = train_masks_reshaped_encoded)
#class_weights[2] = class_weights[2] * 3
print("Class weights are...:", class_weights)


class_weights_dict = dict(enumerate(class_weights))
type(class_weights_dict)
print(class_weights_dict)

#class_weights_list = list(class_weights)
#class_weights_list
#X_train.shape


#Reused parameters in all models

n_classes=4
activation='softmax'

initial_learning_rate = 0.001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=216,
        decay_rate=0.95,
        staircase=True)

optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
focal_loss = sm.losses.CategoricalFocalLoss()
Jaccard_loss = sm.losses.JaccardLoss()
total_loss = dice_loss + (1 * focal_loss) + Jaccard_loss


#class_weights
#total_loss
# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


#train_data = tf.data.Dataset.from_tensor_slices((X_train,y_train_cat))
#val_data = tf.data.Dataset.from_tensor_slices((X_test,y_test_cat))

batchsize=32
#train_data = train_data.batch(batchsize,drop_remainder=True)
#val_data = val_data.batch(batchsize,drop_remainder=True)

from segmentation_models import Unet
#define model

epoch=250
Date_name = 'Sep_14th'

checkpoint_filepath =  "Vanilla_Unet_model_" + str(epoch) + "_epoch_" + "best_save_with_sample_weight_" + str(Date_name) + "_" + str(batchsize) + "_batch_size_" + str(initial_learning_rate) + "_LR_Decay_gpu.hdf5"


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)


print("start training")
print("Try multiple thread on cpu")
from keras import backend as K
from datetime import datetime





def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

def get_unet3plus_model():
    return unet3plus(input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), output_channels = n_classes)

def get_unet3plus_deepsup_cgm():
    return unet3plus_deepsup_cgm(input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), output_channels = n_classes,training=True)

def get_unet3plus_deepsup_cgm():
    return unet3plus_deepsup(input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), output_channels = n_classes,training=True)

def get_unet3plus_RL_model():
    return unet3plusmore(input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), output_channels = n_classes,dropout=0.1)

def get_A_Unet3plus_model():
    return A_unet3plus(input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), output_channels = n_classes,dropout=0.1)



gpus = tf.config.list_physical_devices('GPU')
#strategy = tf.distribute.MirroredStrategy(gpus)
#if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #try:
        #for gpu in gpus:
          #tf.config.experimental.set_virtual_device_configuration(gpu,
          #[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])
          #tf.config.experimental.set_memory_growth(gpu, True)
          #tf.config.set_logical_device_configuration(gpu,[tf.config.LogicalDeviceConfiguration(memory_limit=71680)])

        #logical_gpus = tf.config.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        #print(e)

#X_train1 = X_train/255
#X_test1 = X_test/255


strategy = tf.distribute.MultiWorkerMirroredStrategy()


with strategy.scope():
    model = get_model()
    model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

#with strategy.scope():
  #model = get_unet3plus_model()
  #model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

#with strategy.scope():
  #model = get_unet3plus_RL_model()
  #model.compile(optimizer=optim, loss=total_loss, metrics=metrics)


#with strategy.scope():
  #model = get_A_Unet3plus_model()
  #model.compile(optimizer=optim, loss=total_loss, metrics=metrics)


print("compile completed")
print(model.summary())


#set seed for image augmentation
#seed = 24

#use keras image augmentation utilis
#from keras.preprocessing.image import ImageDataGenerator

#add random rotation, brightness  adjustment, geograohic augmentation
#img_data_gen_args = dict(rotation_range=90,
                         #brightness_range=[0.2,1.0],
                         #width_shift_range=0.3,
                         #height_shift_range=0.3,
                         #shear_range=0.5,
                         #zoom_range=0.3,
                         #vertical_flip=True,
                         #fill_mode='reflect')

#add random rotation, brightness  adjustment, geograohic augmentation
#mask_data_gen_args = dict(rotation_range=90,
                          #width_shift_range=0.3,
                          #height_shift_range=0.3,
                          #shear_range=0.5,
                          #zoom_range=0.3,
                          #vertical_flip=True,
                          #fill_mode='reflect',
                          #)  # Binarize the output again.

#apply adjustment to both training and validation images and masks
#image_data_generator = ImageDataGenerator(**img_data_gen_args)
#image_data_generator.fit(X_train, augment=True, seed=seed)

#image_generator = image_data_generator.flow(X_train, seed=seed)
#valid_img_generator = image_data_generator.flow(X_test, seed=seed)

#mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
#mask_data_generator.fit(y_train, augment=True, seed=seed)
#mask_generator = mask_data_generator.flow(y_train, seed=seed)
#valid_mask_generator = mask_data_generator.flow(y_test, seed=seed)


#def my_image_mask_generator(image_generator, mask_generator):
    #train_generator = zip(image_generator, mask_generator)
    #for (img, mask) in train_generator:
        #yield (img, mask)


#my_generator = my_image_mask_generator(image_generator, mask_generator)

#validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)


print("image augmentation done")



epoch=250
batchsize=32
Date_name = 'Sep_14th'


#print("compile completed")
#print(model.summary())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#now = datetime.now()
#current_time = now.strftime("%H:%M:%S")
#print("Current Time =", current_time)

#history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=steps_per_epoch,
                    #validation_steps=steps_per_epoch, epochs=50)

history = model.fit(X_train, y_train_cat,
                    batch_size=batchsize,
                    verbose=1,
                    epochs=epoch,
                    validation_data=(X_test, y_test_cat),
                    callbacks=model_checkpoint_callback,
                    shuffle=False)

#history = model.fit(my_generator,
                    #batch_size=batchsize,
                    #verbose=1,
                    #epochs=epoch,
                    #validation_data=validation_datagen,
                    #callbacks=model_checkpoint_callback,
                    #shuffle=False)




#history = model.fit(train_data,
                    #verbose=1,
                    #epochs=epoch,
                    #validation_data=val_data,
                    #callbacks=model_checkpoint_callback,
                    #shuffle=False)





#now = datetime.now()
#current_time = now.strftime("%H:%M:%S")
#print("Current Time =", current_time)


#save_name = "Simple_Unet_model_" + str(epoch) + "_epoch_" + "final_weight_" + str(Date_name) + "_" + str(batchsize) + "_batch_size_" + str(initial_learning_rate) + "_LR_Decay_gpu.hdf5"
#model.save(save_name)
#model.save('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')
############################################################
# Evaluate the model
# evaluate model
test_loss, test_acc, *is_anything_else_being_returned = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (test_acc * 100.0), "%")

###
# plot the training and validation accuracy and loss at each epoch
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
val_loss_save_name = "Vanilla_Unet" + "_backbone_model_" + str(epoch) + "_epoch_" + str(Date_name) + "_" + str(batchsize) + "_batch_size_" + str(initial_learning_rate) + "_val_loss.png"
plt.savefig(val_loss_save_name)
plt.clf()


acc = history.history['iou_score']
val_acc = history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
val_IOU_save_name = "Vanilla_Unet" + "_backbone_model_" + str(epoch) + "_epoch_" + str(Date_name) + "_" + str(batchsize) + "_batch_size_" + str(initial_learning_rate) + "_val_IOU.jpg"
plt.savefig(val_IOU_save_name)
plt.clf()


##################################
# model = get_model()
model.load_weights(checkpoint_filepath)
# model.load_weights('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')

# IOU
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)

##################################################

# Using built in keras function
from keras.metrics import MeanIoU

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:, :, :, 0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# To calculate I0U for each class...
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
#print("IoU for class5 is: ", class5_IoU)

#plt.imshow(train_images[0, :, :, 0], cmap='gray')
#plt.imshow(train_masks[0], cmap='gray')

"""
#######################################################################
# Predict on a few images
# model = get_model()
# model.load_weights('???.hdf5')
import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_norm = test_img[:, :, 0][:, :, None]
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
plt.savefig("test on small image")
plt.clf()

#####################################################################

# Predict on large image

# Apply a trained model on large image

from patchify import patchify, unpatchify

large_image = cv2.imread('large_images/large_image.tif', 0)
# This will split the image into small images of shape [3,3]
patches = patchify(large_image, (128, 128), step=128)  # Step=256 for 256 patches means no overlap

predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(i, j)

        single_patch = patches[i, j, :, :]
        #single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1), 2)
        
        single_patch_input = np.expand_dims(single_patch_norm, 0)
        single_patch_prediction = (model.predict(single_patch_input))
        single_patch_predicted_img = np.argmax(single_patch_prediction, axis=3)[0, :, :]

        predicted_patches.append(single_patch_predicted_img)

predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 128, 128))

reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
plt.imshow(reconstructed_image, cmap='gray')
# plt.imsave('data/results/segm.jpg', reconstructed_image, cmap='gray')

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(reconstructed_image, cmap='jet')
plt.show()
plt.savefig("test on large image")
plt.clf()

"""
