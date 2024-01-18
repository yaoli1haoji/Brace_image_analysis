import tensorflow as tf
from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import tifffile as tiff
import focal_loss
from scipy import ndimage
from skimage import measure, color, io
import segmentation_models as sm


for directory_path in glob.glob(f"/home/hl46161/brace_root/2019_annotation_remaster_2/fran pt3 - export(4)/*/"):
    print(directory_path)
    sample_name = directory_path.split("/")[-2]
    braceroot_path = directory_path + sample_name + "_braceroot.ome.tiff"
    Stalk_path = directory_path + sample_name + "_stalk.ome.tiff"
    whitelabel_path = directory_path + sample_name + "_whitelabel.ome.tiff"
    print(braceroot_path)
    #plt.imshow(edges)
    #plt.show()

    brace_mask = cv2.imread(braceroot_path,cv2.IMREAD_UNCHANGED)
    brace_mask = cv2.rotate(brace_mask, cv2.ROTATE_90_CLOCKWISE)
    brace_mask = np.array(brace_mask)

    #convert format
    brace_mask_8bit = (brace_mask / brace_mask.max()) * 255
    brace_mask_8bit = brace_mask_8bit.astype(int)
    brace_mask_8bit = np.uint8(brace_mask_8bit)
    brace_mask_8bit = np.array(brace_mask_8bit)
    #print(np.unique(brace_mask_8bit))

    #plt.imshow(brace_mask_8bit)
    #plt.show()

    #find the contour of brace root
    edges = cv2.Canny(image=brace_mask_8bit, threshold1=0, threshold2=1)

    #dilate the contour edges
    kernel_size = 3
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Kernel to be used for dilation
    dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=2)
    border_index = np.where(dilated_edges == 255)

    #
    mask_with_border = brace_mask_8bit
    mask_with_border[border_index] = 1

    ret, thresh2 = cv2.threshold(mask_with_border, 3, 255, cv2.THRESH_BINARY)
    braceroot_index = np.where(thresh2 == 255)

    whitelabel_mask = cv2.imread(whitelabel_path,cv2.IMREAD_UNCHANGED)
    whitelabel_mask = cv2.rotate(whitelabel_mask, cv2.ROTATE_90_CLOCKWISE)
    whitelabel_mask = np.array(whitelabel_mask)
    # masks = struct.unpack('B' * (4*len(masks)), buffer(masks))
    # np.unique(masks)
    whitelabel_mask_8bit = (whitelabel_mask / whitelabel_mask.max()) * 255
    whitelabel_mask_8bit = whitelabel_mask_8bit.astype(int)
    whitelabel_index = np.where(whitelabel_mask_8bit > 0)
    #print(np.unique(whitelabel_mask_8bit))

    stalk_mask = cv2.imread(Stalk_path,cv2.IMREAD_UNCHANGED)
    stalk_mask = cv2.rotate(stalk_mask, cv2.ROTATE_90_CLOCKWISE)
    stalk_mask = np.array(stalk_mask)

    # masks = struct.unpack('B' * (4*len(masks)), buffer(masks))
    # np.unique(masks)
    stalk_mask_8bit = (stalk_mask / stalk_mask.max()) * 255
    stalk_mask_8bit = stalk_mask_8bit.astype(int)
    stalk_mask_8bit = np.uint8(stalk_mask_8bit)
    stalk_mask_8bit = np.array(stalk_mask_8bit)
    #print("stalk_mask_8bit index")
    #print(np.unique(stalk_mask_8bit))

    stalk_index = np.where(stalk_mask_8bit > 0)

    #fuse brace root, whitlable, brace border together into one picuture
    stalk_mask_8bit[stalk_index] = 1
    stalk_mask_8bit[braceroot_index] = 2
    stalk_mask_8bit[whitelabel_index] = 3
    stalk_mask_8bit[border_index] = 4

    print("after assgin index")
    print(np.unique(stalk_mask_8bit))

    tiff.imwrite("/home/hl46161/brace_root/2019_annotation_remaster_2/fran pt3 - export(4)/" + sample_name + "_braceroot_stalk_border_whitelabel.ome.tiff",stalk_mask_8bit)
    print("/home/hl46161/brace_root/2019_annotation_remaster_2/fran pt3 - export(4)/" + sample_name + "_braceroot_stalk_border_whitelabel.ome.tiff")

#masks = struct.unpack('B' * (4*len(masks)), buffer(masks))
#np.unique(masks)


np.unique(brace_mask_8bit)

edges = cv2.Canny(image=brace_mask_8bit, threshold1=0, threshold2=1)
plt.imshow(edges)
plt.show()

kernel_size = 3
dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Kernel to be used for dilation
dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=2)
border_index = np.where(dilated_edges == 255)

mask_with_border = brace_mask_8bit
mask_with_border[border_index] = 3
np.unique(mask_with_border)

#np.unique(brace_mask_8bit)
plt.imshow(mask_with_border)
plt.show()

ret,thresh2 = cv2.threshold(mask_with_border,4,255,cv2.THRESH_BINARY)
braceroot_index = np.where(thresh2 == 255)

plt.imshow(thresh2)
plt.show()
np.unique(thresh2)

whitelabel_mask = cv2.imread("/home/hl46161/brace_root/2019_new_annotation/fran pt3 - export(2)/1270_plant_5A_whitelabel.ome.tiff",cv2.IMREAD_UNCHANGED)
whitelabel_mask = cv2.rotate(whitelabel_mask, cv2.ROTATE_90_CLOCKWISE)
whitelabel_mask = np.array(whitelabel_mask)
#masks = struct.unpack('B' * (4*len(masks)), buffer(masks))
#np.unique(masks)
whitelabel_mask_8bit = (whitelabel_mask/whitelabel_mask.max()) * 255
whitelabel_mask_8bit = whitelabel_mask_8bit.astype(int)
whitelabel_index = np.where(whitelabel_mask_8bit == 255)


stalk_mask = cv2.imread("/home/hl46161/brace_root/2019_new_annotation/fran pt2 - export(2)/1270_plant_5A_Stalk.ome.tiff",cv2.IMREAD_UNCHANGED)
stalk_mask = cv2.rotate(stalk_mask, cv2.ROTATE_90_CLOCKWISE)
stalk_mask = np.array(stalk_mask)
#masks = struct.unpack('B' * (4*len(masks)), buffer(masks))
#np.unique(masks)

stalk_mask_8bit = (stalk_mask/stalk_mask.max()) * 255
stalk_mask_8bit = stalk_mask_8bit.astype(int)
stalk_mask_8bit = np.uint8(stalk_mask_8bit)
stalk_mask_8bit = np.array(stalk_mask_8bit)

stalk_index = np.where(stalk_mask_8bit == 255)

stalk_mask_8bit[stalk_index] = 1
stalk_mask_8bit[braceroot_index] = 2
stalk_mask_8bit[whitelabel_index] = 3
stalk_mask_8bit[border_index] = 4


np.unique(stalk_mask_8bit)
plt.imshow(stalk_mask_8bit)
plt.show()


whitelabel_mask_8bit[braceroot_index] = 200
whitelabel_mask_8bit[stalk_index] = 150

#np.unique(whitelabel_mask_8bit)
plt.imshow(whitelabel_mask_8bit)
plt.show()








