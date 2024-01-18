import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import cv2
import glob
import os
from PIL import Image



def image_cut_off(input_image):

    test_image = input_image[:,1000:3000]
    old_image_height, old_image_width, channels = test_image.shape
    new_image_height,new_image_width,channels_num = input_image.shape
    color = (0, 0, 0)
    empty_image = np.full((new_image_height, new_image_width,channels), color, dtype=np.uint8)

    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    empty_image[y_center:y_center + old_image_height,
    x_center:x_center + old_image_width] = test_image

    return empty_image

def mask_cut_off(input_mask):

    test_mask = input_mask[:,1000:3000]
    old_mask_height,old_mask_width,channels = test_mask.shape
    new_mask_height,new_mask_width,channels_num = input_mask.shape
    color = (0, 0, 0)
    empty_mask = np.full((new_mask_height, new_mask_width,channels), color, dtype=np.uint8)

    x_center = (new_mask_width - old_mask_width) // 2
    y_center = (new_mask_height - old_mask_height) // 2

    # copy img image into center of result image
    empty_mask[y_center:y_center + old_mask_height,
    x_center:x_center + old_mask_width] = test_mask

    return empty_mask



for directory_path in glob.glob(f"/home/hl46161/brace_root/2019_annotation_remaster_2/2019_final_annotation/*/"):
    print(directory_path)
    #image_path = directory_path
    image_path=directory_path + "images/"
    mask_path = directory_path + "mask/"

    for img_path in glob.glob(os.path.join(image_path, "*.jpg")):
        print(img_path)
        large_image_stack = cv2.imread(img_path)
        #large_image_stack = image_cut_off(large_image_stack)
        #large_image_stack = cv2.rotate(large_image_stack, cv2.cv2.ROTATE_90_CLOCKWISE)
        large_image_stack = cv2.resize(large_image_stack, (1600,2400))
        large_image_stack = cv2.cvtColor(large_image_stack, cv2.COLOR_BGR2RGB)
        #large_image_stack_horizontally = cv2.flip(large_image_stack, 1)
        #cropped_image = large_image_stack[184:5816, 208:3792]
        #large_image_stack = cv2.resize(large_image_stack, (9375, 2560))
        #save the resize images
        if not os.path.exists(directory_path + 'images_resize_1600_2400'):
            os.makedirs(directory_path + 'images_resize_1600_2400')
        tiff.imwrite(
            directory_path + "images_resize_1600_2400/" + directory_path.split("/")[6] + "_resize.tif",large_image_stack)

    #use ome_1, the one without contrast since it has the correct class information
    for mask_path in glob.glob(os.path.join(mask_path, "*.ome.tiff")):
        #print(mask_path)
        large_mask_stack = cv2.imread(mask_path,)
        #large_mask_stack = mask_cut_off(large_mask_stack)
        #cropped_mask = large_mask_stack[184:5816, 208:3792]
        #large_mask_stack = cv2.rotate(large_mask_stack, cv2.cv2.ROTATE_90_CLOCKWISE)
        large_mask_stack = cv2.resize(large_mask_stack, (1600, 2400))
        if not os.path.exists(directory_path + 'mask_resize_1600_2400'):
            os.makedirs(directory_path + 'mask_resize_1600_2400')
        tiff.imwrite(
            directory_path + "mask_resize_1600_2400/" + directory_path.split("/")[6] + "_resize.tif",large_mask_stack)


        #large_mask_stack_horizontally = cv2.flip(large_mask_stack, 1)
        #large_mask_stack = cv2.resize(large_mask_stack,(9375,2560))

    patches_img = patchify(large_image_stack, (800,800,3),step=400)
    patches_mask = patchify(large_mask_stack, (800,800,3),step=400)
    #patches_img.shape
    #patches_mask.shape
    #np.unique(large_mask_stack)

    for i in range(patches_img.shape[0]):
        if i == 1 or i == 3:
            continue
        for j in range(patches_img.shape[1]):
             single_patch_img = patches_img[i, j, :, :]
             if not os.path.exists(directory_path + 'patches_images_resize_800'):
                 os.makedirs(directory_path + 'patches_images_resize_800')
             tiff.imwrite(
                 directory_path + "/patches_images_resize_800/" + 'image_' + '_' + str(
                     i) + str(j) + ".tif", single_patch_img)

    for i in range(patches_mask.shape[0]):
        if i == 1 or i == 3:
            continue
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i, j, :, :]
            if not os.path.exists(directory_path + 'patches_masks_resize_800'):
               os.makedirs(directory_path + 'patches_masks_resize_800')
            tiff.imwrite(
                directory_path + "/patches_masks_resize_800/" + 'mask_' + '_' + str(
                    i) + str(j) + ".tif", single_patch_mask)





for directory_path in glob.glob(f"/home/hl46161/brace_root/2019_annotation_remaster_2/2019_final_annotation/*/"):
    print(directory_path)
    #image_path = directory_path
    image_path=directory_path + "images/"
    mask_path = directory_path + "mask/"

    for img_path in glob.glob(os.path.join(image_path, "*.jpg")):
        print(img_path)
        large_image_stack = cv2.imread(img_path)
        #large_image_stack = image_cut_off(large_image_stack)
        #large_image_stack = cv2.rotate(large_image_stack, cv2.cv2.ROTATE_90_CLOCKWISE)
        large_image_stack = cv2.resize(large_image_stack, (1600,2400))
        large_image_stack = cv2.flip(large_image_stack, 1)
        large_image_stack = cv2.cvtColor(large_image_stack, cv2.COLOR_BGR2RGB)
        #cropped_image = large_image_stack[184:5816, 208:3792]
        #large_image_stack = cv2.resize(large_image_stack, (9375, 2560))
        #save the resize images
        if not os.path.exists(directory_path + 'images_resize_1600_2400_flip'):
            os.makedirs(directory_path + 'images_resize_1600_2400_flip')
        tiff.imwrite(
            directory_path + "images_resize_1600_2400_flip/" + directory_path.split("/")[6] + "_resize.tif",large_image_stack)

    #use ome_1, the one without contrast since it has the correct class information
    for mask_path in glob.glob(os.path.join(mask_path, "*.ome.tiff")):
        #print(mask_path)
        large_mask_stack = cv2.imread(mask_path)
        #cropped_mask = large_mask_stack[184:5816, 208:3792]
        #large_mask_stack = cv2.rotate(large_mask_stack, cv2.cv2.ROTATE_90_CLOCKWISE)
        #large_mask_stack = mask_cut_off(large_mask_stack)
        large_mask_stack = cv2.resize(large_mask_stack, (1600, 2400))
        large_mask_stack = cv2.flip(large_mask_stack, 1)
        #large_mask_stack = cv2.resize(large_mask_stack,(9375,2560))
        if not os.path.exists(directory_path + 'mask_resize_1600_2400_flip'):
            os.makedirs(directory_path + 'mask_resize_1600_2400_flip')
        tiff.imwrite(
            directory_path + "mask_resize_1600_2400_flip/" + directory_path.split("/")[6] + "_resize.tif", large_mask_stack)

    patches_img = patchify(large_image_stack, (800,800,3),step=400)
    patches_mask = patchify(large_mask_stack, (800,800,3),step=400)
    #patches_img.shape
    #patches_mask.shape
    #np.unique(large_mask_stack)

    for i in range(patches_img.shape[0]):
        if i == 1 or i == 3:
            continue
        for j in range(patches_img.shape[1]):
             single_patch_img = patches_img[i, j, :, :]
             if not os.path.exists(directory_path + 'patches_images_resize_800_flip'):
                 os.makedirs(directory_path + 'patches_images_resize_800_flip')
             tiff.imwrite(
                 directory_path + "/patches_images_resize_800_flip/" + 'image_' + '_' + str(
                     i) + str(j) + ".tif", single_patch_img)

    for i in range(patches_mask.shape[0]):
        if i == 1 or i == 3:
            continue
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i, j, :, :]
            if not os.path.exists(directory_path + 'patches_masks_resize_800_flip'):
               os.makedirs(directory_path + 'patches_masks_resize_800_flip')
            tiff.imwrite(
                directory_path + "/patches_masks_resize_800_flip/" + 'mask_' + '_' + str(
                    i) + str(j) + ".tif", single_patch_mask)
