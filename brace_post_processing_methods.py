import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify

def image_readin_prediction(image_path,model):
    # try predict the large image


    large_image = cv2.imread(image_path,1)
    plt.imshow(large_image, cmap='gray')
    plt.show()

    # This will split the image into small images of shape [3,3]
    patches = patchify(large_image, (400, 400, 3), step=400)  # Step=256 for 256 patches means no overlap

    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            print(i, j)

            single_patch = patches[i, j, :, :]
            single_patch_input = single_patch / 255
            single_patch_prediction = (model.predict(single_patch_input))
            single_patch_predicted_img = np.argmax(single_patch_prediction, axis=3)[0, :, :]
            predicted_patches.append(single_patch_predicted_img)

    predicted_patches = np.array(predicted_patches)

    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 400, 400))

    large_image = cv2.imread(image_path,0)

    reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)

    return reconstructed_image


def border_polishing(reconstructed_image,kernel_size):

    #remove white label pixel
    temp_img = reconstructed_image.copy()
    temp_img[np.where(temp_img == 2)] = 0

    kernel = np.ones(kernel_size, np.uint8)

    ret1, thresh = cv2.threshold(temp_img, 2, 255, 0)

    #enlarge the border pixel
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    #use morphology closing and openning to remove noise and close hole
    closing_polished_border = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=1)
    closing_polished_border = cv2.morphologyEx(closing_polished_border, cv2.MORPH_OPEN, kernel, iterations=1)

    reconstructed_image_polished = reconstructed_image.copy()
    reconstructed_image_polished[np.where(closing_polished_border == 255)] = 3

    reconstructed_image_border_index = np.where(reconstructed_image_polished == 3)

    reconstructed_image_polished[reconstructed_image_border_index] = 0

    return reconstructed_image_polished

def white_label_parameter(reconstructed_image_white_label,connectivity):

    mask = np.zeros(reconstructed_image_white_label.shape, dtype="uint8")
    #use build in function for find white label contour
    output = cv2.connectedComponentsWithStats(reconstructed_image_white_label, connectivity, cv2.CV_32S)

    # Get the results
    # The first cell is the number of labels
    numLabels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    white_label_x = None
    white_label_y = None
    white_label_w = None
    white_label_h = None

    # loop over the number of unique connected component labels
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format(i + 1, numLabels)
        # print a status message update for the current connected
        # component
        print("[INFO] {}".format(text))
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        print("examining x of " + str(i) + "th numLabels " + str(x))
        print("examining y of " + str(i) + "th numLabels " + str(y))
        print("examining w of " + str(i) + "th numLabels " + str(w))
        print("examining h of " + str(i) + "th numLabels " + str(h))
        print("examining area of " + str(i) + "th numLabels " + str(area))


        keepWidth = w > 5 and w < 600
        keepArea = area > 200
        # ensure the connected component we are examining passes all
        # three tests
        if all((keepWidth, keepArea)):
            # construct a mask for the current connected component and
            # then take the bitwise OR with the mask
            print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * (i + 10)
            mask = cv2.bitwise_or(mask, componentMask)

            white_label_x = x
            white_label_y = y
            white_label_w = w
            white_label_h = h

    return white_label_x, white_label_y, white_label_w,white_label_h

###############################################################################

def pixelTolength(objectlengthInPixel, whitelabelheightInPixel):
    # how mang inches per pixel?
    reallengthPerPixel = 0.5 / whitelabelheightInPixel
    #print("reallengthPerPixel: ", reallengthPerPixel)
    # object length in inches
    objectlength = reallengthPerPixel * objectlengthInPixel
    return objectlength

###############################################################################

def spreath_width(brace_root_contours,white_label_h):

    threshold_area = 200
    leftmost = 800
    rightmost = 0
    topmost = 0
    bottommost = 0

    topmost_list = []

    for cnt in brace_root_contours:
        area = cv2.contourArea(cnt)
        if area > threshold_area:
            area = cv2.contourArea(cnt)
            temp_leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            temp_rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            temp_topmost_point = tuple(cnt[cnt[:, :, 1].argmin()][0])
            topmost_list.append(temp_topmost_point[1])

            if temp_leftmost[0] < leftmost:
                leftmost = temp_leftmost[0]

            if temp_rightmost[0] > rightmost:
                rightmost = temp_rightmost[0]

    spread_width_pixel = rightmost - leftmost
    spread_width_length = pixelTolength(spread_width_pixel, white_label_h)

    return leftmost,rightmost,spread_width_length

###################################################################################

def brace_root_count(reconstructed_image_brace_root,connectivity):

    y_list = []
    mask = np.zeros(reconstructed_image_brace_root.shape, dtype="uint8")
    #use build in function for find white label contour
    output = cv2.connectedComponentsWithStats(reconstructed_image_brace_root, connectivity, cv2.CV_32S)

    # Get the results
    # The first cell is the number of labels
    numLabels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    white_label_x = None
    white_label_y = None
    white_label_w = None
    white_label_h = None

    # loop over the number of unique connected component labels
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format(i + 1, numLabels)
        # print a status message update for the current connected
        # component
        print("[INFO] {}".format(text))
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        print("examining x of " + str(i) + "th numLabels " + str(x))
        print("examining y of " + str(i) + "th numLabels " + str(y))
        print("examining w of " + str(i) + "th numLabels " + str(w))
        print("examining h of " + str(i) + "th numLabels " + str(h))
        print("examining area of " + str(i) + "th numLabels " + str(area))


        keepWidth = w > 5 and w < 600
        keepArea = area > 200
        # ensure the connected component we are examining passes all
        # three tests
        if all((keepWidth, keepArea)):
            # construct a mask for the current connected component and
            # then take the bitwise OR with the mask
            print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * (i + 10)
            mask = cv2.bitwise_or(mask, componentMask)
            y_list.append(y)
            plt.imshow(mask)
            plt.show()

    return y_list

####################################################################################

def whorls_grouper(iterable):
    prev = None
    group = []
    for item in iterable:
        if prev is None or item - prev <= 50:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def brace_num_per_whorl_count(brace_root_dictionary):
    brace_num_per_whorl_list = []
    for whorl in brace_root_dictionary.values():
        print(whorl)
        whorl_brace_root_num = len(whorl)
        brace_num_per_whorl_list.append(whorl_brace_root_num)

    brace_num_per_whorl_list.reverse()
    return(brace_num_per_whorl_list)