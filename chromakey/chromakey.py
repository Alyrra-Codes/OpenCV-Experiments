import cv2 as cv
import numpy as np
import sys
##################
# Define functions
##################

def rescaleBGround(img):
    '''
    Resize the background to an even number below 640x360
    '''
    # get dimensions
    width   = int(img.shape[1])
    height  = int(img.shape[0])
    # reduce image size to an even number less than 640x360 
    # (it has to be even to place the green screen image directly in the middle later)
    if(640 < width < 500 or 360 < height or width % 2 != 0):
        width += 1
        while (640 < width < 500 or 360 < height):
            width   = int(width*0.9)
            height  = int(height*0.9)
            if (width % 2 != 0):
                width +=1
    # store new dimensions
    dimensions = (width, height)
    # resize image
    newSize = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    # return new image size
    return newSize

def equaliseSize(img1, img2):
    '''
    Resize the green screen image to the same size as the background
    '''
    # green screen
    width1   = int(img1.shape[1])
    height1  = int(img1.shape[0])
    # background
    width2   = int(img2.shape[1])
    height2  = int(img2.shape[0])

    # make green screen smaller than background
    if(width2 < width1 or height2 < height1 or width1 % 2 != 0):
        width1 += 1
        while (width2 < width1 or height2 < height1):
            width1   = int(width1*0.9)
            height1  = int(height1*0.9)
            if (width1 % 2 != 0):
                width1 +=1
        dimensions = (width1, height1)
        newSize = cv.resize(img1, dimensions, interpolation=cv.INTER_AREA)
        src = newSize
    else:
        src = img1

    # pad the green screen image
    # set parameters
    dst = None
    # pad the top and keep image at the bottom
    top = img2.shape[0] - src.shape[0]
    bottom = 0
    # move image to the centre
    left = int(img2.shape[1] - src.shape[1])//2
    right = int(img2.shape[1] - src.shape[1])//2
    # stretch out the green screen to fill the space
    border_type = cv.BORDER_REPLICATE
    value = [0,0,0]
    # execute
    imgPadded = cv.copyMakeBorder(src, top, bottom, left, right, border_type, dst, value)
    # return image sized and positioned in centre bottom
    return imgPadded

def chromakey(img):
    '''
    Remove green background, add white background, store alpha channel data
    '''
    # convert image to LAB for processing
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    # isolate a channel and set threshold for green
    a_channel = lab[:,:,1]
    thresh = cv.threshold(a_channel, 120, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    # create mask of threshold and apply to image
    masked = cv.bitwise_and(img, img, mask=thresh)

    # add alpha channel
    b, g, r = cv.split(masked)
    rgba = [b, g, r, thresh]
    mask_TP = cv.merge(rgba,4)
    alpha_data = mask_TP[:,:,3]
    alpha = cv.cvtColor(alpha_data, cv.COLOR_GRAY2BGR)

    # add white background
    whiteBg= masked.copy()
    whiteBg[thresh==0]=(255,255,255)

    # return masked image, white background image and alpha channel
    return masked, whiteBg, alpha

##################
# Running Sequence
##################

# read images
og_img1 = cv.imread(sys.argv[1])
og_img2 = cv.imread(sys.argv[2])

# resize background
bGround = rescaleBGround(og_img1)

# resize green screen so its smaller than background, then pad it to make it the same size
grnScrn = equaliseSize(og_img2, bGround)


# apply chromakey effect to green screen image and return results
grnScrnMask, grnScrnWhite, alpha = chromakey(grnScrn)

# combine the two images
bGround_grnScrn = np.where(alpha==(0,0,0), bGround, grnScrnMask)

# arrange all four stages together
row1 = np.hstack((grnScrn, grnScrnWhite))
row2 = np.hstack((bGround, bGround_grnScrn))
combined = np.vstack((row1, row2))

# show result
cv.imshow('Final Composition', combined)
print('Final Canvas Size ({0}, {1})'.format(combined.shape[1], combined.shape[0]))

# display setting
cv.waitKey(0)
cv.destroyAllWindows()