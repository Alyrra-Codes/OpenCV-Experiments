import cv2 as cv
import numpy as np
import sys
##################
# Define functions
##################

# colourspace conversions
def cvt2XYZ(img):
    '''
    convert to XYZ
    '''
    XYZ = cv.cvtColor(img, cv.COLOR_BGR2XYZ)
    return XYZ

def cvt2LAB(img):
    '''
    Convert to LAB
    '''
    LAB = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    return LAB

def cvt2YCrCb(img):
    '''
    Convert to YCrCb
    '''
    YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    return YCrCb

def cvt2HSV(img):
    '''
    Convert to HSV
    '''
    HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return HSV

def cvt2RGB(img):
    '''
    Convert to RGB
    '''
    RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return RGB

def imgRescale(img):
    '''
    resize the images
    '''
    # get dimensions
    width   = int(img.shape[1])
    height  = int(img.shape[0])
    # reduce image size to less than 640x360
    while (640 < width or 360 < height):
        width   = int(width*0.9)
        height  = int(height*0.9)
    # store new dimensions
    dimensions = (width, height)
    # resize image
    newSize = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    # return new image size
    return newSize

##################
# Running Sequence
##################

# get colour space and image
colourSpace = sys.argv[1]
og_img = cv.imread(sys.argv[2])

# resize image
imgRescaled = imgRescale(og_img)

# check colour space requirement and convert, and name each channel for later
if (colourSpace == '-XYZ'):
    imgCVTD = cvt2XYZ(imgRescaled)
    TR_name = 'X'
    BL_name = 'Y'
    BR_name = 'Z'
elif (colourSpace == '-Lab' or colourSpace == '-LAB'):
    imgCVTD = cvt2LAB(imgRescaled)
    TR_name = 'L'
    BL_name = 'a'
    BR_name = 'b'
elif (colourSpace == '-YCrCb' or colourSpace == '-YCRCB'):
    imgCVTD = cvt2YCrCb(imgRescaled)
    TR_name = 'Y'
    BL_name = 'Cr'
    BR_name = 'Cb'
elif (colourSpace == '-HSB' or colourSpace == "-HSV"):
    imgCVTD = cvt2HSV(imgRescaled)
    TR_name = 'H'
    BL_name = 'S'
    BR_name = 'B'
else:
    imgCVTD = cvt2RGB(imgRescaled)
    TR_name = 'R'
    BL_name = 'G'
    BR_name = 'B'

# split channels
(channel_1, channel_2, channel_3) = cv.split(imgCVTD)

# convert channels to RGB
ch1_BGR = cv.cvtColor(channel_1, cv.COLOR_GRAY2RGB)
ch2_BGR = cv.cvtColor(channel_2, cv.COLOR_GRAY2RGB)
ch3_BGR = cv.cvtColor(channel_3, cv.COLOR_GRAY2RGB)

# combine images
row1        = np.hstack((imgRescaled, ch1_BGR))
row2        = np.hstack((ch2_BGR, ch3_BGR))
combined    = np.vstack((row1, row2))

# show final composition and label appropriately
title = '{0}: (Original Image, {1}, {2}, {3})'.format(colourSpace.strip('-'), TR_name, BL_name, BR_name)
cv.imshow(title, combined)
print('Final Canvas Size ({0}, {1})'.format(combined.shape[1], combined.shape[0]))

# display settings
cv.waitKey(0)
cv.destroyAllWindows()