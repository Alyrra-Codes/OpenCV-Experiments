import sys
import numpy as np
import cv2 as cv

##################
# DEFINE FUNCTIONS
##################

def scale_to_VGA(frame):
    '''
    A function to rescale a video file down to VGA resolution (640x480)
    '''
    # get dimensions
    width   = int(frame.shape[1])
    height  = int(frame.shape[0])
    # reduce video resolution to less than 640x480
    while (640 < width or 480 < height):
        width   = int(width*0.9)
        height  = int(height*0.9)
    # store new dimensions
    dimensions = (width, height)
    # resize image
    newSize = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    # return new image size
    return newSize

##################
# Running Sequence
##################

# capture video
capture = cv.VideoCapture(sys.argv[1])
if (not capture.isOpened()):
    print('Error: Could not open video')
    exit()
else:
    # create kernal and background subtractors\
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    backsub = cv.createBackgroundSubtractorMOG2()

    # initialise framecount
    frame_count = 0

    # initialise average
    _, frame1 = capture.read()
    VGA_frame1 = scale_to_VGA(frame1)
    avg = np.float32(VGA_frame1)

# read, rescale and display frames
while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        frame_count += 1
        print('Frame {0}:'.format(str(frame_count).zfill(4)), end=' ')
        # rescale
        VGA_vid = scale_to_VGA(frame)
        # convert to grayscale
        gray_frame = cv.cvtColor(VGA_vid, cv.COLOR_BGR2GRAY)
        # apply backsubtraction to frame
        fg_mask = backsub.apply(gray_frame)
        # remove noise using morphology
        fg_morph = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)
        # apply connected components algorithm
        output = cv.connectedComponentsWithStats(fg_morph, 4, cv.CV_32S)
        total_labels, label_ids, values, centroid = output
        # subtract 1 from total labels to account for it finding the video frame
        total_objects = total_labels-1
        if (total_objects == 1):
            print('{0} object'.format(total_objects), end=' ')
        else:
            print('{0} objects'.format(total_objects), end=' ')
        # classify components
        # initialise labels
        persons = 0
        cars = 0
        others = 0
        for i in range(total_objects):
            width = values[i, cv.CC_STAT_WIDTH]
            height = values[i, cv.CC_STAT_HEIGHT]
            if (height != 0):
                ratio = width/height
            else:
                ratio = 0
            # assume persons are much taller than they are wide
            if (ratio < 0.66)and (ratio != 0):
                persons += 1
            # assume cars are much wider than they are tall
            elif (1.33 < ratio):
                cars += 1
            # classify everything else as other
            else:
                others += 1
            # adjust grammar for singular objects
            if (persons == 1):
                p_plr = 'person'
            else:
                p_plr = 'persons'
            if (cars == 1):
                c_plr = 'car'
            else:
                c_plr = 'cars'
            if (others == 1):
                o_plr = 'other'
            else:
                o_plr = 'others'
        print('({0} {1}, {2} {3}, {4} {5})'.format(persons, p_plr, cars, c_plr, others, o_plr))

        # apply forground mask to original video to extract subjects in original colour
        fg_morph = cv.bitwise_and(VGA_vid, VGA_vid, mask=fg_morph)
        
        # calculate the average background
        background = cv.accumulateWeighted(VGA_vid, avg, 0.01)
        background = cv.convertScaleAbs(background)

        # arrange and show images
        # convert fg_mask to BGR
        fg_mask = cv.cvtColor(fg_mask, cv.COLOR_GRAY2BGR)
        row1 = np.hstack((VGA_vid, background))
        row2 = np.hstack((fg_mask, fg_morph))
        final_comp = np.vstack((row1, row2))
        cv.imshow('Video', final_comp)
        
        # establish waitkey
        key = cv.waitKey(25)

        # break loop if esc is pressed
        if (key==27):
            capture.release()
            break
    else:
        break # end of video

capture.release()
cv.destroyAllWindows()