# OpenCV-Experiments
A collection of works I've created using the python library OpenCV

Works include splitting images into their colour channels for image processing, chromakey of still images and background removal of video with object detection.
These were completed in 2023 as part of a course on Computer Vision during my Masters

To Run:<br/>
Colour Channels<br/>
```console
python colour-channels.py -[colourprofile] [img_path]
```

Chromakey<br/>
```console
python chromakey.py [background_img_path] [greenscreen_img_path]
``` 

Background Removal<br/>
```console
python bgr.py [video_path]
```
