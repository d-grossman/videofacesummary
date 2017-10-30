# *Face detection and Chip vectorization containers using dlib and resnet50*
More about dlib: [Site](http://dlib.net/)
[Code](https://github.com/davisking/dlib)

# Detect Faces

This docker container uses dlib to detect faces for each video or image in a mounted volume of media. The output is a pickle file saved in a mounted volume for each image or video found in the media volume. The pickle file contains proposed face bounding boxes for the associated image or video.

## build the container

```Shell
docker build -f Dockerfile.dlib_resnet50 -t vfs.dlib_resnet50 .
```

Note: Run this command from the videofacesummary root folder

### run the CPU container to detect faces with default parameters

```Shell
docker run -v /dirWithMedia:/media -v /outputDir:/bboxes vfs.dlib_resnet50
```

### run the CPU container to detect faces with custom parameters

```Shell
docker run -v /dirWithMedia:/media -v /outputDir:/bboxes vfs.dlib_resnet50 run_dlib.py --reduceby 1.0 
        --every 30 --upsampling 1 --verbose False
```

  * **upsampling** = Number of times for dlib to upsample image to detect faces. (default = 1)
  * **reduceby** = Factor to reduce media resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)")
 
 
# Vectorize Detected Face Chips

This docker container uses resnet50 via dlib to vectorize face chips for each file in a mounted volume of bounding boxes. The output is a pickle file saved in the 'out' mounted volume for each bounding box file found in the 'bboxes' volume. The pickle file contains a dictionary of unique faces, their associated vectors and locations in the original media files.
 
## build the container

```Shell
docker build -f Dockerfile.dlib_resnet50 -t vfs.dlib_resnet50 .
```

Note: Run this command from the videofacesummary root folder

### run the container to vectorize face chips with default parameters

```Shell
docker run -v ~/dirWithMedia/:/media -v ~/dirWithBoundingBoxes/:/bboxes -v ~/outputDir:/out  vfs.dlib_resnet50 run_resnet50.py
```

### run the container to vectorize face chips with custom parameters

```Shell
docker run -v /dirWithMedia:/media -v /outputDir:/bboxes vfs.dlib_resnet50 run_resnet50.py --jitters 1 
        --tolerance 0.6 --chip_size 160 --verbose False
```

  * **jitters** = Perturbations to chip when calculating vector representation (default = 1) 
  * **tolerance** = Threshold for minimum vector distance between different faces, minimum value is 0.0 and maximum value is \
             4.0 (default = 0.6)
  * **chip_size** = Size of face chips used for comparison (default = 160)
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)")
