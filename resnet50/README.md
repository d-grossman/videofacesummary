# *Chip vectorization container using resnet50 (via dlib)*
More about resnet50 (via dlib): [Site](http://dlib.net/)
[Code](https://github.com/davisking/dlib)

# Vectorize Detected Face Chips

This docker container uses resnet50 via dlib to vectorize face chips for each file in a mounted volume of bounding boxes. The output is a pickle file saved in the 'out' mounted volume for each bounding box file found in the 'bboxes' volume. The pickle file contains a dictionary of unique faces, their associated vectors and locations in the original media files.
 
## build the container

```Shell
docker build -f Dockerfile.resnet50 -t vfs.resnet50 .
```

Note: Run this command from the videofacesummary root folder

### run the container to vectorize face chips with default parameters

```Shell
docker run -v ~/dirWithMedia:/media -v ~/dirWithBoundingBoxes:/bboxes -v ~/outputDir:/out  vfs.resnet50 
```

### run the container to vectorize face chips with custom parameters

```Shell
docker run -v /dirWithMedia:/media -v /dirWithBoundingBoxes:/bboxes -v ~/outputDir:/out vfs.resnet50 --jitters 1 
        --tolerance 0.6 --chip_size 160 --verbose False
```

  * **jitters** = Perturbations to chip when calculating vector representation (default = 1) 
  * **tolerance** = Threshold for minimum vector distance between different faces, minimum value is 0.0 and maximum value is \
             4.0 (default = 0.6)
  * **chip_size** = Size of face chips used for comparison (default = 160)
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)")
