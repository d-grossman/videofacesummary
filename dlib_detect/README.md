# *Face detection container using dlib*
More about dlib: [Site](http://dlib.net/)
[Code](https://github.com/davisking/dlib)

# Detect Faces

This docker container uses dlib to detect faces for each video or image in a mounted volume of media. The output is a pickle file saved in a mounted volume for each image or video found in the media volume. The pickle file contains proposed face bounding boxes for the associated image or video.

## build the container

```Shell
docker build -f Dockerfile.dlib_detect -t vfs.dlib_detect .
```

Note: Run this command from the videofacesummary root folder

### run the CPU container to detect faces with default parameters

```Shell
docker run -v ~/dirWithMedia:/media -v ~/outputDir:/bboxes vfs.dlibi_detect
```

### run the CPU container to detect faces with custom parameters

```Shell
docker run -v ~/dirWithMedia:/media -v ~/outputDir:/bboxes vfs.dlib_detect --reduceby 1.0 
        --every 30 --upsampling 1 --verbose False
```

  * **upsampling** = Number of times for dlib to upsample image to detect faces. (default = 1)
  * **reduceby** = Factor to reduce media resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)")
 
