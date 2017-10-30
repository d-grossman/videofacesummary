# *Face detection container using TinyFace*
More about Tiny Face: [Site](https://www.cs.cmu.edu/~peiyunh/tiny/)
[Paper](https://arxiv.org/pdf/1612.04402.pdf)
[Repo](https://github.com/peiyunh/tiny)

This docker container uses TinyFace to detect faces for each video or image in a mounted volume of media.  

The output is a pickle file saved in the bboxes mounted volume for each image or video found in the media volume. The pickle file contains bounding boxes for the associated image or video.

## build the CPU container
```Shell
docker build -f Dockerfile.tinyface -t vfs.tinyface .
```

Note: Run this command from the videofacesummary root folder

### run the CPU container to detect faces with default parameters
```Shell
docker run -v /dirWithMedia:/media -v /outputDir:/bboxes vfs.tinyface
```

### run the CPU container to detect faces with custom parameters
```Shell
docker run -v /dirWithMedia:/media -v /outputDir:/bboxes vfs.tinyface --reduceby 1.0
        --every 30  --prob_thresh 0.7 --nms_thresh 0.1 --verbose False
```

  * **reduceby** = Factor to reduce image/video frame resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)  
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **prob_thresh** = TinyFace Detector threshold for face likelihood (default = 0.7)
  * **nms_thresh** = TinyFace Detector threshold for non-maximum suppression
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)


## build the GPU container
```Shell
docker build -f Dockerfile.tinyface_gpu -t vfs.tinyface_gpu .
```

Note: Run this command from the videofacesummary root folder

### run the GPU container to detect faces with default parameters plus GPU support
```Shell
nvidia-docker run -v /dirWithMedia:/media -v /outputDir:/bboxes vfs.tinyface_gpu --use_gpu True
```

### run the GPU container to detect faces with custom parameters
```Shell
nvidia-docker run -v /dirWithMedia:/media -v /outputDir:/bboxes vfs.tinyface_gpu --use_gpu True --reduceby 1.0
        --every 30 --prob_thresh 0.5 --nms_thresh 0.1 --verbose False
```

  * **use_gpu** = Flag to use gpu. It must be explicitly set to true when submitting custom parameters (default=False). 
  * **reduceby** = Factor to reduce image/video frame resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)  
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **prob_thresh** = TinyFace Detector threshold for face likelihood (default = 0.7)
  * **nms_thresh** = TinyFace Detector threshold for non-maximum suppression
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)
