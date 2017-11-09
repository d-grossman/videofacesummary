# *Face detection container using MTCNN*
More about MTCNN: [Site](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
[Paper](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)
[Code](https://github.com/davidsandberg/facenet/blob/master/src/align/align_dataset_mtcnn.py)

This docker container uses MTCNN to detect faces for each video or image in a mounted volume of media.  

The output is a pickle file saved in the bboxes mounted volume for each image or video found in the media volume. The pickle file contains bounding boxes for the associated image or video.

## build the CPU container

```Shell
docker build -f Dockerfile.mtcnn_detect -t vfs.mtcnn_detect .
```

Note: Run this command from the videofacesummary root folder

### run the CPU container to detect faces with default parameters

```Shell
docker run -v /dirWithMedia:/media -v /outputDir:/bboxes vfs.mtcnn_detect
```

### run the CPU container to detect faces with custom parameters

```Shell
docker run -v /dirWithMedia:/media -v /outputDir:/bboxes vfs.mtcnn_detect --reduceby 1.0 --every 30 
           --threshold 0.85 --scale_factor 0.709  --margin 10 --min_size 40 --verbose False
```

  * **margin** = Pixel padding around face within aligned chip. (default = 10)
  * **min_size** = Minimum pixel size of face to detect. (default = 40)
  * **threshold** = Neural network probability threshold to accept a proposed face detection. (default = 0.85)
  * **scale_factor** = Image scale factor used during MTCNN face detection. (default = 0.709)
  * **reduceby** = Factor to reduce media resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)")
  
 
## build the GPU container

```Shell
docker build -f Dockerfile.mtcnn_gpu_detect -t vfs.mtcnn_gpu_detect .
```

Note: Run this command from the videofacesummary root folder

### run the GPU container to detect faces with default parameters
```Shell
nvidia-docker run -v /dirWithMedia:/media -v /outputDir:/bboxes vfs.mtcnn_gpu_detect 
```

### run the GPU container to detect faces with custom parameters
```Shell
nvidia-docker run -v /dirWithMedia:/media -v /outputDir:/bboxes vfs.mtcnn_gpu_detect --use_gpu True --reduceby 1.0 --every 30 
                  --threshold 0.85 --scale_factor 0.709  --margin 10 --min_size 40 --verbose False
                  --gpu_memory_fraction 0.8 
```

  * **use_gpu** = Use GPU, if available with nvidia-docker. (default = False)   
  * **gpu_memory_fraction** = If use_gpu is True, percentage of GPU memory to use. (default = 0.8)
  * **margin** = Pixel padding around face within aligned chip. (default = 10)
  * **min_size** = Minimum pixel size of face to detect. (default = 40)
  * **threshold** = Neural network probability threshold to accept a proposed face detection. (default = 0.85)
  * **scale_factor** = Image scale factor used during MTCNN face detection. (default = 0.709)
  * **reduceby** = Factor to reduce media resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)")

### test MTCNN face detection on a set of hand labeled images

1. build the container
```Shell
docker build -f Dockerfile.mtcnn_detect -t vfs.mtcnn_detect .
```
2. run the container interactively
```Shell
docker run --entrypoint=/bin/bash -it vfs.mtcnn_detect
```
3. execute test at commmand line
```Shell
python test_mtcnn_detection.py