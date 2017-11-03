# *Face detection container using Faster-RCNN*
More about Faster-RCNN: 
[Paper](http://arxiv.org/abs/1506.01497)
[Dockerface Paper](https://arxiv.org/abs/1708.04370)
[Code](https://github.com/natanielruiz/py-faster-rcnn-dockerface)
[More Code](https://github.com/ShaoqingRen/faster_rcnn)
[Yet More Code](https://github.com/rbgirshick/fast-rcnn)

This docker container uses Faster-RCNN to detect faces for each video or image in a mounted volume of media. Faster-RCNN was presented at NIPS 2015. 

The output is a pickle file saved in the bboxes mounted volume for each image or video found in the media volume. The pickle file contains bounding boxes for the associated image or video.

## build the GPU container

```Shell
docker build -f Dockerfile.frcnn_gpu_detect -t vfs.frcnn_gpu_detect .
```

Note: Run this command from the videofacesummary root folder

### download pretrained caffe model

Download the VGG16 Dockerface pretrained model [here](https://www.dropbox.com/s/dhtawqycd32ca9v/vgg16_dockerface_iter_80000.caffemodel)

### run the GPU container to detect faces with default parameters
```Shell
nvidia-docker run -v /dirWithMedia:/media -v /outputDir:/bboxes -v /models:/models vfs.frcnn_gpu_detect 
```

Note: The pretrained model should be located in the /models volume mount

### run the GPU container to detect faces with custom parameters
```Shell
nvidia-docker run -v /dirWithMedia:/media -v /outputDir:/bboxes -v /models:/models vfs.frcnn_gpu_detect --use_gpu True 
           --caffe_model /models/vgg16_dockerface_iter_80000.caffemodel --reduceby 1.0 --every 30
           --threshold 0.85 --nms 0.15 --verbose False
           --prototxt_file /opt/py-faster-rcnn/models/face/VGG16/faster_rcnn_end2end/test.prototxt
  
```

  * **use_gpu** = Use GPU, if available with nvidia-docker. (default = False)
  * **caffe_model** = Path and file name for pretrained caffe model. (default = /models/vgg16_dockerface_iter_80000.caffemodel)
  * **prototxt_file** = Path and file name for prototxt file to match with caffe model. (default = /opt/py-faster-rcnn/models/face/VGG16/faster_rcnn_end2end/test.prototxt)
  * **threshold** = Neural network probability threshold for Faster-RCNN to accept a proposed face detection. (default = 0.85)
  * **nms** = Non maximum suppression threshold for Faster-RCNN. (default = 0.15)
  * **reduceby** = Factor to reduce media resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)")
   
