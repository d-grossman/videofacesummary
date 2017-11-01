# *Face chip vectorization container using the OpenFace implementation of Facenet via Torch*
More about Facenet and OpenFace: 
[Paper](https://arxiv.org/pdf/1503.03832.pdf)
[Repo](https://github.com/cmusatyalab/openface)

This docker container uses a Torch implementation of Facenet to vectorize face chips for each file in a mounted volume of bounding boxes.  

The output is a pickle file saved in the 'out' mounted volume for each bounding box file found in the 'bboxes' volume. The pickle file contains a dictionary of unique faces, their associated vectors and locations in the original media files.

## build the CPU container

```Shell
docker build -f Dockerfile.openface_vector -t vfs.openface_vector .
```

Note: Run this command from the videofacesummary root folder

### download pretrained torch model and dlib's face landmark model

Download the pretrained Torch model from Carnegie Mellon's openface project [here](https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7). 

### run the CPU container to vectorize face chips with default parameters

```Shell
docker run -v ~/dirWithMedia/:/media -v ~/dirWithBoundingBoxes/:/bboxes -v ~/outputDir:/out -v ~/dirWithModels/:/models vfs.openface_vector 
```

### run the CPU container to vectorize face chips with custom parameters

```Shell
docker run -v ~/dirWithMedia/:/media -v ~/dirWithBoundingBoxes/:/bboxes -v ~/outputDir:/out -v ~/dirWithModels/:/models vfs.openface_vector  
 --facenet_model /models/model_file/ --dlibFacePredictor /models/predictor_file --tolerance 0.8 --chip_size 160  --verbose False
```

  * **facenet_model** = Facenet pretrained model using Torch (default = /models/nn4.small2.v1.t7)
  * **dlibFacePredictor** = Dlib Face Landmarks file (default = /models/shape_predictor_68_face_landmarks.dat)
  * **tolerance** = Threshold for minimum vector distance between different faces, minimum value is 0.0 and maximum value is 4.0 (default = 0.8)")
  * **chip_size** = Pretrained Facenet model from Carnegie Mellon expects face chips of 96x96 (default = 96)")
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)")


## build the GPU container

```Shell
docker build -f Dockerfile.openface_gpu_vector -t vfs.openface_gpu_vector .
```

Note: Run this command from the videofacesummary root folder

### download pretrained torch model and dlib's face landmark model

Download the pretrained Torch model from Carnegie Mellon's openface project [here](https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7). 

### run the GPU container to vectorize face chips with default parameters plus GPU support

```Shell
nvidia-docker run -v ~/dirWithMedia/:/media -v ~/dirWithBoundingBoxes/:/bboxes -v ~/outputDir:/out -v ~/dirWithModels/:/models vfs.openface_gpu_vector 
```

### run the GPU container to vectorize face chips with custom parameters plus GPU support
```Shell
nvidia-docker run -v ~/dirWithMedia/:/media -v ~/dirWithBoundingBoxes/:/bboxes -v ~/outputDir:/out -v ~/dirWithModels/:/models vfs.openface_gpu_vector --use_gpu True
 --facenet_model /models/folderWithModel/ --dlibFacePredictor /models/model_file --tolerance** 0.8 --chip_size 160 --verbose False
 --gpu_memory_fraction 0.8  
```

  * **facenet_model** = Facenet pretrained model using Torch (default = /models/nn4.small2.v1.t7)
  * **dlibFacePredictor** = Dlib Face Landmarks file (default = /models/shape_predictor_68_face_landmarks.dat)
  * **use_gpu** = Use GPU, if available with nvidia-docker. (default = False)    
  * **gpu_memory_fraction** = If use_gpu is True, percentage of GPU memory to use. (default = 0.8)
  * **tolerance** = Threshold for minimum vector distance between different faces, minimum value is 0.0 and maximum value is 4.0 (default = 0.8)")
  * **chip_size** = Pretrained Facenet model from Carnegie Mellon expects face chips of 96x96 (default = 96)")
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)")
