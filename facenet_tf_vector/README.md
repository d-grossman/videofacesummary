# *Face chip vectorization container using Facenet via Tensorflow*
More about Facenet: 
[Paper](https://arxiv.org/pdf/1503.03832.pdf)
[Repo](https://github.com/davidsandberg/facenet)

This docker container uses a Tensorflow implementation of Facenet to vectorize face chips for each file in a mounted volume of bounding boxes.  

The output is a pickle file saved in the 'out' mounted volume for each bounding box file found in the 'bboxes' volume. The pickle file contains a dictionary of unique faces, their associated vectors and locations in the original media files.


## download pretrained model

David Sandberg's GitHub [repository]() includes links to two different pretrained Tensorflow models based on Facenet. Our Docker container uses [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) by default. Unzip the 20170512-110547 file after downloading it.

## build the CPU container

```Shell
docker build -f Dockerfile.facenet_tf_vector -t vfs.facenet_tf_vector .
```

Note: Run this command from the videofacesummary root folder

## run the CPU container to vectorize face chips with default parameters
```Shell
docker run -v ~/dirWithMedia/:/media -v ~/dirWithBoundingBoxes/:/bboxes -v ~/outputDir:/out 
           -v ~/dirWithModels/:/models vfs.facenet_tf_vector  
```

Note: Replace ~/dirWithModels/ with the folder that contains the pretrained model folder 20170512-110547 from the first step.

## run the CPU container to vectorize face chips with custom parameters
```Shell
docker run -v ~/dirWithMedia/:/media -v ~/dirWithBoundingBoxes/:/bboxes -v ~/outputDir:/out 
           -v ~/dirWithModels/:/models vfs.facenet_tf_vector  
           --tolerance 0.8 --chip_size 160 --verbose False --facenet_model /models/20170512-110547
 ```  

  * **use_gpu** = Use GPU, if available with nvidia-docker. (default = False)    
  * **gpu_memory_fraction** = If nvidia-docker is used and use_gpu is True, percentage of GPU memory to use. (default = 0.8)
  * **tolerance** = Threshold for minimum vector distance between different faces, minimum value is 0.0 and maximum value is 4.0 (default = 0.6)")
  * **chip_size** = Pretrained Facenet model from https://github.com/davidsandberg/facenet expects face chips of 160x160 (default = 160)")
  * **verbose** = Print out information related to image processing time and vectorization results (default: False)")
  * **facenet_model** = Pretrained Tensorflow model based on Facenet approach (default: /models/20170512-110547)")


## build the GPU container
```Shell
docker build -f Dockerfile.facenet_tf_gpu_vector -t vfs.facenet_tf_gpu_vector .
```

## run the GPU container to vectorize face chips with default parameters plus GPU support
```Shell
nvidia-docker run -v ~/dirWithMedia/:/media -v ~/dirWithBoundingBoxes/:/bboxes -v ~/outputDir:/out 
                  -v ~/dirWithModels/:/models vfs.facenet_tf_gpu_vector --use_gpu True
```

Note: Replace ~/dirWithModels/ with the folder that contains the pretrained model folder 20170512-110547 from the first step.


## run the GPU container to vectorize face chips with custom parameters
```Shell
nvidia-docker run -v ~/dirWithMedia/:/media -v ~/dirWithBoundingBoxes/:/bboxes -v ~/outputDir:/out 
                  -v ~/dirWithModels/:/models vfs.facenet_tf_gpu_vector --use_gpu True
                  --gpu_memory_fraction 0.8 --tolerance 0.8 --chip_size 160  --verbose False 
                  --facenet_model /models/20170512-110547
```

  * **use_gpu** = Use GPU, if available with nvidia-docker. (default = False)    
  * **gpu_memory_fraction** = If use_gpu is True, percentage of GPU memory to use. (default = 0.8)
  * **tolerance** = Threshold for minimum vector distance between different faces, minimum value is 0.0 and maximum value is 4.0 (default = 0.8)")
  * **chip_size** = Pretrained Facenet model from https://github.com/davidsandberg/facenet expects face chips of 160x160 (default = 160)")
  * **verbose** = Print out information related to image processing time and vectorization results  (default: False)")
  * **facenet_model** = Pretrained Tensorflow model based on Facenet approach (default: /models/20170512-110547)")
