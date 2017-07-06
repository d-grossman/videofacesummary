# videofacesummary
makes a summary of faces seen in a video

# Video Processing container

## build the container
docker build -f Dockerfile.process -t vfs.process .

## run the container
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.process

# Jupyter container

## build the container
docker build -f Dockerfile.notebook -t vfs.notebook .

## run the container
docker run -v /someDirWithdatafiles:/in  -p8888:8888 vfs.notebook

# Tiny Face container
### Counts the number of faces in a picture

## build the CPU container
docker build -f Dockerfile.tinyface -t vfs.tinyface .

## run the CPU container
### (Note: some issues encountered with images > 1mb)
docker run -v /someDirWithimagefiles:/images vfs.tinyface

## build the GPU container
docker build -f Dockerfile.tinyface_gpu -t vfs.tinyface_gpu .

## run the GPU container
### (Note: must use nvidia-docker to execute container)
nvidia-docker run -v /someDirWithimagefiles:/images vfs.tinyface_gpu

# Single Shot Detector (SSD) container
## Object detector 

## build the container
Download to the vfsummary folder the [tar file](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA) containing a pretrained Caffe model            
docker build -f Dockerfile.ssd -t vfs.ssd .

## run the container
docker run -v /someDirWithimagefiles:/images  -p8888:8888 vfs.ssd   
Navigate to 'examples' folder and open ssd_detect.ipynb notebook
