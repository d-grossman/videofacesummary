# videofacesummary
makes a summary of faces seen in a video

# *Video Processing container*

## build the container
docker build -f Dockerfile.process -t vfs.process .

## run the container to process videos with default parameters
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.process

## run the container to process videos with custom parameters
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.process directFeatures.py --**reduceby** 1.0 --**every** 30 --**tolerance** 0.50 --jitters 4  

  * **reduceby** = Factor to reduce video resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)  
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **tolerance** = Different faces are tolerance apart (ex: 0.4->tight 0.6->loose)
  * **jitters** = How many perturberations to use when making face vector

## run the container to resolve processed video output from multiple videos
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.process resolveVideos.py

## run the container to resolve processed video output from multiple videos with custom parameters
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.process resolveVideos.py --**detected_faces_folder** /out --**reference_faces_file** somefile.pkl --**tolerance** 0.50

  * **detected_faces_folder** = Folder containing detected faces pickles files (default:/out)  
  * **reference_faces_file** = Pickle file in '/in' containing reference set of detected faces (default: face_reference_set.pkl)
  * **tolerance** = Different faces are tolerance apart (ex: 0.4->tight 0.6->loose)

# *Jupyter container*
### Interactive playing with dat processed by the *Video Processing container*

## build the container
docker build -f Dockerfile.notebook -t vfs.notebook .

## run the container
docker run -v /someDirWithdatafiles:/in -p8888:8888 vfs.notebook

# *Tiny Face container*
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

# *Single Shot Detector (SSD) container*
## Object detector 

## build the container
Download to the ssd folder the [tar file](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA) containing a pretrained Caffe model            

docker build -f Dockerfile.ssd -t vfs.ssd .

## run the container
docker run -v /someDirWithimagefiles:/images  -p8888:8888 vfs.ssd   

Navigate to 'examples' folder and open ssd_detect.ipynb notebook
