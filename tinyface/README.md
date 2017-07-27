# *Video Processing container using Tiny Face*
More about Tiny Face: [Site](https://www.cs.cmu.edu/~peiyunh/tiny/)
[Paper](https://arxiv.org/pdf/1612.04402.pdf)
[Repo](https://github.com/peiyunh/tiny)

## build the container
**(Note: run this command from the videofacesummary folder)**  

docker build -f tinyface/Dockerfile.tinyface -t vfs.tinyface .

## run the container to process videos with default parameters
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.tinyface

## run the container to process videos with custom parameters
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.tinyface directFeatures_tinyface.py --**reduceby** 1.0
        --**every** 30 --**tolerance** 0.50 --**jitters** 4  --**prob_thresh**=0.5 --**nms_thresh**=0.1 --**gpu**=False

  * **reduceby** = Factor to reduce video resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)  
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **tolerance** = Different faces are tolerance apart (ex: 0.4->tight 0.6->loose)
  * **jitters** = How many perturberations to use when making face vector
  * **prob_thresh** = Tiny Face Detector threshold for face likelihood
  * **nms_thresh** = Tiny Face Detector threshold for non-maximum suppression
  * **gpu** = Flag to use GPU (note: Must use nvidia-docker to run container")

## run the container to resolve processed video output from multiple videos
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.tinyface resolveVideos.py

## run the container to resolve processed video output from multiple videos with custom parameters
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.tinyface resolveVideos.py --**detected_faces_folder** /out --**reference_faces_file** somefile.pkl --**tolerance** 0.50

  * **detected_faces_folder** = Folder containing detected faces pickles files (default:/out)  
  * **reference_faces_file** = Pickle file in '/in' containing reference set of detected faces (default: face_reference_set.pkl)
  * **tolerance** = Different faces are tolerance apart (ex: 0.4->tight 0.6->loose)

# *Jupyter container*
### Interactive playing with data processed by the *Video Processing container*

## build the container
**(Note: run this command from the videofacesummary folder)**  

docker build -f tinyface/Dockerfile.tinyface_notebook -t vfs.tinyface_notebook .

## run the container
docker run -v /someDirWithdatafiles:/in -p 8888:8888 vfs.tinyface_notebook