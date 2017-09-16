# *Video Processing container using Tiny Face*
More about Tiny Face: [Site](https://www.cs.cmu.edu/~peiyunh/tiny/)
[Paper](https://arxiv.org/pdf/1612.04402.pdf)
[Repo](https://github.com/peiyunh/tiny)

## build the CPU container
docker build -f Dockerfile.tinyface -t vfs.tinyface .

Note: Run this command from the videofacesummary root folder

## run the CPU container to process videos with default parameters
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.tinyface

## run the CPU container to process videos with custom parameters
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.tinyface --**reduceby** 1.0
        --**every** 30 --**tolerance** 0.50 --**jitters** 4  --**prob_thresh** 0.5 --**nms_thresh** 0.1 

  * **reduceby** = Factor to reduce video resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)  
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **tolerance** = Different faces are tolerance apart (ex: 0.4->tight 0.6->loose)
  * **jitters** = How many perturberations to use when making face vector
  * **prob_thresh** = Tiny Face Detector threshold for face likelihood
  * **nms_thresh** = Tiny Face Detector threshold for non-maximum suppression

## build the GPU container
docker build -f Dockerfile.tinyface_gpu -t vfs.tinyface_gpu .

Note: Run this command from the videofacesummary root folder

## run the GPU container to process videos with default parameters
nvidia-docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.tinyface_gpu

## run the GPU container to process videos with custom parameters
nvidia-docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.tinyface_gpu --**gpu** True --**reduceby** 1.0
        --**every** 30 --**tolerance** 0.50 --**jitters** 4  --**prob_thresh** 0.5 --**nms_thresh** 0.1 

  * **gpu** = Flag to use gpu. It must be explicitly set to true when submitting custom parameters. 
  * **reduceby** = Factor to reduce video resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)  
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **tolerance** = Different faces are tolerance apart (ex: 0.4->tight 0.6->loose)
  * **jitters** = How many perturberations to use when making face vector
  * **prob_thresh** = Tiny Face Detector threshold for face likelihood
  * **nms_thresh** = Tiny Face Detector threshold for non-maximum suppression