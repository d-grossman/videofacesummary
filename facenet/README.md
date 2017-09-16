# *Video Processing container using Facenet and MTCNN*
More about Facenet and MTCNN: [Site](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
[Paper](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)
[Repo](https://github.com/davidsandberg/facenet)

## build the CPU container
docker build -f Dockerfile.facenet -t vfs.facenet .

Note: Run this command from the videofacesummary root folder

## run the CPU container to process videos with default parameters
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.facenet

## run the CPU container to process videos with custom parameters
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.facenet --**reduceby** 1.0 
        --**every** 30 --**tolerance** 0.50 --**jitters** 4  --**model** /folder/with/model 

  * **detector** = Face detection algorithm, Dlib is 1 and MTCNN is 2 (default=1)   
  * **vectorizer** = Chip vectorizer, Dlib FaceNet50 is 1 and Facenet ResNetInception is 2 (default=1)
  * **reduceby** = Factor to reduce video resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **tolerance** = Different faces are tolerance apart (ex: 0.4->tight 0.6->loose)
  * **jitters** = When vectorizer=1, how many perturberations to use when making face vector 
  * **model** = When vectorizer=2, pretrained model to use. Download a Facenet Inception ResNet pretrained model [here](https://github.com/davidsandberg/facenet)*
 
## build the GPU container
1. Edit line 40 of facenet/align.py and change 'USE_GPU = False' to 'USE_GPU = True'
2. Run command: docker build -f Dockerfile.facenet_gpu -t vfs.facenet_gpu .

## run the GPU container to process videos with default parameters
nvidia-docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.facenet_gpu

## run the GPU container to process videos with custom parameters
nvidia-docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.facenet_gpu --**reduceby** 1.0 
        --**every** 30 --**tolerance** 0.50 --**jitters** 4  --**model** /folder/with/model 

  * **detector** = Face detection algorithm, Dlib is 1 and MTCNN is 2 (default=1)   
  * **vectorizer** = Chip vectorizer, Dlib FaceNet50 is 1 and Facenet ResNetInception is 2 (default=1)
  * **reduceby** = Factor to reduce video resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **tolerance** = Different faces are tolerance apart (ex: 0.4->tight 0.6->loose)
  * **jitters** = When vectorizer=1, how many perturberations to use when making face vector 
  * **model** = When vectorizer=2, pretrained model to use. Download a Facenet Inception ResNet pretrained model [here](https://github.com/davidsandberg/facenet)*
