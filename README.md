# videofacesummary
makes a summary of faces seen in a video or image


# *Face Detection and Chip Vectorization*

The process to analyze a batch of media (videos or images) using this repository is:
1. Select a face detection technique to run on the media
2. Select a chip vectorization technique to run on the media
3. (optional) Resolve the output into a reference set of unique faces and when/where they appeared

Instructions for building and running the containers for each technique can be found in the associated repository folder.

## Face Detection Techniques
1. dlib_detect - Based on the dlib library's facial landmarks models
2. mtcnn_detect - Based on the Multi Task CNN algorithm
3. tinyface_detect - Based on the TinyFace algorithm
4. frcnn_detect - Based on Faster R-CNN 

Output of the Face Detection Techniques is a pickle file for each video or image. The pickle files contains a list of bounding boxes for detected faces found in the video or image.

## Chip Vectorization Techniques
1. resnet50_vector - Make vectors from chips using resnet50 model
2. facenet_tf_vector - Make vectors from chips using the FaceNet algorithm via Tensorflow
3. openface_vector - Make vectors from chips using the FaceNet algorithm via Torch

Output of the Chip Vectorization Techniques is a pickle file for each video or image. The pickle file contains a dictionary of unique faces and when/where they appeared in the video or image.

## (optional) Create a Reference Set of Unique Faces
Resolving the output into a reference set of unique faces can be done by building the Dockerfile.process container and following the instructions below marked 'run the container to build a reference set of faces from processed media with custom parameters'.  

# *Video Processing container*

Dockerfile.process combines face detection (dlib) and chip vectorization (resnet50) components into one pipeline for ease of use. You may find better performance and accuracy on your media from using a different combination of the modular detection and vectorization techniques contained in the subfolders of this repository. 

## build the container

```Shell
docker build -f Dockerfile.process -t vfs.process .
```

## run the container to process videos with default parameters

```Shell
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.process
```

## run the container to process videos with custom parameters

```Shell
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs.process directFeatures.py --reduceby 1.0 
            --every 30 --tolerance 0.50 --jitters 4
```

  * **reduceby** = Factor to reduce video resolution (ex: 1.0 = original resolution, 2.0 -> reduce horizontal and vertical resolution by 2)  
  * **every** = Process every nth frame (ex: 30 = every 30th frame of video)
  * **tolerance** = Different faces are tolerance apart (ex: 0.4->tight 0.6->loose)
  * **jitters** = How many perturberations to use when making face vector


## run the container to build a reference set of faces from processed media 
**Note: This will try to process all pickle files that were vectorized via resnet50 in the "/out" volume by default and build a reference set of faces at /reference/face_reference_set_resnet50.pkl and a hash table of filenames and content hashes at /reference/hash_table.pkl**

```Shell
docker run -v /dirWithDetectedFaces:/out -v /referenceDir:/reference vfs.process resolveVideos.py
```

## run the container to to build a reference set of faces from processed media with custom parameters
```Shell
docker run -v /dirWithDetectedFaces:/out -v /referenceDir:/reference vfs.process resolveVideos.py 
             --vectors_used openface --detected_faces_folder /out --reference_faces_file face_reference_set_openface.pkl 
             --hash_table_file anotherfile.pkl --tolerance 0.6  --verbose False
```

  * **vectors_used** = Vectorization technique used to generate pickle files (resnet50, facenet_tf, openface). Note that 
          vector distance comparison between pickle files created with different vectorization techniques does not provide meaningful results. (default = resnet50)  
  * **detected_faces_folder** = Folder containing 'detected_faces' pickles files (default:/out)  
  * **reference_faces_file** = Pickle file in '/reference' containing reference set of detected faces (default: face_reference_set_resnet50.pkl)
  * **hash_table_file** = Pickle file in '/reference' containing hash table of filenames and content hashes (default: hash_table.pkl)
  * **tolerance** = Different faces are tolerance apart (ex: 0.4->tight 0.6->loose)
  * **verbose** = Print out detailed information related to processing time and results (default: False)")

# *Jupyter container*
### Interactive playing with data processed by the *Video Processing container*

## build the container

```Shell
docker build -f Dockerfile.notebook -t vfs.notebook .
```

## run the container

```Shell
docker run -v /someDirWithdatafiles:/in -p8888:8888 vfs.notebook
```
