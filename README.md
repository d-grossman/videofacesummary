# videofacesummary
makes a summary of faces seen in a video

# build the container
docker build -f Dockerfile -t vfs .

#run the container
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs
