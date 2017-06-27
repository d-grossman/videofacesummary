# videofacesummary
makes a summary of faces seen in a video

# Video Processing container

## build the container
docker build -f Dockerfile.processing -t vfs.processing .

## run the container
docker run -v /dirWith1movie:/in -v /outputDir:/out vfs

# Jupyter container

## build the container
docker build -f Dockerfile.notebook -t vfs.notebook .

## run the container
docker run -v /someDirWithdatafiles:/in  -p8888:8888 vfs.notebook
