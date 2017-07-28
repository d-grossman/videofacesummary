# *Face Counting container*
### Counts the number of faces in a picture using Tiny Face
More about Tiny Face: [Site](https://www.cs.cmu.edu/~peiyunh/tiny/)
[Paper](https://arxiv.org/pdf/1612.04402.pdf)
[Repo](https://github.com/peiyunh/tiny)

## build the CPU container
docker build -f Dockerfile.countfaces -t vfs.countfaces .

## run the CPU container
**Note: Requires 4GB-8GB of RAM for Docker, depending on picture size**  

docker run -v /someDirWithimagefiles:/images vfs.countfaces

## build the GPU container
docker build -f Dockerfile.countfaces_gpu -t vfs.countfaces_gpu .

## run the GPU container
**Note: must use nvidia-docker to execute container**  
**Note: Requires 4GB-8GB of RAM for Docker, depending on image size**
  
nvidia-docker run -v /someDirWithimagefiles:/images vfs.countfaces_gpu