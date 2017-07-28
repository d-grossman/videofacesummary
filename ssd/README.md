# *Single Shot Detector (SSD) container*

More about Single Shot Detector: [Paper](http://arxiv.org/abs/1512.02325)
[Slides](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf)
[Repo](https://github.com/weiliu89/caffe/tree/ssd)

## Object detector

## build the container
Download to the ssd folder the [tar file](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA) containing a pretrained Caffe model

docker build -f Dockerfile.ssd -t vfs.ssd .

## run the container
docker run -v /someDirWithimagefiles:/images  -p8888:8888 vfs.ssd

Open Jupyter in your browser, navigate to the 'examples' folder and open ssd_detect.ipynb notebook
