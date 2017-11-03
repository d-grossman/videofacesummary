from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import numpy as np
import os
import caffe

import glob
import sys
from os.path import splitext, isdir
import cv2
from tqdm import tqdm
from time import time

from utils.get_md5 import file_digest
from utils.match_to_faces import write_out_pickle


# Process an image for faces
def process_image(image_file, threshold, nms, reduceby, net, verbose=False):

    filename = image_file.split('/')[-1]

    file_content_hash = file_digest(image_file)
    image = cv2.imread(image_file)
    frame_number = -1

    # Find bounding boxes for face chips in this image
    face_locations, num_detections = identify_chips(
        image, frame_number, threshold, nms, reduceby, net)

    # Only save pickle if faces were detected
    if num_detections > 0:
        results = (filename, file_content_hash, [face_locations])
        write_out_pickle(filename, results, "/bboxes", "frcnn", "bboxes")

    if verbose:
        print("{0} face detections in {1}".format(num_detections, filename))


# getframe
def get_frame_inefficient(filename, frame_number):
    camera = cv2.VideoCapture(filename)
    camera.set(1, frame_number)
    keep_going, image = camera.read()
    camera.release()
    return (keep_going, image)

# get movie length


def get_movie_length(filename):
    camera = cv2.VideoCapture(filename)
    ret_val = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    camera.release()
    return ret_val

# Process a video for faces


def process_video(image_file, threshold, nms, reduceby, net, every):

    frame_number = 0
    num_detections = 0
    filename = image_file.split('/')[-1]

    #camera = cv2.VideoCapture(image_file)

    #capture_length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    capture_length = get_movie_length(image_file)
    progress = tqdm(total=capture_length)

    file_content_hash = file_digest(image_file)
    combined_face_locations = list()
    keep_going = True
    first = True

    while keep_going:

        if not first:
            if (every + frame_number) > capture_length:
                keep_going = False
                progress.close()
                break
            frame_number += every
            #camera.set(1, frame_number)
            progress.update(every)
        else:
            first = False

        #keep_going, image = camera.read()
        keep_going, image = get_frame_inefficient(image_file, frame_number)

        # only face detect every once in a while
        progress.set_description(
            'Processing video: {0} detections: {1}'.format(
                filename[
                    0:30] + "...",
                num_detections))
        progress.refresh()

        # verify that there is a video frame to process
        if image is None:
            progress.refresh()
            progress.write('end of capture:IMG')
            progress.close()
            break
        if frame_number > capture_length:
            progress.refresh()
            progress.write('end of capture:Length')
            progress.close()
            break
        if not keep_going:
            progress.refresh()
            progress.write('end of capture:camera.read')
            progress.close()
            break

        # Find bounding boxes for face chips in this frame
        face_locations, detections = identify_chips(
            image, frame_number, threshold, nms, reduceby, net)
        if detections > 0:
            combined_face_locations += [face_locations]
            num_detections += detections

    # Only save pickle if faces were detected
    if num_detections > 0:
        results = (filename, file_content_hash, combined_face_locations)
        write_out_pickle(filename, results, "/bboxes", "frcnn", "bboxes")


# Detect faces and vectorize chips based on input parameters
def identify_chips(image, frame_number, threshold, nms_thresh, reduceby, net):

    resized_image = cv2.resize(image, (0, 0),
                               fx=1.0 / reduceby,
                               fy=1.0 / reduceby)

    # Modified from https://github.com/natanielruiz/py-faster-rcnn-dockerface
    # Model expects BGR cv2 image.
    # # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, resized_image)

    cls_ind = 1
    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    dets = dets[keep, :]

    keep = np.where(dets[:, 4] > threshold)
    dets = dets[keep]

    # dets are the upper left and lower right coordinates of bbox
    # dets[:, 0] = x_ul, dets[:, 1] = y_ul
    # dets[:, 2] = x_lr, dets[:, 3] = y_lr

    dets[:, 2] = dets[:, 2]
    dets[:, 3] = dets[:, 3]

    # list_face_locations = [(dets[j,0], dets[j, 1], dets[j, 2], dets[j, 3],
    # dets[j, 4])

    # Align face locations with original image
    transformed_face_locations = [[int(dets[j, 1] * reduceby),
                                   int(dets[j, 2] * reduceby),
                                   int(dets[j, 3] * reduceby),
                                   int(dets[j, 0] * reduceby)]
                                  for j in xrange(dets.shape[0])]

    frame_with_face_locations = (frame_number, transformed_face_locations)

    return frame_with_face_locations, len(transformed_face_locations)


def main(
        use_gpu,
        caffe_model,
        prototxt_file,
        threshold=0.85,
        nms=0.15,
        reduceby=1,
        every=30,
        verbose=False):

    # Look for files at /media folder
    files = [item for item in glob.glob('/media/*') if not isdir(item)]

    if verbose:
        durations = list()
        kickoff = time()

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    if use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        cfg.GPU_ID = 0
    else:
        caffe.set_mode_cpu()

    print('\n\nLoading Caffe model...')
    net = caffe.Net(prototxt_file, caffe_model, caffe.TEST)
    #net = caffe.Net(prototxt_file, caffe.TEST, "weights="+caffe_model)

    for f in files:
        ext = splitext(f)[1]

        # videos
        if ext in ['.avi', '.mov', '.mp4']:
            process_video(f, threshold, nms, reduceby, net, every)

        # images
        elif ext in ['.jpg', '.png', '.jpeg', '.bmp', '.gif']:
            if verbose:
                start = time()
                process_image(f, threshold, nms, reduceby, net, verbose)
                duration = time() - start
                durations.append(duration)
                print("{0} seconds to process {1}\n".format(
                    '%.3f' % duration, f.split('/')[-1]))
            else:
                process_image(f, threshold, nms, reduceby, net)

        sys.stdout.flush()
        sys.stderr.flush()

    final = time()

    if verbose and len(durations) > 0:
        average = sum(durations) / len(durations)
        print(
            "\nAverage elapsed time to detect faces in images = {0}".format(
                '%.3f' %
                average))
        print(
            "Total time to detect faces in {0} images = {1}".format(
                len(durations), '%.3f' %
                (final - kickoff)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Identify bounding boxes for faces in video and images using Faster-RCNN')

    # Optional args
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help='Use GPU, if available with nvidia-docker. Default = False.')

    parser.add_argument(
        '--caffe_model',
        type=str,
        default="/models/vgg16_dockerface_iter_80000.caffemodel",
        help='Pretrained caffe faster-rcnn model. (default = vgg16_dockerface_iter_80000.caffemodel)')

    parser.add_argument(
        '--prototxt_file',
        type=str,
        default="/opt/py-faster-rcnn/models/face/VGG16/faster_rcnn_end2end/test.prototxt",
        help='Prototxt file paired with pretrained caffe faster-rcnn model. (default = /opt/py-faster-rcnn/models/face/VGG16/faster_rcnn_end2end/test.prototxt)')

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Probability threshold to included a proposed face detection. (default = 0.85)')

    parser.add_argument(
        '--nms',
        type=float,
        default=0.15,
        help='Non maximum suppression threshold for Faster-RCNN. (default = 0.15)')

    parser.add_argument(
        '--reduceby',
        type=float,
        default=1.0,
        help='Factor by which to reduce image/frame resolution to increase processing speed (ex: 1 = original resolution)')

    parser.add_argument(
        '--every',
        type=int,
        default=30,
        help='Analyze every nth frame_number of video (ex: 30 = process only every 30th frame_number of video')

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Flag to print number of faces detected per image and elapsed time to detect faces per image")

    args = parser.parse_args()
    print(
        "F-RCNN parameters set as: \n \
           Use GPU = {0} \n \
           Caffe Model = {1} \n \
           Prototxt File = {2} \n \
           Threshold = {3} \n \
           NMS = {4} \n \
           Media reduced by {5}x \n \
           Analyzing every {6}th frame of video \n \
           Verbose = {7} \n"
        .format(
            args.use_gpu,
            args.caffe_model,
            args.prototxt_file,
            args.threshold,
            args.nms,
            args.reduceby,
            args.every,
            args.verbose))

    sys.stdout.flush()
    sys.stderr.flush()

    main(
        args.use_gpu,
        args.caffe_model,
        args.prototxt_file,
        args.threshold,
        args.nms,
        args.reduceby,
        args.every,
        args.verbose)

    print("Finished processing all media.")
