import glob
import os
import pickle
import sys
from collections import defaultdict
from os.path import join
import cv2
import tensorflow as tf
from time import time

from facenet_tf_vector import facenet
from utils.get_md5 import file_digest
from utils.prewhiten import prewhiten
from utils.match_to_faces import match_to_faces, write_out_pickle
from utils.get_cropped import get_cropped


# Detect faces and vectorize chips based on input parameters
def extract_chips(filename, bounding_boxes, chip_size, file_content_hash, tolerance, images_placeholder,
                  phase_train_placeholder, embeddings, sess, verbose=False):

    people = defaultdict(dict)

    for frame_number, box_list in bounding_boxes:

        cropped = get_cropped(filename, frame_number, box_list)

        vectorized_faces = vectorize_chips(cropped, chip_size, images_placeholder, phase_train_placeholder,
                                               embeddings, sess)

        people = match_to_faces(vectorized_faces, cropped, box_list, people, frame_number, filename,
                                    file_content_hash, tolerance)

        # frame_number of -1 indicates image
        if verbose and frame_number == -1:
            print("{0} bounding boxes resulted in {1} unique face vectors from {2}".format(len(cropped), len(people), filename))

    write_out_pickle(filename, people, "/out", "facenet_tf", "face_detected")


# Process extracted face chips through pretrained Tensorflow model to obtain vector represenations
def vectorize_chips(chips, chip_size, images_placeholder, phase_train_placeholder, embeddings, sess):

    # Adjust colors, prewhiten and resize to chip size appropriate for Facenet model input
    rgb_colors = [cv2.cvtColor(chip, cv2.COLOR_BGR2RGB) for chip in chips]
    prewhitened = [prewhiten(rgb_color) for rgb_color in rgb_colors]
    faces = [cv2.resize(prew, (chip_size, chip_size), interpolation=cv2.INTER_LINEAR) for prew in prewhitened]

    # Run forward pass in mtcnn to calculate embeddings
    feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
    vectorized_faces = sess.run(embeddings, feed_dict=feed_dict)

    return vectorized_faces


def main(model, use_gpu=False, gpu_memory_fraction=0.8, tolerance=0.6, chip_size=160, verbose=False):

    if use_gpu:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    else:
        sess = tf.Session()

    # Load the model
    print("Loading Facenet Model...")
    sys.stdout.flush()
    sys.stderr.flush()
    facenet.load_model(model,sess)
    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Gather files names
    files = [item for item in glob.glob('/bboxes/*') if not os.path.isdir(item)]

    if verbose:
        durations = list()
        kickoff = time()

    for f in files:

        print('Processing:', f.split('/')[-1])
        filename, file_content_hash, bounding_boxes = pickle.load(open(f,"rb"))
        file_with_path = join('/media', filename)
        # Verify original file exists on disk and has same content hash
        if os.path.isfile(file_with_path) and file_digest(file_with_path) == file_content_hash:
            if len(bounding_boxes) > 0 and bounding_boxes[0][0] == -1:
                if verbose:
                    start = time()
                    extract_chips(filename, bounding_boxes, chip_size, file_content_hash, tolerance, images_placeholder,
                              phase_train_placeholder, embeddings, sess, verbose)
                    duration = time() - start
                    durations.append(duration)
                    print("{0} seconds to process {1}\n".format('%.3f' % duration, f.split('/')[-1]))
                else:
                    extract_chips(filename, bounding_boxes, chip_size, file_content_hash, tolerance, images_placeholder,
                              phase_train_placeholder, embeddings, sess)
            else:
                print("There were no faces detected in {0}".format(f))
        else:
            print("\tWarning - {0} has bounding box file in /bboxes but was not found in /media.\n".format(filename))

        sys.stdout.flush()
        sys.stderr.flush()

    sess.close()

    if verbose and len(durations)>0:
        average = sum(durations)/len(durations)
        print("\nAverage elapsed time to vectorize faces in an image = {0}".format('%.3f' % average))
        print("Total time to vectorize faces in {0} images = {1}".format(len(durations), '%.3f' % (time() - kickoff)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Vectorize and group previously detected faces in video and images')

    parser.add_argument(
        '--facenet_model',
        type=str,
        default="/models/20170512-110547",
        help='Facenet model to vectorize chips of detected faces; Could be either a directory containing the meta_file \
              and ckpt_file or a model protobuf (.pb) file')

    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help='Use GPU, if available with nvidia-docker. Default = False.')

    parser.add_argument(
        '--gpu_memory_fraction',
        type=float,
        default=0.8,
        help='If use_gpu is True, percentage of GPU memory to use. Default = 0.8.')

    parser.add_argument(
        "--tolerance",
        type=float,
        default= 0.6,
        help="Threshold for minimum vector distance between different faces, minimum value is 0.0 and maximum value is 4.0 (default = 0.6)")

    parser.add_argument(
        "--chip_size",
        type=int,
        default= 160,
        help="Pretrained Facenet model from https://github.com/davidsandberg/facenet expects face chips of 160x160 (default = 160)")

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Flag to print number of bounding boxes input and unique face vectors output per image")

    args = parser.parse_args()
    print(
        "FacenetTF parameters set as: \n \
           Model = {0} \n \
           Use GPU = {1} \n \
           GPU Memory Fraction = {2} \n \
           Face matching tolerance = {3} \n \
           Chip size = {4} \n \
           Verbose = {5} \n" \
            .format(
            args.facenet_model,
            args.use_gpu,
            args.gpu_memory_fraction,
            args.tolerance,
            args.chip_size,
            args.verbose))

    sys.stdout.flush()
    sys.stderr.flush()

    main(args.facenet_model, args.use_gpu, args.gpu_memory_fraction, args.tolerance, args.chip_size, args.verbose)
    print("Finished processing all media.")
