import glob
import os
import pickle
import sys
from collections import defaultdict
from os.path import join
import cv2
from time import time

from utils.get_md5 import file_digest
from utils.match_to_faces import match_to_faces, write_out_pickle
from utils.get_cropped import get_cropped

from align_dlib import AlignDlib
from torch_neural_net import TorchNeuralNet
#from openface_vector.align_dlib import AlignDlib
#from openface_vector.torch_neural_net import TorchNeuralNet


# Find face chips in images based on previously discovered bounding boxes
def extract_chips(filename, bounding_boxes, chip_size, file_content_hash, tolerance,net, aligner, verbose=False):

    people = defaultdict(dict)

    for frame_number, box_list in bounding_boxes:

        cropped = get_cropped(filename, frame_number, box_list)

        vectorized_faces = vectorize_chips(cropped, chip_size, net, aligner)

        people = match_to_faces(vectorized_faces, cropped, box_list, people, frame_number, filename,
                                    file_content_hash, tolerance)

        # frame_number of -1 indicates image
        if verbose and frame_number == -1:
            print("{0} bounding boxes resulted in {1} unique face vectors from {2}".format(len(cropped), len(people), filename))

    write_out_pickle(filename, people, "/out", "openface", "face_detected")


# Process extracted face chips through pretrained Torch model to obtain vector represenations
def vectorize_chips(chips, chip_size, net, aligner):

    # Torch network from OpenFace expects BGR images
    chips = [cv2.cvtColor(chip, cv2.COLOR_BGR2RGB) for chip in chips]

    # OpenFace model expects images to be aligned via dlib
    faces = list()
    for chip in chips:
       alignedface = aligner.align(chip_size, chip, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
       # Catch scenario when dlib alignment fails
       if alignedface is None:
           alignedface = cv2.resize(chip,(chip_size,chip_size))
       faces.append(alignedface)

    # Run forward pass in mtcnn to calculate embeddings
    vectorized_faces = [net.forward(face) for face in faces]

    return vectorized_faces


def main(model, dlibFacePredictor, use_gpu=False, tolerance=0.6, chip_size=96, verbose=False):

    print("Loading Facenet Model...")
    sys.stdout.flush()
    sys.stderr.flush()

    # Load the Facenet model
    if use_gpu:
        net = TorchNeuralNet(model, chip_size, cuda = True)
    else:
        net = TorchNeuralNet(model, chip_size, cuda = False)

    # Load dlib image aligner
    aligner = AlignDlib(dlibFacePredictor)

    # Gather file names
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
                    extract_chips(filename, bounding_boxes, chip_size, file_content_hash, tolerance,net,aligner, verbose)
                    duration = time() - start
                    durations.append(duration)
                    print("{0} seconds to process {1}\n".format('%.3f' % duration, f.split('/')[-1]))
                else:
                    extract_chips(filename, bounding_boxes, chip_size, file_content_hash, tolerance, net, aligner)
            else:
                print("There were no faces detected in {0}".format(f))
        else:
            print("\tWarning - {0} has bounding box file in /bboxes but was not found in /media.\n".format(filename))

        sys.stdout.flush()
        sys.stderr.flush()

    if verbose and len(durations)>0:
        average = sum(durations)/len(durations)
        print("\nAverage elapsed time to vectorize faces in an image = {0}".format('%.3f' % average))
        print("Total time to vectorize faces in {0} images = {1}".format(len(durations), '%.3f' % (time() - kickoff)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Vectorize and group previously detected faces in video and images'
                                                 'via Facenet model running in Torch')

    parser.add_argument(
        '--facenet_model',
        type=str,
        default="/models/nn4.small2.v1.t7",
        help='Facenet model to vectorize chips of detected faces')

    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        default="/opt/conda/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat",
        help='dlib face landmark model for image alignment')

    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help='Use GPU, if available with nvidia-docker. Default = False.')

    parser.add_argument(
        "--tolerance",
        type=float,
        default= 0.6,
        help="Threshold for minimum vector distance between different faces, minimum value is 0.0 and maximum value is \
             4.0 (default = 0.6)")

    parser.add_argument(
        "--chip_size",
        type=int,
        default= 96,
        help="Pretrained Facenet model from https://github.com/cmusatyalab/openface expects face chips of 96x96 \
             (default = 96)")

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Flag to print number of bounding boxes input and unique face vectors output per image")

    args = parser.parse_args()
    if not os.path.isfile(args.facenet_model):
        print("Error - Required Torch model was not found. Please verify your Torch model is located at {0}".format(args.facenet_model))
        quit()
    elif not os.path.isfile(args.dlibFacePredictor):
        print("Error - Required dlib landmark model was not found. Please verify your landmark model is located at {0}".format(args.dlibFacePredictor))
        quit()

    print(
        "Openface parameters set as: \n \
           Facenet Model = {0} \n \
           Dlib Landmark Model = {1} \n \
           Use GPU = {2} \n \
           Face matching tolerance = {3} \n \
           Chip size = {4} \n \
           Verbose = {5} \n" \
            .format(
            args.facenet_model,
            args.dlibFacePredictor,
            args.use_gpu,
            args.tolerance,
            args.chip_size,
            args.verbose))

    sys.stdout.flush()
    sys.stderr.flush()

    main(args.facenet_model, args.dlibFacePredictor, args.use_gpu, args.tolerance, args.chip_size, args.verbose)
    print("Finished processing all media.")
