import glob
import os
import pickle
import sys
from collections import defaultdict
from os.path import join
import dlib
from time import time

from face import face
from normalizeface import align_face_to_template, get_face_landmarks
from utils.get_md5 import file_digest
from utils.match_to_faces import match_to_faces, write_out_pickle
from utils.get_cropped import get_cropped


# Find face chips in images based on previously discovered bounding boxes
def extract_chips(
        filename,
        bounding_boxes,
        chip_size,
        file_content_hash,
        tolerance,
        jitters,
        verbose=False):

    people = defaultdict(dict)

    for frame_number, box_list in bounding_boxes:

        cropped = get_cropped(filename, frame_number, box_list)

        vectorized_faces = vectorize_chips(cropped, chip_size, jitters)

        people = match_to_faces(
            vectorized_faces,
            cropped,
            box_list,
            people,
            frame_number,
            filename,
            file_content_hash,
            tolerance)

        # frame_number of -1 indicates image
        if verbose and frame_number == -1:
            print(
                "{0} bounding boxes resulted in {1} unique face vectors from {2}".format(
                    len(cropped),
                    len(people),
                    filename))

    write_out_pickle(filename, people, "/out", "resnet50", "face_detected")


# Process extracted face chips through resnet50 to obtain vector represenations
def vectorize_chips(chips, chip_size, jitters):

    vectorized_faces = list()
    for chip in chips:
        adjusted_face = normalize_faces(chip, chip_size)
        vectorized_faces += face.face_encodings(
            adjusted_face, [(0, chip_size, chip_size, 0)], jitters)

    return vectorized_faces


# Align chips using dlib landmarks
def normalize_faces(pic, chip_size):

    landmarks = get_face_landmarks(
        face.pose_predictor, pic, dlib.rectangle(
            0, 0, pic.shape[1], pic.shape[0]))

    adjusted_face = align_face_to_template(pic, landmarks, chip_size)

    return adjusted_face


def main(jitters=1, tolerance=0.6, chip_size=160, verbose=False):

    # Gather file names
    files = [item for item in glob.glob(
        '/bboxes/*') if not os.path.isdir(item)]

    if verbose:
        durations = list()
        kickoff = time()

    for f in files:

        print('Processing:', f.split('/')[-1])
        filename, file_content_hash, bounding_boxes = pickle.load(
            open(f, "rb"))
        file_with_path = join('/media', filename)

        # Verify original file exists on disk and has same content hash
        if os.path.isfile(file_with_path) and file_digest(
                file_with_path) == file_content_hash:
            if len(bounding_boxes) > 0:
                # only allow verbose flag for images
                if verbose and bounding_boxes[0][0] == -1:
                    start = time()
                    extract_chips(
                        filename,
                        bounding_boxes,
                        chip_size,
                        file_content_hash,
                        tolerance,
                        jitters,
                        verbose)
                    duration = time() - start
                    durations.append(duration)
                    print("{0} seconds to process {1}\n".format(
                        '%.3f' % duration, f.split('/')[-1]))
                else:
                    extract_chips(
                        filename,
                        bounding_boxes,
                        chip_size,
                        file_content_hash,
                        tolerance,
                        jitters)

            else:
                print("There were no faces detected in {0}".format(f))
        else:
            print(
                "\tWarning - {0} has bounding box file in /bboxes but was not found in /media.\n".format(filename))

        sys.stdout.flush()
        sys.stderr.flush()

    if verbose and len(durations) > 0:
        average = sum(durations) / len(durations)
        print(
            "\nAverage elapsed time to vectorize faces in an image = {0}".format(
                '%.3f' %
                average))
        print(
            "Total time to vectorize faces in {0} images = {1}".format(
                len(durations), '%.3f' %
                (time() - kickoff)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Vectorize and group previously detected faces in video and images \
                                    via resnet50')

    parser.add_argument(
        "--jitters",
        type=int,
        default=1,
        help="Perturbations to chip when calculating vector representation (default = 1)")

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="Threshold for minimum vector distance between different faces, minimum value is 0.0 and maximum value is \
             4.0 (default = 0.6)")

    parser.add_argument(
        "--chip_size",
        type=int,
        default=160,
        help="Size of face chips used for comparison (default = 160)")

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Flag to print number of bounding boxes input and unique face vectors output per image")

    args = parser.parse_args()

    print(
        "Resnet50 parameters set as: \n \
           Jitters = {0} \n \
           Face matching tolerance = {1} \n \
           Chip size = {2} \n \
           Verbose = {3} \n"
        .format(
            args.jitters,
            args.tolerance,
            args.chip_size,
            args.verbose))

    sys.stdout.flush()
    sys.stderr.flush()

    main(args.jitters, args.tolerance, args.chip_size, args.verbose)
    print("Finished processing all media.")
