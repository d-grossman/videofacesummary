import glob
import sys
from tqdm import tqdm
import cv2
from os.path import splitext, isdir
from time import time

from tinyface_face_extractor import extract_tinyfaces
from utils.get_md5 import file_digest
from utils.match_to_faces import write_out_pickle


# Process an image for faces
def process_image(
        image_file,
        reduceby,
        prob_thresh,
        nms_thresh,
        use_gpu,
        verbose=False):

    filename = image_file.split('/')[-1]

    file_content_hash = file_digest(image_file)
    image = cv2.imread(image_file)
    frame_number = -1

    # Find bounding boxes for face chips in this image
    face_locations, num_detections = identify_chips(
        image, frame_number, reduceby, prob_thresh, nms_thresh, use_gpu)

    # Only save pickle if faces were detected
    if num_detections > 0:
        results = (filename, file_content_hash, [face_locations])
        write_out_pickle(filename, results, "/bboxes", "tinyface", "bboxes")

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


def process_video(
        image_file,
        reduceby,
        every,
        prob_thresh,
        nms_thresh,
        use_gpu):

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
            image, frame_number, reduceby, prob_thresh, nms_thresh, use_gpu)

        if detections > 0:
            combined_face_locations += [face_locations]
            num_detections += detections

    # Only save pickle if faces were detected
    if num_detections > 0:
        results = (filename, file_content_hash, combined_face_locations)
        write_out_pickle(filename, results, "/bboxes", "tinyface", "bboxes")


# Detect faces and vectorize chips based on input parameters
def identify_chips(
        image,
        frame_number,
        reduceby,
        prob_thresh,
        nms_thresh,
        use_gpu):

    resized_image = cv2.resize(image, (0, 0),
                               fx=1.0 / reduceby,
                               fy=1.0 / reduceby)

    list_face_locations = extract_tinyfaces(
        resized_image, prob_thresh, nms_thresh, use_gpu)

    list_face_locations = [(int(x[1]), int(x[2]), int(
        x[3]), int(x[0])) for x in list_face_locations]

    # Align face locations with original image
    transformed_face_locations = [[int(face_location[0] * reduceby),
                                   int(face_location[1] * reduceby),
                                   int(face_location[2] * reduceby),
                                   int(face_location[3] * reduceby)]
                                  for face_location in list_face_locations]

    frame_with_face_locations = (frame_number, transformed_face_locations)

    return frame_with_face_locations, len(list_face_locations)


def main(
        use_gpu=False,
        prob_thresh=0.7,
        nms_thresh=0.1,
        reduceby=1,
        every=30,
        verbose=False):

    # Look for files at /media folder
    files = [item for item in glob.glob('/media/*') if not isdir(item)]

    if verbose:
        durations = list()
        kickoff = time()

    for f in files:
        ext = splitext(f)[1]

        # videos
        if ext in ['.avi', '.mov', '.mp4']:
            process_video(f, reduceby, every, prob_thresh, nms_thresh, use_gpu)

        # images
        elif ext in ['.jpg', '.png', '.jpeg', '.bmp', '.gif']:
            if verbose:
                start = time()
                process_image(f, reduceby, prob_thresh,
                              nms_thresh, use_gpu, verbose)
                duration = time() - start
                durations.append(duration)
                print("{0} seconds to process {1}\n".format(
                    '%.3f' % duration, f.split('/')[-1]))
            else:
                process_image(f, reduceby, prob_thresh, nms_thresh, use_gpu)

        sys.stdout.flush()
        sys.stderr.flush()

    final = time()

    if verbose and len(durations) > 0:
        average = sum(durations) / len(durations)
        print("\nAverage elapsed time to detect faces in images = {0}".format(
            '%.3f' % average))
        print("Total time to detect faces in {0} images = {1}".format(
            len(durations), '%.3f' % (final - kickoff)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Process video for faces using TinyFace')

    # Required args
    parser.add_argument(
        '--reduceby',
        type=float,
        default=1.0,
        help='Factor by which to reduce video resolution to increase processing speed (ex: 1 = original resolution)')

    parser.add_argument(
        '--every',
        type=int,
        default=30,
        help='Analyze every nth frame_number (ex: 30 = process only every 30th frame_number of video')

    parser.add_argument(
        "--prob_thresh",
        type=float,
        default=0.85,
        help="Tiny Face Detector threshold for face likelihood (default = 0.85)")

    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.1,
        help="Tiny Face Detector non-maximum suppression threshold (default = 0.1)")

    parser.add_argument(
        "--use_gpu",
        type=bool,
        default=False,
        help="Flag to use GPU (note: Use nvidia-docker to run container")

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Flag to print number of faces detected per image and elapsed time to detect faces per image")

    args = parser.parse_args()

    print(
        "Tinyface parameters set as: \n \
           Use GPU = {0} \n \
           TinyFace Probability Threshold = {1} \n \
           TinyFace NMS Threshold = {2} \n \
           Media reduced by {3}x \n \
           Analyzing every {4}th frame of video \n \
           Verbose =  {5} \n"
        .format(
            args.use_gpu,
            args.prob_thresh,
            args.nms_thresh,
            args.reduceby,
            args.every,
            args.verbose))

    sys.stdout.flush()
    sys.stderr.flush()

    main(args.use_gpu, args.prob_thresh, args.nms_thresh,
         args.reduceby, args.every, args.verbose)

    print("Finished processing all media.")
