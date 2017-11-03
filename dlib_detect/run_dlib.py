import glob
import sys
from os.path import splitext, isdir
import cv2
from tqdm import tqdm
from time import time

from utils.get_md5 import file_digest
from utils.match_to_faces import write_out_pickle
from face import face


# Process an image for faces
def process_image(image_file, reduceby, upsampling, verbose=False):

    filename = image_file.split('/')[-1]

    file_content_hash = file_digest(image_file)
    image = cv2.imread(image_file)
    frame_number = -1

    # Find bounding boxes for face chips in this image
    face_locations, num_detections = identify_chips(
        image, frame_number, reduceby, upsampling)

    # Only save pickle if faces were detected
    if num_detections > 0:
        results = (filename, file_content_hash, [face_locations])
        write_out_pickle(filename, results, "/bboxes", "dlib", "bboxes")

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
def process_video(image_file, reduceby, every, upsampling):

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
            image, frame_number, reduceby, upsampling)
        if detections > 0:
            combined_face_locations += [face_locations]
            num_detections += detections

    # Only save pickle if faces were detected
    if num_detections > 0:
        results = (filename, file_content_hash, combined_face_locations)
        write_out_pickle(filename, results, "/bboxes", "dlib", "bboxes")


# Detect faces and vectorize chips based on input parameters
def identify_chips(image, frame_number, reduceby, upsampling):

    resized_image = cv2.resize(image, (0, 0),
                               fx=1.0 / reduceby,
                               fy=1.0 / reduceby)

    # Detect faces with DLib, return aligned chips and bounding boxes
    list_face_locations = face.face_locations(resized_image, upsampling)

    # Align face locations with original image
    transformed_face_locations = [[int(face_location[0] * reduceby),
                                   int(face_location[1] * reduceby),
                                   int(face_location[2] * reduceby),
                                   int(face_location[3] * reduceby)]
                                  for face_location in list_face_locations]

    frame_with_face_locations = (frame_number, transformed_face_locations)

    return frame_with_face_locations, len(list_face_locations)


def main(reduceby=1, every=30, upsampling=1, verbose=False):

    # Look for files at /media folder
    files = [item for item in glob.glob('/media/*') if not isdir(item)]

    if verbose:
        durations = list()
        kickoff = time()

    for f in files:
        ext = splitext(f)[1]

        # videos
        if ext in ['.avi', '.mov', '.mp4']:
            process_video(f, reduceby, every, upsampling)

        # images
        elif ext in ['.jpg', '.png', '.jpeg', '.bmp', '.gif']:
            if verbose:
                start = time()
                process_image(f, reduceby, upsampling, verbose)
                duration = time() - start
                durations.append(duration)
                print("{0} seconds to process {1}\n".format(
                    '%.3f' % duration, f.split('/')[-1]))
            else:
                process_image(f, reduceby, upsampling)

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
        description='Identify bounding boxes for faces in video and images using dlib')

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
        '--upsampling',
        type=int,
        default=1,
        help='Number of times dlib should upsample image to scan for faces (default = 1')

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Flag to print number of faces detected per image and elapsed time to detect faces per image")

    args = parser.parse_args()
    print(
        "Dlib parameters set as: \n \
           Media reduced by {0}x \n \
           Analyzing every {1}th frame of video \n \
           Upsampling = {2} \n \
           Verbose =  {3} \n"
        .format(
            args.reduceby,
            args.every,
            args.upsampling,
            args.verbose))

    sys.stdout.flush()
    sys.stderr.flush()

    main(args.reduceby, args.every, args.upsampling, args.verbose)

    print("Finished processing all media.")
