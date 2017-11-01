import glob
import sys
from os.path import splitext, isdir
import cv2
import tensorflow as tf
from tqdm import tqdm
from time import time

#from mtcnn_detect import align, detect_face
import align, detect_face
from utils.get_md5 import file_digest
from utils.match_to_faces import write_out_pickle


# Process an image for faces
def process_image(image_file, margin, min_size, threshold, scale_factor, reduceby, pnet, rnet, onet, verbose=False):

    filename = image_file.split('/')[-1]

    file_content_hash = file_digest(image_file)
    image = cv2.imread(image_file)
    frame_number = -1

    # Find bounding boxes for face chips in this image
    face_locations, num_detections = identify_chips(image, frame_number, margin, min_size, threshold,
                                         scale_factor, reduceby, pnet, rnet, onet)

    # Only save pickle if faces were detected
    if num_detections > 0:
        results = (filename, file_content_hash,[face_locations])
        write_out_pickle(filename, results, "/bboxes","mtcnn","bboxes")

    if verbose:
        print("{0} face detections in {1}".format(num_detections, filename))


# Process a video for faces
def process_video(image_file, margin, min_size, threshold, scale_factor, reduceby, pnet, rnet, onet, every):

    frame_number = 0
    num_detections = 0
    filename = image_file.split('/')[-1]

    camera = cv2.VideoCapture(image_file)

    capture_length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
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
            camera.set(1, frame_number)
            progress.update(every)
        else:
            first = False

        keep_going, image = camera.read()

        # only face detect every once in a while
        progress.set_description(
            'Processing video: {0} detections: {1}'.format(filename[0:30]+"...", num_detections))
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
        face_locations, detections = identify_chips(image, frame_number, margin, min_size, threshold,
                                                   scale_factor, reduceby, pnet, rnet, onet)
        if detections > 0:
            combined_face_locations += [face_locations]
            num_detections += detections

    # Only save pickle if faces were detected
    if num_detections > 0:
        results = (filename, file_content_hash, combined_face_locations)
        write_out_pickle(filename, results, "/bboxes","mtcnn","bboxes")


# Detect faces and vectorize chips based on input parameters
def identify_chips(image, frame_number, margin, min_size, threshold, scale_factor, reduceby, pnet, rnet, onet):

    resized_image = cv2.resize(image, (0, 0),
                               fx=1.0 / reduceby,
                               fy=1.0 / reduceby)

    # Detect faces with MTCNN, return aligned chips and bounding boxes
    _, list_face_locations = align.load_and_align_data(resized_image, margin, min_size, threshold,
                                                    scale_factor, pnet, rnet, onet)

    # Align face locations with original image
    transformed_face_locations = [[int(face_location[0] * reduceby),
                         int(face_location[1] * reduceby),
                         int(face_location[2] * reduceby),
                         int(face_location[3] * reduceby)]
                         for face_location in list_face_locations]

    frame_with_face_locations = (frame_number, transformed_face_locations)

    return frame_with_face_locations, len(list_face_locations)


def main(use_gpu=False, gpu_memory_fraction=0.8, margin=10, min_size=40, threshold=0.85, scale_factor=0.709,
         reduceby=1, every=30, verbose=False):

    # Look for files at /media folder
    files = [item for item in glob.glob('/media/*') if not isdir(item)]

    if verbose:
        durations = list()
        kickoff = time()

    # Establish MTCNN threshold to be the same for all three networks
    threshold = [threshold,threshold,threshold]

    with tf.Graph().as_default():
        # Load tensorflow session and mtcnn
        if use_gpu:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        else:
            sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

            for f in files:
                ext = splitext(f)[1]

                # videos
                if ext in ['.avi', '.mov', '.mp4']:
                    process_video(f, margin, min_size, threshold, scale_factor, reduceby, pnet, rnet, onet, every)

                # images
                elif ext in ['.jpg', '.png', '.jpeg', '.bmp', '.gif']:
                    if verbose:
                        start = time()
                        process_image(f, margin, min_size, threshold, scale_factor, reduceby, pnet, rnet, onet, verbose)
                        duration = time() - start
                        durations.append(duration)
                        print("{0} seconds to process {1}\n".format('%.3f' % duration, f.split('/')[-1]))
                    else:
                        process_image(f, margin, min_size, threshold, scale_factor, reduceby, pnet, rnet, onet)

                sys.stdout.flush()
                sys.stderr.flush()

            final = time()

    if verbose and len(durations)>0:
        average = sum(durations)/len(durations)
        print("\nAverage elapsed time to detect faces in images = {0}".format('%.3f' % average))
        print("Total time to detect faces in {0} images = {1}".format(len(durations), '%.3f' % (final - kickoff)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Identify bounding boxes for faces in video and images using MTCNN')

    # Optional args
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
        '--margin',
        type=int,
        default=10,
        help='Pixel padding around face within aligned chip. Default = 10.')

    parser.add_argument(
        '--min_size',
        type=int,
        default=40,
        help='Minimum pixel size of face to detect. Default = 40.')

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Probability threshold to included a proposed face detection. Default = 0.85.')

    parser.add_argument(
        '--scale_factor',
        type=float,
        default=0.709,
        help='Image scale factor used during MTCNN face detection. Default = 0.709.')

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
        "MTCNN parameters set as: \n \
           Use GPU = {0} \n \
           GPU Memory Fraction = {1} \n \
           Margin = {2} \n \
           Min Size = {3} \n \
           Threshold = {4} \n \
           Scale Factor = {5} \n \
           Media reduced by {6}x \n \
           Analyzing every {7}th frame of video \n \
           Verbose = {8} \n" \
           .format(
           args.use_gpu,
           args.gpu_memory_fraction,
           args.margin,
           args.min_size,
           args.threshold,
           args.scale_factor,
           args.reduceby,
           args.every,
           args.verbose))

    sys.stdout.flush()
    sys.stderr.flush()

    main(args.use_gpu, args.gpu_memory_fraction, args.margin, args.min_size, args.threshold, args.scale_factor, \
         args.reduceby, args.every, args.verbose)

    print("Finished processing all media.")
