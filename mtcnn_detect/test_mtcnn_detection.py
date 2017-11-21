import cv2
from os.path import basename
from utils.get_test_images import get_test_images, get_iou_images
from mtcnn_detect.run_mtcnn import identify_chips
from mtcnn_detect.detect_face import create_mtcnn
import timeit
import tensorflow as tf
from utils.get_iou import get_iou


def main(test_file, iou_url, test_folder, test_type, use_gpu, gpu_memory_fraction, margin, min_size, threshold, scale_factor):

    # Establish MTCNN threshold to be the same for all three networks
    threshold = [threshold, threshold, threshold]

    with tf.Graph().as_default():
        # Load tensorflow session and mtcnn
        if use_gpu:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(
                config=tf.ConfigProto(
                    gpu_options=gpu_options,
                    log_device_placement=False))
        else:
            sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = create_mtcnn(sess, None)

            if test_type == 'facecount':
                # Get dictionary with ground truth for test images from test_file
                pic_to_faces = get_test_images(test_file, test_folder)

                detection_results = list()
                total_truth = sum(list(pic_to_faces.values()))
                exact_matches = 0

                # Run detections
                for item in pic_to_faces:
                    image = cv2.imread(item)
                    truth = pic_to_faces[item]
                    file = basename(item)
                    _, num_detections = identify_chips(image, frame_number=-1, margin=0, min_size=40,
                                                       threshold=threshold, scale_factor=0.709, reduceby=1, pnet=pnet,
                                                       rnet=rnet, onet=onet)
                    detection_results.append(num_detections)
                    if truth == num_detections:
                        star = '* Exact Match'
                        exact_matches +=1
                    else:
                        star = ''
                    print("File {0} - Detected {1} - Truth {2}  {3}".format(file, num_detections, truth, star))

                exact_match_percentage = '%.2f' % (float(exact_matches)/len(pic_to_faces))
                print("\nExact Matches {0} - Total Tests {1} - Percentage {2}".format(
                    exact_matches, len(pic_to_faces), exact_match_percentage))

                total_detections = sum(detection_results)
                print("Total Faces Detected {0} - Total True Faces {1}".format(total_detections, total_truth))

            elif test_type == 'duration':
                # Get dictionary with ground truth for test images from test_file
                pic_to_faces = get_test_images(test_file, test_folder)

                duration_results = list()
                # Run detections
                for item in pic_to_faces:
                    image = cv2.imread(item)
                    truth = pic_to_faces[item]
                    file = basename(item)
                    setup = "from mtcnn_detect.run_mtcnn import identify_chips"
                    t = timeit.Timer(stmt='''identify_chips(image, frame_number=-1, margin=10, min_size=40,
                        threshold=threshold, scale_factor=0.709,reduceby=1, pnet=pnet, rnet=rnet, onet=onet)''',
                        setup=setup, globals=locals())
                    durations = t.repeat(10, 1)
                    durations_avg = float(sum(durations)/len(durations))
                    duration_results.append(durations_avg)
                    width, height = image.shape[0:2]
                    print("File {0} ({1}x{2}) - Average Detection Duration {3}".format(file, height, width, '%.2f' % durations_avg))

                total_durations = sum(duration_results)
                final_duration_avg = '%.2f' % (float(total_durations)/len(duration_results))
                print("Overall Average Detection Duration {0}".format(final_duration_avg))

            elif test_type == 'iou':
                pic_to_bbox = get_iou_images(iou_url, test_folder)
                iou_results = list()
                good_matches = 0
                bad_matches = 0

                # Run iou
                for item in pic_to_bbox:
                    image = cv2.imread(item)
                    truth = pic_to_bbox[item]
                    file = basename(item)
                    face_locations, _ = identify_chips(image, frame_number=-1, margin=0, min_size=40,
                                                       threshold=threshold, scale_factor=0.709, reduceby=1, pnet=pnet,
                                                       rnet=rnet, onet=onet)
                    # Only one face bounding box for iou test set but some pictures have multiple people; take max iou
                    best = 0.0
                    for face_location in face_locations[1]:
                        t, r, b, l = face_location
                        formatted_face_location = [l, t, r, b]
                        result = get_iou(formatted_face_location, truth)
                        if result > best:
                            best = result

                    iou_results.append(best)
                    if best > 0.5:
                        good_matches += 1
                    print("File {0} - IOU {1}".format(file, '%.2f' % best))

                print("\nBounding Boxes IOU > 0.50 for {0} Test Images: {1}".format(len(pic_to_bbox),good_matches))
                #print("Bounding Boxes IOU <= 0.50 for {0} Test Images: {1}".format(len(pic_to_bbox), bad_matches))
                iou_average = '%.2f' % (float(sum(iou_results))/len(pic_to_bbox))
                print("Average IOU for {0} Test Images = {1}".format(len(pic_to_bbox), iou_average))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Test mtcnn face detection on dataset with hand-labeled counts of detected faces')

    # Optional args
    parser.add_argument(
        '--test_file',
        type=str,
        default="/prog/test_data/url_file_faces_hash.txt",
        help='Comma delimited file with urls and number of faces in image (manually labeled). (default = url_numfaces.txt')

    parser.add_argument(
        '--iou_url',
        type=str,
        default="http://www.cs.columbia.edu/CAVE/databases/pubfig/download/eval_urls.txt",
        help='URL for PubFig dataset tab delimited evaluation file (format: person imagenum url rect md5sum). (default = http://www.cs.columbia.edu/CAVE/databases/pubfig/download/eval_urls.txt')

    # Optional args
    parser.add_argument(
        '--test_images',
        type=str,
        default="/test_images",
        help='Folder to store test images retrieved from input_file. (default = /test_images')

    parser.add_argument(
        '--test_type',
        type=str,
        default="facecount",
        help="Type of test to run. Options are 'facecount', 'iou' or 'duration' (default = facecount")

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
        default=0,
        help='Pixel padding around face within aligned chip. Default = 0.')

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

    args = parser.parse_args()

    print(
        "Test parameters set as: \n \
           Test input file = {0} \n \
           IOU url = {1} \n \
           Test image folder = {2} \n \
           Test Type = {3} \n \
           Use GPU = {4} \n \
           GPU Memory Fraction = {5} \n \
           Margin = {6} \n \
           Min Size = {7} \n \
           Threshold = {8} \n \
           Scale Factor = {9} \n "
        .format(
            args.test_file,
            args.iou_url,
            args.test_images,
            args.test_type,
            args.use_gpu,
            args.gpu_memory_fraction,
            args.margin,
            args.min_size,
            args.threshold,
            args.scale_factor))
    main(args.test_file, args.iou_url, args.test_images, args.test_type, args.use_gpu, args.gpu_memory_fraction,
         args.margin, args.min_size, args.threshold, args.scale_factor)
    print("Finished test.")

