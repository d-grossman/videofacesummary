import cv2
from os.path import join

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

# Detect faces and vectorize chips based on input parameters


def get_cropped(filename, frame_number, box_list):

    # Frame number of -1 indicates an image so we can manipulate it directly
    if frame_number == -1:
        image = cv2.imread(join('/media', filename))
        cropped = [image[box[0]:box[2], box[3]:box[1]] for box in box_list]

    # It's a video so we need to pull the specific frame
    else:
        cropped = list()
        location = join('/media', filename)
        ret, frame = get_frame_inefficient(location, frame_number)
        #cap = cv2.VideoCapture(join('/media', filename))
        #cap.set(1, frame_number);
        #ret, frame = cap.read()
        if ret:
            cropped = [frame[box[0]:box[2], box[3]:box[1]] for box in box_list]
        else:
            print("Error with recalling frame {0} from video {1}".format(
                frame_number, filename))

    return cropped
