import sys
from collections import defaultdict

import dlib
import numpy as np
from skimage import io
import pickle
import glob
from os.path import join 
import sys

import cv2

import face_recognition_models


face_detector = dlib.get_frontal_face_detector()

predictor_model = face_recognition_models.pose_predictor_model_location()
pose_predictor = dlib.shape_predictor(predictor_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


detector = dlib.get_frontal_face_detector()
options = dlib.get_frontal_face_detector()
options.num_threads = 4
options.be_verbose = True


def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def load_image_file(filename, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array
    :param filename: image file to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    return scipy.misc.imread(filename, mode=mode)


def _raw_face_locations(img, number_of_times_to_upsample=1):
    """
    Returns an array of bounding boxes of human faces in a image
    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of dlib 'rect' objects of found face locations
    """
    return face_detector(img, number_of_times_to_upsample)


def face_locations(img, number_of_times_to_upsample=1):
    """
    Returns an array of bounding boxes of human faces in a image
    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample)]


def _raw_face_landmarks(face_image, face_locations=None):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location)
                          for face_location in face_locations]

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_landmarks(face_image, face_locations=None):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image
    :param face_image: image to search
    :param face_locations: Optionally provide a list of face locations to check.
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """
    landmarks = _raw_face_landmarks(face_image, face_locations)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()]
                           for landmark in landmarks]

    # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmarks_as_tuples]


def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.
    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimentional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)

    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.4):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.
    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def process_vid(filename):
    filename = filename.split('/')[-1]
    print('about to process:', filename)
    sys.stdout.flush()
    in_filename = join('/in',filename)
    
    camera = cv2.VideoCapture(in_filename)
    capture_length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    print("there are {0} frames".format(capture_length))
    sys.stdout.flush()

    list_face_locations = []
    list_face_encodings = []
    list_face_names = []

    people = defaultdict(dict)
    currentUnknown = 0
    every = 30 
    frame = 0

    keepGoing = True
    while keepGoing: 
        for x in range(every):
            frame += 1
            keepGoing, img = camera.read()
            if img is None:
                print('end of capture:IMG')
                sys.stdout.flush()
                keepGoing = False
                break
            if frame > capture_length:
                print('end of capture:Length')
                sys.stdout.flush()
                keepGoing = False
                break
            if not keepGoing:
                print('end of capture:keepGoing')
                sys.stdout.flush()
                keepGoing = False
                break
            if (frame % 1000 == 0):
                print('processing frame:',frame)
                print('faces found :',len(people))
                sys.stdout.flush()

        try:
            small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        except:
            print('end of capture:image resize')
            sys.stdout.flush()
            break
 

        list_face_locations = face_locations(small_img)
        list_face_encodings = face_encodings(small_img, list_face_locations)

        list_face_names = []
        #if len(list_face_locations) > 0 and len(list_face_encodings) > 0:
        #    print('here1',list_face_locations,list_face_encodings[0][:10])
        #    sys.stdout.flush()
        for face_encoding, face_location in zip(list_face_encodings, list_face_locations):

            name = ''
            exists = False
            #print('here2')

            for person in people.keys():
                #print('here3')
                #sys.stdout.flush()
                current = people[person]
                facevec = current['face_vec']
                times = current['times']
                match = compare_faces([facevec], face_encoding)

                if match[0]:
                    exists = True
                    name = person
                    times.append((frame, face_location))

            if not exists:
                name = '{0}'.format(currentUnknown)
                currentUnknown += 1
                current = people[name]
                current['face_vec'] = face_encoding

                #print('name:',name)
                #sys.stdout.flush()
                
                (top, right, bottom, left) = face_location
                current['pic'] = small_img[top:bottom, left:right]
                current['times'] = list()

                #correct to original resolution
                top *= 4
                bottom *= 4
                left *= 4
                right *= 4
                current['times'].append((frame, (top,left,bottom,right)))

            list_face_names.append(name)

            for (top, right, bottom, left), name in zip(list_face_locations, list_face_names):
                cv2.rectangle(small_img, (left, top),
                              (right, bottom), (255, 0, 0), 2)

    print('processing completed writing outputfile\n')
    out_file = join('/out','{0}.face_detected.pickle'.format(filename))
    print('writting output to',out_file)
    sys.stdout.flush()
    pickle.dump(people, open(out_file,'wb'))

    print('done\n')
    sys.stdout.flush()

if __name__ == '__main__':
    files = glob.glob('/in/*')
    for f in files:
        ext = f.split('.')[-1]
        if ext in ['avi','mov','mp4']:
            process_vid(f)
