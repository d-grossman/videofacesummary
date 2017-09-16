# Modified version of compare.py and align_dataset_mtcnn.py by David Sandberg
# See https://github.com/davidsandberg/acenet.git
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import facenet
import align
import face.face as face_operations

import glob
import hashlib
import pickle
import sys
from collections import defaultdict
from os.path import join, splitext

import cv2
import dlib
import numpy as np
from tqdm import tqdm
from normalizeface import align_face_to_template, get_face_landmarks

# Set parameters for MTCNN face detection and alignment
# Output size of aligned chip; Note that pretrained Facenet models expect 160x160
image_size = 160
# Pixel padding for face within aligned chip
margin = 5
# Flag for MTCNN to detect and align multiple face detection chips
detect_multiple_faces = True

# Get MD5 hash of file
def file_digest(in_filename):
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(in_filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()

# Group variables for passing into match_to_faces function
def make_constants(filename, file_hash, reduceby, tolerance,jitters):
    return (filename, file_hash, reduceby, tolerance, jitters)

# Logic to categorize face chips into known or unknown people
def match_to_faces(
        list_face_encodings,
        list_face_locations,
        people,
        resized_image,
        frame_number,
        constants):
    filename, filecontent_hash, reduceby, tolerance, jitters = constants
    list_face_names = []
    filename_hash = hashlib.md5(str(filename).encode('utf-8')).hexdigest()

    for face_encoding, face_location in zip(
            list_face_encodings, list_face_locations):
        name = ''
        exists = False
        next_unknown = len(people.keys())

        for person in people.keys():

            # see if face_encoding matches listing of people
            current = people[person]
            facevec = current['face_vec']
            times = current['times']
            #match = face_operations.compare_faces([facevec], face_encoding, tolerance)
            match = face_operations.compare_faces(
                facevec, face_encoding, tolerance)  # normal

            if match[0]:
                exists = True
                name = person
                times.append((frame_number, face_location))

        # didnt find face, make a new entry
        if not exists:
            name = '{0}'.format(next_unknown)
            next_unknown += 1
            current = people[name]
            current['face_vec'] = face_encoding
            current['file_name'] = filename
            current['file_name_hash'] = filename_hash
            current['file_content_hash'] = filecontent_hash

            # Deprecated - Keeping sample frame for demo purposes instead of pulling the frame from disk
            current['frame_pic'] = None

            (top, right, bottom, left) = face_location
            current['face_pic'] = resized_image[top:bottom, left:right]
            current['times'] = list()

            # correct to original resolution
            top *= reduceby
            bottom *= reduceby
            left *= reduceby
            right *= reduceby
            current['times'].append((frame_number, (top, left, bottom, right)))

        list_face_names.append(name)

        #for (top, right, bottom, left), name in zip(
        #        list_face_locations, list_face_names):
        #    cv2.rectangle(resized_image, (left - 5, top - 5),
        #                  (right + 5, bottom + 5), (255, 0, 0), 2)

# Use dlib to align face chip for consistency
def normalize_faces(pic, places, jitters):
    ret_val = list()

    for place in places:
        top, right, bottom, left = place
        landmarks = get_face_landmarks(
            face_operations.pose_predictor, pic, dlib.rectangle(left, top, right, bottom))
        # Use face chip image_size from MTCNN here for consistency
        adjusted_face = align_face_to_template(pic, landmarks, image_size)
        encoding = np.array(face_operations.face_encodings(
            adjusted_face, [(0, image_size, image_size, 0)], jitters))
        ret_val.append(encoding)

    return ret_val

# Process an image
def process_img(image_file, tolerance, jitters, detector, vectorizer, image_size=160, margin=10, reduceby=1,\
    detect_multiple_faces=True, images_placeholder=None, phase_train_placeholder=None, embeddings=None, sess=None):

    filename = image_file.split('/')[-1]
    print('Processing image:', filename)

    file_hash = file_digest(image_file)
    image = cv2.imread(image_file)

    resized_image = cv2.resize(image, (0, 0),
                               fx=1.0 / reduceby,
                               fy=1.0 / reduceby)

    list_face_encodings, list_face_locations = identify_chips(resized_image, jitters, detector, vectorizer, \
        image_size, margin, detect_multiple_faces, images_placeholder, phase_train_placeholder, embeddings, sess)

    constants = make_constants(filename, file_hash, reduceby, tolerance, jitters)
    people = defaultdict(dict)

    match_to_faces(list_face_encodings, list_face_locations, people, resized_image, -1, constants)

    # finished processing file for faces, write out pickle
    write_out_pickle(filename, people, detector, vectorizer)

# Process a video
def process_vid(image_file, tolerance=0.5, jitters=1, detector=1, vectorizer=1, image_size = 160, margin = 10, reduceby = 1, every = 15,  \
                detect_multiple_faces = True, images_placeholder = None, phase_train_placeholder = None, embeddings = None, sess = None):

    frame_number = 0
    num_detections = 0
    filename = image_file.split('/')[-1]

    camera = cv2.VideoCapture(image_file)

    capture_length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm(total=capture_length)

    file_hash = file_digest(image_file)

    constants = make_constants(
        filename, file_hash, reduceby, tolerance, jitters)

    people = defaultdict(dict)

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

        keep_going, img = camera.read()

        # only face detect every once in a while
        progress.set_description(
            'Processing video: {0} faces:{1} detections:{2}'.format(filename[0:30]+"...", len(people), num_detections))
        progress.refresh()

        if img is None:
            progress.refresh()
            progress.write('end of capture:IMG')
            progress.close()
            keep_going = False
            break
        if frame_number > capture_length:
            progress.refresh()
            progress.write('end of capture:Length')
            progress.close()
            keep_going = False
            break
        if not keep_going:
            progress.refresh()
            progress.write('end of capture:keep_going')
            progress.close()
            keep_going = False
            break

        if not keep_going:
            progress.refresh()
            progress.close()
            break

        resized_image = cv2.resize(img, (0, 0),
                                   fx=1.0 / reduceby,
                                   fy=1.0 / reduceby)

        list_face_encodings, list_face_locations = identify_chips(resized_image, jitters, detector, vectorizer, \
            image_size, margin, detect_multiple_faces, images_placeholder, phase_train_placeholder, embeddings, sess)

        num_detections += len(list_face_locations)

        match_to_faces(list_face_encodings, list_face_locations, people,
                       resized_image, frame_number, constants)

    # finished processing file for faces, write out pickle
    write_out_pickle(filename, people, detector, vectorizer)

# Detect faces and vectorize chips based on input parameters
def identify_chips(image, jitters=1, detector=1, vectorizer=1, image_size = 160, \
                margin = 10, detect_multiple_faces = True, images_placeholder = None, \
                phase_train_placeholder = None, embeddings = None, sess = None):

    if detector == 1:
        # Use dlib for face detection
        list_face_locations = face_operations.face_locations(image, 2)
        faces = list()
        for location in list_face_locations:
            cropped = image[location[0]:location[2], location[3]:location[1], :]
            scaled = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            faces.append(facenet.prewhiten(scaled))
    else:
        # Detect faces with MTCNN, return aligned chips and bounding boxes
        faces, list_face_locations = align.load_and_align_data(image, image_size, margin, detect_multiple_faces)

    if vectorizer == 1:
        # Face embeddings through Dlib followed by normalization
        list_face_locations = [(int(item[0]), int(item[1]), int(item[2]), int(item[3])) for item in list_face_locations]
        #list_face_encodings = face_operations.face_encodings(image, list_face_locations, jitters)
        list_face_encodings = normalize_faces(image,list_face_locations,jitters)
    else:
        # Run forward pass in mtcnn to calculate embeddings
        if len(faces) > 0:
            feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
            list_face_encodings = [np.array([row]) for row in sess.run(embeddings, feed_dict=feed_dict)]
        else:
            list_face_encodings = list()

    return list_face_encodings, list_face_locations

# Write pickle file
def write_out_pickle(filename, people, detector, vectorizer):
    if detector == 1 and vectorizer == 1:
        out_file = join('/out', '{0}.dlib_resnet50_face_detected.pickle'.format(filename))
    elif detector == 1 and vectorizer !=1:
        out_file = join('/out', '{0}.dlib_resnetinception_face_detected.pickle'.format(filename))
    elif detector != 1 and vectorizer ==1:
        out_file = join('/out', '{0}.mtcnn_resnet50_face_detected.pickle'.format(filename))
    else:
        out_file = join('/out', '{0}.mtcnn_resnetinception_face_detected.pickle'.format(filename))

    pickle.dump(people, open(out_file, 'wb'))

def main(detector=1, vectorizer=1, model=None, reduceby=1, every=15, tolerance=0.5, jitters=1, use_gpu=False):

    files = glob.glob('/in/*')

    if vectorizer != 1:
        import tensorflow as tf
        # Start tensorflow model
        tf.Graph().as_default()
        sess = tf.Session()
        # Load the model
        print("Loading Resnet Inception Model...")
        sys.stdout.flush()
        facenet.load_model(model,sess)
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    else:
        sess = None
        images_placeholder = None
        embeddings = None
        phase_train_placeholder = None

    for f in files:
        ext = splitext(f)[1]

        # videos
        if ext in ['.avi', '.mov', '.mp4']:
            process_vid(f, tolerance, jitters, detector, vectorizer, image_size, margin, reduceby, every, \
                        detect_multiple_faces, images_placeholder, phase_train_placeholder, embeddings, sess)

        # images
        elif ext in ['.jpg', '.png', '.jpeg', '.bmp', '.gif']:
            process_img(f, tolerance, jitters, detector, vectorizer, image_size, margin, reduceby, \
                    detect_multiple_faces, images_placeholder, phase_train_placeholder, embeddings, sess)

        sys.stdout.flush()
        sys.stderr.flush()

    if vectorizer != 1:
        sess.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process video and images for faces')

    # Optional args
    parser.add_argument(
        '--detector',
        type=int,
        default=1,
        help='Technique for detecting faces. 1=Dlib (default) or 2=MTCNN.') 

    parser.add_argument(
        '--vectorizer',
        type=int,
        default=1,
        help='Technique for calculating vectors for chips. 1=Dlib/Resnet50 (default) or 2=MTCNN/ResnetInception. Note: Only compare face vectors generated from same vectorizer')

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='MTCNN model if using MTCNN as vectorizer; Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')

    parser.add_argument(
        '--reduceby',
        type=float,
        default=1.0,
        help='Factor by which to reduce image/frame resolution to increase processing speed (ex: 1 = original resolution)')

    parser.add_argument(
        '--every',
        type=int,
        default=15,
        help='Analyze every nth frame_number of video (ex: 30 = process only every 30th frame_number of video')

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="different faces are tolerance apart, 0.4->tight 0.6->loose (default = 0.5)")

    parser.add_argument(
        "--jitters",
        type=int,
        default=1,
        help="how many perturberations to use when making face vector with Dlib")

    args = parser.parse_args()
    print(
        "Parameters set as: \n \
           Detector set to {0} \n \
           Vectorizer set to {1} \n \
           Model set to {2} \n \
           Reducing image/frame resolution by {3}x \n \
           Analyzing every {4}th frame of video \n \
           Face matching at tolerance {5} \n \
           Jitters set to {6} \n " \
            .format(
            args.detector,
            args.vectorizer,
            args.model,
            args.reduceby,
            args.every,
            args.tolerance,
            args.jitters
             ))
    if args.vectorizer == 2 and args.model == None:
        print("Error - A model file name must be provided in order to use MTCNN/ResnetInception. Exiting.")
        quit()
    main(args.detector, args.vectorizer, args.model, args.reduceby, args.every, args.tolerance, args.jitters)
    print("done")
