import sys
from collections import defaultdict

import dlib
import numpy as np
from skimage import io
import pickle
import glob
from os.path import join
import sys
from face import face
from tqdm import tqdm

import cv2
import hashlib
import face_recognition_models


def file_digest(in_filename):
 # Get MD5 hash of file
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(in_filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()

def make_constants(filename, file_hash,  reduceby, tolerance, jitters):
    return (filename, file_hash, reduceby, tolerance, jitters)

def match_to_faces(list_face_encodings, list_face_locations, people, resized_image, frame_number,  constants):
    filename, file_hash, reduceby, tolerance, jitters = constants

    for face_encoding, face_location in zip(list_face_encodings, list_face_locations):

        list_face_names = []
        name = ''
        exists = False
        next_unknown = len(people.keys())

        for person in people.keys():

            # see if face_encoding matches listing of people
            current = people[person]
            facevec = current['face_vec']
            times = current['times']
            match = face.compare_faces([facevec], face_encoding, tolerance)

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
            current['video_name'] = filename
            current['video_hash'] = file_hash

            (top, right, bottom, left) = face_location
            current['pic'] = resized_image[top:bottom, left:right]
            current['times'] = list()

            # correct to original resolution
            top *= reduceby
            bottom *= reduceby
            left *= reduceby
            right *= reduceby
            current['times'].append((frame_number, (top, left, bottom, right)))

        list_face_names.append(name)

        for (top, right, bottom, left), name in zip(list_face_locations, list_face_names):
            cv2.rectangle(resized_image, (left, top),
                          (right, bottom), (255, 0, 0), 2)

def process_vid(filename, reduceby, every, tolerance, jitters):

    list_face_locations = []

    frame_number = 0
    filename = filename.split('/')[-1]
    print('about to process:', filename)
    sys.stdout.flush()
    in_filename = join('/in', filename)

    camera = cv2.VideoCapture(in_filename)
    capture_length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

    progress = tqdm(total=capture_length)

    print('there are {0} frame_numbers'.format(capture_length))
    sys.stdout.flush()

    file_hash = file_digest(in_filename)

    constants = make_constants(filename, file_hash, reduceby, tolerance, jitters)

    people = defaultdict(dict)

    keep_going = True
    while keep_going:
        for _ in range(every):
            # only face detect every once in a while
            progress.update(1)
            progress.set_description('faces:{0} '.format(len(people)))
            progress.refresh()
            frame_number += 1
            keep_going, img = camera.read()
            if img is None:
                print('end of capture:IMG')
                sys.stdout.flush()
                keep_going = False
                break
            if frame_number > capture_length:
                print('end of capture:Length')
                sys.stdout.flush()
                keep_going = False
                break
            if not keep_going:
                print('end of capture:keep_going')
                sys.stdout.flush()
                keep_going = False
                break

        if not keep_going:
            break

        resized_image = cv2.resize(img, (0, 0),
                                   fx=1.0 / reduceby,
                                   fy=1.0 / reduceby)

        list_face_locations = face.face_locations(resized_image)
        list_face_encodings = face.face_encodings(resized_image, list_face_locations, jitters)

        match_to_faces(list_face_encodings, list_face_locations, people,
                       resized_image, frame_number, constants)


    # finished processing file for faces write out pickle
    print('processing completed writing outputfile\n')
    out_file = join('/out', '{0}.face_detected.pickle'.format(filename))
    print('writting output to', out_file)
    sys.stdout.flush()
    pickle.dump(people, open(out_file, 'wb'))

    print('done\n')
    sys.stdout.flush()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process video for faces')

    # Required args
    parser.add_argument('--reduceby',
                        type=float,
                        default=2.0,
                        help='Factor by which to reduce video resolution to increase processing speed (ex: 1 = original resolution)')

    parser.add_argument('--every',
                        type=int,
                        default=15,
                        help='Analyze every nth frame_number (ex: 30 = process only every 30th frame_number of video')

    parser.add_argument("--tolerance",
                        type=float,
                        default=0.5,
                        help="different faces are tolerance apart, 0.4->tight 0.6->loose")

    parser.add_argument("--jitters",
                        type=int,
                        default=1,
                        help="how many perturberations to use when making face vector")

    args = parser.parse_args()
    print("Reducing videos by %dx and Analyzing every %dth frame_number" %
          (args.reduceby, args.every))
    sys.stdout.flush()

    files = glob.glob('/in/*')
    for f in files:
        ext = f.split('.')[-1]
        if ext in ['avi', 'mov', 'mp4']:
            process_vid(f, args.reduceby, args.every,
                        args.tolerance, args.jitters)
