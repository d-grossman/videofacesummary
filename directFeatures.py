from collections import defaultdict
import pickle
import glob
from os.path import join
import sys
from face import face
from tqdm import tqdm

import cv2
import hashlib

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
    list_face_names = []
    for face_encoding, face_location in zip(list_face_encodings, list_face_locations):
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
            current['file_name'] = filename
            current['file_name_hash'] = hashlib.md5(str(filename).encode('utf-8')).hexdigest()
            current['file_content_hash'] = file_hash

            # Keeping sample frame for demo purposes instead of pulling the frame from disk
            # TODO - Resize frames that are larger than a given threshold
            current['frame_pic'] = resized_image

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

        for (top, right, bottom, left), name in zip(list_face_locations, list_face_names):
            cv2.rectangle(resized_image, (left-5, top-5),
                          (right+5, bottom+5), (255, 0, 0), 2)

def process_vid(filename, reduceby, every, tolerance, jitters):

    frame_number = 0
    filename = filename.split('/')[-1]
    in_filename = join('/in', filename)

    camera = cv2.VideoCapture(in_filename)
    capture_length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

    progress = tqdm(total=capture_length)

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
                progress.close()
                print('\nend of capture:IMG')
                keep_going = False
                break
            if frame_number > capture_length:
                progress.close()
                print('\nend of capture:Length')
                keep_going = False
                break
            if not keep_going:
                progress.close()
                print('\nend of capture:keep_going')
                keep_going = False
                break

        if not keep_going:
            progress.close()
            break

        resized_image = cv2.resize(img, (0, 0),
                                   fx=1.0 / reduceby,
                                   fy=1.0 / reduceby)

        list_face_locations = face.face_locations(resized_image)
        list_face_encodings = face.face_encodings(resized_image, list_face_locations, jitters)

        match_to_faces(list_face_encodings, list_face_locations, people,
                       resized_image, frame_number, constants)


    # finished processing file for faces, write out pickle
    out_file = join('/out', '{0}.face_detected.pickle'.format(filename))
    sys.stdout.flush()
    pickle.dump(people, open(out_file, 'wb'))
    print('Wrote output to ' + out_file)
    sys.stdout.flush()

def process_img(filename, reduceby, tolerance, jitters):

    filename = filename.split('/')[-1]
    in_filename = join('/in', filename)

    file_hash = file_digest(in_filename)

    constants = make_constants(filename, file_hash, reduceby, tolerance, jitters)

    people = defaultdict(dict)

    img = cv2.imread(in_filename)
    resized_image = cv2.resize(img, (0, 0),
                                   fx=1.0 / reduceby,
                                   fy=1.0 / reduceby)

    list_face_locations = face.face_locations(resized_image)
    list_face_encodings = face.face_encodings(resized_image, list_face_locations, jitters)

    match_to_faces(list_face_encodings, list_face_locations, people,
                       resized_image, -1, constants)

    # finished processing file for faces, write out pickle
    out_file = join('/out', '{0}.face_detected.pickle'.format(filename))
    pickle.dump(people, open(out_file, 'wb'))
    print('Wrote output to ' + out_file)
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
    print("Reducing media by {0}x, Analyzing every {1}th frame of video, Face matching at tolerance {2}".format(args.reduceby, args.every, args.tolerance))

    files = glob.glob('/in/*')
    for f in files:
        ext = f.split('.')[-1]
        if ext in ['avi', 'mov', 'mp4']:
            print('Start processing video:', f.split('/')[-1])
            sys.stdout.flush()
            process_vid(f, args.reduceby, args.every,
                        args.tolerance, args.jitters)
        elif ext in ['jpg', 'png', 'jpeg', 'bmp', 'gif']:
            print('Start processing image:', f.split('/')[-1])
            process_img(f, args.reduceby, args.tolerance, args.jitters)