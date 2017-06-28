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

import cv2

import face_recognition_models

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
    reduceby = 4

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


        if not keepGoing:
            break

        small_img = cv2.resize(img, (0, 0), fx=1.0/reduceby, fy=1.0/reduceby)
 

        list_face_locations = face.face_locations(small_img)
        list_face_encodings = face.face_encodings(small_img, list_face_locations)

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
                match = face.compare_faces([facevec], face_encoding)

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
                top *= reduceby
                bottom *= reduceby
                left *= reduceby
                right *= reduceby
                current['times'].append((frame, (top,left,bottom,right)))

            list_face_names.append(name)

            for (top, right, bottom, left), name in zip(list_face_locations, list_face_names):
                cv2.rectangle(small_img, (left, top),
                              (right, bottom), (255, 0, 0), 2)

    #finished processing file for faces write out pickle
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
