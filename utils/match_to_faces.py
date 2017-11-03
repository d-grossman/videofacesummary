import hashlib
import numpy as np
from os.path import join
import pickle

# Logic to categorize face chips into known or unknown people


def match_to_faces(
        vectorized_faces,
        cropped,
        box_list,
        people,
        frame_number,
        filename,
        file_content_hash,
        tolerance):

    list_face_names = []
    filename_hash = hashlib.md5(str(filename).encode('utf-8')).hexdigest()

    for face_encoding, face_location, face_pic in zip(
            vectorized_faces, box_list, cropped):
        name = ''
        exists = False
        next_unknown = len(people.keys())
        top, right, bottom, left = face_location
        for person in people.keys():
            # see if face_encoding matches listing of people
            current = people[person]
            facevec = current['face_vec']
            times = current['times']
            match = list(
                np.linalg.norm(
                    [facevec] -
                    face_encoding,
                    axis=1) <= tolerance)

            if match[0]:
                exists = True
                name = person
                times.append((frame_number, (top, left, bottom, right)))

        # didnt find face, make a new entry
        if not exists:
            name = '{0}'.format(next_unknown)
            next_unknown += 1
            current = people[name]
            current['face_vec'] = face_encoding
            current['file_name'] = filename
            current['file_name_hash'] = filename_hash
            current['file_content_hash'] = file_content_hash
            current['face_pic'] = face_pic
            current['times'] = list()
            current['times'].append((frame_number, (top, left, bottom, right)))

        list_face_names.append(name)

    return people


# Write pickle file with vector and grouping results
def write_out_pickle(
        filename,
        results,
        destination="/bboxes",
        technique="mtcnn",
        purpose="bboxes"):
    out_file = join(
        destination,
        '{0}.{1}_{2}.pickle'.format(
            filename,
            technique,
            purpose))
    pickle.dump(results, open(out_file, 'wb'))
