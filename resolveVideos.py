import os
import pickle
from collections import defaultdict
from face import face

# Format an item for addition as new entity to reference set
def add_face_to_reference(key, new_item):
    reference_face = defaultdict(dict)
    id_with_hash = key + "_" + new_item['video_hash']
    reference_face[id_with_hash]['face_vec'] = new_item['face_vec']
    reference_face[id_with_hash]['pic'] = new_item['pic']
    reference_face[id_with_hash]['videos'] = defaultdict(set)
    name_with_hash = new_item['video_name'] + "__" + new_item['video_hash']
    #reference_face[id_with_hash]['videos'][name_with_hash] += new_item['times']
    reference_face[id_with_hash]['videos'][name_with_hash].update(new_item['times'])
    reference_face[id_with_hash]['label'] = "unknown"
    return reference_face

# Format an item for merging with entity already in reference set
def merge_face_to_reference(orig_key, orig_item, merge_key, merge_item):
    name_with_hash = merge_item['video_name'] + "__" + merge_item['video_hash']
    #orig_item['videos'][name_with_hash] += merge_item['times']
    orig_item['videos'][name_with_hash].update(merge_item['times'])
    reference_face = defaultdict(dict)
    reference_face[orig_key] = orig_item
    return reference_face

def main(detected_faces_folder,reference_faces_file, tolerance):

    in_reference_faces = os.path.join('/reference', reference_faces_file.split('/')[-1])

    # Load reference set if available
    if os.path.isfile(in_reference_faces):
        reference_faces = pickle.load(open(in_reference_faces, "rb"))
        print("Loaded reference set from {0} with {1} faces".format(in_reference_faces, len(reference_faces)))
        cold_start = False
    else:
        print("No reference set pickle file found at {0}, starting with blank set".format(in_reference_faces))
        reference_faces = defaultdict(dict)
        cold_start = True

    filtered_folder = [file for file in os.listdir(detected_faces_folder) if os.path.splitext(file)[1] in ['.pkl', \
                                                                                                           '.pickle']]
    total_faces_reviewed = 0
    total_duplicates = 0

    for current_file in filtered_folder:
        current_faces = pickle.load(open(os.path.join(detected_faces_folder, current_file), "rb"))
        total_faces_reviewed += len(current_faces)

        # Establish a reference set of faces from first file if no reference set exists
        if cold_start:
            for current_face_id in current_faces:
                reference_faces.update(add_face_to_reference(current_face_id, current_faces[current_face_id]))
            cold_start = False
            print("Built reference set from {0} with {1} faces".format(current_file, len(reference_faces)))
        else:
            print("Loaded {0} with {1} faces".format(current_file, len(current_faces)))
            reference_vectors_tuples = [(reference_faces[x]['face_vec'], x) for x in reference_faces]
            reference_vectors_only = [x[0] for x in reference_vectors_tuples]
            pop_list = set()

            # Look for duplicate faces in reference set
            for current_face_id in current_faces:
                match = face.compare_faces(reference_vectors_only, current_faces[current_face_id]['face_vec'], tolerance)
                match_indexes = [i for i, x in enumerate(match) if x == True]

                # Merge matches into reference set
                # TODO - Should multiple reference face matches be averaged with the current face as a new reference entity?
                for index in match_indexes:
                    key = reference_vectors_tuples[index][1]
                    reference_faces.update(merge_face_to_reference(key, reference_faces[key], current_face_id, \
                                                                   current_faces[current_face_id]))
                    pop_list.add(current_face_id)

            # Pop off merged faces and add remaining new faces into reference set
            total_duplicates += len(pop_list)
            for item_to_pop in pop_list:
                current_faces.pop(item_to_pop)
            for current_face_id in current_faces:
                reference_faces.update(add_face_to_reference(current_face_id, current_faces[current_face_id]))

    # Summarize and save output
    print("Reviewed {0} faces and merged {1} duplicate faces".format(total_faces_reviewed, total_duplicates))
    if not os.path.isdir("/reference"): os.mkdir("/reference")
    out_reference_file = os.path.join('/reference/', reference_faces_file.split('/')[-1])
    print("Saving {0} faces as reference set to {1}".format(len(reference_faces), out_reference_file))
    pickle.dump(reference_faces, open(out_reference_file, "wb"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Merge faces detected from video via directFeatures')

    # Optional args
    parser.add_argument("--detected_faces_folder",
                        type=str,
                       default="/out",
                        help="Folder with pickle files of detected faces via directFeatures (default = /out)")
    parser.add_argument("--reference_faces_file",
                        type=str,
                        default="face_reference_set.pkl",
                        help="Pickle file of reference set for faces in /reference folder (default = face_reference_set.pkl)")
    parser.add_argument("--tolerance",
                        type=float,
                        default=0.5,
                        help="different faces are tolerance apart, 0.4->tight 0.6->loose")
    args = parser.parse_args()
    main(args.detected_faces_folder, args.reference_faces_file, args.tolerance)