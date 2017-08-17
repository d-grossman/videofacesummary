import os
import pickle
from collections import defaultdict
from face import face

# Format an item for addition as new entity to reference set
def add_face_to_reference(key, new_item):
    reference_face = defaultdict(dict)
    id_with_hash = key + "_" + new_item['file_content_hash']
    reference_face[id_with_hash]['face_vec'] = new_item['face_vec']
    reference_face[id_with_hash]['frame_pic'] = new_item['frame_pic']
    reference_face[id_with_hash]['face_pic'] = new_item['face_pic']
    reference_face[id_with_hash]['videos'] = defaultdict(set)
    contenthash_namehash = new_item['file_content_hash'] + "_" + new_item['file_name_hash']
    reference_face[id_with_hash]['videos'][contenthash_namehash].update(new_item['times'])
    reference_face[id_with_hash]['label'] = "unknown"
    return reference_face

# Format an item for merging with entity already in reference set
def merge_face_to_reference(orig_key, orig_item, merge_item):
    contenthash_namehash = merge_item['file_content_hash'] + "_" + merge_item['file_name_hash']
    orig_item['videos'][contenthash_namehash].update(merge_item['times'])
    reference_face = defaultdict(dict)
    reference_face[orig_key] = orig_item
    return reference_face

def main(detected_faces_folder,reference_faces_file, hash_table_file, tolerance):

    in_reference_faces = os.path.join('/reference', reference_faces_file.split('/')[-1])
    in_hash_table = os.path.join('/reference', hash_table_file.split('/')[-1])

    # Load reference set if available
    if os.path.isfile(in_reference_faces):
        reference_faces = pickle.load(open(in_reference_faces, "rb"))
        print("Loaded reference set from {0} with {1} faces".format(in_reference_faces, len(reference_faces)))
        cold_start = False
    else:
        print("No reference set pickle file found at {0}, starting with blank set".format(in_reference_faces))
        reference_faces = defaultdict(dict)
        cold_start = True

    # Load file name/content hash table if available
    if os.path.isfile(in_hash_table):
        hash_table = pickle.load(open(in_hash_table, "rb"))
        print("Loaded hash table from {0} with {1} entries".format(in_hash_table, len(hash_table['hash_to_file'])))
    else:
        print("No hash table pickle file found at {0}, starting with blank table".format(in_hash_table))
        hash_table = defaultdict(dict)

    filtered_folder = [file for file in os.listdir(detected_faces_folder) if file.endswith('face_detected.pickle')]
    total_faces_reviewed = 0
    total_duplicates = 0

    for current_file in filtered_folder:
        current_faces = pickle.load(open(os.path.join(detected_faces_folder, current_file), "rb"))
        total_faces_reviewed += len(current_faces)
        write_to_hash = True

        # Establish a reference set of faces from first file if no reference set exists
        if cold_start:
            for current_face_id in current_faces:
                reference_faces.update(add_face_to_reference(current_face_id, current_faces[current_face_id]))
                if write_to_hash:
                    contenthash = current_faces[current_face_id]['file_content_hash']
                    filename = current_faces[current_face_id]['file_name']
                    hash_table['hash_to_file'][contenthash] = filename
                    hash_table['file_to_hash'][filename] = contenthash
                    write_to_hash = False
            cold_start = False
            print("Built reference set from {0} with {1} faces".format(current_file, len(reference_faces)))
        else:
            print("Loaded {0} with {1} faces".format(current_file, len(current_faces)))
            reference_vectors_tuples = [(reference_faces[x]['face_vec'], x) for x in reference_faces]
            reference_vectors_only = [x[0] for x in reference_vectors_tuples]
            pop_list = set()

            # Look for duplicate faces in reference set
            for current_face_id in current_faces:

                # Write filename and content hash to dict on first pass
                if write_to_hash:
                    contenthash = current_faces[current_face_id]['file_content_hash']
                    filename = current_faces[current_face_id]['file_name']
                    hash_table['hash_to_file'][contenthash] = filename
                    hash_table['file_to_hash'][filename] = contenthash
                    write_to_hash = False

                # Compare reference vectors to current face vector
                match = face.compare_faces(reference_vectors_only, current_faces[current_face_id]['face_vec'], tolerance)
                match_indexes = [i for i, x in enumerate(match) if x == True]

                # Merge matches into reference set
                # TODO - Should multiple reference face matches be averaged with the current face as a new reference entity?
                for index in match_indexes:
                    key = reference_vectors_tuples[index][1]
                    reference_faces.update(merge_face_to_reference(key, reference_faces[key], current_faces[current_face_id]))
                    pop_list.add(current_face_id)

            # Pop off merged faces and add remaining new faces into reference set
            total_duplicates += len(pop_list)
            for item_to_pop in pop_list:
                current_faces.pop(item_to_pop)
            for current_face_id in current_faces:
                reference_faces.update(add_face_to_reference(current_face_id, current_faces[current_face_id]))

    # Summarize and save face reference set output
    print("Reviewed {0} faces and merged {1} duplicate faces".format(total_faces_reviewed, total_duplicates))
    if not os.path.isdir("/reference"): os.mkdir("/reference")
    out_reference_file = os.path.join('/reference/', reference_faces_file.split('/')[-1])
    print("Saving {0} faces as reference set to {1}".format(len(reference_faces), out_reference_file))
    pickle.dump(reference_faces, open(out_reference_file, "wb"))

    # Summarize and save filename/content hash table output
    out_hashtable_file = os.path.join('/reference/', hash_table_file.split('/')[-1])
    print("Saving filename/content hash table to {0}".format(out_hashtable_file))
    pickle.dump(hash_table, open(out_hashtable_file, "wb"))

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
    parser.add_argument("--hash_table_file",
                        type=str,
                        default="hash_table.pkl",
                        help="Pickle file of hash lookups for file names and contents in /reference folder (default = hash_table.pkl)")
    parser.add_argument("--tolerance",
                        type=float,
                        default=0.5,
                        help="different faces are tolerance apart, 0.4->tight 0.6->loose")
    args = parser.parse_args()
    print("Face matching at tolerance {0}".format(args.tolerance))
    main(args.detected_faces_folder, args.reference_faces_file, args.hash_table_file, args.tolerance)