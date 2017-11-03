import os
import pickle
from collections import defaultdict
import numpy as np
import sys

# Format an item for addition as new entity to reference set


def add_face_to_reference(key, new_item):
    reference_face = defaultdict(dict)
    id_with_hash = key + "_" + new_item['file_content_hash']
    reference_face[id_with_hash]['face_vec'] = new_item['face_vec']
    reference_face[id_with_hash]['face_pic'] = new_item['face_pic']
    reference_face[id_with_hash]['videos'] = defaultdict(set)
    contenthash_namehash = new_item[
        'file_content_hash'] + "_" + new_item['file_name_hash']
    reference_face[id_with_hash]['videos'][
        contenthash_namehash].update(new_item['times'])
    reference_face[id_with_hash]['label'] = "unknown"
    return reference_face


# Format an item for merging with entity already in reference set
def merge_face_to_reference(orig_key, orig_item, merge_item):
    contenthash_namehash = merge_item[
        'file_content_hash'] + "_" + merge_item['file_name_hash']
    orig_item['videos'][contenthash_namehash].update(merge_item['times'])
    reference_face = defaultdict(dict)
    reference_face[orig_key] = orig_item
    return reference_face


# Load reference files if available or create new variables
def retrieve_reference_files(
        reference_faces_file,
        hash_table_file,
        current_file,
        current_faces):

    # Set up reference file and hash table file paths
    in_reference_faces = os.path.join(
        '/reference', reference_faces_file.split('/')[-1])
    in_hash_table = os.path.join('/reference', hash_table_file.split('/')[-1])

    # Load reference set if available
    if os.path.isfile(in_reference_faces):
        reference_faces = pickle.load(open(in_reference_faces, "rb"))
        print("Loaded reference set from {0} with {1} faces".format(
            in_reference_faces, len(reference_faces)))
        cold_start = False
    else:
        print("No reference set pickle file found at {0}, starting with blank set".format(
            in_reference_faces))
        reference_faces = defaultdict(dict)
        cold_start = True

    # Load file name/content hash table if available
    if os.path.isfile(in_hash_table):
        hash_table = pickle.load(open(in_hash_table, "rb"))
        print("Loaded hash table from {0} with {1} entries\n".format(
            in_hash_table, len(hash_table['hash_to_file'])))
    else:
        print("No hash table pickle file found at {0}, starting with blank table\n".format(
            in_hash_table))
        hash_table = defaultdict(dict)

    if cold_start:
        reference_faces, hash_table, write_to_hash = create_reference_faces(
            current_faces, reference_faces, hash_table, current_file)

    return reference_faces, hash_table, cold_start


# Create a reference face dictionary if no previous reference exists
def create_reference_faces(
        current_faces,
        reference_faces,
        hash_table,
        current_file):

    write_to_hash = True
    for current_face_id in current_faces:
        reference_faces.update(
            add_face_to_reference(
                current_face_id,
                current_faces[current_face_id]))
        if write_to_hash:
            hash_table = add_to_hash_table(
                current_faces, current_face_id, hash_table)
            write_to_hash = False
    print(
        "Built reference set from {0} with {1} faces".format(
            current_file,
            len(reference_faces)))

    return reference_faces, hash_table, write_to_hash


def add_to_hash_table(current_faces, current_face_id, hash_table):

    contenthash = current_faces[current_face_id]['file_content_hash']
    filename = current_faces[current_face_id]['file_name']

    hash_table['hash_to_file'][contenthash] = filename
    hash_table['file_to_hash'][filename] = contenthash

    return hash_table


def main(vectors_used,
         detected_faces_folder,
         reference_faces_file,
         hash_table_file,
         tolerance,
         verbose=False):

    # Return only files with faces that were vectorized according to
    # vectors_used parameter
    if vectors_used == 'dlib':
        filtered_folder = [file for file in os.listdir(
            detected_faces_folder) if file.endswith('.face_detected.pickle') or
            file.endswith(vectors_used + "_" + 'face_detected.pickle')]
    else:
        filtered_folder = [
            file for file in os.listdir(detected_faces_folder) if file.endswith(
                vectors_used + "_" + 'face_detected.pickle')]

    if len(filtered_folder) == 0:
        print("No files detected for {0} vectorization technique; quitting".format(
            vectors_used))
        quit()
    else:
        print(
            "{0} files detected that used {1} vectorization technique".format(
                len(filtered_folder),
                vectors_used))

    total_faces_reviewed = 0
    total_duplicates = 0
    setup_completed = False

    for current_file in filtered_folder:

        current_faces = pickle.load(
            open(
                os.path.join(
                    detected_faces_folder,
                    current_file),
                "rb"))
        total_faces_reviewed += len(current_faces)

        # If not set up already, establish face reference set and hash table
        if not setup_completed:
            reference_faces, hash_table, cold_start = retrieve_reference_files(
                reference_faces_file, hash_table_file, current_file, current_faces)
            setup_completed = True
            # If we made the reference set from the first file, move on to next
            # file
            if cold_start:
                continue

        if verbose:
            print(
                "Loaded {0} with {1} faces".format(
                    current_file,
                    len(current_faces)))

        reference_vectors_tuples = [
            (reference_faces[x]['face_vec'],
             x) for x in reference_faces]
        reference_vectors_only = [x[0] for x in reference_vectors_tuples]

        pop_list = set()
        write_to_hash = True

        # Look for duplicate faces in reference set
        for current_face_id in current_faces:

            # Write filename and content hash to dict on first pass
            if write_to_hash:
                hash_table = add_to_hash_table(
                    current_faces, current_face_id, hash_table)
                write_to_hash = False

            # Compare reference vectors to current face vector
            match = list(np.linalg.norm(np.array([current_faces[current_face_id][
                         'face_vec']]) - np.array(reference_vectors_only), axis=1) <= tolerance)

            # Create list of indexes for face matches
            match_indexes = [i for i, x in enumerate(match) if x]

            # Merge matches into reference set
            for index in match_indexes:
                key = reference_vectors_tuples[index][1]
                reference_faces.update(
                    merge_face_to_reference(
                        key,
                        reference_faces[key],
                        current_faces[current_face_id]))
                pop_list.add(current_face_id)

        # Pop off merged faces and add remaining new faces into reference set
        total_duplicates += len(pop_list)
        for item_to_pop in pop_list:
            current_faces.pop(item_to_pop)
        for current_face_id in current_faces:
            reference_faces.update(
                add_face_to_reference(
                    current_face_id,
                    current_faces[current_face_id]))

    # Summarize and save face reference set output
    print(
        "\nReviewed {0} faces and located {1} similar face vectors at threshold {2}".format(
            total_faces_reviewed,
            total_duplicates,
            tolerance))

    out_reference_file = os.path.join(
        '/reference/', reference_faces_file.split('/')[-1])
    print("Saving {0} faces as reference set to {1}".format(
        len(reference_faces), out_reference_file))
    pickle.dump(reference_faces, open(out_reference_file, "wb"))

    # Summarize and save filename/content hash table output
    out_hashtable_file = os.path.join(
        '/reference/', hash_table_file.split('/')[-1])
    print(
        "Saving filename/content hash table to {0}".format(out_hashtable_file))
    pickle.dump(hash_table, open(out_hashtable_file, "wb"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Merge faces detected from images and video via videofacesummary')

    # Optional args
    parser.add_argument(
        "--vectors_used",
        type=str,
        default="resnet50",
        help='''Vectorization technique used to generate pickle files (resnet50, facenet_tf, openface). Note that
          vector distance comparison between different vectorization techniques does not work. (default = resnet50)''')
    parser.add_argument(
        "--detected_faces_folder",
        type=str,
        default="/out",
        help="Folder with pickle files of detected faces via videofacesummary (default = /out)")
    parser.add_argument(
        "--reference_faces_file",
        type=str,
        default="face_reference_set_resnet50.pkl",
        help="Pickle file of reference set for faces in mounted /reference folder (default = face_reference_set_resnet50.pkl)")
    parser.add_argument(
        "--hash_table_file",
        type=str,
        default="hash_table.pkl",
        help="Pickle file of hash lookups for file names and contents in mounted /reference folder (default = hash_table.pkl)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="different faces are tolerance apart, 0.4->tight 0.6->loose")
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Flag to print number of bounding boxes input and unique face vectors output per image")
    args = parser.parse_args()

    print(
        "resolveVideos parameters set as: \n \
           Vectors used = {0} \n \
           Detected Faces Folder = {1} \n \
           Reference Faces File = {2} \n \
           Hash Table File = {3} \n \
           Tolerance = {4} \n \
           Verbose = {5} \n"
        .format(
            args.vectors_used,
            args.detected_faces_folder,
            args.reference_faces_file,
            args.hash_table_file,
            args.tolerance,
            args.verbose))

    sys.stdout.flush()
    sys.stderr.flush()

    main(
        args.vectors_used,
        args.detected_faces_folder,
        args.reference_faces_file,
        args.hash_table_file,
        args.tolerance,
        args.verbose)
