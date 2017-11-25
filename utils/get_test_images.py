import os
from csv import DictReader
from os.path import isdir, isfile, join
from sys import exc_info
from utils.get_md5 import file_digest
from collections import OrderedDict
import random

try:
    from urllib.request import urlretrieve
    py3 = True
except:
    from urllib2 import urlopen
    py3 = False


def get_test_images(test_file, test_folder):

    print("Getting test images...")
    if not isdir(test_folder):
        os.mkdir(test_folder)

    pic_to_faces = OrderedDict()

    # Read in test_file contents
    with open(test_file,"r") as csvfile:
        test_items = DictReader(csvfile,fieldnames=['url','file_name','num_faces','md5_hash'])

        for row in test_items:

            test_file_name = join(test_folder, row['file_name'])

            try:

                # File has already been downloaded
                if isfile(test_file_name) and row['md5_hash'] == file_digest(test_file_name):
                    pic_to_faces[test_file_name] = int(row['num_faces'])
                    #print("Verified {0}".format(test_file_name))
                    continue

                # Download file and test hash
                else:
                    if py3:
                        urlretrieve(row['url'],test_file_name)

                    else:
                        response = urlopen(row['url'])
                        with open(test_file_name, "wb") as local_file:
                            local_file.write(response.read())

                    if row['md5_hash'] != file_digest(test_file_name):
                        os.remove(test_file_name)
                        print("Warning - {0} did not match expected hash".format(test_file_name))
                    else:
                        pic_to_faces[test_file_name] = int(row['num_faces'])
                        #print("Downloaded {0}".format(test_file_name))

            except:
                print("Error - ", exc_info())
                continue

    print("{0} test images verified\n".format(len(pic_to_faces)))
    return pic_to_faces


def get_iou_images(iou_url, test_folder, num_test_images=20, seed=41):

    print("Downloading list of test images...")
    if not isdir(test_folder):
        os.mkdir(test_folder)

    pic_to_bbox = OrderedDict()

    # Go get the iou_url file
    iou_file_name = "../test_data/iou_test_file.txt"
    try:
        if py3:
            urlretrieve(iou_url, iou_file_name)
        else:
            response = urlopen(iou_url)
            with open(iou_file_name, "wb") as local_file:
                local_file.write(response.read())
    except:
        print("Error retrieving {0} - {1}".format(iou_url,exc_info()))
        quit()


    # Read in iou_file_name contents
    with open(iou_file_name,"r") as csvfile:
        iou_items = DictReader(csvfile,fieldnames=['person','imagenum','url','rect','md5_hash'],delimiter='\t')
        marker = 0
        candidate_list = list()
        for item in iou_items:
            # Skip first two lines of header info
            if item['md5_hash'] is None:
                continue
            candidate_list.append(item)

        # Sample for variety of faces
        random.seed(seed)
        candidate_list = random.sample(candidate_list,num_test_images*20)

        print("Looking for previously downloaded test images...")
        pop_list = list()
        for i in range(len(candidate_list)):
            test_file_name = join(test_folder, candidate_list[i]['person'].replace(" ", "") + candidate_list[i]['imagenum'] + ".iou")
            if isfile(test_file_name) and candidate_list[i]['md5_hash'] == file_digest(test_file_name):
                bbox = [int(coordinate) for coordinate in candidate_list[i]['rect'].split(",")]
                pic_to_bbox[test_file_name] = bbox
                pop_list.append(i)
        # Remove already downloaded files from candidate_list
        for index_to_pop in pop_list:
            candidate_list.pop(index_to_pop)

        print("Downloading test images (slow)...")
        for row in candidate_list:

            test_file_name = join(test_folder, row['person'].replace(" ", "")+row['imagenum']+".iou")

            try:

                # File has already been downloaded
                #if isfile(test_file_name) and row['md5_hash'] == file_digest(test_file_name):
                #    bbox = [int(coordinate) for coordinate in row['rect'].split(",")]
                #    pic_to_bbox[test_file_name] = bbox
                    #print("Verified {0}".format(test_file_name))
                #    continue

                # Download file and test hash
                #else:
                    if py3:
                        urlretrieve(row['url'],test_file_name)
                    else:
                        response = urlopen(row['url'])
                        with open(test_file_name, "wb") as local_file:
                            local_file.write(response.read())

                    if row['md5_hash'] != file_digest(test_file_name):
                        os.remove(test_file_name)
                        #print("Warning - {0} did not match expected hash".format(test_file_name))
                    else:
                        bbox = [int(coordinate) for coordinate in row['rect'].split(",")]
                        pic_to_bbox[test_file_name] = bbox
                        #print("Downloaded {0}".format(test_file_name))

            except:
                #print("Error - ", exc_info())
                continue

            # Check is enough test images downloaded
            if len(pic_to_bbox) % 5 == 0 and len(pic_to_bbox) > marker:
                print("{0} of {1} test images downloaded and verified".format(len(pic_to_bbox),num_test_images))
                marker = len(pic_to_bbox)
            if len(pic_to_bbox) == num_test_images:
                break

    #print("{0} test images verified\n".format(len(pic_to_bbox)))
    return pic_to_bbox
