import os
from csv import DictReader
from os.path import isdir, isfile, join
from sys import exc_info
from utils.get_md5 import file_digest
from collections import OrderedDict

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
