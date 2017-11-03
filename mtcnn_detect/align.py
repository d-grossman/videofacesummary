# Modified version of align_dataset_mtcnn.py by David Sandberg at
# https://github.com/davidsandberg/facenet/blob/master/src/align/align_dataset_mtcnn.py
#
# MIT License
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

import detect_face
import numpy as np
import cv2


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def load_and_align_data(
        img,
        margin,
        minsize,
        threshold,
        factor,
        pnet,
        rnet,
        onet):
    # Pretrained Facenet model we use expects 160x160
    image_size = 160
    # Allow MTCNN to detect multiple faces in one image
    detect_multiple_faces = True
    # detect_face function expects RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Filter out any layers beyond first three (RGB)
    if img.shape[2] > 3:
        img = img[:, :, 0:3]

    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    faces = list()
    bboxes = list()
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (
                    det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack([(det[:, 0] +
                                      det[:, 2]) /
                                     2 -
                                     img_center[1], (det[:, 1] +
                                                     det[:, 3]) /
                                     2 -
                                     img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(
                    bounding_box_size -
                    offset_dist_squared *
                    2.0)  # some extra weight on the centering
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))

        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = cv2.resize(
                cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            faces.append(prewhiten(scaled))
            bboxes.append((bb[1], bb[2], bb[3], bb[0]))
        return faces, bboxes
    else:
        #print('Warning - No faces detected in image')
        return faces, bboxes
