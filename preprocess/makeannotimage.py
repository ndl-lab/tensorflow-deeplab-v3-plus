import numpy as np
import os
import glob
import pandas as pd
import cv2
import random
from xml.etree import ElementTree as ET

random.seed(777)
class XML_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.num_classes = 3
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = glob.glob(self.path_prefix+"/*")
        random.shuffle(filenames)
        ft = open('train.txt', 'w')
        fv = open('val.txt', 'w')
        for filename in filenames:
            imgfilename=os.path.basename(filename[:-4])+".jpg"
            tree = ET.parse(filename)
            root = tree.getroot()
            flag = False
            size_tree = root.find('size')
            width = int(size_tree.find('width').text)
            height = int(size_tree.find('height').text)
            print(width,height)
            annotimg=np.zeros((height, width, 1), np.uint8)
            #o_xmin,o_xmax,o_ymin,o_ymax
            for object_tree in root.findall('object'):
                flag=True
                class_name = object_tree.find('name').text
                xmin = int(object_tree.find("bndbox").find("xmin").text)
                ymin = int(object_tree.find("bndbox").find("ymin").text)
                xmax = int(object_tree.find("bndbox").find("xmax").text)
                ymax = int(object_tree.find("bndbox").find("ymax").text)
                if class_name=="4_illustration":
                    for h in range(ymin, ymax):
                        for w in range(xmin, xmax):
                            annotimg[h, w] = max(1,annotimg[h, w])
                elif class_name!="1_overall":
                    for h in range(ymin, ymax):
                        for w in range(xmin, xmax):
                            annotimg[h, w] = 2
                else:
                    o_xmin, o_xmax, o_ymin, o_ymax =xmin,xmax,ymin,ymax
            if random.random()<0.1:
                fv.write(imgfilename + "\n")
            else:
                ft.write(imgfilename + "\n")
            cv2.imwrite(os.path.join("annotimg",imgfilename), annotimg)
        ft.close()
        fv.close()

## example on how to use it
import pickle
XML_preprocessor('annotxml')
