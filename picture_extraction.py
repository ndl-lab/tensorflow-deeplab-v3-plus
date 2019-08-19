#!/usr/bin/env python3
"""Run inference a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import argparse
import os
import sys
import glob
import tensorflow as tf

import deeplab_model
from utils import preprocessing
from utils import dataset_util

from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tensorflow.python import debug as tf_debug

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', type=str,
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str,
                    help='Path to the directory to generate the inference results')

parser.add_argument('--model_dir', type=str, default='./model50',
        help="Base directory for the model. "
        "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
        "with checkpoint name.")
parser.add_argument('--base_architecture', type=str, default='resnet_v2_50',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 3

OFFSET=10


def make_xml(filepath,width,height,cordinates_lst):
    filename=os.path.basename(filepath)
    root_xml = ET.Element('annotation')
    ET.SubElement(root_xml, 'folder').text='annot'
    
    ET.SubElement(root_xml, 'filename').text=filename
    
    ET.SubElement(root_xml, 'path').text=filename
    
    xml_source=ET.SubElement(root_xml, 'source')
    ET.SubElement(xml_source, 'database').text="Unknown"
    
    xml_size=ET.SubElement(root_xml, 'size')
    ET.SubElement(xml_size, 'width').text=str(width)
    ET.SubElement(xml_size, 'height').text=str(height)
    ET.SubElement(xml_size, 'depth').text=str(3)
    
    ET.SubElement(root_xml, 'segmented').text=str(0)

    for index,cordinates in enumerate(cordinates_lst):
        #print(cordinates)
        xml_obj=ET.SubElement(root_xml, 'object')
        ET.SubElement(xml_obj, 'name').text=cordinates[0]
        ET.SubElement(xml_obj, 'pose').text=str(index)
        ET.SubElement(xml_obj, 'truncated').text=str(0)
        ET.SubElement(xml_obj, 'difficult').text=str(0)

        xml_bndbox=ET.SubElement(xml_obj, 'bndbox')
        ET.SubElement(xml_bndbox, 'xmin').text=str(cordinates[1])
        ET.SubElement(xml_bndbox, 'ymin').text=str(cordinates[2])
        ET.SubElement(xml_bndbox, 'xmax').text=str(cordinates[3])
        ET.SubElement(xml_bndbox, 'ymax').text=str(cordinates[4])
    
    tree=ET.ElementTree(root_xml)
    #print(tree)
    tree.write(os.path.join(FLAGS.output_dir,filename)+".xml",xml_declaration=False)

def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  pred_hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    pred_hooks = [debug_hook]

  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_plus_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
      })

  #examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
  image_files = glob.glob(os.path.join(FLAGS.input_dir,'*'))

  predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files,1600,1600),
        hooks=pred_hooks)

  output_dir = FLAGS.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for pred_dict, image_path in zip(predictions, image_files):
    img_raw = cv2.imread(image_path,1)
    height_r,width_r=img_raw.shape[:2]
    img=cv2.resize(img_raw,(1600,1600))
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    mask = pred_dict['decoded_labels']
    mask_g = mask[:,:,0]
    ret,mask_g = cv2.threshold(mask_g,120,255,0)
    _, contours, hierarchy = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    height,width=img.shape[:2]
    cord_lst=[]
    for index,c in enumerate(contours):
        if cv2.contourArea(c)<mask.shape[0]*mask.shape[1]/100:
            break
        x, y, w, h = cv2.boundingRect(c)
        xmar_min=max(x-OFFSET,0)
        xmar_max=min((x + w+OFFSET),width)
        ymar_min=max(y-OFFSET,0)
        ymar_max=min((y + h+OFFSET),height)
        figure_cropped_margin= img[ymar_min:ymar_max, xmar_min:xmar_max].copy()
        fh,fw=figure_cropped_margin.shape[:2]
        cord_lst.append(["graphic",xmar_min,ymar_min,xmar_max,ymar_max])
        figure_cropped_margin=cv2.resize(figure_cropped_margin,(fw*width_r//1600,fh*height_r//1600))
        cv2.imwrite(os.path.join(output_dir,os.path.basename(image_path)+"_"+str(index)+".jpg"),figure_cropped_margin)
    #pascal VOC形式でアノテーションの付与された推論結果が必要な場合、以下の2行のコメントアウトを外す。
    #make_xml(image_basename,1600,1600,cord_lst)
    #cv2.imwrite(os.path.join(FLAGS.output_dir,os.path.basename(image_path)),img)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
