#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv
from PIL import Image
from io import BytesIO
from urllib.request import urlopen


def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def DownloadImage(key_url):
    
    # Manually set output directory
    out_dir = '/Users/margaretsabelhaus/Documents/GitHub/CSCI575-Group12/example-project/google-data/images'
    (key, url) = key_url
    filename = os.path.join(out_dir, '%s.jpg' % key)
    
    if os.path.exists(filename):
      print('Image %s already exists. Skipping download.' % filename)
      return
    
    try:
      response = urlopen(url)
      image_data = response.read()
    except:
      print('Warning: Could not download image %s from %s' % (key, url))
      return
    
    try:
      pil_image = Image.open(BytesIO(image_data))
    except:
      print('Warning: Failed to parse image %s' % key)
      return
  
    try:
      pil_image_rgb = pil_image.convert('RGB')
    except:
      print('Warning: Failed to convert image %s to RGB' % key)
      return
    
    try:
      pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
      print('Warning: Failed to save image %s' % filename)
      return


def Run():
#  if len(sys.argv) != 3:
#    print('Syntax: %s <data_file.csv> <output_dir/>' % sys.argv[0])
#    sys.exit(0)
#  (data_file, out_dir) = sys.argv[1:]

#  if not os.path.exists(out_dir):
#    os.mkdir(out_dir)

    # Manually set csv to input
    data_file = '/Users/margaretsabelhaus/Documents/GitHub/CSCI575-Group12/example-project/google-data/train_subset.csv'
    
    key_url_list = ParseData(data_file)
    pool = multiprocessing.Pool(processes=50)
    pool.map(DownloadImage, key_url_list)


if __name__ == '__main__':
  Run()
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  