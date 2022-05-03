# CSCI575-Group12
Final project for CSCI 575: Advanced Machine Learning. 

Contributors: Paul Carau, Mark Higger, Elissa Himes, Margaret Sabelhaus



### Downloading the images

This involves roughly three steps: (1) getting a subsetted list of images to download, (2) running a python script to download them, and (3) correcting the list of subsetted images to remove images with broken urls.

I include all steps in the readme in case we need to recreate said subsetted list of images, but note that if  `/data/google-data/train_subset.csv` and `/data/google-data/df_final.csv` exist on your device already, you <u>ONLY</u> need to perform step (2). These files should be in the Git repo.



Step 1:  Create `/data/google-data/train_subset.csv`

- Using `/data/createDataSubsetCsv.ipynb`, read in the data and create a CSV of subsetted images we want.
- SKIP if `train_subset.csv` exists.



Step 2: Download images

`downloadImages.py` is a python script that downloads the actual images from the the subsetted list of image URLs (`train_subset.csv`). It must be run from your terminal to work properly. 

- NOTE: If you do not have an `images` folder in `/data/google-data`, just create an empty folder and call it 'images'.

- Note that you have to change two file paths in order for this to work on your system: 
  - Line 27: `out_dir = '/Users/margaretsabelhaus/Documents/GitHub/CSCI575-Group12/data/google-data/images'` 
    - Change base directory to where you have cloned your repo.
  - Line 71: `data_file = '/Users/margaretsabelhaus/Documents/GitHub/CSCI575-Group12/data/google-data/train_subset.csv'`
    - Change base directory to where you have cloned your repo.
- Running from your terminal:
  - Open up terminal, cd into your cloned repo and then into the `data` folder, and run the python script. So, perform the two steps:
  - `cd Documents/Github/CSCI575-Group12/data`
  - `python downloadImages.py`
- This should take around 45 minutes to run. You will get some error messages about images not being found; that's fine. Those refer to broken URLs.



Step 3: Create `/data/google-data/df_final.csv` (the corrected subsetted list of images)

- Using `/data/fixDataSubsetCsv.ipynb`, read in the full list of subsetted URLs that we tried to download (`/data/google-data/train_subset.csv`). The script will compare the image ids from this CSV to the image ids of those in the `images` folder. It will isolate image ids for images that were not successfully downloaded and remove them from the dataframe. Then it saves the corrected list to a new CSV that we will use in model training.

- SKIP if `df_final.csv` exists.

  

### Data Exploration

Data exploration led to the creation of two significant scripts:

1. `/data/createDataSubsetCsv.ipynb` is used to take a subset of landmarks to use in the project due to computational restrictions.
2. `/data/fixDataSubsetCsv.ipynb` is used to correct the CSV of downloaded images to include only those without broken links.



### Data Modeling

- `FNN_basic.py` is used to generate our baseline Feedforward Neural Network model.
- `FNN_hog.py` is used to incorporate the HOG feature extraction method.
- All scripts found in the `CNN` folder were used to create our Convolutional Neural Network.
