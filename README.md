# CSCI575-Group12
Final project for CSCI 575: Advanced Machine Learning.



### Folders

<u>example-project</u>

- The *example-project* folder contains files that intend to (roughly) reconstruct results from another individual's model for the Google landmark data.
  - The *google-data* folder contains the data needed. Note that we likely can remove quite a few of these files as they are not being used. 
    - I created this by <u>manually downloading</u> the data from the Kaggle competition ([link](https://www.kaggle.com/google/google-landmarks-dataset)) and <u>placing all of those files</u> in a subfolder within *example-project* named *google-data*. I can't push these files to our repo because they are too big. The *google-data* folder currently exists in our repo and contains the subsetted data I created (see below.) So the user must manually download the data from that Kaggle link and place all of the files in the already-created *google-data* folder.
  - *example-code.ipynb* is Margaret's data exploration notebook. So far it includes: reading in the raw data, reducing it to only include images in classes that have a certain number of images in them, and saving a csv of said subsetted images. Currently working on getting the example model to work.
    - This csv is *google-data/train_subset.csv* and at the moment only contains images in classes with 900-950 images. There are just under 11k images in 12 classes. This was done to test if we could successfully download images given the space limitations on our laptops. If we want to extend the problem to more classes, we will likely need to use a cloud system such as google cloud. 
  - *downloadImages.py* is a python script that downloads the actual images from the the subsetted list of image URLs (*train_subset.csv*). It must be run from your terminal to work properly. 
    - Note that you have to change two file paths in order for this to work on your system: 
      - Line 27: `out_dir = '/Users/margaretsabelhaus/Documents/GitHub/CSCI575-Group12/example-project/google-data/images'` 
        - Change base directory to where you have cloned your repo.
      - Line 71: `data_file = '/Users/margaretsabelhaus/Documents/GitHub/CSCI575-Group12/example-project/google-data/train_subset.csv'`
        - Change base directory to where you have cloned your repo.
    - Running from your terminal:
      - Open up terminal, cd into your cloned repo and then into the *example-project* folder, and run the python script. So, perform the two steps:
      - `cd Documents/Github/CSCI575-Group12/example-project`
      - `python downloadImages.py`
    - This should take around 45 minutes to run. You will get some error messages about images not being found; that's fine. Those refer to broken URLs.

