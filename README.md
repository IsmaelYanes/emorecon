# emorecon
FER2013 adapted project using 5 emotions and pytorch

Changes:

-Removed emotions number 1 and 4 (disgust and sadness)

-Deleted unsuitable images like full black or icons, those didnt show any face or emotion.

Data folder contains 3 folders:

-Train: used to train the model

-Test: constains the images to test accuracy.

-Val: used to validate results.

Each one of the folders above has a group of folders, named with numbers form 0 to 6 initially, but in my case as I deleted 1 and 4 each one of the folders constains 5 folders with images of different emotions, 48x48 pixels in grayscale.

//TODO
