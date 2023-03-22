import numpy as np
import os
import pickle
import cv2

# specify the path to the directory containing the images and labels
data_dir=(r'C:\Users\Admin\OneDrive\Desktop\elephant/')

# specify the class labels
class_labels = ['elephant']

# create empty lists to store the images and labels
images = []
labels = []

# loop over the images in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.jpg'):
        # read the image
        img = cv2.imread(os.path.join(data_dir, filename))
       
        # preprocess the image
        img_resized = cv2.resize(img, (224, 224))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
       
        # add the preprocessed image and its label to the lists
        images.append(img_gray)
        labels.append(class_labels.index('elephant'))
       
# convert the lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# save the images and labels as a pickle file
with open('data.pkl', 'wb') as f:
    pickle.dump((images, labels), f)
