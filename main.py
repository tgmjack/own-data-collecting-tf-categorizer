from data_collector import get_data
from data_handler import *
from ml_systems import *
from display_data import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from multiprocessing import Process, Manager , freeze_support
import io
import PIL.Image as Image
import os
import numpy as np
import os
import random
import datetime


# controls
actual_dir = os.getcwd()
categories = ["cow" , "spanner"]  
STANDARDIZED_IMAGE_SIZE = 150
LR = 1e-3
layers = 0
experiment_epochs = 4
epochs = 55
number_of_images_to_collect_in_total = 200
COLLECT_NEW_DATA = True        #### you can save time if the webscrappers have allready collected data and just skip to the ml systems using this
#normalize_amounts_of_data = True
########## end of controls


MODEL_NAME = 'model_to_categorize_'     #    ) # just so we remember which saved model is which, sizes must match
for c in categories:
    MODEL_NAME +="_"+ c

##########################   directory stuff below

raw_data_dir = actual_dir+'/data'
isExist = os.path.exists(raw_data_dir)
if not isExist:
    os.makedirs(raw_data_dir)
for cat in categories:
    path = raw_data_dir +'//'+str(cat)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print("The new directory is created!")
    else:
        pass



# webscrapper below
if COLLECT_NEW_DATA:
    for thing in categories:
        get_data(thing, 1 , number_of_images_to_collect_in_total , raw_data_dir)

################################## proccess and load data below
train,test = proccess_and_seperate_data(categories , raw_data_dir , STANDARDIZED_IMAGE_SIZE)
X,Y,test_x,test_y = seperate_data_into_x_and_y(train,test,STANDARDIZED_IMAGE_SIZE)

##################################  quickly check a few different models to see which is best amount of convolutedness for your objects
model = get_best_model(STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME , test_x, test_y, X, Y , experiment_epochs , LR)

############################ more heavily train the best model from above
model.fit({'input': X}, {'targets': Y},validation_set=({'input': test_x}, {'targets': test_y}),  n_epoch=epochs , snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(actual_dir+"/"+MODEL_NAME)
########################### show a sample of test data and their predictions 
display_a_test(raw_data_dir, STANDARDIZED_IMAGE_SIZE , model , categories)
