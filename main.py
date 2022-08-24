


import base64
import numpy as np
import matplotlib.pyplot as plt

import cv2
from tqdm import tqdm
import random
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
from selenium.webdriver.chrome.options import Options
import requests
from multiprocessing import Process, Manager , freeze_support
import io
import PIL.Image as Image
import os

# controls
experiment_epochs = 2
categories = ["pig", "cow"]
STANDARDIZED_IMAGE_SIZE = 150
LR = 1e-3
layers = 0
epochs = 35
number_of_images_to_collect_in_total = 40
COLLECT_NEW_DATA = False        #### you can dsave time if the webscrappers have allready done their thing
#normalize_amounts_of_data = True
folder_to_save_data = 'C://Users//tgmjack//Desktop//fully automated ml area'


for cat in categories:
    path =folder_to_save_data+'//'+str(cat)

    isExist = os.path.exists(path)

    print(isExist)

    if not isExist:
        os.makedirs(path)
        print("The new directory is created!")


MODEL_NAME = 'crazy name'     #    ) # just so we remember which saved model is which, sizes must match
actual_dir = "C:/logs/cvd4/"
raw_data_dir = 'C://Users//tgmjack//Desktop//fully automated ml area'



# webscrapper below

def accept_cookies(driver):
    accept_xp = '//*[@id="L2AGLb"]'
    WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,accept_xp)))
    accept = driver.find_element_by_xpath(accept_xp)
    accept.click()
def setup_driver():
    driver_xpath2 = "chromedriver.exe"
    leaseplan_url = 'https://images.google.co.uk/'
    driver = webdriver.Chrome(driver_xpath2)
    driver.get(leaseplan_url)
#    driver.maximize_window()

    return driver

def search_for_image(driver, keyword):
    search_box_xp = '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input'
    WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,search_box_xp)))
    search_box = driver.find_element_by_xpath(search_box_xp)
    search_box.send_keys(keyword)
    time.sleep(0.5)

    search_button_xp = '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button'
    WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,search_button_xp)))
    search_button = driver.find_element_by_xpath(search_button_xp)
    search_button.click()

    first_image_xp = '//*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img'
    WebDriverWait(driver,20).until(EC.presence_of_element_located((By.XPATH,first_image_xp)))
    first_image = driver.find_element_by_xpath(first_image_xp)

def check_if_this_index_image_is_visible(driver, index):
    #//*[@id="islrg"]/div[1]/div[72]/a[1]/div[1]/img
    img_xp_p1 = '//*[@id="islrg"]/div[1]/div['
    img_xp_p2 = ']/a[1]/div[1]/img'
    img_xp = img_xp_p1 + str(index) + img_xp_p2
    print("looking for "+str(img_xp))
    try:
        WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,img_xp)))
        img = driver.find_element_by_xpath(img_xp)
        return True
    except:
        return False

def scroll_down_until_this_index_is_visible(driver, index):
    times_scrolled = 0
    num_to_scroll_to = 0
    y_pixels_per_screen = 1000
    #times_scrolled = 0
   # for t in range(times_scrolled):
  #      num_to_scroll_to = times_scrolled * y_pixels_per_screen
 #       driver.execute_script("window.scrollTo(0, "+str(num_to_scroll_to)+")")
#    last_point = num_to_scroll_to
    while not check_if_this_index_image_is_visible(driver, index):
        num_to_scroll_to = times_scrolled * y_pixels_per_screen
        times_scrolled+=1
        driver.execute_script("window.scrollTo(0, "+str(num_to_scroll_to)+")")
        print("num_to_scroll_to  = "+ str(num_to_scroll_to))
        if times_scrolled%6 == 0:
            print("done")
            break;
    print("its visible")



def save_image(driver, index , category):
    print("top of save image func")
    img_xp_p1 = '//*[@id="islrg"]/div[1]/div['
    img_xp_p2 = ']/a[1]/div[1]/img'
    img_xp = img_xp_p1 + str(index) + img_xp_p2
    filename_to_save = folder_to_save_data +"/"+str(category)+"/"+str(category)+" "+str(index)+".png"#
    if index > 30:
        print(img_xp)
        WebDriverWait(driver,5).until(EC.presence_of_element_located((By.XPATH,img_xp)))
        img = driver.find_element_by_xpath(img_xp)
    try:
        WebDriverWait(driver,5).until(EC.presence_of_element_located((By.XPATH,img_xp)))
        img = driver.find_element_by_xpath(img_xp)
    except:
        print("couldnt find image")
        return False


    data = img.get_attribute("src")
    try:
        head, encodedData = data.split(',', 1)
        file_ext = head.split(';')[0].split('/')[1]
        base = head.split(';')[1]
        decodedData = encodedData
        if base == "base64":
            decodedData = base64.b64decode(encodedData)
        elif base == "SomeOtherBase":    # Add any other base you want
            pass
        with open(str(category)+"/"+str(index)+"."+file_ext, 'wb') as f:
            f.write(decodedData)
    except:
        response = requests.get(data)
        if response.status_code == 200:
            filename_to_save = str(category)+"/"+str(index)+".png"
            image = Image.open(io.BytesIO(response.content))
            image.save(filename_to_save)



def more_button(driver):
    xp = '//*[@id="islmp"]/div/div/div[2]/div[1]/div[2]/div[2]/input'
    WebDriverWait(driver,1).until(EC.presence_of_element_located((By.XPATH,xp)))
    button = driver.find_element_by_xpath(xp)
    button.click()

def get_data(keyword, start_index_num , end_index_num):

    driver = setup_driver()
    accept_cookies(driver)
    search_for_image(driver, keyword)


    ##### scroll a bit
    time.sleep(1)
    num_to_scroll_to = 1200
    driver.execute_script("window.scrollTo(0, "+str(num_to_scroll_to)+")")
    print("   lkjjlkjlkj    ")
    time.sleep(1)


    ##### make sure its visible
    scroll_down_until_this_index_is_visible(driver, start_index_num)
    times_scrolled = 0
    done = False
    consequitive_fails_to_load_more = 0
    for i in range(end_index_num-start_index_num):
        if not done:
            print("up to index  =  "+str(i) )
            index = start_index_num+i
            if index % 10 == 0:  # scroll down periodically
                num_to_scroll_to = times_scrolled * 1200
                times_scrolled+=1
                driver.execute_script("window.scrollTo(0, "+str(num_to_scroll_to)+")")
                if index%20==0: # click load more images button
                    try:
                        more_button(driver)
                        consequitive_fails_to_load_more = 0
                    except:
                        consequitive_fails_to_load_more += 1
                        print("no load more button")
                        if consequitive_fails_to_load_more > 20:   # done
                            done = True
            try:
                save_image(driver, index , keyword)
            except:
                print("failed to save image ")

    driver.quit()





if COLLECT_NEW_DATA:
    for thing in categories:
        get_data(thing, 1 , 2000 )



#######################      multiproccessing stuff below to save time webscrapping

multiproces = """
print("brc 4.3")
processes = []
print("brc 4.4")

print("yo")
print(__name__)
if __name__ == "__main__":
    freeze_support()
    manager = Manager()
    print("brc 4.5")
    d = manager.dict()
    print("brc 4.6")
    proc_nums_over_2 = os.cpu_count () /2

    print("well "+str(proc_nums_over_2))
    number_of_images_per_process = int(number_of_images_to_collect_in_total / proc_nums_over_2)
    for word in categories:
        for proc_num in proc_nums_over_2:
            print("brc 5")
            start_index_num = number_of_images_per_process* helper
            end_index_num = number_of_images_per_process* (helper+1)
            if word == category[1]:
                proc_num = proc_num + proc_nums_over_2
            d[proc_num] = []
            p1 = Process( target = get_data , args = (word, start_index_num , end_index_num ))
            processes.append(p1)
        for proc in processes:
            print("got a process going")
            proc.start()
        for proc in processes:
            proc.join()

print("done got data :)")

print(9/0)
# In[ ]:
"""

################################################# proccess and load data below


def show_image(im):
    plt.imshow(im, cmap='gray')  # graph it
    plt.show()  # display!

def proccess_and_seperate_data():
    proccessed_data = []


    fail_counter = 0
    for c in categories:
        for path in tqdm(os.listdir(raw_data_dir+"\\"+str(c))):
            try:
                path = raw_data_dir+"\\"+str(c)+"\\"+path
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (STANDARDIZED_IMAGE_SIZE,STANDARDIZED_IMAGE_SIZE))
                if categories[0].lower() in c.lower():
                    proccessed_data.append([np.array(img) , np.array([1,0])])
                elif categories[1].lower() in c.lower():
                    proccessed_data.append([ np.array(img) , np.array([0,1])])
            except:
               # print(fail_counter)
                fail_counter+= 1

   # A = int(int(len(proccessed_data)-1)/100)
    A = int(len(proccessed_data)-1)
    b = int(A * 0.8)
    c = int(A * 0.2)
    np.random.shuffle(proccessed_data)
    train = proccessed_data[:b]
    test = proccessed_data[b:A]

    print("    --------- data info below ----------         ")
    print(fail_counter)
    print(len(train))
    print(len(test))
    np.random.shuffle(test)
    np.random.shuffle(train)
    np.save(raw_data_dir+'/test_data.npy', test)
    np.save(raw_data_dir+'/train_data.npy', train)

    return train,test;


train,test = proccess_and_seperate_data()






#######   make neural net below



import numpy as np
import os
import random
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import datetime
print(tf.__version__)



file_name = actual_dir+'/'+'{}.meta'.format(MODEL_NAME)




def make_model(layers , model_type):

    tf.compat.v1.reset_default_graph()

    convnet = input_data(shape=[None,STANDARDIZED_IMAGE_SIZE,STANDARDIZED_IMAGE_SIZE, 1], name='input')

    if model_type == 1:
        #                   incoming, nb_filter, filter_size
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = fully_connected(convnet, 1024, activation='relu')
    if model_type == 2:
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = fully_connected(convnet, 512, activation='relu')

    if model_type == 3:
        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = fully_connected(convnet, 1024, activation='relu')

    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(convnet, tensorboard_dir=actual_dir, tensorboard_verbose=3)
    model.save(actual_dir+"/"+MODEL_NAME)
    return model


X = np.array([i[0] for i in train]).reshape(-1,STANDARDIZED_IMAGE_SIZE,STANDARDIZED_IMAGE_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,STANDARDIZED_IMAGE_SIZE,STANDARDIZED_IMAGE_SIZE,1)
test_y = [i[1] for i in test]


print(X)
print(X.ndim)
print(X.shape)


print("1 start ")
model_1 = make_model(layers , 1)
model_1.fit({'input': X}, {'targets': Y},validation_set=({'input': test_x}, {'targets': test_y}),  n_epoch=experiment_epochs , snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

print("2 start ")
model_2 = make_model(layers , 2)
model_2.fit({'input': X}, {'targets': Y},validation_set=({'input': test_x}, {'targets': test_y}),  n_epoch=experiment_epochs , snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

print("3 start ")
model_3 = make_model(layers , 3)
model_3.fit({'input': X}, {'targets': Y},validation_set=({'input': test_x}, {'targets': test_y}),  n_epoch=experiment_epochs , snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


print(9/0)


#
#
#   compar the accuracies and choose the best
#
#



#try:
#    if os.path.exists(file_name):
     #   model = model.load(model_file = file_name)
    #    print('model loaded!')
   # else:
  #      model.fit({'input': X}, {'targets': Y},validation_set=({'input': test_x}, {'targets': test_y}),  n_epoch=3 , snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
 #       model.save(actual_dir+"/"+MODEL_NAME)
#except:


model.fit({'input': X}, {'targets': Y},validation_set=({'input': test_x}, {'targets': test_y}),  n_epoch=epochs , snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


model.save(actual_dir+"/"+MODEL_NAME)
print("the load needs fiximg to save time")

print("fing done ")



# %tensorboard --logdir C://logs//cvd --port 5567


########################### show examples below


import matplotlib.pyplot as plt


test_data = np.load(raw_data_dir+'/test_data.npy', allow_pickle= True)

np.random.shuffle(test_data)
fig=plt.figure()





for num,data in enumerate(test_data[:12]):

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(STANDARDIZED_IMAGE_SIZE,STANDARDIZED_IMAGE_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label=categories[1].lower()
    else: str_label=categories[0].lower()

    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

