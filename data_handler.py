import numpy as np 
import cv2
from tqdm import tqdm
import os

def proccess_and_seperate_data(categories , raw_data_dir , STANDARDIZED_IMAGE_SIZE):
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
                fail_counter+= 1

    A = int(len(proccessed_data)-1)
    b = int(A * 0.8)
    c = int(A * 0.2)

    np.random.shuffle(proccessed_data)
    train = proccessed_data[:b]
    test = proccessed_data[b:A]
    np.random.shuffle(test)
    np.random.shuffle(train)
    np.save(raw_data_dir+'/test_data.npy', test)
    np.save(raw_data_dir+'/train_data.npy', train)

    return train,test;

def seperate_data_into_x_and_y(train,test,STANDARDIZED_IMAGE_SIZE):
    X = np.array([i[0] for i in train]).reshape(-1,STANDARDIZED_IMAGE_SIZE,STANDARDIZED_IMAGE_SIZE,1)
    Y = np.array([i[1] for i in train])

    test_x = np.array([i[0] for i in test]).reshape(-1,STANDARDIZED_IMAGE_SIZE,STANDARDIZED_IMAGE_SIZE,1)
    test_y = np.array([i[1] for i in test])
    return X,Y,test_x,test_y