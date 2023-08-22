import matplotlib.pyplot as plt
import numpy as np

def display_a_test(raw_data_dir, STANDARDIZED_IMAGE_SIZE , model , categories):

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

def show_image(im):
    plt.imshow(im, cmap='gray')
    plt.show() 