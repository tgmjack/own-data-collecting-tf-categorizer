from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import tflearn



def make_model(model_type , STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME,LR):

    tf.compat.v1.reset_default_graph()

    convnet = input_data(shape=[None,STANDARDIZED_IMAGE_SIZE,STANDARDIZED_IMAGE_SIZE, 1], name='input')

    if model_type == 1:
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
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = fully_connected(convnet, 1024, activation='relu')

    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(convnet, tensorboard_dir=actual_dir, tensorboard_verbose=3)
    model.save(actual_dir+"/"+MODEL_NAME)
    return model

def find_model_accuracy(model , test_features, test_labels):
    test_results = {}
    test_results['model'] = model.evaluate(test_features, test_labels)
    print(f" Accuracy: {test_results}")
    return float(test_results['model'][0]);


def get_best_model(STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME , test_x, test_y, X, Y , experiment_epochs  , LR):
    # try a few models with and do a brief test to find quickest learner
    model_1 = make_model(1 , STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME , LR)
    model_1.fit({'input': X}, {'targets': Y},validation_set=({'input': test_x}, {'targets': test_y}),  n_epoch=experiment_epochs , snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    acc1 = find_model_accuracy(model_1 , test_x, test_y)

    model_2 = make_model(2 , STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME , LR)
    model_2.fit({'input': X}, {'targets': Y},validation_set=({'input': test_x}, {'targets': test_y}),  n_epoch=experiment_epochs , snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    acc2 = find_model_accuracy(model_2 , test_x, test_y)

    model_3 = make_model(3 , STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME , LR)
    model_3.fit({'input': X}, {'targets': Y},validation_set=({'input': test_x}, {'targets': test_y}),  n_epoch=experiment_epochs , snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    acc3 = find_model_accuracy(model_3 , test_x, test_y)


    print(acc1)
    print(acc2)
    print(acc3)

    ### choose best model
    if acc1 > acc2  and acc1 > acc3:
        model_1 = make_model( 1  , STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME , LR)
        model = model_1
        best_model_num = 1
        print("         1               ")
    elif acc2 > acc1  and acc2 > acc3:
        model_2 = make_model(2 , STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME , LR)
        model = model_2
        best_model_num = 2
        print("         2               ")
    elif acc3 > acc1  and acc3 > acc2:
        model_3 = make_model(3 , STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME , LR)
        model = model_3
        best_model_num = 3
        print("         3               ")
    elif acc3 == acc1:
        model_1 = make_model( 1  , STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME , LR)
        model = model_1
        best_model_num = 1
    elif acc3 == acc2:
        model_3 = make_model(3 , STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME,LR)
        model = model_3
        best_model_num = 3
    elif acc1 == acc2:
        model_1 = make_model( 1  , STANDARDIZED_IMAGE_SIZE , actual_dir , MODEL_NAME,LR)
        model = model_1
        best_model_num = 1
    else:
        raise Exception("trouble finding best accuracy out of =   "+str(acc1)+" , "+str(acc2)+" , "+str(acc3))
    return model