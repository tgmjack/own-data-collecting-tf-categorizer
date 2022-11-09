# fully-automated-cnn


this system takes in 2 keywords in a list eg) ["cow", "pig" ] or ["spanners" , "hammers"]
it then searches google images for lots of images of these things. 
a convolutional neural network then learns the difference between these 2 objects

a series of cnn's differeing in convolutedness are tried on the data , the best of these models is then reused for more intense training. I can get > 95% accuracy with this system. 
