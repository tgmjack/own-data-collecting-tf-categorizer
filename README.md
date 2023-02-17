# fully-automated-cnn


this system takes in 2 keywords in a list eg) ["cow", "pig" ] or ["spanners" , "hammers"]
it then searches google images for lots of images of these things. 
a categorizer is trained up (using tensorflow) using that data collected

a series of cnn's differeing in convolutedness are tried on the data , the best of these models is then reused for more intense training. I can get > 95% accuracy with this system. 
