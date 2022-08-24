# fully-automated-cnn


this system takes in 2 keywords in a list eg) ["cow", "pig" ] or ["spanners" , "hammers"]
it then searches google images for lots of images of these things. 
a convolutional neural network then learns the difference between these 2 objects


next step is vary the convolutedness of the cnn , right now ["cow", "pig" ] can easy get 95% accuracy but ["spanners" , "hammers"] gives ~65% , probably because  the resolution makes detecting the needle like structre of the screwdriver becomes difficult or alternativlet the angular sicle like structre at the top of the spanner isnt being broken into correctly sized chunks that coherently describe the wrench part.

i think i might try a test epoch to see how that improves accuracy, 

if test epoch not good enouugh 
  change things 
  else 
    run propper test
