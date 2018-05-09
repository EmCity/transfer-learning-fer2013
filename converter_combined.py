import scipy.misc
from dataset_loader import DatasetLoader
import numpy as np
dataset = DatasetLoader()
dataset.load_from_save()
counter = [0,0,0,0,0,0]

#Converts npy to folder based structure for Keras ImageDataGenerator

for i,img in enumerate(dataset.images_test):
    x=dataset.labels_test[i]
    print(x)
    scipy.misc.toimage(scipy.misc.imresize(np.reshape(img,(128,128)),(224,224)), cmin=0.0).save('./new_data/imgs_test_new/'+str(x)+'/image'+ str("%05d" %counter[x])+'.jpg')
    counter[x] = counter[x] + 1

counter = [0,0,0,0,0,0]
counter_val = [0,0,0,0,0,0]
for i,img in enumerate(dataset.images):
     if(i%10 != 0):
         x=dataset.labels[i]
         scipy.misc.toimage(scipy.misc.imresize(np.reshape(img,(128,128)),(224,224)), cmin=0.0).save('./new_data/imgs_train_new/'+str(x)+'/image'+ str("%05d" %counter[x])+'.jpg')
         counter[x] = counter[x] + 1
     else:
         x=dataset.labels[i]
         scipy.misc.toimage(scipy.misc.imresize(np.reshape(img,(128,128)),(224,224)), cmin=0.0).save('./new_data/imgs_val_new/'+str(x)+'/image'+ str("%05d" %counter_val[x])+'.jpg')
         counter_val[x] = counter_val[x] + 1
