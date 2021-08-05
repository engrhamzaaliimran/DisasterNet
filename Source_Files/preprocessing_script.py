import os
import numpy as np
import shutil

from skimage import data, color, exposure, io
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from skimage.transform import rescale, resize



#Following function creates directories (test, vali,train) for each class
def diractory_creation(root_dir = './Cyclone_Wildfire_Flood_Earthquake_Database', class_name='Cyclone'):
    os.makedirs(root_dir +'/train/'+class_name)
    os.makedirs(root_dir +'/val/'+class_name)
    os.makedirs(root_dir +'/test/'+class_name)

#Following function divides data into (test, vali,train) for each class
def create_partition(root_dir='Cyclone_Wildfire_Flood_Earthquake_Database',currentCls = 'Cyclone',final_dir='.'):
    src = root_dir +"/"+ currentCls # Folder to copy images from
    #print (src)
    allFileNames = os.listdir(src)
    #print (allFileNames)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])
    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

    print('Total images=> ', len(allFileNames))
    print('Training=> ', len(train_FileNames))
    print('Validation=> ', len(val_FileNames))
    print('Testing=> ', len(test_FileNames))
    # Copy-pasting images
    print ('for class ', currentCls)
    unique = 0
    for name in train_FileNames:
        new_name=final_dir+"/train/"+currentCls+"/"+str(unique)+".PNG"
        unique=unique+1
        img = io.imread(name, as_gray=False)
        roi = resize(img,(100,100),anti_aliasing=True)
        io.imsave(new_name, roi)
    unique=0
    for name in val_FileNames:
        new_name=final_dir+"/val/"+currentCls+"/"+str(unique)+".PNG"
        unique=unique+1
        img = io.imread(name, as_gray=False)
        roi = resize(img,(100,100),anti_aliasing=True)
        io.imsave(new_name, roi)
    unique=0
    for name in test_FileNames:
        new_name=final_dir+"/test/"+currentCls+"/"+str(unique)+".PNG"
        unique=unique+1
        img = io.imread(name, as_gray=False)
        roi = resize(img,(100,100),anti_aliasing=True)
        io.imsave(new_name, roi)

diractory_creation('.','Cyclone')
diractory_creation('.','Earthquake')
diractory_creation('.','Flood')
diractory_creation('.','Wildfire')
create_partition('Cyclone_Wildfire_Flood_Earthquake_Database','Cyclone',".")
create_partition('Cyclone_Wildfire_Flood_Earthquake_Database','Earthquake',".")
create_partition('Cyclone_Wildfire_Flood_Earthquake_Database','Flood',".")
create_partition('Cyclone_Wildfire_Flood_Earthquake_Database','Wildfire',".")