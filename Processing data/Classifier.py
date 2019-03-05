# Needed libirarires
import numpy as np
import h5py
import os
import imgaug as ia
from imgaug import augmenters as iaa

# Global variable 
data_dir = '/content/drive/My Drive/AgentHuman/SeqTrain'                                  # Data directory 

high_level = ["FOLLOW", "LEFT", "RIGHT", "STRAIGHT"]                                      # High level commands
total_counter = np.zeros(4)                                                               # Used to count up images

def write_data(imags_matrix, targets_matrix, directory, filename):
    with h5py.File(directory + filename, 'w') as hdf:
        hdf.create_dataset('rgb',data=imags_matrix)
        hdf.create_dataset('targets',data=targets_matrix)
        

def classify(output_dir, command): 
    imags_counter = 0
    file_number = 0
    for _ in range(3663, 6952, 1):
        if((_ - 3663)%500 == 0 and (_ != 3663)):
            print("Done classifying 500 files ..")
        if(_ == 3663):
            imgs_data = []
            targets_data = []
        filename = data_dir + '/data_0' + str(_) + '.h5'
        statinfo = os.stat(filename)
        if statinfo.st_size != 10584544:
            continue
        with h5py.File(filename, 'r') as hdf:
            imgs = hdf.get('rgb')
            imgs = np.array(imgs[:,:,:])
            targets = hdf.get('targets')
            targets = np.array(targets)
            for i in range (0,200):
                if(targets[i][10] < -1):
                	continue 
                img = imgs[i]
                target = targets[i]
                if(target[24] == command ):
                    imgs_data.append(img)
                    targets_data.append(target)
                    imags_counter = imags_counter + 1
                    if(imags_counter == 32):
                        output_filename = '/data_0' + str(file_number) + '.h5'
                        write_data(imgs_data, targets_data, output_dir, output_filename)

                        total_counter[int(command - 2)] = total_counter[int(command - 2)] + 32 
                        file_number = file_number + 1
                        imags_counter = 0
                        imgs_data = []
                        targets_data = []
        if(_ == 6951):
            print("Additional ", int(imags_counter), " images with the command ", high_level[int(command - 2)], " were neglected! ..")
            print("The total number of images associated with command ", high_level[int(command - 2)], " equals = ", int(total_counter[int(command - 2)]))
            print("."*50)

            imgs_data = []
            targets_data = []


# Classifying data into five categories
output_directories = ['/content/drive/My Drive/AgentHuman/SeqTrain/Follow',
                      '/content/drive/My Drive/AgentHuman/SeqTrain/Left', 
                      '/content/drive/My Drive/AgentHuman/SeqTrain/Right', 
                      '/content/drive/My Drive/AgentHuman/SeqTrain/Straight']

commands = [2.0, 3.0, 4.0, 5.0]
for i, directory in enumerate(output_directories):
    print("Classifying the ", high_level[i], " command ...")
    classify(directory, commands[i])
