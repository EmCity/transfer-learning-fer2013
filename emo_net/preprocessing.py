import os
import numpy as np
from PIL import Image
import pandas as pd
import cv2
from constants import *

def read_fer():
    data = pd.read_csv(FER_PATH)
    train_data = data.pixels.loc[(data.Usage == 'Training') & (data.emotion != 1)]
    test_data = data.pixels.loc[(data.Usage == 'PrivateTest') & (data.emotion != 1)]
    train_label = np.array(data.emotion.loc[(data.Usage == 'Training') & (data.emotion != 1)])
    test_label = np.array(data.emotion.loc[(data.Usage == 'PrivateTest') & (data.emotion != 1)])
    
    # parse training set
    train_img = []
    for i, data in enumerate(train_data):
        img = np.reshape(np.array([int(item) for item in data.split(" ")]), (48, 48)).astype('uint8')
        img = Image.fromarray(img).resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img)
        train_img.append(np.reshape(img, (1, IMAGE_SIZE * IMAGE_SIZE)))
    train_img = np.concatenate(train_img, axis=0)
    # parse testing set
    test_img = []
    for i, data in enumerate(test_data):
        img = np.reshape(np.array([int(item) for item in data.split(" ")]), (48, 48)).astype('uint8')
        img = Image.fromarray(img).resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img)
        test_img.append(np.reshape(img, (1, IMAGE_SIZE * IMAGE_SIZE)))
    test_img = np.concatenate(test_img, axis=0)
    
    # filter emotion: disgust
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    for i in range(train_label.shape[0]):
        if train_label[i] > 1:
            train_label[i] -= 1

    for i in range(test_label.shape[0]):
        if test_label[i] > 1:
            test_label[i] -= 1
    
    # flip train
    train_flip = flip(train_img)
    train_img = np.concatenate([train_img, train_flip])
    train_label = np.concatenate([train_label, train_label])
    
    # normalize
    train_img = (train_img - np.min(train_img)) / (np.max(train_img) - np.min(train_img))
    test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))
    
    return train_img, test_img, train_label, test_label

def flip(data):
    temp = np.zeros(data.shape)
    for i in range(data.shape[0]):
        img = Image.fromarray(np.reshape(data[i, :], (int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1])))))
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        temp[i, :] = np.reshape(np.array(img), (1, data.shape[1]))
    return temp

def read_jaffe():
    filelist = os.listdir(JAFFE_PATH)
    face_cascade = cv2.CascadeClassifier(CAS_PATH)
    jaffe_data = []
    jaffe_label = []
    emo_dict = {
        'AN': 0,
        'FE': 1,
        'HA': 2,
        'SA': 3,
        'SU': 4,
        'NE': 5
    }
    for file in filelist:
        emotion = file.split(".")[1][:-1]
        if emotion != 'DI':
            img = np.array(Image.open(JAFFE_PATH + file).convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            # if face is detected
            if len(faces) > 0:
                # chop the image
                img = img[faces[0][1]:(faces[0][1] + faces[0][2]), faces[0][0]:(faces[0][0] + faces[0][3])]
            img = Image.fromarray(np.array(img))
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = np.reshape(np.array(img), (1, IMAGE_SIZE * IMAGE_SIZE))
            jaffe_data.append(img)
            jaffe_label.append(emo_dict[emotion])

    jaffe_data = np.concatenate(jaffe_data, axis=0)
    jaffe_label = np.array(jaffe_label)
    
    # data augmentation
    jaffe_flip = flip(jaffe_data)
    jaffe_data = np.concatenate([jaffe_data, jaffe_flip], axis=0)
    jaffe_label = np.concatenate([jaffe_label, jaffe_label], axis=0)
    
    # normalize
    jaffe_data = (jaffe_data - np.min(jaffe_data)) / (np.max(jaffe_data) - np.min(jaffe_data))
    
    # split train and test
    jaffe_data_train = jaffe_data[:300]
    jaffe_data_test = jaffe_data[300:]
    jaffe_label_train = jaffe_label[:300]
    jaffe_label_test = jaffe_label[300:]
    
    return jaffe_data_train, jaffe_data_test, jaffe_label_train, jaffe_label_test

def read_ck():
    CK_IMAGE_PATH = CK_PATH + "cohn-kanade-images/"
    CK_LABEL_PATH = CK_PATH + "Emotion/"
    
    # get emotion label filenames
    filenames = []
    dir_1 = os.listdir(CK_LABEL_PATH)
    for file_1 in dir_1:
        dir_2 = os.listdir(CK_LABEL_PATH + file_1)
        for file_2 in dir_2:
            dir_3 = os.listdir(CK_LABEL_PATH + file_1 + '/' + file_2)
            if dir_3:
                filenames.append(dir_3[0])

    # parse images
    temp_images = []
    exp_filename = []
    face_cascade = cv2.CascadeClassifier(CAS_PATH)
    for filename in filenames:
        names = filename.split("_")
        temp_name = '_'.join(names[:-1])
        img = np.array(Image.open(CK_IMAGE_PATH + names[0] + '/' + names[1] + '/' + temp_name + '.png').convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        if len(faces) > 0:
            img = img[faces[0][1]:(faces[0][1] + faces[0][2]), faces[0][0]:(faces[0][0] + faces[0][3])]
            img = Image.fromarray(img)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = np.reshape(np.array(img), (1, IMAGE_SIZE * IMAGE_SIZE))
            temp_images.append(img)
        else:
            exp_filename.append(filename)
    temp_images = np.concatenate(temp_images)
    
    # parse labels
    temp_labels = []
    for filename in filenames:
        if filename not in exp_filename:
            names = filename.split("_")
            with open(CK_LABEL_PATH + names[0] + '/' + names[1] + '/' + filename) as file:
                temp_labels.append(int(file.read()[3]))
               
    # filter out contempt and disgust
    # 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
    # convert to 0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral
    em_map = {
        0: 5,
        1: 0,
        4: 1,
        5: 2,
        6: 3,
        7: 4
    }
    images = []
    labels = []
    for i in range(len(temp_labels)):
        if temp_labels[i] != 2 and temp_labels[i] != 3:
            images.append(np.reshape(temp_images[i], (1, IMAGE_SIZE * IMAGE_SIZE)))
            labels.append(em_map[temp_labels[i]])
    images = np.concatenate(images)
    labels = np.array(labels)
    
    # flip the images
    images = np.concatenate([images, flip(images)])
    labels = np.concatenate([labels, labels])
    
    # normalization
    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    
    # split train and test
    train_images = images[:400]
    test_images = images[400:]
    train_labels = labels[:400]
    test_labels = labels[400:]
    
    return train_images, test_images, train_labels, test_labels
