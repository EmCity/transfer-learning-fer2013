# Inspired by https://github.com/isseu/emotion-recognition-neural-networks

from __future__ import division, absolute_import
import re
from data_augmentation import augment
import numpy as np
from dataset_loader import DatasetLoader
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization, local_response_normalization
from tflearn.layers.estimator import regression
from constants import *
from os.path import isfile, join
import random
import sys
import tensorflow as tf

class EmotionRecognition:

  def __init__(self):
    self.dataset = DatasetLoader()
    #self.augmentor = DataAugmentation()
  def build_network(self):
    # Smaller 'AlexNet'
    # https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
    print('[+] Building CNN')
    #tf.reset_default_graph()
    #tf.set_random_seed(343)
    np.random.seed(343)
    tf.logging.set_verbosity(tf.logging.INFO)

    #tflearn.init_graph(num_cores=4,gpu_memory_fraction=0.5)
    self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1])
    self.network = conv_2d(self.network, 64, 5, activation = 'relu')
    self.network = batch_normalization(self.network)
    #self.network = local_response_normalization(self.network)
    self.network = max_pool_2d(self.network, 3, strides = 2)
    self.network = conv_2d(self.network, 64, 5, activation = 'relu')
    self.network = batch_normalization(self.network)
    self.network = max_pool_2d(self.network, 3, strides = 2)
    self.network = conv_2d(self.network, 64, 4, activation = 'relu')
    self.network = batch_normalization(self.network)
    self.network = dropout(self.network, 0.3)
    #self.network = fully_connected(self.network, 3072, activation = 'relu')
    self.network = fully_connected(self.network, 128, activation = 'relu')
    #self.network = fully_connected(self.network, len(EMOTIONS), activation = 'softmax')
    self.network = fully_connected(self.network, 6, activation = 'softmax')
    self.network = regression(self.network,
      optimizer = 'momentum',
      loss = 'categorical_crossentropy')
    #with tf.device('/device:GPU:0'):
    tflearn.config.init_graph(log_device=True, soft_placement=True)
    with tf.device('/device:GPU:0'):
      self.model = tflearn.DNN(
        self.network,
        checkpoint_path = SAVE_DIRECTORY + '/emotion_recognition',
        max_checkpoints = 1,
        tensorboard_verbose = 2
      )
      #self.load_model()

  def load_saved_dataset(self):
    self.dataset.load_from_save()
    print('[+] Dataset found and loaded')

  def start_training(self):
    self.load_saved_dataset()
    self.build_network()
    if self.dataset is None:
      self.load_saved_dataset()
    # Training
    print('[+] Training network')
    #with tf.device('/device:GPU:0'):
    # These can be any tensors of matching type and dimensions.
    #images, labels = augment(self.dataset.images, self.dataset.labels,
    #                         horizontal_flip=True, rotate=15, crop_probability=0.8, mixup=4)
    print('train images, test images, train labels, test labels', self.dataset.images.shape, self.dataset.images_test.shape, self.dataset.labels.shape, self.dataset.labels_test.shape)
    with tf.device('/device:GPU:0'):
      self.model.fit(
        self.dataset.images, self.dataset.labels,
        validation_set = (self.dataset.images_test, self.dataset.labels_test),
        n_epoch = 50,
        batch_size = 32,
        shuffle = True,
        show_metric = True,
        snapshot_step = 200,
        snapshot_epoch = True,
        run_id = 'emotion_recognition'
      )

  def predict(self, image):
    if image is None:
      return None
    image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    return self.model.predict(image)

  def save_model(self):
    self.model.save(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
    print('[+] Model trained and saved at ' + SAVE_MODEL_FILENAME)

  def load_model(self):
    if isfile(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME)):
      self.model.load(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
      print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)


def show_usage():
  # I din't want to have more dependecies
  print('[!] Usage: python emotion_recognition.py')
  print('\t emotion_recognition.py train \t Trains and saves model with saved dataset')


if __name__ == "__main__":
  if len(sys.argv) <= 1:
    show_usage()
    exit()

  network = EmotionRecognition()
  if sys.argv[1] == 'train':
    network.start_training()
    network.save_model()

  else:
    show_usage()
