from os.path import join
import numpy as np
from constants import *
import cv2

class DatasetLoader(object):

  def __init__(self):
    pass

  def load_from_save(self):
    self._images      = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_FILENAME))
    self._labels      = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_FILENAME))
    self._images_test = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_TEST_FILENAME))
    self._labels_test = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_TEST_FILENAME))
    self._images      = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    self._images_test = self._images_test.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    #self._labels      = self._labels.reshape([-1, len(EMOTIONS)])
    #self._labels_test = self._labels_test.reshape([-1, len(EMOTIONS)])
    #self._labels      = self._labels.reshape([-1, 6])
    #self._labels_test = self._labels_test.reshape([-1, 6])
    print('Dataset shapes')
    print(self._images_test.shape)
    print(self._labels.shape)
    print(self._images.shape)
    print(self._labels_test.shape)

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def images_test(self):
    return self._images_test

  @property
  def labels_test(self):
    return self._labels_test

  @property
  def num_examples(self):
    return self._num_examples
