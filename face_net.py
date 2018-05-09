from __future__ import division, absolute_import

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras_vggface.vggface import VGGFace
from constants import *
from keras.utils import plot_model
import keras
from sklearn.metrics import *
from keras.engine import Model
from keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPool2D, BatchNormalization, Dropout, MaxPooling2D
from os.path import isfile, join
import random
import sys
import tensorflow as tf

import matplotlib.pyplot as plt
import pickle

import itertools

class EmotionRecognition:


    def build_network(self):
        # VGG16 Facenet (v1)
        print('[+] Building CNN')



        vgg_notop = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
        last_layer = vgg_notop.get_layer('pool5').output
        x = Conv2D(filters=64, kernel_size=1, activation='relu')(last_layer)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
        x = Conv2D(filters=64, kernel_size=2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc6')(x)
        x = Dense(2048, activation='relu', name='fc7')(x)
        print("Emotions count", len(EMOTIONS))

        out = Dense(6, activation='softmax', name='classifier')(x)

        custom_vgg_model = Model(vgg_notop.input, out)

        # adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

        custom_vgg_model.compile(optimizer='adam',
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])
        plot_model(custom_vgg_model, to_file='model2.png', show_shapes=True)
        self.model = custom_vgg_model

    def build_network_resnet(self):
        # Resnet Facenet (v2)

        print('[+] Building CNN')

        vgg_notop = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        last_layer = vgg_notop.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dense(1024, activation='relu', name='fc7')(x)
        print("Emotions count", len(EMOTIONS))
        l=0
        for layer in vgg_notop.layers:
            print(layer,"["+str(l)+"]")
            l=l+1
        for i in range(101):
             vgg_notop.layers[i].trainable = False

        out = Dense(6, activation='softmax', name='classifier')(x)

        custom_resnet = Model(vgg_notop.input, out)


        optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        #optim = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        custom_resnet.compile(optimizer='sgd',
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])
        plot_model(custom_resnet, to_file='model2.png', show_shapes=True)
        self.model = custom_resnet


    def plot_confusion_matrix(self,cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap='Blues'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def start_training(self):
        continue_train = False
        continue_epoch = 0
        learn = True

        if continue_train:
            self.load_model()
        else:
            self.build_network_resnet()

        if learn:
            # Training
            print('[+] Training network')

            train_datagen = ImageDataGenerator(
                rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2)
            train_generator = train_datagen.flow_from_directory(
                '/media/nicolas/BA70BA1070B9D37D/train/imgs_train_new',
                target_size=(224, 224),
                batch_size=32,
                color_mode='rgb',
                class_mode='categorical')

            validation_datagen = ImageDataGenerator(
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.3
            )
            validation_generator = validation_datagen.flow_from_directory(
                '/media/nicolas/BA70BA1070B9D37D/train/imgs_val_new',
                target_size=(224, 224),
                batch_size=32,
                color_mode='rgb',
                class_mode='categorical')

            filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
            #early_stopping = EarlyStopping(verbose=1, patience=10, monitor='val_acc',mode="max")
            plateau= keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, verbose=0, mode='max',
                                              epsilon=0.0001, cooldown=0, min_lr=0)

            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [plateau,checkpoint]

            with tf.device('/device:GPU:0'):
                history = self.model.fit_generator(
                    train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    initial_epoch=continue_epoch,
                    epochs=1,
                    shuffle=True,
                    verbose=1,
                    max_queue_size=100,
                    callbacks=callbacks_list
                    , use_multiprocessing=False
                )
                with open('trainHistoryDict', 'wb') as file_pi:
                    pickle.dump(history.history, file_pi)
                    history=history.history
        else:
            with open('trainHistoryDict', 'rb') as file_pi:
                history=pickle.load(file_pi)

        print(history.keys())

        self.visualizeLearning(history)
        self.test()




    def visualizeLearning(self, history):
        # Compute confusion matrix

        # summarize history for accuracy
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("plot1.png")

        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        # summarize history for loss
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("plot2.png")
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()



    def predict(self, image):
        if image is None:
            return None
        image = image.reshape([-1, 224, 224, 3])
        prediction = self.model.predict(image)

        return prediction



    def test(self):

        test_datagen = ImageDataGenerator(   )
        test_generator = test_datagen.flow_from_directory(
            'imgs_test_fer',
            target_size=(224, 224),
            shuffle=False,
            batch_size=32,
            color_mode='rgb',
            class_mode='categorical')
        loss = self.model.evaluate_generator(
            test_generator
        )

        prediction = self.model.predict_generator(
            test_generator
        )
        plot_model(self.model, to_file='model_test.png', show_shapes=True)
        print(self.model.metrics_names,loss)

        cnf_matrix = confusion_matrix(test_generator.classes , prediction.argmax(axis=1))
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral'],
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=['angry', 'fearful', 'happy', 'sad', 'surprised', 'neutral'], normalize=True,
                              title='Normalized confusion matrix')
        plt.savefig("confusion_matrix.png")

        plt.show()


    def load_model(self):
        if isfile("./weights-improvement-29-0.72.hdf5"):
            self.model= load_model("./weights-improvement-29-0.72.hdf5")


def show_usage():
    print('[!] Usage: python emotion_recognition.py')
    print('\t emotion_recognition.py train \t Trains and saves model with saved dataset')
    print('\t emotion_recognition.py test \t Tests the model with saved dataset')


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        show_usage()
        exit()

    network = EmotionRecognition()
    if sys.argv[1] == 'train':
        network.start_training()
    elif sys.argv[1] == 'test':
        network.load_model()
        network.test()
    else:
        show_usage()
