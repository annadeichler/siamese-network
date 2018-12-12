
'''convert input images to tfrecords and regress angle between training pair images using Siamese network
    test dataset is MNIST
    solution  based on RotNet

'''
import os
import copy
import time
import cv2

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.layers import merge
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.objectives import categorical_crossentropy
from keras.losses import mean_squared_error
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
from keras import callbacks as cbks
from keras import optimizers, objectives
from keras import metrics as metrics_module

from keras.datasets import mnist

if K.backend() != 'tensorflow':
    raise RuntimeError('only run with the TensorFlow backend ')


tf.reset_default_graph()
logs_path = "/tmp/mnist/2"



def _distance(x0_in, x1_in):
    x0_out= base_network(x0_in)
    x1_out = base_network(x1_in)   
    L1_distance = lambda x: K.abs(x[0]-x[1])
    distance = merge([x0_out,x1_out], mode = L1_distance, output_shape=lambda x: x[0])
    return distance


def crop_center(img, width, height):

    img_size = (img.shape[1], img.shape[0])
    img_center = (int(img_size[0] * 0.5), int(img_size[1] * 0.5))

    if(width > img_size[0]):
        width = img_size[0]

    if(height > img_size[1]):
        height = img_size[1]

    x1 = int(img_center[0] - width * 0.5)
    x2 = int(img_center[0] + width * 0.5)
    y1 = int(img_center[1] - height * 0.5)
    y2 = int(img_center[1] + height * 0.5)

    return img[y1:y2, x1:x2]

def rotate(img, angle):
    """
    rotate image around it's centre by the given angle, 
    returned img will have the same size as the new img.
    """
    img_size = (img.shape[1], img.shape[0])
    img_center = tuple(np.array(img_size) / 2 - 0.5)
    rot_mat = np.vstack([cv2.getRotationMatrix2D(img_center, angle, 1.0), [0, 0, 1]]
    )
    trans_mat = np.matrix([
        [1, 0, int(img_size[0] * 0.5 - img_center[1])],
        [0, 1, int(img_size[0] * 0.5 - img_center[1])],
        [0, 0, 1]
    ])

    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(
        img,
        affine_mat,
        img_size,
        flags=cv2.INTER_LINEAR
    )

    return result


def imgs_to_tfrecord(imgs, labels, filename):
    """
    convert images to tfrecord format for efficiency

    """
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    """ Save data into TFRecord """
    if not os.path.isfile(filename):
        num_examples = imgs.shape[0]
        
        rows = imgs.shape[1]
        cols = imgs.shape[2]
        depth = imgs.shape[3]
        
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            img_0 = imgs[index].tostring()
            angle = np.random.randint(360)
            angle = np.expand_dims(angle, axis=1)
            img_crop = crop_around_center(imgs[index], rows, cols)
            img_rot = rotate(img_crop, angle)
            img_1 = img_rot.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width':  _int64_feature(cols),
                'depth':  _int64_feature(depth),
                'angle':  _int64_feature(angle),
                'label': _int64_feature(int(labels[index])),
                'img_0': _bytes_feature(img_0),
                'img_1': _bytes_feature(img_1)}))
            writer.write(example.SerializeToString())
        writer.close()
    else:
        print('tfrecord %s already exists' % filename)


def read_and_decode_recordinput(tf_glob, one_hot=True, classes=None, is_train=None,
                                batch_shape=[1000, 28, 28, 1], parallelism=1):

    """ Return tensor to read from TFRecord """

    print 'Creating graph for loading %s TFRecords...' % tf_glob
    with tf.variable_scope("TFRecords"):
        record_input = data_flow_ops.RecordInput(
            tf_glob, batch_size=batch_shape[0], parallelism=parallelism)
        records_op = record_input.get_yield_op()
        records_op = tf.split(records_op, batch_shape[0], 0)
        records_op = [tf.reshape(record, []) for record in records_op]
        progbar = Progbar(len(records_op))

        imgs0 = []
        imgs1 = []
        labels  = []
        angles  = []
        for i, serialized_example in enumerate(records_op):
            progbar.update(i)
            with tf.variable_scope("parse_imgs", reuse=True):
                features = tf.parse_single_example(
                    serialized_example,
                    features={
                        'label': tf.FixedLenFeature([], tf.int64),
                        'angle': tf.FixedLenFeature([], tf.int64),
                        'img_0': tf.FixedLenFeature([], tf.string),
                        'img_1': tf.FixedLenFeature([], tf.string),

                    })
                img0 = tf.decode_raw(features['img_0'], tf.uint8)
                img0.set_shape(batch_shape[1] * batch_shape[2])
                img0 = tf.reshape(img0, [1] + batch_shape[1:])
                img0 = tf.cast(img0, tf.float32) * (1. / 255) - 0.5

                img1 = tf.decode_raw(features['img_1'], tf.uint8)
                img1.set_shape(batch_shape[1] * batch_shape[2])
                img1 = tf.reshape(img1, [1] + batch_shape[1:])
                img1 = tf.cast(img1, tf.float32) * (1. / 255) - 0.5
                
                # label = tf.cast(features['label'], tf.float32)
                label = tf.cast(features['label'],  tf.int32)
                if one_hot and classes:
                    label = tf.one_hot(label, classes)
                angle = tf.cast(features['angle'], tf.float32)

                imgs0.append(img0)
                imgs1.append(img1)
                labels.append(label)
                angles.append(angle)

        imgs0 = tf.parallel_stack(imgs0, 0)
        imgs0 = tf.cast(imgs0, tf.float32)
        imgs0 = tf.reshape(imgs0, shape=batch_shape)

        imgs1 = tf.parallel_stack(imgs1, 0)
        imgs1 = tf.cast(imgs1, tf.float32)
        imgs1 = tf.reshape(imgs1, shape=batch_shape)

        labels = tf.parallel_stack(labels, 0)

        angles = tf.parallel_stack(angles, 0)
        angles = tf.expand_dims(angles, -1)

        # staging area  will store tensors across multiple steps to speed up execution
        imgs0_shape = imgs0.get_shape()
        imgs1_shape = imgs1.get_shape()
        labels_shape  = labels.get_shape()
        angles_shape  = angles.get_shape()
        copy_stage = data_flow_ops.StagingArea(
            [tf.float32, tf.float32, tf.float32, tf.float32],
            shapes=[imgs0_shape, imgs1_shape, labels_shape, angles_shape])
        copy_stage_op = copy_stage.put(
            [imgs0, imgs1, labels, angles])
        staged_imgs0,staged_imgs1, staged_labels, staged_angles = copy_stage.get()

        return imgs0, imgs1, labels, angles


def save_mnist_as_tfrecord():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    imgs_to_tfrecord(imgs=X_train, labels=y_train, filename='train.mnist.tfrecord')
    imgs_to_tfrecord(imgs=X_test, labels=y_test, filename='test.mnist.tfrecord')


def cnn_layers(x_train_input):
    x = Conv2D(32, (3, 3), activation='relu', padding='valid')(x_train_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_train_out = Dense(classes,
                        activation='softmax')(x)
    return x_train_out

def get_convnet(input_dim):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_dim, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add( Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add( MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1,activation='softmax'))
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',help = 'path to save images')
    args = parser.parse_args()  
    return args

def main(**kwargs):
    sess = tf.Session()
    img_path = int(args.image_path)
    K.set_session(sess)

    save_mnist_as_tfrecord()
    batch_size = 32 
    batch_shape = [batch_size, 28, 28, 1]
    epochs = 100
    classes = 1
    parallelism = 10

    x0_train_batch, x1_train_batch, y_train_batch, angles_train_batch = read_and_decode_recordinput(
        str(img_path) + 'train.mnist.tfrecord',
        one_hot=True,
        classes=classes,
        is_train=True,
        batch_shape=batch_shape,
        parallelism=parallelism)

    x0_test_batch, x1_test_batch, y_test_batch, angles_test_batch = read_and_decode_recordinput(
        str(img_path) + 'test.mnist.tfrecord',
        one_hot=True,
        classes=classes,
        is_train=True,
        batch_shape=batch_shape,
        parallelism=parallelism)


    x0_batch_shape = x0_train_batch.get_shape().as_list()
    x1_batch_shape = x1_train_batch.get_shape().as_list()
    y_batch_shape = y_train_batch.get_shape().as_list()

    x0_train_input = Input(tensor=x0_train_batch, batch_shape=x0_batch_shape)
    x1_train_input = Input(tensor=x1_train_batch, batch_shape=x1_batch_shape)

    input_shape = (28, 28, 1)
    base_network = get_convnet(input_shape)
    x0_train_out= base_network(x0_train_input)
    x1_train_out = base_network(x1_train_input)   

    print x0_train_input.get_shape().as_list()

    inputs = [x0_train_input, x1_train_input]

    distance = _distance(x0_train_input, x1_train_input)
    print distance.get_shape().as_list()
    mse = mean_squared_error(angles_train_batch, distance)
    train_model = Model(inputs=[x0_train_input, x1_train_input], outputs=_distance(x0_train_input, x1_train_input))
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    train_model.add_loss(mse)
    train_model.compile(optimizer='rmsprop',
                        loss=None,
                        metrics=['accuracy'])   
    train_model.summary()


    initial_epoch = 0


    train_model.fit(batch_size=batch_size,
                    epochs=epochs, verbose = True)  # callbacks=[tensorboard])


if __name__ == '__main__':
    args = parse_args()
    kwargs = vars(args)
    main(**kwargs)

