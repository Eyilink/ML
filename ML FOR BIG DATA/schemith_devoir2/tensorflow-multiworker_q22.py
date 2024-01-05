import tensorflow as tf
import numpy as np

import argparse

from tensorflow.python.distribute.cluster_resolver.slurm_cluster_resolver import expand_hostlist
from tensorflow.train import ClusterSpec

import os


parser = argparse.ArgumentParser(description='cifar10 classification models, tensorflow MultiWorkerMirrored test')
parser.add_argument('--lr', default=0.001, help='')
parser.add_argument('--batch_size', type=int, default=256, help='')

args = parser.parse_args()

host_list = expand_hostlist(os.environ['SLURM_NODELIST'])

port = 3456

host_list = [f"{host}:{port}" for host in host_list]

host_dict = {'worker': host_list}

cluster_config = tf.distribute.cluster_resolver.SimpleClusterResolver(ClusterSpec(host_dict),
                                                                                task_type="worker",
                                                                                task_id=int(os.environ['SLURM_PROCID']),
                                                                                rpc_layer='grpc')
comm_options = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.RING)

strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_config, communication_options=comm_options)

with strategy.scope():

    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3),padding='same'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10,activation='softmax'),
  ])
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
         optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr),metrics=['accuracy'])

### This next line will attempt to download the CIFAR10 dataset from the internet if you don't already have it stored in ~/.keras/datasets. 
### Run this line on a login node prior to submitting your job, or manually download the data from 
### https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz, rename to "cifar-10-batches-py.tar.gz" and place it under ~/.keras/datasets

(x_train, y_train),_ = tf.keras.datasets.cifar10.load_data()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)

EPOCHS = 50

from time import time

t0 = time()
model.fit(dataset, epochs=EPOCHS)
tt = time() - t0
print("classifier trained in {} seconds".format(round(tt,3)))
