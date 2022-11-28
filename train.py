# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow as tf
import argparse
import os
import pandas as pd
import numpy as np
import json


def model(x_train, y_train, epochs=1):
    """Generate a simple model"""
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
        ])  

     

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    model.fit(x_train,
        y_train,
        epochs=epochs,
        )
   

    return model


def _load_data(base_dir_train):
    """Load MNIST training and testing data"""
    # Bucket name
    data = pd.read_csv('s3://sagemaker-us-east-1-410677554255/hand_gesture_tensorflow/data/keypoint.csv',header = None)
    X_train = data.iloc[:,1:].to_numpy()
    y_train = data[0].to_numpy() 
    
    
    return X_train, y_train

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--epochs', type=int, default=1)

    return parser.parse_known_args()

# List devices available to TensorFlow
from tensorflow.python.client import device_lib

if __name__ == "__main__":
    args, unknown = _parse_args()

    print(args)
    #print('SN_MODEL_DIR: {}\n\n'.format(args.SM_MODEL_DIR))
    
    print('\n\nDEVICES\n\n')    
    print(device_lib.list_local_devices())
    print('\n\n')
    
    print('Loading Fashion MNIST data..\n')
    train_data, train_labels = _load_data(args.train)

    print('Training model for {} epochs..\n\n'.format(args.epochs))
    mnist_classifier = model(train_data, train_labels, epochs=args.epochs)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        mnist_classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')