import numpy as np
import tensorflow as tf
if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
    tf = tf.compat.v1

# Compatibility shim for environments with TensorFlow 2.16+ where tf.estimator was removed
if not hasattr(tf, 'estimator') or not hasattr(getattr(tf, 'estimator', None), 'DNNClassifier'):
    class _EstimatorInputs:
        @staticmethod
        def numpy_input_fn(x, y=None, batch_size=128, num_epochs=None, shuffle=True):
            def input_fn():
                feat_data = list(x.values())[0] if isinstance(x, dict) else x
                return feat_data, y, batch_size, num_epochs, shuffle
            return input_fn

    class _DNNClassifier:
        def __init__(self, feature_columns=None, hidden_units=None, n_classes=2, activation_fn=None, optimizer=None):
            self.hidden_units = hidden_units or [100, 100, 100]
            self.n_classes = n_classes
            self.activation_fn = activation_fn if activation_fn is not None else tf.nn.relu
            self.optimizer = optimizer
            self.model = None

        def _build_model(self, in_dim):
            layers = [tf.keras.layers.Input(shape=(in_dim,))]
            for h in self.hidden_units:
                layers.append(tf.keras.layers.Dense(h, activation=self.activation_fn))
            layers.append(tf.keras.layers.Dense(self.n_classes, activation='softmax'))
            self.model = tf.keras.Sequential(layers)
            
            opt = self.optimizer
            if opt is None:
                opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
            elif hasattr(opt, '_learning_rate'):
                lr = float(getattr(opt, '_learning_rate', 0.01))
                mom = float(getattr(opt, '_momentum', 0.9))
                opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=mom)
            elif not isinstance(opt, (str, tf.keras.optimizers.Optimizer)):
                opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
                
            self.model.compile(
                optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )

        def train(self, input_fn, steps=None):
            feat, y, batch_size, num_epochs, shuffle = input_fn()
            if self.model is None:
                self._build_model(feat.shape[1])
            
            n_samples = len(feat)
            if steps is not None:
                epochs = max(1, int(round(steps * batch_size / n_samples)))
            else:
                epochs = num_epochs or 1
            
            self.model.fit(
                feat, y,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=shuffle,
                verbose=2
            )

        def predict(self, input_fn):
            feat, _, _, _, _ = input_fn()
            preds = self.model.predict(feat, batch_size=256, verbose=0)
            class_ids = np.argmax(preds, axis=-1)
            for c in class_ids:
                yield {'classes': [np.array(str(c).encode('utf-8'))]}

    class _EstimatorModule:
        DNNClassifier = _DNNClassifier
        inputs = _EstimatorInputs()

    tf.estimator = _EstimatorModule()

import matplotlib.pyplot as plt
from util import * 
from dnn_np import test
import pdb


def bat_classification():
    # Load data from file
    # Make sure that bat.dat is in data/
    train_x, train_y, test_x, test_y = get_bat_data()
    train_x, _, test_x = normalize(train_x, train_x, test_x)    

    test_y  = test_y.flatten().astype(np.int32)
    train_y = train_y.flatten().astype(np.int32)
    num_class = (np.unique(train_y)).shape[0]
 
    # DNN parameters
    hidden_layers = [100, 100, 100]
    learning_rate = 0.005
    batch_size = 200
    steps = 12000
   
    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[train_x.shape[1]])]

    # Available activition functions
    # https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_
    # tf.nn.relu
    # tf.nn.elu
    # tf.nn.sigmoid
    # tf.nn.tanh
    activation = tf.nn.relu
    
    # [TODO 1.7] Create a neural network and train it using estimator

    # Some available gradient descent optimization algorithms
    # https://www.tensorflow.org/api_guides/python/train#Optimizers
    # tf.train.GradientDescentOptimizer
    # tf.train.AdadeltaOptimizer
    # tf.train.AdagradOptimizer
    # tf.train.AdagradDAOptimizer
    # tf.train.MomentumOptimizer
    # tf.train.AdamOptimizer
    # tf.train.FtrlOptimizer
    # tf.train.ProximalGradientDescentOptimizer
    # tf.train.ProximalAdagradOptimizer
    # tf.train.RMSPropOptimizer
    # Create optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    
    # build a deep neural network
    # https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_layers,
        n_classes=num_class,
        activation_fn=activation,
        optimizer=optimizer
    )
    
    # Define the training inputs
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )
    
    # Train model.
    classifier.train(
        input_fn=train_input_fn,
        steps=steps)
    
    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={"x": test_x},
                                    y=test_y,
                                    num_epochs=1,
                                    shuffle=False)
    
    # Evaluate accuracy. 
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_x},
      num_epochs=1,
      shuffle=False)
    y_hat = classifier.predict(input_fn=predict_input_fn)
    y_hat = list(y_hat)
    y_hat = np.asarray([int(x['classes'][0]) for x in y_hat]) 
    test(y_hat, test_y)


def mnist_classification():
    # Load data from file
    # Make sure that fashion-mnist/*.gz is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data(1)
    train_x, val_x, test_x = normalize(train_x, train_x, test_x)    

    train_y = train_y.flatten().astype(np.int32)
    val_y = val_y.flatten().astype(np.int32)
    test_y = test_y.flatten().astype(np.int32)
    num_class = (np.unique(train_y)).shape[0]

    # DNN parameters
    hidden_layers = [100, 100, 100]
    learning_rate = 0.001
    batch_size = 200
    steps = 15000
   
    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[train_x.shape[1]])]

    # Choose activation function
    activation = tf.nn.relu
    
    # Some available gradient descent optimization algorithms 
    # TODO: [YC1.7] Create optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    
    # build a deep neural network
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_layers,
        n_classes=num_class,
        activation_fn=activation,
        optimizer=optimizer
    )
    
    # Define the training inputs
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )
    
    # Train model.
    classifier.train(
        input_fn=train_input_fn,
        steps=steps)
    
    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={"x": test_x},
                                    y=test_y,
                                    num_epochs=1,
                                    shuffle=False)
    
    # Evaluate accuracy. 
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_x},
      num_epochs=1,
      shuffle=False)
    y_hat = classifier.predict(input_fn=predict_input_fn)
    y_hat = list(y_hat)
    y_hat = np.asarray([int(x['classes'][0]) for x in y_hat]) 
    test(y_hat, test_y)


problem1 = bat_classification
problem2 = mnist_classification


if __name__ == '__main__':
    np.random.seed(2026) 

    plt.ion()
    bat_classification()
    mnist_classification()
