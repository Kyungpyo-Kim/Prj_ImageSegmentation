# Prj_ImageSegmentation
Minimal codes for ShuffleSeg with citiscape dataset

## Dependencies
* tensorboard (1.14.0)
* tensorflow-estimator (1.14.0)
* tensorflow-gpu (1.14.0)
* tensorflow-tensorboard (0.1.8)
* tensorrt (5.0.2.6)
* numpy (1.16.0)

## Dataset
dataset
--weights.npy: class balancing 을 위한 정보, 각 클래스에 대한 weight 정보가 들어있음
--X_train.npy: Training data, rgb
--Y_train.npy: Labels for training data
--X_val.npy: Validation data
--Y_val.npy: Labels for validataion data
--ynames_val.npy: Names for validation data

## ShuffleSeg
* Shufflenet 은 ImageNet 의 pretrained weights 를 사용한다.

## TODO
* tensorflow 2.0 으로 옮기기


## Warnings
2020-05-25 21:35:00.946924: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
* Tensorflow 를 다시 빌드해서 사용하기를 권장, 학습 속도 향상에 도움이 된다.

py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
Instructions for updating:
tf.py_func is deprecated in TF V2. Instead, there are two
    options available in V2.
    - tf.py_function takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    - tf.numpy_function maintains the semantics of the deprecated tf.py_func
    (it is not differentiable, and manipulates numpy arrays). It drops the
    stateful argument making all functions stateful.
    











