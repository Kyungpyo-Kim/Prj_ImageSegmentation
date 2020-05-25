import tensorflow as tf
import numpy as np
import os


class Tf2ShuffleNet:
  MEAN = [73.29132098, 83.04442645, 72.5238962]

  def __init__(self, x_input, num_classes, pretrained_path, train_flag, batchnorm_enabled=True, num_groups=3,
                 weight_decay=4e-5,
                 bias=0.0):
    self.x_input = x_input
    self.train_flag = train_flag
    self.num_classes = num_classes
    self.num_groups = num_groups
    self.bias = bias
    self.wd = weight_decay
    self.batchnorm_enabled = batchnorm_enabled
    self.pretrained_path = os.path.realpath(os.getcwd()) + "/" + pretrained_path
    self.score_fr = None
    self.stage2 = None
    self.stage3 = None
    self.stage4 = None
    self.max_pool = None
    self.conv1 = None

    # These feed layers are for the decoder
    self.feed1 = None
    self.feed2 = None

    # A number stands for the num_groups
    # Output channels for conv1 layer
    self.output_channels = {'1': [144, 288, 576], '2': [200, 400, 800], '3': [240, 480, 960], '4': [272, 544, 1088],
                            '8': [384, 768, 1536], 'conv1': 24}
  
  def build(self):
    print ( 'build model')

    print ( 'preprocessing inputs')
    red, green, blue = tf.split(self.x_input, num_or_size_splits=3, axis=3)
    preprocessed_input = tf.concat([
                    tf.subtract(blue, Tf2ShuffleNet.MEAN[0]) / tf.constant(255.0),
                    tf.subtract(green, Tf2ShuffleNet.MEAN[1]) / tf.constant(255.0),
                    tf.subtract(red, Tf2ShuffleNet.MEAN[2]) / tf.constant(255.0),
                ], 3)
    print ( preprocessed_input.shape[1:] )
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=self.output_channels['conv1'],
                       kernel_size=(3,3),
                       strides=(2,2),
                       padding='valid',
                       activation='relu',
                       input_shape=list(preprocessed_input.shape[1:])
                       ),
      tf.keras.layers.BatchNormalization(),

    ])


# self.conv1 = conv2d('conv1', x=preprocessed_input, w=None, 
#                                l2_strength=self.wd, bias=self.bias,
#                                 batchnorm_enabled=self.batchnorm_enabled, is_training=self.train_flag,
#                                 activation=tf.nn.relu,)
            # _debug(self.conv1)
            # padded = tf.pad(self.conv1, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")
            # self.max_pool = max_pool_2d(padded, size=(3, 3), stride=(2, 2), name='max_pool')
            # _debug(self.max_pool)
            # self.stage2 = self.stage(self.max_pool, stage=2, repeat=3)
            # _debug(self.stage2)
            # self.stage3 = self.stage(self.stage2, stage=3, repeat=7)
            # _debug(self.stage3)
            # self.stage4 = self.stage(self.stage3, stage=4, repeat=3)
            # _debug(self.stage4)

            # self.feed1 = self.stage3
            # self.feed2 = self.stage2

            # # First Experiment is to use the regular conv2d
            # self.score_fr = conv2d('conv_1c_1x1', self.stage4, num_filters=self.num_classes, l2_strength=self.wd,
            #                        kernel_size=(1, 1))

            # print("\nEncoder ShuffleNet is built successfully\n\n")
    


if __name__ == '__main__':
  x_input = np.random.uniform(0.0 ,255.0 ,[10, 512, 1024, 3]).astype(np.float32)
  print ( x_input.shape )
  shufflenet = Tf2ShuffleNet(x_input, 20, '../pretrained_weights/weights.npy', True)
  shufflenet.build()
  
      

