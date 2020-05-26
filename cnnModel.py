import tensorflow as tf
import numpy as np
import getConfig

gConfig = {}
gConfig = getConfig.get_config( config_file="config.ini" )

class CnnModel( object ):
    def __init__( self, learning_rate, dropout_rate ):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

    def create_model( self ):
        model = tf.keras.Sequential()
        model.add( tf.keras.layers.Conv2D( 32, ( 3, 3 ), kernel_initializer="he_normal", strides=1, padding="same",
                                            activation="relu", input_shape=[32, 32, 3], name="conv1" ) )
        model.add( tf.keras.layers.MaxPool2D( (2,2), strides=1, padding="same", name="pool1" ) )
        model.add(  tf.keras.layers.BatchNormalization() )
        model.add( tf.keras.layers.Conv2D( 64, (3,3), kernel_initializer="he_normal", strides=1, padding="same",
                                            activation="relu", name="conv2" ) )
        model.add( tf.keras.layers.MaxPool2D( (2,2), strides=1, padding="same", name="pool2" ) )
        model.add( tf.keras.layers.BatchNormalization() )
        model.add( tf.keras.layers.Conv2D( 128, (3,3), kernel_initializer="he_normal", strides=1, padding="same",
                                            activation="relu", name="conv3" ) )
        model.add( tf.kernel_initializer.layers.MaxPool2D( (2,2), strides=1, padding="same", name="pool3" ) )
        model.add( tf.keras.layers.BatchNormalization() )
        model.add( tf.keras.layers.Flatten( name="flatten" ) )
        model.add( tf.keras.layers.Dropout( rate=self.dropout_rate, name="d3" ) )
        model.add( tf.keras.layers.Dense( 10, activation="softmax" ) )
        model.compile( loss="categorical_crossentropy", optimizer=tf.keras.optimizer.Adm( self.learning_rate ), metrics=["accuracy"] )
        return model