import os, sys, time, pickle, random
import numpy as np
import tensorflow as tf
import getConfig
from cnnModel import CnnModel

g_config = getConfig.get_config( config_file="config.nin" )

def unpickle_patch( file ):
    patch_bin_file = open( file, "rb" )
    patch_dict = pickle.load( patch_bin_file, encoding="bytes" )
    return patch_dict

def read_data( dataset_path, im_dim, num_channels, num_files, images_per_file ):
    files_names = os.listdir( dataset_path )
    dataset_array = np.zeros( shape=( num_files * images_per_file, im_dim, im_dim, num_files ) )
    dataset_labels = np.zeros( shape=( num_files * images_per_file ), dtype=np.uint8 )
    index = 0
    for file_name in files_names:
        if file_name[ 0:len( file_name )-1 ] == "data_batch_":
            print( "Now processing data：", file_name )
            data_dict = unpickle_patch( dataset_path + file_name )
            images_data = data_dict[b"data"]
            print( images_data.shape )
            images_data_reshaped = np.reshape( images_data, newshape=( len(images_data), im_dim, im_dim, num_channels ) )
            dataset_array[ index * images_per_file : (index+1) * images_per_file, :, :, : ] = images_data_reshaped
            dataset_labels[ index * images_per_file : (index+1) * images_per_file ] = data_dict[ b"labels" ]
            index += 1
    return dataset_array, dataset_labels

dataset_array, dataset_labels = read_data( dataset_path=g_config["dataset_path"], im_dim=g_config["im_dim"],
                                        num_channels=g_config["num_channels"], num_files=g_config["num_files"],
                                        images_per_file=g_config["images_per_file"] )
dataset_array = dataset_array.astype( "float32" ) / 255
dataset_labels = tf.keras.utils.to_categorical( dataset_labels, 10 )

def create_model():
    if "pretrained_model" in g_config:
        model = tf.keras.models.load_model( g_config[ "pretrained_model" ] )
        return model
    ckpt = tf.io.gfile.listdir( g_config[ "working_directory" ] )
    if ckpt:
        model_file = os.path.join( g_config[ "working_directory" ], ckpt[-1] )
        print( "Reading model parameters from %s" % model_file )
        model = tf.keras.models.load_model( model_file )
        return model
    else:
        model = CnnModel( g_config["learning_rate"], g_config["dropout_rate"] )
        model = model.create_model()
        return model

def train():
    model = create_model()
    history = model.fit( dataset_array, dataset_labels, verbose=1, epochs=100, validation_split=0.2 )
    file_name = "cnn_model.h5"
    checkpoint_path = os.path.join( g_config[ "working_directory" ], file_name )
    model.save( checkpoint_path )

def predict( data ):
    chpt = os.listdir( g_config[ "working_directory" ] )
    checkpoint_path = os.path.join( g_config[ "working_directory" ], "cnn_model.h5" )
    model = tf.keras.models.load_model( checkpoint_path )
    prediction = model.predict( data )
    index = tf.math.argmax( prediction[0] ).numpy()
    return label_names_dict[ index ]

if __name__ == "__main__":
    g_config = getConfig.get_config()
    if g_config["mode"] == "train":
        train()
    elif g_config["mode"] == "server":
        print( "Please use: python app.py" )