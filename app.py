import os, sys, time, pickle
import flask, werkzeug, requests
from flask import request, jsonify
import numpy as np
from PIL import Image
import execute, getConfig

g_config = getConfig.get_config( config_file="config.ini" )
app = flask.Flask( "TF2_ImageClassifierWeb" )

def cnn_predict():
    file = g_config[ "dataset_path" ] + "batches.meta"
    patch_bin_file = open( file, "rb" )
    label_names_dict = pickle.load( patch_bin_file )[ "label_names" ]
    global secure_filename
    img = Image.open( os.path.join( app.root_path, secure_filename ) )
    r, g, b = img.split()
    r_arr = np.array( r )
    g_arr = np.array( g )
    b_arr = np.array( b )
    image = img.reshape( [1, 32, 32, 3] ) / 255
    predicted_class = execute.predict( image )
    return flask.render_template( template_name_or_list="prediction_result.html",
                                    predicted_class = predicted_class )

app.add_url_rule( rule="/predict/", endpoint="predict", view_func=cnn_predict )
def upload_image():
    global secure_filename
    if flask.request.method == "POST":
        img_file = flask.request.files[ "image_file" ]
        secure_filename = werkzeug.secure_filename( img_file.filename )
        img_path = os.path.join( app.root_path, secure_filename )
        img_file.save( img_path )
        print( "Images upload success." )
        return flask.redirect( flask.url_for( endpoint="predict" ) )
    return "Images upload failed."

app.add_url_rule( rule="/upload/", endpoint="upload", view_func=upload_image, methods=["POST"] )
def redirect_upload():
    return flask.render_template( template_name_or_list="upload_image.html" )

app.add_url_rule( rule="/", endpoint="homepage", view_func=redirect_upload )

if __name__ == "__main__":
    app.run( host="0.0.0.0", port=7777, debug=False )
    pass
