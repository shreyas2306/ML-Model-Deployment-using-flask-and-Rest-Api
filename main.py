from source.main.prediction.predict import ModelPrediction
from source.database.db import DataBase
from source.main.training.train import Train
from resource import config as cf
from source.Rest_Api.rest_api import PredictDomain
from flask import Flask, render_template, request
from flask_restful import abort, Api
from werkzeug.utils import secure_filename
import os
from glob import glob
import argparse


def train():

    train_obj = Train(cf.PATH_RESUME_TRAIN, cf.MODEL_PATH)
    train_obj.extract_resume_content()
    train_obj.train_model()
    train_obj.test_model()


def predict(path, model_path):

    modelprediction_obj = ModelPrediction(path, model_path)
    df = modelprediction_obj.predict() 
    return df


def rest_api():

    app = Flask(__name__)
    api = Api(app)

    api.add_resource(PredictDomain, '/')
    app.run(debug = True)


def api():

    app = Flask(__name__)

    

    @app.route("/")
    def index():
        return render_template('index.html')
    
    @app.route('/getfile', methods=['GET','POST'])
    def getfile():
        UPLOAD_FOLDER = cf.PATH_UPLOAD_FOLDER

        files = glob(UPLOAD_FOLDER + '/*')

        for file in files:
            os.remove(file)
        
        if request.method == 'POST':
            
            # for secure filenames. Read the documentation.
            file = request.files['myfile']
            filename = secure_filename(file.filename) 

            # os.path.join is used so that paths work in every operating system
            file.save(os.path.join(UPLOAD_FOLDER, filename))

            df = predict(UPLOAD_FOLDER, cf.MODEL_PATH)
            text = df['Predicted Type'][0]
            
            
       
        return render_template('index.html', prediction_text='Predicted Domain of resume is : {}'.format(text))

    app.run(debug = True)



if __name__ == '__main__':
    
    rest_api()