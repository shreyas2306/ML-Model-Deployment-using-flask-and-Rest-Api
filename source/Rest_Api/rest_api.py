from flask_restful import reqparse, abort, Resource
from source.main.prediction.predict import ModelPrediction
from resource import config as cf

parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictDomain(Resource):

    def get(self):

        
        modelprediction_obj = ModelPrediction(cf.PATH_RESUME_TEST, cf.MODEL_PATH)
        df = modelprediction_obj.predict()
        data = df.set_index('FileName').T.to_dict('list') 


        return data

        


