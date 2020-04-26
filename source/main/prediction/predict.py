from source.main.utils import stopwords as st


import warnings
warnings.filterwarnings('ignore')
import re
import pickle
import numpy as np
import pandas as pd
from glob import glob
from tika import parser

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

class ModelPrediction:
    
    
    def __init__(self, path, model_path):


        self.path = path
        self.le = LabelEncoder()
        self.clf = pickle.load(open(model_path + 'latest_model.pkl', 'rb'))
        self.le.classes_ = np.load(model_path + 'classes.npy', allow_pickle=True)
        self.count_vect = pickle.load(open(model_path + 'count_vect.pkl', 'rb'))
        self.df = pd.DataFrame()
    
    def predict_extract_preprocessed_text(self, path):

        '''
        This method predicts the 'Resume Type' using text from resume.

        @param path :: string: Path to resume file

        @return:
            :filename :: string: Name of the file
            :resume_type_pred :: string: Predicted value
        '''


        filename = path.split('/')[-1]
        try:
            txt = parser.from_file(path)['content'].lower().strip()
            txt = re.sub(st.PREPROCESS_STRING, " ", txt)
            txt_list = re.sub(st.SPACE_STRING, " ", txt).split(' ')    
            txt_list = [ word for word in txt_list if word not in st.STOP ]
            text = [' '.join(txt_list)]

            test_count = self.count_vect.transform(text)
            resume_type_pred = self.le.inverse_transform(self.clf.predict(test_count))[0]

        except:
            resume_type_pred = 'Data not present'
        
        return filename, resume_type_pred




    def predict(self):

        self.df['FileName'], self.df['Predicted Type'] = zip(*pd.Series(glob(self.path + '*/*')).apply(self.predict_extract_preprocessed_text))
        return self.df