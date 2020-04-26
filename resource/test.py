#path = r'/Users/shreyasnanaware/Desktop/resumeparser/resources/Resumes'
path = r'/Users/shreyasnanaware/Desktop/resumeparser/resource/Resumes'
import pandas as pd
from glob import glob

df = pd.DataFrame()

df['Path'] = glob(path + '*/*')
print(df['Path'][0])