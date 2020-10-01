import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

df = load_breast_cancer()

    
medical_df = pd.DataFrame(df.data,columns=df.feature_names)
medical_df['target'] = df.target


malignant = len(medical_df[medical_df['target']==0])
benign = len(medical_df[medical_df['target']==1])

target_data = {'malignant':malignant, 'benign':benign}
target = pd.Series(target_data,index=['malignant','benign'])


print(target)
