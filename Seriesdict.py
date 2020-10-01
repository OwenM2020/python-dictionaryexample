import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

#Read the dataset from sklearn with breast cancer data
df = load_breast_cancer()
# Create a dataframe with the keys and data
medical_df = pd.DataFrame(df.data,columns=df.feature_names)
# Target is the filed with the information on the cases that are benign or malignant
medical_df['target'] = df.target

# Create a series that show the umber of bening and malignant cases from the dataset
malignant = len(medical_df[medical_df['target']==0])
benign = len(medical_df[medical_df['target']==1])

target_data = {'malignant':malignant, 'benign':benign}
target = pd.Series(target_data,index=['malignant','benign'])

#Display the results
print(target)
