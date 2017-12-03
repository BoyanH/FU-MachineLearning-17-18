import os
import numpy as np
import pandas as pd
from LogisticRegression import LogisticRegression
from Parser import extract_classes_from_data_set


file_name = os.path.join(os.path.dirname(__file__), './Dataset/boolfunc.data')
data = np.array(pd.read_csv(file_name, header=None).as_matrix())
X = np.array(data[:,:-1], dtype=np.float64)
y = data[:,-1]

and_or_X, and_or_y = extract_classes_from_data_set(X, y, ['and', 'or'])
and_or_y = np.array([1 if x == 'and' else 0 for x in and_or_y], dtype=np.float64)
lr_and_or = LogisticRegression(and_or_X, and_or_y, iterations=1000)
print('beta and vs or: {}'.format(lr_and_or.beta))
print('transformed X and vs or: {}'.format(lr_and_or.transform(and_or_X, and_or_y)[0]))
print('Predicted: {}; Expected: {}'.format(lr_and_or.predict(and_or_X), and_or_y))

and_xor_X, and_xor_y = extract_classes_from_data_set(X, y, ['and', 'xor'])
and_xor_y = np.array([1 if x == 'and' else 0 for x in and_xor_y], dtype=np.float64)
lr_and_xor = LogisticRegression(and_xor_X, and_xor_y, iterations=1000)
print('beta and vs xor: {}'.format(lr_and_xor.beta))
print('transformed X and vs xor: {}'.format(lr_and_xor.transform(and_xor_X, and_xor_y)[0]))
print('Predicted: {}; Expected: {}'.format(lr_and_xor.predict(and_xor_X), and_xor_y))

or_xor_X, or_xor_y = extract_classes_from_data_set(X, y, ['or', 'xor'])
or_xor_y = np.array([1 if x == 'or' else 0 for x in or_xor_y], dtype=np.float64)
lr_or_xor = LogisticRegression(or_xor_X, or_xor_y, iterations=1000)
print('beta or vs xor: {}'.format(lr_or_xor.beta))
print('transformed X or vs xor: {}'.format(lr_or_xor.transform(or_xor_X, or_xor_y)[0]))
print('Predicted: {}; Expected: {}'.format(lr_or_xor.predict(or_xor_X), or_xor_y))
