# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:18:56 2023

@author: lidya
"""

import numpy as np
import pickle


loaded_model = pickle.load(open('C:/Users/lidya/Desktop/Gold Price Prediction/trained_model_gold.sav','rb'))


input_data = (1447.1,78.47,15.18,1.471692)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)