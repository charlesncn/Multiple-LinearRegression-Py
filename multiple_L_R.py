import pandas as pd
import numpy as np
import io
from google.colab import files
upload = files.upload()
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv(io.BytesIO(upload['wine.csv']))
dataset.info()
y=dataset.iloc[:,:1].values
x=dataset.iloc[:,1:].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()

regressor.fit(x_train,y_train)

pred = regressor.predict(x_test)
pred
