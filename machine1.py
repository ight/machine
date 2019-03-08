# first machine learning program 
# regresion for a straight line
# equation for a stright line y = x + p (where p is constant and and x and y variables)
# "XYZ" is the api key that quandl provides after registration

import pandas as pd
import quandl, math, datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinerRegression
from matplotlib import style

style.use('ggplot')

df = quandl.get("CHRIS/MGEX_IH1", authtoken="XYZ")
df = df[['Open', 'High', 'Low', 'Last', 'Volume']]
df['HI_PCT'] = (df['High'] - df['Last']) / df['Last'] * 100.0
df['CHN_PCT'] = (df['Last'] - df['Open']) / df['Open'] * 100.0

df = df[['Last', 'HI_PCT', 'CHN_PCT', 'Volume']]

forecast_col = 'Last'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['lable'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# features "X"
# labels "Y"

X = np.array(df.drop(['lable'], 1))
# scaling x
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

Y = np.array(df['lable'])
Y = np.array(df['lable'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
clf = LinearRegression()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test,y_test)

print(accuracy)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
  next_date = datetime.datetime.fromtimestamp(next_unix)
  next_unix += one_day
  df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i]


df['Last'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()