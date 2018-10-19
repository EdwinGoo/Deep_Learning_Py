import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.models import Sequential

from sklearn.metrics import r2_score

airline=pd.read_csv('data/international-airline-passengers.csv')
airline = airline.dropna()

# airline.set_index('Month').plot()
# plt.show()
# print(pd.DataFrame(airline))

X = airline.iloc[:,1].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1,1))
y = X[1:].flatten()

X = X[:-1]
X = X.reshape(-1, 1, 1)
#시계열 데이터는 셔플을 하면 바보
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

time_step = X.shape[1] # 1
특징수 = X.shape[2] 

model = Sequential()
model.add(LSTM(4, input_shape=(time_step, 특징수)))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train,y_train, epochs=10, batch_size=2)

훈련결과 = pd.DataFrame(history.history)
훈련결과['loss'].plot()

y_pred_train = model.predict(X_train)
print(r2_score(y_train, y_pred_train))