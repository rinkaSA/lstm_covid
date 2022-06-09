import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from numpy import hstack
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

m = 90 # досліджуваний період
n = 367  # перший досліджуваний день

N = 126264931  # населення Японії 2019
N_0 = 126529100  # населення 2018
M_0 = 1351035  # deaths in 2019
G = 921734  # birth rate 2019

sliced_growth = [[], [], []]  # confirmed death recovered but by day growth

confirmed = pd.read_csv('time_series_covid19_confirmed_global.csv')
dead = pd.read_csv('time_series_covid19_deaths_global.csv')
recovered = pd.read_csv('time_series_covid19_recovered_global.csv')


def data_process(dataset):
    dataset.rename(columns={'Country/Region': 'Country'}, inplace=True)
    index_Japan = dataset.loc[dataset['Country'] == 'Japan'].index[0]
    return index_Japan


end_date = '4/1/21'


def make_into_growth(list_of_country_index_ds):
    sliced = list_of_country_index_ds[0].loc[[list_of_country_index_ds[1]], '12/31/20':end_date]
    growth_by_day = np.diff(sliced.values, axis=1)
    begin_conditions = list_of_country_index_ds[0].loc[list_of_country_index_ds[1], '12/31/20']
    return growth_by_day, begin_conditions


list_of_country_index_confirmed_with_ds = [confirmed, data_process(confirmed)]
list_of_country_index_dead_with_ds = [dead, data_process(dead)]
list_of_country_index_recovered_with_ds = [recovered, data_process(recovered)]

growth_confirmed, begin_con_confirmed = make_into_growth(list_of_country_index_confirmed_with_ds)
growth_dead, begin_con_dead = make_into_growth(list_of_country_index_dead_with_ds)
growth_recovered, begin_con_recovered = make_into_growth(list_of_country_index_recovered_with_ds)
# print(growth_confirmed[0])

dates = []
start = datetime.datetime.strptime("01-01-2021", "%d-%m-%Y")
end = datetime.datetime.strptime("01-05-2022", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

for date in date_generated:
    dates.append(date.strftime("%d-%m-%Y"))

croped_dates = [dates[i] if i % 30 == 0 else ' ' for i in range(len(dates))]
labels = ['Сприйнятливі до захворювання', 'Кількість інфікованих', 'Кількість одужалих', 'Померлі']


# convert an array of values into a dataset matrix
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


dataset_1 = growth_confirmed[0][0:-1]
dataset_1 = np.reshape(dataset_1, (dataset_1.shape[0], 1))
dataset_2 = growth_dead[0][0:-1]
dataset_2 = np.reshape(dataset_2, (dataset_2.shape[0], 1))
dataset_3 = growth_recovered[0][0:-1]
dataset_3 = np.reshape(dataset_3, (dataset_3.shape[0], 1))
dataset = hstack((dataset_1, dataset_2, dataset_3))



scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

n_steps = 3
n_features = 3
X, y = split_sequences(dataset, n_steps)
# split into train and test sets
train_size = int(len(X) * 0.9)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size, :], X[train_size:len(X), :]
Y_train, Y_test = y[0:train_size, :], y[train_size:len(y), :]
#print("x train ",X_train)
#print(" y train ", Y_train)
# create  network

model = Sequential(name='covid_forecast')
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(n_features))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])
model.summary()
# fit model
history = model.fit(X_train, Y_train, epochs=50, batch_size=1, verbose=2, validation_data=(X_test, Y_test))
# make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
unscaled_train = scaler.inverse_transform(trainPredict)
unscaled_test = scaler.inverse_transform(testPredict)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

plot_time = np.arange(78, 78+len(unscaled_test), 1)

plt.plot(dataset_1, color='navy')
plt.plot(unscaled_train[:, 0:1], color='green')
plt.plot(plot_time, unscaled_test[:, 0:1], color='orange')
plt.show()
plt.plot(dataset_2, color='navy')
plt.plot(unscaled_train[:, 1:2], color='green')
plt.plot(plot_time, unscaled_test[:, 1:2], color='orange')
plt.show()
plt.plot(dataset_3, color='navy')
plt.plot(unscaled_train[:, 2:3], color='green')
plt.plot(plot_time, unscaled_test[:, 2:3], color='orange')
plt.show()