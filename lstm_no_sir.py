import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

m = 90  # досліджуваний період
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
end = datetime.datetime.strptime("01-04-2021", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

for date in date_generated:
    dates.append(date.strftime("%d-%m-%Y"))

croped_dates = [dates[i] if i % 10 == 0 else ' ' for i in range(len(dates))]
labels = ['Сприйнятливі до захворювання', 'Кількість інфікованих', 'Кількість одужалих', 'Померлі']


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

growth_confirmed = np.reshape(growth_confirmed[0], (len(growth_confirmed[0]), 1))

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(growth_confirmed)
# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(25, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2, validation_data=(testX, testY))
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
