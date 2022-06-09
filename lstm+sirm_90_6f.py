import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import math
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


m = 90  # досліджуваний період
n = 367  # перший досліджуваний день

N =  126264931 # населення Японії 2019
N_0 = 126529100 # населення 2018
M_0 =  1351035  # deaths in 2019
G =  921734  # birth rate 2019

sliced_growth = [[], [], []]  # confirmed death recovered but by day growth

begin = [[], [], []]

confirmed = pd.read_csv('time_series_covid19_confirmed_global.csv')
dead = pd.read_csv('time_series_covid19_deaths_global.csv')
recovered = pd.read_csv('time_series_covid19_recovered_global.csv')


def data_process(dataset):
    dataset.rename(columns={'Country/Region': 'Country'}, inplace=True)
    index_Japan = dataset.loc[dataset['Country'] == 'Japan'].index[0]
    return index_Japan


list_of_country_index_confirmed_with_ds = [confirmed, data_process(confirmed)]  # датасет и нужные индексы для стран
list_of_country_index_dead_with_ds = [dead, data_process(dead)]
list_of_country_index_recovered_with_ds = [recovered, data_process(recovered)]


end_date = '4/1/21'
start_date = '12/31/20'

def make_into_growth(list_of_country_index_ds):
    sliced = list_of_country_index_ds[0].loc[[list_of_country_index_ds[1]], start_date:end_date]
    growth_by_day = np.diff(sliced.values, axis=1)
    begin_conditions = list_of_country_index_ds[0].loc[list_of_country_index_ds[1], start_date]
    return growth_by_day, begin_conditions


def sir_one_country():
    growth_confirmed, begin_con_confirmed = make_into_growth(list_of_country_index_confirmed_with_ds)
    growth_dead, begin_con_dead = make_into_growth(list_of_country_index_dead_with_ds)
    growth_recovered, begin_con_recovered = make_into_growth(list_of_country_index_recovered_with_ds)
    I = [begin_con_confirmed - begin_con_dead - begin_con_recovered]
    R = [begin_con_recovered]
    M = [begin_con_dead]
    S = [(N - (n - 1) * M_0 * N / (N_0 * 365) + (n - 1) * G * N /
          (N_0 * 365) - I[0] - R[0] - M[0])]

    m_1 = m_3 = M_0 / (365 * N_0)
    gamma = (G * N/ (365 * N_0))
    for i in range(m - 1):
        S.append(int((1 - m_1) * S[i] - growth_confirmed[0][i] + gamma))
        M.append(int(M[i] + growth_dead[0][i]))
        R.append(int((1 - m_3) * R[i] + growth_recovered[0][i]))
        I.append(int(I[i] + growth_confirmed[0][i] - growth_recovered[0][i] - growth_dead[0][i]))

    return S, I, R, M, m_1, gamma


SIRM = sir_one_country()

dates = []
start = datetime.datetime.strptime("01-01-2021", "%d-%m-%Y")
end = datetime.datetime.strptime("01-04-2021", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

for date in date_generated:
    dates.append(date.strftime("%d-%m-%Y"))

croped_dates = [dates[i] if i % 10 == 0 else ' ' for i in range(len(dates))]
labels = ['Сприйнятливі до захворювання', 'Кількість інфікованих', 'Кількість одужалих', 'Померлі']


def draw(comp, i, var_label):
    plt.xlabel('Date')
    plt.ylabel('Number of people')
    plt.xticks(range(len(croped_dates)), croped_dates, rotation='vertical', size='small')
    plt.plot(dates, comp, label=var_label[i], color='black')
    plt.legend()
    plt.show()


draw(SIRM[1], 1, labels)

def calculate_F(a,b,matrix_f, P, F):
    v4_list = []
    flagg = True
    for i in range(a, b):
        v1 = P[i].dot(matrix_f[i].transpose())
        v2 = matrix_f[i].dot(P[i])
        v3 = v2.dot(matrix_f[i].transpose())
        try:
            v4 = np.linalg.inv(v3 + np.eye(4))
        except:
            flagg = False
        if flagg is False:
            F.append(F[i-1])
        else:
            v4_list.append(v4)
            v5 = v1.dot(v4)
            F.append(v5)
        tmp = np.eye(3) - F[i].dot(matrix_f[i])
        tmp1 = tmp.dot(P[i]).dot(tmp.transpose())
        P.append(tmp1 + np.eye(3) + F[i].dot(F[i].transpose()))


def calculate_mark_a(SIRM):
    mark_a = [np.array([[0],
                        [0],
                        [0]])]
    matrix_f = []
    matrix_g = []
    P = [np.zeros((3, 3))]
    F = []
    y = []
    for i in range(m):
        matrix_f.append(np.array([[-SIRM[0][i] * SIRM[1][i], 0, 0],
                                  [SIRM[0][i] * SIRM[1][i], -SIRM[1][i], -SIRM[1][i]],
                                  [0, SIRM[1][i], 0],
                                  [0, 0, SIRM[1][i]]]))

        matrix_g.append(np.array([[(1 - SIRM[4]) * SIRM[0][i] + SIRM[5]],
                                  [SIRM[1][i]],
                                  [(1 - SIRM[4]) * SIRM[2][i]],
                                  [SIRM[3][i]]]))

    calculate_F(0,m-1,matrix_f, P, F)

    for i in range(m-1):
        y.append(np.array([[SIRM[0][i + 1]],
                           [SIRM[1][i + 1]],
                           [SIRM[2][i + 1]],
                           [SIRM[3][i + 1]]]) - matrix_g[i])
    for i in range(m-1):
        etw = mark_a[i] + F[i].dot(y[i] - matrix_f[i].dot(mark_a[i]))
        if etw[0] < 0:
            etw[0] = 0
        if etw[1] < 0:
            etw[1] = 0
        if etw[2] < 0:
            etw[2] = 0
        mark_a.append(etw)
    return mark_a

lab = ['Оптимальна оцінка альфа', 'Оптимальна оцінка бета', 'Оптимальна оцінка мю_2']

mark_a_for_certain_country = calculate_mark_a(SIRM)
list_mark_alpha_for_certain_country = []
list_mark_beta_for_certain_country = []
list_mark_mu2_for_certain_country = []
for i in range(m):
    list_mark_alpha_for_certain_country.append(mark_a_for_certain_country[i][0].tolist()[0])
    list_mark_beta_for_certain_country.append(mark_a_for_certain_country[i][1].tolist()[0])
    list_mark_mu2_for_certain_country.append(mark_a_for_certain_country[i][2].tolist()[0])
list_of_marks = [list_mark_alpha_for_certain_country, list_mark_beta_for_certain_country,
                 list_mark_mu2_for_certain_country]
#for i in range(3):
#    draw(list_of_marks[i], i, lab)



plt.xlabel('Date')
plt.ylabel('Number of people')
plt.xticks(range(len(croped_dates)), croped_dates, rotation='vertical', size='small')
plt.plot(dates[0:96], list_mark_alpha_for_certain_country[0:96], color='black')
plt.legend()
plt.show()

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

dataset_4 = list_mark_alpha_for_certain_country[0:90]
dataset_4 = np.reshape(dataset_4, (len(dataset_4), 1))
dataset_5 = list_mark_beta_for_certain_country[0:90]
dataset_5 = np.reshape(dataset_5, (len(dataset_5), 1))
dataset_6 = list_mark_mu2_for_certain_country[0:90]
dataset_6 = np.reshape(dataset_6, (len(dataset_6), 1))

f = open("dataset.txt")

dataset123 = np.ndarray((90, 3))
for i in range(90):
    str_tmp = f.readline()
    arr_tmp = (str_tmp[1:-2]).split()
    dataset123[i][0] = int(arr_tmp[0])
    dataset123[i][1] = int(arr_tmp[1])
    dataset123[i][2] = int(arr_tmp[2])

dataset1_2_3 = dataset123

dataset = hstack((dataset123, dataset_4, dataset_5, dataset_5))

scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset)

n_steps = 2
n_features = 6
X, y = split_sequences(dataset, n_steps)
# split into train and test sets
train_size = int(len(X) * 0.9)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size, :], X[train_size:len(X), :]
Y_train, Y_test = y[0:train_size, :], y[train_size:len(y), :]
#print("x train ",X_train)
#print(" y train ", Y_train)
# create  network
print('xtrain ',len(X_train))
print(len(X_test))
model = Sequential(name='covid_forecast_s')
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(n_features))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])
model.summary()
# fit model
history = model.fit(X_train, Y_train, epochs=300, batch_size=1, verbose=2, validation_data=(X_test, Y_test))
# make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
unscaled_train = scaler.inverse_transform(trainPredict)
unscaled_test = scaler.inverse_transform(testPredict)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

plot_time = np.arange(80, 80+len(unscaled_test), 1)
green_plot_time = np.arange(2, 2+len(unscaled_train), 1)
plt.plot(dataset1_2_3[:,0:1], color='navy')
plt.plot(green_plot_time,unscaled_train[:, 0:1], color='green')
plt.plot(plot_time, unscaled_test[:, 0:1], color='orange')
plt.show()
plt.plot(dataset1_2_3[:,1:2], color='navy')
plt.plot(green_plot_time,unscaled_train[:, 1:2], color='green')
plt.plot(plot_time, unscaled_test[:, 1:2], color='orange')
plt.show()
plt.plot(dataset1_2_3[:,2:3], color='navy')
plt.plot(green_plot_time,unscaled_train[:, 2:3], color='green')
plt.plot(plot_time, unscaled_test[:, 2:3], color='orange')
plt.show()
