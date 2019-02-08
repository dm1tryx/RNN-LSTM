import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM


with open('BTCUSDT_2017-08-01_2019-01-23_1m.dat') as file:
    data = pd.read_json(file)


# исследуем данные на NaN и дубликаты (с этим все ок)
data.describe()
data.info()
data['duplicated'] = data.duplicated()
print(data[(data['duplicated'] == True)])
data.drop(columns=['duplicated'], inplace=True)


# проверяем данные на пропущенные свечи
print(data['time'].iloc[0])
print(data['time'].iloc[-1])


# Первый timestamp - 1502942400000 = Thursday, 17 August 2017 г., 4:00:00
# Последний - 1548225540000 = Wednesday, 23 January 2019 г., 6:39:00
# Строк в таблице - 750247, а должно быть 754720, т.е. пропущено 4473 минутных свечи.
# В идеале эти пропуски нужно как-то заполнить, поскольку модель будет тренироваться с ошибками.

# В колонке time в исходных данных есть какой-то косяк - в рядах [156661:177060] и [251167:252367] timestamps
# сдвинуты во времени, и заканчиваются на 799, а не 000 как обынчо. По этой причине мы корректируем timestamps,
# сдвигая их к ближайшим значениям - таким, которые делятся на 60000 (60 сек) без остатка.


data['time'].loc[156661:177060] = data['time'].loc[156661:177060] + 39201
data['time'].loc[251167:252367] = data['time'].loc[251167:252367] - 14789


# Далее генерируем новый временной ряд без пропусков
new_time = [i for i in range(1502942400000, 1548225600000, 60000)]
df = pd.DataFrame(new_time, columns=['time'])


# И объединяем таблицы, чтобы на месте изначально пропущенных свечей были NaN
data = pd.merge(df, data, how='left', on='time')


# OHLC заполняем методом interpolate(), а trades и volume - по средней, рассчитываемой для каждых 5000 свеч
data['close'].interpolate(inplace=True)
data['high'].interpolate(inplace=True)
data['low'].interpolate(inplace=True)
data['open'].interpolate(inplace=True)

for x in range(0, 754719, 5000):
    mean = round(data['trades'].loc[x:(x+5000)].mean())
    data['trades'].loc[x:(x+5000)].fillna(value=mean, inplace=True)

for x in range(0, 754719, 5000):
    mean = data['volume'].loc[x:(x+5000)].mean()
    data['volume'].loc[x:(x+5000)].fillna(value=mean, inplace=True)


# Трансформируем минутные свечи в 15-минутные
data['time'] = data['time'] / 1000
data['time'] = pd.to_datetime(data['time'], unit='s')
data.set_index('time', inplace=True)
ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'trades': lambda x: x.sum(), 'volume':
                                                                                            lambda x: x.sum()}
data = data.resample('15Min').apply(ohlc_dict)


# Добавляем технические индикаторы
data['MA'] = data['close'].rolling(window=30).mean()
data['Std_dev'] = data['close'].rolling(5).std()
data['MACD'], data['macdsignal'], data['macdhist'] = talib.MACD(data['close'].values, fastperiod=12, slowperiod=26,
                                                                                                    signalperiod=9)
data['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
data['RSI'] = talib.RSI(data['close'], timeperiod=14)


# Будем прогнозировать цену закрытия следующей свечи
data['y'] = data['close'].shift(-1)
data.dropna(inplace=True)


# Задаем константы
timesteps = 100  # сколько свеч назад будем анализировать при тренировке и прогнозе

optimizer = 'adam'
batches = 5
epochs = 3
features = 13


# Инициализируем класс для нормализации данных
scaler = StandardScaler()


# Создаем модель
model = Sequential()

model.add(LSTM(96, input_shape=(timesteps, features)))
model.add(Dense(1))

score = []
predictions = []


# Прогнозируем в 5 этапов - каждый этап последовательно прогнозирует 20% тестовой части данных на основе
# всей имеющейся предыдущей истории
for period in (0.5, 0.6, 0.7, 0.8, 0.9):

    # Нормализуем данные и готовим их для модели
    split = int(len(data)*period)

    train = data[0:split]
    test = data[split - timesteps + 1: int(len(data)*(period + 0.1))]

    nd_train = scaler.fit_transform(train)
    nd_test = scaler.transform(test)

    x_train = nd_train[:, 0:-1]
    y_train = nd_train[:, -1]

    x_test = nd_test[:, 0:-1]
    y_test = nd_test[:, -1]

    x_tr_list = [np.expand_dims(x_train[i:timesteps+i, :], axis=0) for i in range(len(x_train) - timesteps + 1)]
    x_train = np.concatenate(x_tr_list, axis=0)
    y_train = y_train[99:]

    x_test_list = [np.expand_dims(x_test[i:timesteps+i, :], axis=0) for i in range(len(x_test) - timesteps + 1)]
    x_test = np.concatenate(x_test_list, axis=0)
    y_test = y_test[99:]

    # Компилируем и тренируем модель
    model.compile(loss='mse', optimizer=optimizer)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batches)

    # Генерируем предсказания и возвращаем их в исходный вид
    scaled_pred = model.predict(x_test, batch_size=batches)
    pred_batch = scaler.inverse_transform(np.concatenate((nd_test[99:, :-1], scaled_pred), axis=1))

    score.append(model.evaluate(x_test, y_test, batch_size=batches))
    predictions.append(pred_batch[:, -1:])

print('The score is:', np.mean(score))


# Объединяем предсказанные значения с реальными в одной таблице для тестирования стратегии
predictions = np.vstack(predictions)
pred_values = pd.DataFrame(predictions[:, -1:], columns=['pred'])
all_test = data[int(len(data)*0.5):]
all_test.reset_index(inplace=True)
all_test = all_test[['time', 'open', 'high', 'low', 'close', 'y']]
bt = pd.concat((all_test, pred_values), axis=1)
bt.rename(index=str, columns={"y": "close_t1"}, inplace=True)


# Считаем MSE на данных в исходном виде
real_y = all_test['y']
real_vs_pred = pd.concat([real_y, pred_values], axis=1)
real_vs_pred['se'] = (real_vs_pred['y'] - real_vs_pred['pred'])**2
real_vs_pred['se'].describe()


# Если надо нарисовать график ошибок
plt.plot(real_vs_pred['se'])


# Если надо выгрузить файл
bt.to_csv('backtest.csv')
