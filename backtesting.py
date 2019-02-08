import pandas as pd
from random import randint
import json

bt = pd.read_csv('backtest.csv', index_col=0)


# Считаем потенциальную доходность
bt['pred_return'] = (bt['pred'] - bt['close']) / bt['close']
bt['time'] = bt['time'].shift(1)


# Задаем параметры стратегии - при каких условиях бот входит в сделку
# (1) - сигнал на покупку, (-1) - на продажу
bt['signal'] = 0
bt.loc[bt['pred_return'] > 0.03, 'signal'] = 1
bt.loc[bt['pred_return'] < -0.03, 'signal'] = -1
takeprofit = 0.03
stoploss = 0.02

open_trade = 0
orders = {}


# Простейшая стратегия на выход из трейда по стоплоссу или тейкпрофиту
for index, row in bt.iterrows():
    if open_trade == 1:
        if orders[order_id]['type'] == 'long':
            if row['low'] < orders[order_id]['stoploss']:
                orders[order_id]['status'] = 'closed'
                orders[order_id]['close_price'] = orders[order_id]['stoploss']
                orders[order_id]['close_time'] = row['time']
                open_trade = 0
                continue
            elif row['high'] > orders[order_id]['takeprofit']:
                orders[order_id]['status'] = 'closed'
                orders[order_id]['close_price'] = orders[order_id]['takeprofit']
                orders[order_id]['close_time'] = row['time']
                open_trade = 0
                continue
            else:
                orders[order_id]['current_price'] = row['close']
        else:
            if row['high'] > orders[order_id]['stoploss']:
                orders[order_id]['status'] = 'closed'
                orders[order_id]['close_price'] = orders[order_id]['stoploss']
                orders[order_id]['close_time'] = row['time']
                open_trade = 0
                continue
            elif row['low'] < orders[order_id]['takeprofit']:
                orders[order_id]['status'] = 'closed'
                orders[order_id]['close_price'] = orders[order_id]['takeprofit']
                orders[order_id]['close_time'] = row['time']
                open_trade = 0
                continue
            else:
                orders[order_id]['current_price'] = row['close']
    elif row['signal'] == 1:
        order_id = randint(0, 10**6)
        orders[order_id] = {
            'open_price': row['close'],
            'current_price': None,
            'close_price': None,
            'type': 'long',
            'open_time': row['time'],
            'close_time': None,
            'status': 'open',
            'stoploss': row['close'] * (1 - stoploss),
            'takeprofit': row['close'] * (1 + takeprofit)
        }
        open_trade = 1
        continue
    elif row['signal'] == -1:
        order_id = randint(0, 10 ** 6)
        orders[order_id] = {
            'open_price': row['close'],
            'current_price': None,
            'close_price': None,
            'type': 'short',
            'open_time': row['time'],
            'close_time': None,
            'status': 'open',
            'stoploss': row['close'] * (1 + stoploss),
            'takeprofit': row['close'] * (1 - takeprofit)
        }
        open_trade = 1
        continue
    else:
        continue


# Считаем результаты
results = pd.DataFrame.from_dict(orders)
results = results.T
results['profit'] = 0
for index, row in results.iterrows():
    if row['status'] == 'closed':
        if row['type'] == 'long':
            results.loc[index, 'profit'] = (row['close_price'] - row['open_price']) / row['open_price']
        else:
            results.loc[index, 'profit'] = (row['open_price'] - row['close_price']) / row['open_price']
    else:
        if row['type'] == 'long':
            results.loc[index, 'profit'] = (row['current_price'] - row['open_price']) / row['current_price']
        else:
            results.loc[index, 'profit'] = (row['open_price'] - row['current_price']) / row['open_price']

cumulative = 0
for entry in results['profit']:
    if cumulative == 0:
        cumulative = 100 * (1 + entry)
    else:
        cumulative = cumulative * (1 + entry)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(results)

print('Cumulative profitability = ', cumulative - 100, '%')

bt_export = bt[['time', 'close', 'signal']].copy(deep=True)
bt_export['time'] = bt_export['time'].shift(-1)
bt_export = bt_export[(bt_export['signal'] != 0)]


# Задаем функцию для изменения формата времени, требуемого для выгрузки
def change_time_format(input):
    x, y = input.split(' ')
    y = y[:-3]
    y = y + 'Z[UTC]'
    output = 'T'.join((x,y))
    return output

report = []
for index, row in bt_export.iterrows():
    new_time = change_time_format(row['time'])
    trade_type = 'BUY' if row['signal'] == 1 else 'SELL'
    report.append({"time": new_time, "type": trade_type, "price": row['close']})

with open('report.json', 'w') as outfile:
    json.dump(report, outfile)
