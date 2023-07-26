import json
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import mean_absolute_error as MAE
# from prophet import Prophet
# import pandas as pd
# import cython
from pystanmodel import *
# from orbit.orbit.models.lgt import LGT as LGTFull
# from orbit.orbit.models.dlt import DLT as DLTFull
# from sktime.forecasting.naive import NaiveForecaster
# from orbit.orbit.diagnostics.plot import plot_predicted_data

pd.options.display.width = 0


def get_data():
    data = pd.read_csv('data.csv')
    # print(pd.read_csv('data.csv'))
    k = []
    for i, j, t in zip(data['ticker'], data['last_prices'], data['time']):
        last_pr = list(json.loads(j))
        time = list(json.loads(t))
        t = {pd.to_datetime(data): float(price) for data, price in zip(time, last_pr)}
        k.append(t)
    df_time_tick = pd.DataFrame.from_records(k, index=data['ticker'])
    df_time_tick = df_time_tick.transpose().sort_index()
    df_time_tick_without_nulls = df_time_tick.fillna(method='ffill', inplace=False).fillna(method='bfill')
    return df_time_tick_without_nulls


def split_into_train_and_test(data: pd.DataFrame, proportion=0.8) -> dict:
    point = int(proportion * data.shape[0])
    return {'train': data.loc[:point], 'test':  data.loc[point:]}


def get_pred(data, ticker, solver=naiveSolver): # перем дату, сплитим, отправляем в солвер
    company = data[ticker]
    company = company.reset_index(drop=False).rename(columns={'index': 'date'})
    company[ticker] = np.log(company[ticker])
    data = split_into_train_and_test(company)
    data_end = solver(data, ticker)
    if solver == naiveSolver:
        data['test'][ticker].plot()
        data_end['test_prediction']['y_naiv'].plot()
        data_end['test_prediction']['y_naiv_season'].plot()
        data['train'][ticker].plot()
        plt.show()
    return data_end


def start():
    data = get_data()
    yandex = get_pred(data, 'YNDX', solver=pystanModel)
    print(yandex)
    return 0
    # yandex.index = pd.PeriodIndex(yandex.index, freq='D')
    # yandex_df['date'] = yandex_df['date'].to_timestamp()
    # print(yandex_df.shape)
    # train_df11 = yandex_df.loc[:int(0.8 * yandex_df.shape[0])]
    # test_df11 = yandex_df.loc[int(0.8 * yandex_df.shape[0]):]
    # prophnet(train_df11, test_df11)
    # print(df_time_tick.pivot_table(index=df_time_tick.columns, columns=df_time_tick.index))
    # print(t)
    # time_data = pd.DataFrame(t)
    # plt.plot(t['YNDX'])
    plt.show()
    # make_plot()
    # print(j['Цена'])
    # print(time_data)
    # print(statistics(t['YNDX']))
    # j = pd.read_csv('Прошлые данные - YNDX.csv')
    # j['data'] = j['Дата'].astype(np.datetime64)
    # j['data'] = pd.to_datetime(j['data'])
    # date_from_inv = j.set_index('data').sort_index(ascending=True)
    # date_from_inv['price'] = date_from_inv['Цена'].apply(lambda x: float(x.replace('.', '').replace(',', '.')))
    # print(date_from_inv['price'].reset_index()['price'].plot())
    # print(date_from_inv)
    # print(date_from_inv['2022-12-16'])
    # plt.show()
    # print(date_from_inv.std())
    return 0


def make_plot():
    j = pd.read_csv('Прошлые данные - YNDX.csv')
    print(j)
    date_from_inv = j.set_index(pd.to_datetime(j['Дата'], format='YYYY-MM-DD'))['Цена'].sort_index(
        ascending=True).apply(
        lambda x: float(x.replace('.', '').replace(',', '.')))
    plt.plot(np.array(date_from_inv))
    plt.show()


def preparing(data):
    data = np.array(data)
    g = len(data)
    train = data[:int(g * 0.3)]
    test = data[int(g * 0.3):]

    return data.std()
