import json

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from test.pystanmodel import pystanModel
from trol import t
# import os
from datetime import timedelta

# from tinkoff.invest import CandleInterval, Client, PortfolioRequest, PortfolioResponse, InstrumentIdType
# from tinkoff.invest.utils import now

TOKEN = t


def main():
    with Client(TOKEN) as client:
        data = []
        # print(get_price_data(client=client, figi='dfdf', bool=False))
        # return 0

        accounts = client.users.get_accounts()
        account_id = accounts.accounts[0].id
        # p = PortfolioRequest(account_id)
        # p = PortfolioResponse()
        p = client.operations.get_positions(account_id=account_id)
        data = pd.DataFrame(p.securities)
        print(data.columns)
        data = data[['figi', 'balance', 'instrument_type']]
        print(data)
        pp = client.operations.get_portfolio(account_id=account_id)
        # print(pp) # тут видим количество денег на счете и другие параметры счета
        tt = client.market_data.get_last_prices(figi=data['figi'].to_list())
        print(tt.last_prices)
        data['price'] = [i.price.units for i in tt.last_prices]
        data['ticker'] = data['figi'] + '_' +  data['instrument_type']
        data['ticker'] = data['ticker'].apply(lambda x : get_attr(x, client, True))
        # print(data)
        # data['name'] = (data['figi'] + '_' +  data['instrument_type']).apply(lambda x : get_attr(x, client, False))
        data['time'] = [i.time.time().isoformat()[:8] for i in tt.last_prices]
        # f = client.instruments.share_by(id_type= InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id='BBG004S688G4')
        data['last_prices'] = data.figi.apply(lambda x : get_price_data(client=client, figi=x, bool=True, bool1=False, days=30))
        data['time'] = data.figi.apply(lambda x : get_price_data(client=client, figi=x, bool=True, bool1 = True, days=30))
        data.to_csv('data.csv')
        # for candle in client.get_all_candles(
        #     figi="BBG004730N88",
        # BBG000000002 - нужно мне
        #     from_=now() - timedelta(days=365),
        #     interval=CandleInterval.CANDLE_INTERVAL_HOUR,
        # ):
        #     t = create_df(candle)
        #     data.append(t)
        # k = pd.DataFrame(data)
        # print(k.shape[0]/14)
        # print(k.head(16))
        # k.plot(x='time', y='close')
        # plt.show()



    return 0
def create_df(candle):
    return pd.Series({'open': candle.open.units,
                                      'high': candle.high.units,
                                      'low': candle.low.units,
                                      'close': candle.close.units,
                                      'time': candle.time,
                                      'volume': candle.volume
               })


def get_attr(tr,  client, bool):
    figi, tip = tr.split('_')
    if bool:
        if tip == 'share':
            return client.instruments.share_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi).instrument.ticker
        elif tip == 'share':
            return client.instruments.etf_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi).instrument.ticker
        elif tip == 'bond':
            return client.instruments.bond_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi).instrument.ticker
        else:
            return 'not found/Tinkoffofsky'
    else:
        if tip == 'share':
            return client.instruments.share_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi).instrument.name
        elif tip == 'share':
            return client.instruments.etf_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi).instrument.name
        elif tip == 'bond':
            return client.instruments.bond_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi).instrument.name
        else:
            return 'not found/Tinkoffofsky'


def get_price_data(figi, client, bool, bool1,  days):
    data = []
    # ffi = 'BBG00178PGX3'
    ffi = figi
    intr = CandleInterval.CANDLE_INTERVAL_HOUR
    if bool:
        intr = CandleInterval.CANDLE_INTERVAL_DAY
    for candle in client.get_all_candles(
            figi=ffi,
            from_=now() - timedelta(days=days),
            interval=intr,
    ):
        d = candle
        t = create_df(candle)
        data.append(t)
    if len(data) == 0:
        return 'NotData/Tinkoffofsy'
    k = pd.DataFrame(data)
    q = json.dumps(list(k['close']))
    if bool1:
        q = json.dumps(list( np.datetime_as_string(k['time'].values, unit='D')))
        # print(q)
    # return 0
    return q


def main1():
    fi = ['BBG004RVFFC0','BBG004731032','BBG0047315D0','BBG004731354', 'BBG004731354']
    ti = ['LKOH','ROSN','TATN','SNGS', 'NVTK']

    for ffi, tti in zip(fi, ti):
        with Client(TOKEN) as client:
            data = []
            for candle in client.get_all_candles(
                figi = ffi,
                from_=now() - timedelta(days=365 * 9 - 110),
                interval=CandleInterval.CANDLE_INTERVAL_DAY,
            ):
                t = create_df(candle)
                data.append(t)
            k = pd.DataFrame(data)
            k.to_csv(f'{tti}.csv')

            # print(k.shape[0]/14)
            # print(k.head(16))
            # k.plot(x='time', y='close', xlabel='LKOH')
            # plt.show()
            #

    return 0



def main2():
    with Client(TOKEN) as client:
        data = []
        a = client.instruments.shares(instrument_status = 0)
        a = list(a.instruments)
        for i in a:
            if i.ticker in ['LKOH','ROSN','TATN','SNGS', 'NVTK']:
                print(i.figi)
                # main2(i.figi, i.name)

            # data = pd.DataFrame(np.array(data), columns=['name, figi, uid'])

        #     data.to_csv('data.csv')
        # data = pd.read_csv('data.csv')
        print(data)


    return 0
import pystan
# from analyse import start
if __name__ == "__main__":
    # main()
    model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
    # start()
    model = pystan.StanModel(model_code=model_code)
    y = model.sampling().extract()['y']
    print(y)
    print(sum(y)/len(y))
    # pystanModel()

