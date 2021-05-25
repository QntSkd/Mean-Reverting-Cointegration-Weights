import pandas as pd
import numpy as np
import finnhub
import datetime
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class Cointegration(object):
    def __init__(self, key, frequency, start, end):
        self.key = key
        self.frequency = frequency
        self.start = int(datetime.datetime.timestamp(datetime.datetime.strptime(start, '%Y-%m-%d')))
        self.end = int(datetime.datetime.timestamp(datetime.datetime.strptime(end, '%Y-%m-%d')))

    def api_call(self, ticker):
        client = finnhub.Client(api_key=self.key)
        price_data = client.stock_candles(ticker, self.frequency, self.start, self.end)
        df = pd.DataFrame({'Timestamp':price_data['t'],
                           'Open':price_data['o'],
                           'High':price_data['h'],
                           'Low':price_data['l'],
                           'Close':price_data['c'],
                           'Volume':price_data['v']
                           })
        df = df.set_index('Timestamp')
        return df

    def johansen_test(self, ticker1, ticker2):
        returns_1 = self.api_call(ticker1)['Close'].values
        returns_2 = self.api_call(ticker2)['Close'].values
        df = pd.DataFrame({ticker1:returns_1, ticker2:returns_2})
        results = coint_johansen(df,0,1)
        trace = results.trace_stat[0]
        crit_val_95th = results.trace_stat_crit_vals[0][1]
        if trace < crit_val_95th:
            return "Not Cointegrated"
        else:
            weights = results.evec[:, 0]
            return f"{ticker1} weight: {weights[0]}, {ticker2} weight: {weights[1]}"

run = Cointegration('<ENTER FINNHUB API KEY>','D','2020-01-01','2021-05-25')
print(run.johansen_test("FTGC","ENVA"))
