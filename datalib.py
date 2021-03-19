#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:37:56 2018

@author: igor
"""

import requests
import mysecrets as s
import datetime as dt
import params as p
import os
import pickle
import pandas as pd
import numpy as np
from flatlib.datetime import Datetime
from flatlib.geopos import GeoPos
from flatlib.chart import Chart
import flatlib.const as flc


def load_data_cc(ticker, currency):
    retry = True
    while retry: # This is to avoid issue when only 31 rows are returned
        r = requests.get('https://min-api.cryptocompare.com/data/histo'+p.bar_period
                         +'?fsym='+ticker+'&tsym='+currency
                         +'&allData=true&e='+p.exchange
                         +'&api_key='+s.cryptocompare_key)
        df = pd.DataFrame(r.json()['Data'])
        if len(df) > p.min_data_size: 
            retry = False
        else:
            print("Incomplete price data. Retrying ...")
    df = df.rename(columns={'volumefrom': 'volume'})
    
    return df


def load_data_kr(ticker, currency):
    pairs = {
        'ETHUSD': 'XETHZUSD',
        'BTCUSD': 'XXBTZUSD',
        'ETHBTC': 'XETHXXBT'
    }
    pair = pairs[ticker+currency]
    url = 'https://api.kraken.com/0/public/OHLC?pair='+pair+'&interval=1440'
    df = pd.DataFrame(requests.get(url).json()['result'][pair])
    df.columns = ['time','open','high','low','close','vwap','volume','count']
    df = df[['time','open','high','low','close','volume']]
    df = df.apply(pd.to_numeric)
    
    return df


# Load Historical Price Data from Cryptocompare
# API Guide: https://medium.com/@agalea91/cryptocompare-api-quick-start-guide-ca4430a484d4
def load_data(ticker, currency, file=''):
    if file == '':
        file = p.file

    now = dt.datetime.today().strftime('%Y-%m-%d')
    if (not p.reload) and os.path.isfile(file):
        df = pickle.load(open(file, "rb"))
        # Return loaded price data if it is up to date
        if df.date.iloc[-1].strftime('%Y-%m-%d') == now:
            print('Using loaded prices for ' + now)
            return df
    
    if p.datasource == 'cc':
        df = load_data_cc(ticker, currency)
    elif p.datasource == 'kr':
        df = load_data_kr(ticker, currency)
    else:
        print('Invalid Data Source: '+p.datasource)

    df = df.set_index('time')
    df = df.sort_index()
    df['date'] = pd.to_datetime(df.index, unit='s')

    os.makedirs(os.path.dirname(file), exist_ok=True)
    pickle.dump(df, open(file, "wb"))
    print('Loaded '+ticker+currency+' prices from '+p.exchange+' via '+p.datasource
          + ' Rows:' + str(len(df))+' Date:'+str(df.date.iloc[-1]))
    print('Last complete '+p.bar_period+' close: '+str(df.close.iloc[-2]))

    if p.max_bars > 0: df = df.tail(p.max_bars).reset_index(drop=True)

    return df


def get_calendar(start, end):
    dates = pd.date_range(start=start, end=end)
    calendar = []
    for d in dates:
        date = Datetime(d.strftime("%Y/%m/%d"), '12:00', '+00:00')
        pos = GeoPos('51n23', '0w18')
        chart = Chart(date, pos)
        moon = chart.getObject(flc.MOON)
        sun = chart.getObject(flc.SUN)
        mercury = chart.getObject(flc.MERCURY)
        venus = chart.getObject(flc.VENUS)
        mars = chart.getObject(flc.MARS)
        calendar.append({
            'date': d,
            'moon_lon': int(moon.lon),
            'sun_lon': int(sun.lon),
            'mercury_lon': int(mercury.lon),
            'venus_lon': int(venus.lon),
            'mars_lon': int(mars.lon)
        })

    calendar = pd.DataFrame(calendar)
    return calendar


def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data
