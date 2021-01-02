#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 20:28:17 2018

@author: igor
"""

import params as p
import backtest as bt
import talib
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras import backend as K
from keras.layers import Dense, LSTM, Activation, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
import pandas as pd
import datalib as dl
from joblib import dump, load
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def get_signal_str(s=''):
    if s == '': s = get_signal()
    txt = p.pair + ':'
    txt += ' NEW' if s['new_trade'] else ' Same' 
    txt += ' Signal: ' + s['action'] 
    if p.short and s['action'] == 'Sell': txt += ' SHORT'
    txt += ' Open: '+str(s['open'])
    if s['action'] != 'Cash': txt += ' P/L: '+str(s['pnl'])+'%'
    if s['tp']: txt += ' TAKE PROFIT!'
    if s['sl']: txt += ' STOP LOSS!'
    
    return txt 


def get_signal(offset=-1):
    s = td.iloc[offset]
    pnl = round(100*(s.ctrf - 1), 2)
    sl = p.truncate(s.sl_price, p.price_precision)
    tp = p.truncate(s.tp_price, p.price_precision)
    
    return {'new_trade':s.new_trade, 'action':s.signal, 
            'open':s.open, 'open_ts':s.date, 
            'close':s.close, 'close_ts':s.date_to, 'pnl':pnl, 
            'sl':s.sl, 'sl_price':sl, 'tp':s.tp, 'tp_price':tp}


def add_features(ds):
    ds['VOL'] = ds['volume']/ds['volume'].rolling(window = p.vol_period).mean()
    ds['HH'] = ds['high']/ds['high'].rolling(window = p.hh_period).max() 
    ds['LL'] = ds['low']/ds['low'].rolling(window = p.ll_period).min()
    ds['DR'] = ds['close']/ds['close'].shift(1)
    ds['MA'] = ds['close']/ds['close'].rolling(window = p.sma_period).mean()
    ds['MA2'] = ds['close']/ds['close'].rolling(window = 2*p.sma_period).mean()
    ds['STD']= ds['close'].rolling(p.std_period).std()/ds['close']
    ds['RSI'] = talib.RSI(ds['close'].values, timeperiod = p.rsi_period)
    ds['WR'] = talib.WILLR(ds['high'].values, ds['low'].values, ds['close'].values, p.wil_period)
    ds['DMA'] = ds.MA/ds.MA.shift(1)
    ds['MAR'] = ds.MA/ds.MA2
    ds['ADX'] = talib.ADX(ds['high'].values, ds['low'].values, ds['close'].values, timeperiod = p.adx_period)
    ds['Price_Rise'] = np.where(ds['DR'] > 1, 1, 0)

    ds = ds.dropna()
    
    return ds


def get_train_test(X, y):
    # Separate train from test
    train_split = int(len(X)*p.train_pct)
    test_split = p.test_bars if p.test_bars > 0 else int(len(X)*p.test_pct)
    X_train, X_test, y_train, y_test = X[:train_split], X[-test_split:], y[:train_split], y[-test_split:]
    
    # Feature Scaling
    # Load scaler from file for test run
#    from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
    scaler = p.cfgdir+'/sc.dmp'
    if p.train:
#        sc = QuantileTransformer(10)
#        sc = MinMaxScaler()
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        dump(sc, scaler)
    else:
        sc = load(scaler)
        # Uncomment if you need to upgrade scaler
        # dump(sc, scaler)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
        
    return X_train, X_test, y_train, y_test


def plot_fit_history(h):
    # Plot model history
    # Accuracy: % of correct predictions 
#    plt.plot(h.history['acc'], label='Train Accuracy')
#    plt.plot(h.history['val_acc'], label='Test Accuracy')
    plt.plot(h.history['loss'], label='Train')
    plt.plot(h.history['val_loss'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()


def train_model(X_train, X_test, y_train, y_test, file):
    print('*** Training model with '+str(p.units)+' units per layer ***')
    nn = Sequential()
    nn.add(Dense(units = p.units, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
    nn.add(Dense(units = p.units, kernel_initializer = 'uniform', activation = 'relu'))
    nn.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    cp = ModelCheckpoint(file, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    nn.compile(optimizer = 'adam', loss = p.loss, metrics = ['accuracy'])
    history = nn.fit(X_train, y_train, batch_size = 100, 
                             epochs = p.epochs, callbacks=[cp], 
                             validation_data=(X_test, y_test), 
                             verbose=0)

    # Plot model history
    plot_fit_history(history)

    # Load Best Model
    nn = load_model(file) 
    
    return nn


# TODO: Use Long / Short / Cash signals
def gen_signal(ds, y_pred_val):
    td = ds.copy()
    td = td[-len(y_pred_val):]
    td['y_pred_val'] = y_pred_val
    td['y_pred'] = (td['y_pred_val'] >= p.signal_threshold)
    td = td.dropna()

    td['y_pred_id'] = np.trunc(td['y_pred_val'] * p.signal_scale)
    td['signal'] = td['y_pred'].map({True: 'Buy', False: 'Sell'})
    if p.ignore_signals is not None:
        td['signal'] = np.where(np.isin(td.y_pred_id, p.ignore_signals), np.NaN, td.signal)
        td['signal'] = td.signal.fillna(method='ffill')
    if p.hold_signals is not None:
        td['signal'] = np.where(np.isin(td.y_pred_id, p.hold_signals), 'Cash', td.signal)
    if p.adjust_signal:
        td['signal'] = np.where(td.ADX < 26, 'Cash', td.signal)

    return td


# Inspired by:
# https://www.quantinsti.com/blog/artificial-neural-network-python-using-keras-predicting-stock-price-movement/
def runNN():
    global td
    global ds
    
    ds = dl.load_data()
    ds = add_features(ds)
    
    # Separate input from output. Exclude last row
    X = ds[p.feature_list][:-1]
#    y = ds[['DR']].shift(-1)[:-1]
    y = ds[['Price_Rise']].shift(-1)[:-1]

    # Split Train and Test and scale
    X_train, X_test, y_train, y_test = get_train_test(X, y)    
    
    K.clear_session() # Required to speed up model load
    if p.train:
        file = p.cfgdir+'/model.nn'
        nn = train_model(X_train, X_test, y_train, y_test, file)
    else:
        file = p.model
        nn = load_model(file) 
#        print('Loaded best model: '+file)
     
    # Making prediction
    y_pred_val = nn.predict(X_test)

    # Generating Signals
    td = gen_signal(ds, y_pred_val)

    # Backtesting
    td = bt.run_backtest(td, file)

    print(str(get_signal_str()))


def runNN1():
    global td
    global ds
    
    ds = dl.load_data()

    ds['VOL'] = ds['volume']/ds['volume'].rolling(window = p.vol_period).mean()
    ds['HH'] = ds['high']/ds['high'].rolling(window = p.hh_period).max() 
    ds['LL'] = ds['low']/ds['low'].rolling(window = p.ll_period).min()
    ds['DR'] = ds['close']/ds['close'].shift(1)
    ds['MA'] = ds['close']/ds['close'].rolling(window = p.sma_period).mean()
    ds['MA2'] = ds['close']/ds['close'].rolling(window = 2*p.sma_period).mean()
    ds['STD']= ds['close'].rolling(p.std_period).std()/ds['close']
    ds['RSI'] = talib.RSI(ds['close'].values, timeperiod = p.rsi_period)
    ds['WR'] = talib.WILLR(ds['high'].values, ds['low'].values, ds['close'].values, p.wil_period)
    ds['DMA'] = ds.MA/ds.MA.shift(1)
    ds['MAR'] = ds.MA/ds.MA2
    ds['ADX'] = talib.ADX(ds['high'].values, ds['low'].values, ds['close'].values, timeperiod = p.adx_period)
    ds['Price_Rise'] = np.where(ds['DR'] > 1, 1, 0)
    
    if p.btc_data:
        p.currency = 'BTC'
        p.kraken_pair = 'XETHXXBT'
        ds1 = dl.load_data()
        ds = ds.join(ds1, rsuffix='_btc')
        ds['RSI_BTC'] = talib.RSI(ds['close_btc'].values, timeperiod = p.rsi_period)
        p.feature_list += ['RSI_BTC']
    
    ds = ds.dropna()

    # Separate input from output. Exclude last row
    X = ds[p.feature_list][:-1]
    y = ds[['DR']].shift(-1)[:-1]

    # Split Train and Test and scale
    train_split = int(len(X)*p.train_pct)
    test_split = p.test_bars if p.test_bars > 0 else int(len(X)*p.test_pct)
    X_train, X_test, y_train, y_test = X[:train_split], X[-test_split:], y[:train_split], y[-test_split:]
    
    # Feature Scaling
#    from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
    scaler = p.cfgdir+'/sc.dmp'
    scaler1 = p.cfgdir+'/sc1.dmp'
    if p.train:
#        sc = QuantileTransformer(10)
#        sc = MinMaxScaler()
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        dump(sc, scaler)
 
        sc1 = MinMaxScaler()
        y_train = sc1.fit_transform(y_train)
        y_test = sc1.transform(y_test)
        dump(sc1, scaler1)

    else:
        sc = load(scaler)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

        sc1 = load(scaler1)
        y_train = sc1.transform(y_train)
        y_test = sc1.transform(y_test)
    
    K.clear_session() # Required to speed up model load
    if p.train:
#        Custom Loss Function
#        def stock_loss(t, p):
#            loss = K.switch(K.less((t-1)*(p-1), 0), K.abs(t-p), 0.1*K.abs(t-p))
#            return K.mean(loss, axis=-1)
#        p.loss = stock_loss
    
        file = p.cfgdir+'/model.nn'
        print('*** Training model with '+str(p.units)+' units per layer ***')
        nn = Sequential()
        nn.add(Dense(units = p.units, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
        nn.add(Dense(units = p.units, kernel_initializer = 'uniform', activation = 'relu'))
        nn.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))

        cp = ModelCheckpoint(file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        nn.compile(optimizer = 'adam', loss = p.loss, metrics = ['accuracy'])
        history = nn.fit(X_train, y_train, batch_size = len(X_train) if p.batch_size == 0 else p.batch_size,
                             epochs = p.epochs, callbacks=[cp], 
                             validation_data=(X_test, y_test), 
                             verbose=0)

        # Plot model history
        plot_fit_history(history)

        # Load Best Model
#        nn = load_model(file, custom_objects={'stock_loss': stock_loss}) 
        nn = load_model(file) 
    else:
        file = p.model
        nn = load_model(file) 
     
    # Making prediction
    y_pred_val = nn.predict(X_test)
    y_pred_val = sc1.inverse_transform(y_pred_val)

    # Generating Signals
    td = gen_signal(ds, y_pred_val)

    # Backtesting
    td = bt.run_backtest(td, file)
    print(str(get_signal_str()))


# See: 
# https://towardsdatascience.com/predicting-ethereum-prices-with-long-short-term-memory-lstm-2a5465d3fd
def runLSTM():
    global ds
    global td

    ds = dl.load_data()
    ds = add_features(ds)
   
    lag = 3
    n_features = 1
    X = pd.DataFrame()
    for i in range(1, lag+1):
        X['RSI'+str(i)] = ds['RSI'].shift(i)
#        X['MA'+str(i)] = ds['MA'].shift(i)
#        X['VOL'+str(i)] = ds['VOL'].shift(i)
    X = X.dropna()
    
    y = ds['DR']

    X_train, X_test, y_train, y_test = get_train_test(X, y) 

    X_train_t = X_train.reshape(X_train.shape[0], lag, n_features)
    X_test_t = X_test.reshape(X_test.shape[0], lag, n_features)

    file = p.model
    if p.train:
        file = p.cfgdir+'/model.nn'
        nn = Sequential()
        nn.add(LSTM(p.units, input_shape=(X_train_t.shape[1], X_train_t.shape[2]), return_sequences=True))
        nn.add(Dropout(0.2))
        nn.add(LSTM(p.units, return_sequences=False))
        nn.add(Dense(1))
        
        optimizer = RMSprop(lr=0.005, clipvalue=1.)
#        optimizer = 'adam'
        nn.compile(loss=p.loss, optimizer=optimizer)
        
        cp = ModelCheckpoint(file, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        h = nn.fit(X_train_t, y_train, batch_size = 10, epochs = p.epochs, 
                             verbose=0, callbacks=[cp], validation_data=(X_test_t, y_test))

        plot_fit_history(h)
    
    # Load Best Model
    nn = load_model(file)

    y_pred = nn.predict(X_test_t)    
    td = gen_signal(ds, y_pred)

    # Backtesting
    td = bt.run_backtest(td, file)
    
    print(str(get_signal_str()))


def runModel(conf):
    p.load_config(conf)
    globals()[p.model_type]()
        

def check_missing_dates(td):
    from datetime import timedelta
    date_set = set(td.date.iloc[0] + timedelta(x) for x in range((td.date.iloc[-1] - td.date.iloc[0]).days))
    missing = sorted(date_set - set(td.date))
    print(missing)


# Tuning
#runModel('BTCUSDNN')
#runModel('ETHBTCNN')
# Using ETHBTC data
#runModel('ETHUSDNN3')

# Best SR / Less Sortino / Worse on Kraken Data
#runModel('ETHUSDNN2')

# runModel('ETHUSDNN1')

# PROD Year SR: 3.61 CCCAGG: 9747, Kraken: 5589
runModel('ETHUSDNN')
ds.to_csv('ds.csv')
