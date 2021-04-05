#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:40:08 2017

@author: imonahov
"""

import math


def truncate(n, digits):
    return math.trunc(n*(10**digits))/(10**digits)


def load_config(config):
    global conf
    conf = config
    global random_scale
    random_scale = 0.00001  # Defines standard deviation for random Q values
    global start_balance
    start_balance = 1.0
    global short
    short = False # Short trading
    global actions
    actions = 2 # Number of actions (% of stock holdings) 2 for long only, 3 to add short
    # α ∈ [0, 1] (alpha) is the learning rate used to vary the weight given to new experiences compared with past Q-values.
    global alpha
    alpha = 0.2
    # γ ∈ [0, 1] (gamma) is the discount factor used to progressively reduce the value of future rewards. Best: 0.9
    global gamma
    gamma = 0.9
    # Probability to chose random action instead of best action from Q Table. Best values: 0.2 - 0.5
    global epsilon
    epsilon = 0.5
    global train
    train = False # Train model
    global reload
    reload = False # Force to reload price data. False means reload only if data is old 
    global charts
    charts = True # Plot charts
    global stats
    stats = True # Show model stats
    global epochs
    epochs = 30 # Number of iterations for training (best 50)
    global features
    features = 4 # Number of features in state for Q table
    global feature_bins
    feature_bins = 3 # Number of bins for feature (more bins tend to overfit)
    global max_r
    max_r = 0
    global ticker
    ticker = conf[0:3]
    global currency
    currency = conf[3:6]
    global pair # Used for Exchange
    pair = ticker+'/'+currency
    global cfgdir
    cfgdir = 'data/'+conf
    global version
    version = 2 # Model version
    global sma_period
    sma_period = 50 # Best: 50 Alternative: 21, 55, 89, 144, 233 / 50, 100, 200
    global adr_period
    adr_period = 20 # Average Daily Return period
    global hh_period
    hh_period = 50 # Window for Highest High (best: 20 - 50)
    global ll_period
    ll_period = 50 # Window for Lowest Low (best: 20 - 50)
    global rsi_period
    rsi_period = 50
    global vol_period
    vol_period = 30
    global std_period
    std_period = 7
    global wil_period
    wil_period = 7
    global order_size # Order size in equity. 0 means to use order_pct
    order_size = 0
    global max_short # Max Order size for short position
    max_short = 0
    global order_pct # % of balance to use for position
    order_pct = 1
    global order_precision # Number of digits after decimal for order size
    order_precision = 2
    global result_size
    result_size = 0
    global order_wait # Wait time in seconds for order to be filled
    order_wait = 5*60
    global order_type # Order Type: market or limit 
    order_type = 'limit'
    global min_cash
    min_cash = 1
    global min_equity # Minimum order size
    min_equity = 0.001
    global bar_period
    bar_period = 'day' # Price bar period: day or hour
    global max_bars
    max_bars = 0 # Number of bars to use for training
    global train_goal
    train_goal = 'R' # Maximize Return
    global limit_fee # Exchange fee
    limit_fee = 0.0010 # Kraken Maker fee
    global market_fee # Kraken Taker fee
    market_fee = 0.0016
    global margin # Daily Margin fee for short positions
    margin = 0.0012 # Kraken 24 margin fee
    global margin_open # Kraken Margin Open fee
    margin_open = 0.0002
    global ratio
    ratio = 0 # Min ratio for Q table to take an action
    global units
    units = 16
    global train_pct
    train_pct = 0.8 # % of data used for training
    global test_pct
    test_pct = 0.2 # % of data used for testing
    global model
    model = cfgdir+'/model.nn'
    global test_bars # Number of bars to test. Overrides test_pct if > 0
    test_bars = 0
    global time_lag # Number of hours to offset price data. 0 means no offset
    time_lag = 0 # best 0: 3.49 4: 2.59 6: 1.6 7: 1.49 8: 2.71 20: 0.87
    global trade_interval
    trade_interval = 60*24 # Trade interval in minutes
    global sleep_interval
    sleep_interval = 60 # Bot sleep interval in seconds when waiting for new signal / network error
    global ignore_signals
    ignore_signals = None # list of y_pred_id to ignore. None to disable 
    global hold_signals # list of y_pred_id to HOLD. None to disable
    hold_signals = None
    global min_data_size # Minimum records expected from Cryptocompare API
    min_data_size = 100
    global take_profit # Take Profit % Default 1 which is no TP
    take_profit = 1
    global stop_loss
    stop_loss = 0
    global buy_sl # Enables SL for Buy
    buy_sl = False # Buy SL is disabled as not profitable
    global sell_sl # Enables SL for Sell
    sell_sl = False
    global buy_tp # Enables TP for Buy
    buy_tp = False
    global sell_tp # Enables TP for Sell
    sell_tp = False # Sell TP is disabled as cannot have both SL and TP on Kraken
    global leverage # Leverage used for margin trading. 0 means - no leverage
    leverage = 2
    global feature_list # List of features to use for NN (ordered by importance)
    feature_list = ['VOL','HH','LL','DR','MA','MA2','STD','RSI','WR','DMA','MAR'] 
#    features ordered by importance: ['RSI','MA','MA2','STD','WR','MAR','HH','VOL','LL','DMA','DR']
    global loss # Loss function for NN: mse, binary_crossentropy, mean_absolute_error etc
    loss = 'mse'
    global signal_threshold
    signal_threshold = 0.5
    global model_type # Model Type to run: NN, LSTM
    model_type = 'runNN'
    global price_precision # Number of decimals for price
    price_precision = 2
    global breakout # Use Breakout strategy
    breakout = False
    global signal_scale # Used for signal grouping by y_pred_id 
    signal_scale = 1000
    # Risk-off mode. More profitable if disabled in bull market. Enables ASR and ADX
    global adjust_signal
    adjust_signal = True
    global batch_size
    batch_size = 0
    global btc_data # Use asset price in BTC for model
    btc_data = False
    global position_sizing
    position_sizing = False
    # ADX period for trend strength
    global adx_period
    adx_period = 6
    # Min level of ADX to enter the trade
    global adx_lo_threshold
    adx_lo_threshold = 0  # Best for KR data: 40
    # Max level of ADX to exit the trade
    global adx_hi_threshold
    adx_hi_threshold = 1000  # Disabled, 75 is best for some models
    # Average strategy return period
    global asr_period
    asr_period = 10
    # Min ASR to exit the trade
    global asr_threshold
    asr_threshold = 0
    global roc_period
    global roc_threshold
    global signal_delay
    global datasource # Data Source for price data. Options cc: CryptoCompare, kr: Kraken, dr: DataReader, ql: Quandl
    # datasource = 'cc'
    datasource = 'kr'
    global models  # Defines list of models for Ensemble model
    global exchange
    # exchange = 'CCCAGG' # Average price from all exchanges
    exchange = 'KRAKEN'

    if conf == 'BTCUSD': # R: 180.23 SR: 0.180 QL/BH R: 6.79 QL/BH SR: 1.80
#        train = True
        max_r = 18
        version = 1
    elif conf == 'ETHUSD': # R: 6984.42 SR: 0.164 QL/BH R: 8.94 QL/BH SR: 1.30
#        6508 / 1.25
        max_r = 6508
#        train = True
#        epsilon = 0
    elif conf == 'BTCUSDLSTM':
#        model = cfgdir+'/model.top'
#        model = 'data/ETHUSDLSTM/model.nn'
        model_type = 'LSTM'
        signal_threshold = 1
#        short = True
        train = True
        train_pct = 1
        test_pct = 1
        units = 32
        epochs = 20
        limit_fee = 0.002 # Taker
        order_type = 'market'
        rsi_period = 50
    elif conf == 'ETHUSDLSTM':
# Accuracy: 0.57, Win Ratio: 0.68, Strategy Return: 1.77
#        train = True
#        train_pct = 1
#        test_pct = 1
#        test_bars = 365
#        test_pct = 1
        units = 32
        epochs = 20
        model_type = 'LSTM'
        signal_threshold = 1
        model = cfgdir+'/model.top'
        take_profit = 0.15  # Best TP 0.15: 1.77 No: 1.45
        limit_fee = 0.002 # Taker
        order_type = 'market'
        order_pct = 0.99 # Reserve 1% for slippage
#        !!! Short only in Bear market !!!
#        short = True
#        max_short = 250
    elif conf == 'ETHUSDLSTM1':
        train = True
        train_pct = 0.7
        test_pct = 0.3
#        test_pct = 1
        model_type = 'LSTM'
        units = 20
        epochs = 20
        signal_threshold = 1
    elif conf == 'ETHBTCNN':
        datasource = 'cc'
        feature_list = ['MA','MA2']
        reload = True
#        train = True
#        test_bars = 272
#        test_pct = 1
        model = cfgdir+'/model.top'
        units = 20
        epochs = 20
#        breakout = True
#        sell_sl = True
        limit_fee = 0.0008 # Maker
#        short = True
    elif conf == 'BTCUSDNN':
        breakout = True
        order_pct = 1
        short = True
        reload = True
#        train = True
#        test_pct = 1
        test_bars = 365
        units = 20
        epochs = 30
        model = cfgdir+'/model.top'
        limit_fee = 0.0008 # Maker
    elif conf == 'ETHUSDNN':
        buy_sl = True
        min_equity = 0.02
        order_precision = 0
        reload = True
#        train = True
        test_pct = 1
#         test_pct = 0.2
#         test_bars = 365
        units = 32
        epochs = 20
        model = cfgdir+'/model.215'
        order_type = 'market'
        # Estimated fees including slippage and margin
        limit_fee = 0.002
        market_fee = 0.004
    elif conf == 'BTCUSDROC':
        model_type = 'runNN3'
        reload = True
        adjust_signal = False
        limit_fee = 0.002
        market_fee = 0.004
        order_type = 'market'
        roc_period = 15
        roc_threshold = 0
        signal_delay = 4
    # ****************** Active Models ************************************************
    # !!! Do not tune Active models - use new conf for tuning !!!
    # !!! DO NOT trade Short unless you want to get REKT !!!
    # !!! Do not us breakout !!!
    # !!! Use Market order to avoid unfilled order losses
    # !!! Scaler will be updated when tuning is run
    elif conf == 'ETHUSDROC':
        model_type = 'runNN3'
        reload = True
        adjust_signal = False
        limit_fee = 0.002
        market_fee = 0.004
        order_type = 'market'
        roc_period = 17
        roc_threshold = -5
        signal_delay = 1
    elif conf in ['ETHUSDNN1','ETHUSDNN1S']:
        cfgdir = 'data/ETHUSDNN1'
        min_equity = 0.02
        order_precision = 0
        # Kraken data on CC is different from Kraken Exchange
        reload = True
        # train = True
        # train_pct = 0.8
        # test_pct = 0.2
        test_pct = 1
        # test_bars = 365
        units = 32
        epochs = 200
        batch_size = 100
        model = cfgdir+'/model.826'
        # Estimated fee including slippage and margin
        order_type = 'market'
        limit_fee = 0.002
        market_fee = 0.004
        signal_threshold = 1
        signal_scale = 100
        model_type = 'runNN1'
        btc_data = True
        feature_list = ['VOL', 'HH', 'LL', 'DR', 'MA', 'MA2', 'STD', 'RSI', 'WR', 'DMA', 'MAR', 'RSI_BTC', 'BTC/ETH']
        adx_period = 6
        # Safe-Mode
        if conf == 'ETHUSDNN1S':
            adx_lo_threshold = 40
            asr_threshold = 0.99
            buy_sl = True
    elif conf == 'ETHUSDNN2':
        min_equity = 0.02
        order_precision = 0
        reload = True
        # train = True
        test_pct = 1
        # test_bars = 365
        units = 32
        epochs = 30
        batch_size = 100
        # model = cfgdir + '/model.387'
        # adr_period = 14
        # Better model but does not improve Ensemble performance for some reason
        model = cfgdir + '/model.410'
        adr_period = 19
        order_type = 'market'
        limit_fee = 0.002
        # Estimated fee including slippage and margin
        market_fee = 0.004
        signal_threshold = 1
        signal_scale = 100
        model_type = 'runNN2'
        feature_list = [
            'DR',
            'ADR',
            'moon_lon_sin',
            'moon_lon_cos'
        ]
    elif conf == 'ETHUSDENS':
        min_equity = 0.02
        order_precision = 0
        order_type = 'market'
        limit_fee = 0.002
        market_fee = 0.004
        model_type = 'run_ensemble'
        signal_threshold = 0.5
        adjust_signal = False
        # models = ['ETHUSDNN1', 'ETHUSDNN1S', 'ETHUSDNN2']  # Best on full data, great at bear market, but overfit!
        models = ['ETHUSDNN1', 'ETHUSDROC']  # Best in last 720 days
    elif conf == 'ETHBTCROC':
        min_equity = 0.02
        order_precision = 0
        model_type = 'runNN3'
        reload = True
        adjust_signal = False
        limit_fee = 0.002
        market_fee = 0.004
        order_type = 'market'
        roc_period = 16
        roc_threshold = 11
        signal_delay = 6

    if order_type == 'market':
        limit_fee = market_fee
    
    global file
    file = cfgdir+'/price.pkl'
    global q
    q = cfgdir+'/q.pkl'
    global tl
    tl = cfgdir+'/tl.pkl'
    print('')
    print('**************** Loaded Config for '+conf+' ****************')
