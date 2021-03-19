#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:47:41 2018

@author: igor
"""

import nn
import tele as t
import exchange as ex
import datetime as dt
import time
import params as p
import ccxt


def send(msg, public=False):
    print(msg)
    t.send_msg(str(msg), public)


def get_signal(conf):
    while True:
        td = nn.runModel(conf)
        signal = nn.get_signal(td)
        if dt.datetime.today() > signal['close_ts']:
            send('Signal has expired. Waiting for new one ...')
            time.sleep(p.sleep_interval)
        else:
            return signal


def send_results(x, res, msg):
    send(msg+' of '+str(res['filled'])+' '+p.pair+' with price '+str(res['average']))
    send('Balance: '+x.get_balance_str())


def execute(s):
    x = ex.Exchange()
    action = s['action']
    position = x.get_position()    
    is_open = (position == 'Buy' or position == 'Sell' and p.short)
    
    # Cancel all open orders
    x.cancel_orders()
    
    # Close position if new trade
    if is_open and action != position:
        res = x.close_position(position)
        send_results(x, res, 'Closed '+position+' Position')
        is_open = False
    
    # TODO: Open position with SL/TP and no wait
    # TODO: Handle partly open position
    if not is_open and (action == 'Buy' or action == 'Sell' and p.short):
        res = x.open_position(action)
        send_results(x, res, 'Opened '+action+' Position')
        is_open = True

    """ SL/TP can only be set AFTER order is executed if margin is not used """
    if is_open and (action == 'Buy' and p.buy_sl or p.short and action == 'Sell' and p.sell_sl): 
        res = x.stop_loss(action, s['sl_price'])
        send(res, True)

    if is_open and (action == 'Buy' and p.buy_tp or p.short and action == 'Sell' and p.sell_tp):
        res = x.take_profit(action, s['tp_price'])
        send(res, True)
        
    # Breakout Order
    if p.breakout and action == 'Sell':
        x.open_position('Buy', ordertype='stop-loss', price=s['sl_price'], wait=False)
        send('Breakout SL set at '+str(s['sl_price']), True)        


def run(conf, live=False):
    done = False
    while not done:
        try:
            s = get_signal(conf)
            send(nn.get_signal_str(s), True)
            if live:
                execute(s)
            done = True
        except ccxt.NetworkError as e:
            send('Network Error has occurred. Retrying ...')
            send(e)
            time.sleep(p.sleep_interval)

        
def test_execute():
    conf = 'ETHUSDNN'
    p.load_config(conf)
    s = get_signal(conf)
    p.order_size = 0.02
    s = {}
    s['action'] = 'Buy'
    s['new_trade'] = False
    s['sl'] = False
    s['tp'] = False
    s['sl_price'] = 100
    s['tp_price'] = 200
    
    execute(s)


# Trading
try:
    t.init()
    send('*** Maxi Model *** ', True)
    run('ETHBTCROC', live=True)

    send('*** Combo Model *** ', True)
    run('ETHUSDENS', live=True)

    send('*** Risk Model *** ', True)
    run('ETHUSDNN1')

    send('*** Moon Model *** ', True)
    run('ETHUSDNN2')

    send('*** Rock Model *** ', True)
    run('ETHUSDROC')
    run('BTCUSDROC')
except Exception as e:
    send('An error has occurred. Please investigate!')
    send(e)
    raise
finally:
    t.cleanup()
