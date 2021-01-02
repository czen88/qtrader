#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Kraken API
# https://github.com/dominiktraxl/pykrakenapi

# CCXT API
# https://github.com/ccxt/ccxt/wiki/Manual#overriding-unified-api-params

"""
Created on Mon Dec 25 18:06:07 2017

@author: imonahov
"""
import ccxt
import time
import params as p
import mysecrets as s


class Exchange:
    """
    Executes Market Order on exchange
    Example: Buy BTC with 100 EUR
    order = market_order('buy', 'BTC', 'EUR', 100)
    Example: Sell 0.0001 BTC
    order = market_order('sell', 'BTC', 'EUR', 0.0001)
    """
    ex = None

    def __init__(self):
        self.ex = ccxt.kraken({
            #  'verbose': True,
            'apiKey': s.exchange_api_key,
            'secret': s.exchange_sk,
            'timeout': 20000,
            # 'session': cfscrape.create_scraper(), # To avoid Cloudflare block => still fails with 520 Origin Error
            'enableRateLimit': True,
            'rateLimit': 1000  # Rate Limit set to 1 sec to avoid issues
        })
        self.ex.load_markets()

    # Returns current price
    def get_price(self, item='last'):
        ticker = self.ex.fetch_ticker(p.pair)
        return ticker[item]

    def get_ticker(self):
        ticker = self.ex.fetch_ticker(p.pair)
        return ticker

    def get_balance(self, asset=''):
        if asset == '': asset = p.currency
        balance = self.ex.fetch_balance()['total']
        return balance[asset]

    def get_balance_str(self):
        balance = self.ex.fetch_balance()['total']
        return p.currency+': '+str(balance[p.currency])+', '+p.ticker+': '+str(balance[p.ticker])

    def get_total_value(self):
        bal = self.ex.fetch_balance()['total']
        amt = 0
        for c in bal:
            if c == 'USD' or bal[c] == 0: price = 1
            else: price = self.ex.fetch_ticker(c+'/USD')['last']

            amt = amt + bal[c] * price

        return p.truncate(amt, 2)

    def fetchOrder(self, order_id):
        order = {}
        try:
            order = self.ex.fetchOrder(order_id)
        except Exception as e:
            print(e)

        return order

    def wait_order(self, order_id):
        print('Waiting for order ' + order_id + ' to be executed ...')
        while True:
            order = self.fetchOrder(order_id)
            if order != {} and order['status'] in ['closed', 'canceled', 'expired']:
                print('***** Order ' + order['status'] + ' *****')
                print(order)
                return order
            time.sleep(p.order_wait)

    def create_order(self, side, amount=0, price=0, ordertype='', leverage=1, wait=True):
        params = {}
        if ordertype == '': ordertype = p.order_type
        if leverage > 1: params['leverage'] = leverage
        # TODO: Use 0% if price is better than order price to avoid market order
        if price == 0 and ordertype == 'limit': params['price'] = '#0%'

        order = self.ex.create_order(p.pair, ordertype, side, amount, price, params)
        order = self.fetchOrder(order['id'])
        print('***** Order Created *****')
        print(order)

        # Wait till order is executed
        if wait: order = self.wait_order(order['id'])

        return order


    #def get_order_price(order_type):
    #    orders = self.ex.fetchClosedOrders(p.pair)
    #    return orders[0]['info']['price']

    def get_order_size(self, action, price=0):
        # Calculate position size based on portfolio value %
        if price == 0: price = self.get_price()
        amount = self.get_balance() * p.order_pct
        size = p.truncate(amount/price, p.order_precision)

        # Applying order size limit
        if p.order_size > 0: size = min(size, p.order_size)
        if action == 'Sell' and p.max_short > 0: size = min(size, p.max_short)
        return size

    def close_position(self, action, price=0, ordertype='', wait=True):
        res = {}
        if ordertype == '': ordertype = p.order_type

        if action == 'Sell':
            res = self.create_order('buy', 0, price, ordertype, p.leverage, wait)
        elif action == 'Buy':
            amount = self.get_balance(p.ticker)
            res = self.create_order('sell', amount, price, ordertype, 1, wait)

        return res

    def open_position(self, action, price=0, ordertype='', wait=True):
        res = {}
        amount = self.get_order_size(action, price)
        if amount == 0:
            raise Exception('Not enough funds to open position')
        lot = amount
        side = action.lower()

        if action == 'Sell':
            leverage = p.leverage
        elif action == 'Buy':
            leverage = 1
        else:
            raise Exception('Invalid action provided: '+action)

        while lot >= p.min_equity:
            try:
                res = self.create_order(side, lot, price, ordertype, leverage, wait)
                print('Created order of size '+str(lot))
                if lot == amount: break # Position opened as expected
            except ccxt.InsufficientFunds:
                lot = p.truncate(lot/2, p.order_precision)
                print('Insufficient Funds. Reducing order size to '+str(lot))

        return res

    def take_profit(self, action, price):
        ticker = self.get_ticker()
        if action == 'Buy' and price >= ticker['ask'] or action == 'Sell' and price <= ticker['bid']:
            self.close_position(action, ordertype='take-profit', price=price, wait=False)
            return 'TP set at %s' % price
        return 'TP is not set'

    def stop_loss(self, action, price):
        ticker = self.get_ticker()
        if action == 'Buy' and price <= ticker['bid'] or action == 'Sell' and price >= ticker['ask']:
            self.close_position(action, ordertype='stop-loss', price=price, wait=False)
            return 'SL set at %s' % price
        return 'SL is not set'

    def has_orders(self, types=[]):
        if types == []: types = [p.order_type]
        for order in self.ex.fetchOpenOrders(p.pair):
            if order['type'] in types: return True
        return False

    def wait_orders(self, types=[]):
        if types == []: types = [p.order_type]
        for order in self.ex.fetchOpenOrders(p.pair):
            if order['type'] in types: self.wait_order(order['id'])

    def has_sl_order(self):
        return self.has_orders(['stop-loss'])

    def has_tp_order(self):
        return self.has_orders(['take-profit'])

    def get_position(self):
        if self.get_balance(p.ticker) > p.min_equity: return 'Buy'
        if not p.short: return 'Sell'

        # Check short position
        res = self.ex.privatePostOpenPositions()
        if len(res['result']) > 0: return 'Sell'

        return 'Cash'

    def cancel_orders(self, types=[]):
        for order in self.ex.fetchOpenOrders(p.pair):
            if types == [] or order['type'] in types:
                print("Cancelling Order:")
                print(order)
                self.ex.cancelOrder(order['id'])

    def cancel_sl(self):
        self.cancel_orders(['stop-loss'])

    def cancel_tp(self):
        self.cancel_orders(['take-profit'])

    def test_order1(self):
        p.load_config('ETHUSDNN')
        p.order_size = 0.02
        # Print available API methods
        print(dir(self.ex))

        # Buy
        self.ex.fetch_balance()['total']
        # Close SL Order
        self.cancel_orders()

        self.ex.fetchOpenOrders()

        self.ex.fetchClosedOrders('ETH/USD')

        # Get Open Positions
        self.ex.privatePostOpenPositions()

        # Limit Order with current price
        self.create_order('Buy', 'limit', 0.02, {'price':'+0%'})

        self.ex.createOrder('ETH/USD', 'market', 'buy', 0.02)

    def test_order2(self):
        p.load_config('ETHUSDNN')

        # Create Market Order
        self.ex.createOrder('ETH/USD', 'market', 'buy', 0.02)
        self.ex.createOrder('ETH/USD', 'market', 'sell', 0.02)
        self.ex.createOrder('ETH/USD', 'market', 'buy', 0.02, 0) # Price is ignored

        # Create Limit Order for fixed price
        self.ex.createOrder('ETH/USD', 'limit', 'buy', 0.02, 100)
        # Create Limit Order for -1% to market price
        self.ex.createOrder('ETH/USD', 'limit', 'buy', 0.02, 0, {'price':'-1%'})

        # Fetch Open Orders
        orders = self.ex.fetchOpenOrders()
        # Order Size
        orders[0]['amount']

        self.ex.fetchBalance()['ETH']

    def test_order3(self):
        p.load_config('ETHUSDNN')
        p.order_size = 0.02
        p.order_wait = 10
        self.open_position('Buy')
        print(self.get_balance())

        res = self.take_profit('Buy', 200)
        res = self.stop_loss('Buy', 100)
        res = self.close_position('Buy', wait=False)
        self.get_balance('ETH')
        self.ex.fetchOpenOrders()
        self.cancel_sl()
        self.cancel_tp()
        self.cancel_orders()
        self.get_price()

        res = self.ex.privatePostOpenPositions()
        len(res['result'])
        self.open_position('Sell')
        self.close_position('Sell', wait=False)
        self.ex.fetchOpenOrders()
        self.get_price()
        self.create_order('buy', 10, 215.19, 'stop-loss', 1, False)
