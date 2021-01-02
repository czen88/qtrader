import numpy as np
import params as p
import pandas as pd
import stats as s
import matplotlib.pyplot as plt
import math
import datetime as dt
import talib


def get_stats(ds):
    def my_agg(x):
        names = {
            'SRAvg': x['SR'].mean(),
            'SRTotal': x['SR'].prod(),
            'DRAvg': x['DR'].mean(),
            'DRTotal': x['DR'].prod(),
            'Count': x['y_pred_id'].count()
        }

        return pd.Series(names)

    st = ds.groupby(ds['y_pred_id']).apply(my_agg)

    # Calculate Monthly Stats
    def my_agg(x):
        names = {
            'MR': x['DR'].prod(),
            'SR': x['SR'].prod()
        }

        return pd.Series(names)

    st_mon = ds.groupby(ds['date'].map(lambda x: x.strftime('%Y-%m'))).apply(my_agg)
    st_mon['CMR'] = np.cumprod(st_mon['MR'])
    st_mon['CSR'] = np.cumprod(st_mon['SR'])

    return st, st_mon


def get_stats_mon(ds):
    def my_agg(x):
        names = {
            'MR': x['DR'].prod(),
            'SR': x['SR'].prod()
        }

        return pd.Series(names)

    return ds.groupby(ds['date'].map(lambda x: x.strftime('%m'))).apply(my_agg)


def gen_trades(ds):
    def trade_agg(x):
        names = {
            'action': x.signal.iloc[0],
            'open_ts': x.date.iloc[0],
            'close_ts': x.date_to.iloc[-1],
            'open': x.open.iloc[0],
            'close': x.close.iloc[-1],
            'duration': x.date.count(),
            'sl': x.sl.max(),
            'tp': x.tp.max(),
            'high': x.high.max(),
            'low': x.low.min(),
            'mr': x.DR.prod(),
            'sr': x.SR.prod()
        }

        return pd.Series(names)

    tr = ds.groupby(ds.trade_id).apply(trade_agg)
    #    if not p.short: tr = tr[tr.action=='Buy']
    tr['win'] = (tr.sr > 1) | ((tr.sr == 1) & (tr.mr < 1))
    tr['CMR'] = np.cumprod(tr['mr'])
    tr['CSR'] = np.cumprod(tr['sr'])
    tr = tr.dropna()

    return tr


def plot_chart(df, title, date_col='date'):
    td = df.dropna().copy()
    td = td.set_index(date_col)
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    fig.autofmt_xdate()
    ax.plot(td['CSR'], color='g', label='Strategy Return')
    ax.plot(td['CMR'], color='r', label='Market Return')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()


def show_stats(td, trades):
    avg_loss = 1 - trades[trades.win == False].sr.mean()
    avg_win = trades[trades.win].sr.mean() - 1
    r2r = 0 if avg_loss == 0 else avg_win / avg_loss
    win_ratio = len(trades[trades.win]) / len(trades)
    trade_freq = len(trades) / len(td)
    adr = trade_freq * (win_ratio * avg_win - (1 - win_ratio) * avg_loss)
    exp = 365 * adr
    sr = math.sqrt(365) * s.sharpe_ratio((td.SR - 1).mean(), td.SR - 1, 0)
    srt = math.sqrt(365) * s.sortino_ratio((td.SR - 1).mean(), td.SR - 1, 0)
    dur = trades.duration.mean()
    slf = len(td[td.sl]) / len(td)
    tpf = len(td[td.tp]) / len(td)
    print('Strategy Return: %.2f' % td.CSR.iloc[-1])
    print('Market Return: %.2f' % td.CMR.iloc[-1])
    print('Sortino Ratio: %.2f' % srt)
    print('Bars in Trade: %.0f' % dur)
    print('Buy Pct: %.2f' % (len(td[td.signal == 'Buy']) / len(td)))
    print('Accuracy: %.2f' % (len(td[td.y_pred.astype('int') == td.Price_Rise]) / len(td)))
    print('Win Ratio: %.2f' % win_ratio)
    print('Avg Win: %.2f' % avg_win)
    print('Avg Loss: %.2f' % avg_loss)
    print('Risk to Reward: %.2f' % r2r)
    print('Expectancy: %.2f' % exp)
    print('Sharpe Ratio: %.2f' % sr)
    print('Average Daily Return: %.3f' % adr)
    print('SL: %.2f TP: %.2f' % (slf, tpf))


def run_pnl(td):
    bt = td[['date', 'open', 'high', 'low', 'close', 'signal']].copy()
    if p.position_sizing:
        bt['size'] = td['size']

    bt['ADX'] = talib.ADX(bt['high'].values, bt['low'].values, bt['close'].values, timeperiod=p.adx_period)
    bt = bt.dropna()

    # Calculate Pivot Points
    bt['PP'] = (bt.high + bt.low + bt.close) / 3
    bt['R1'] = 2 * bt.PP - bt.low
    bt['S1'] = 2 * bt.PP - bt.high
    bt['R2'] = bt.PP + bt.high - bt.low
    bt['S2'] = bt.PP - bt.high + bt.low
    bt['R3'] = bt.high + 2 * (bt.PP - bt.low)
    bt['S3'] = bt.low - 2 * (bt.high - bt.PP)
    bt['R4'] = bt.high + 3 * (bt.PP - bt.low)
    bt['S4'] = bt.low - 3 * (bt.high - bt.PP)

    # Calculate SL price
    bt['sl_price'] = np.where(bt.signal == 'Buy', bt.close.rolling(50).mean().shift(1), 0)
    bt['sl_price'] = np.where(bt.signal == 'Sell', bt.R2.shift(1), bt.sl_price)
    bt['sl'] = False
    if p.buy_sl:
        bt['sl'] = np.where((bt.signal == 'Buy') & (bt.sl_price <= bt.open) & (bt.sl_price >= bt.low), True, bt.sl)
    if p.sell_sl:
        bt['sl'] = np.where((bt.signal == 'Sell') & (bt.sl_price >= bt.open) & (bt.sl_price <= bt.high), True, bt.sl)

    # Calculate TP price
    bt['tp_price'] = np.where(bt.signal == 'Buy', bt.open * (1 + p.take_profit), 0)
    bt['tp_price'] = np.where(bt.signal == 'Sell', bt.open * (1 - p.take_profit), bt.tp_price)
    bt['tp'] = False
    if p.buy_tp:
        bt['tp'] = np.where((bt.signal == 'Buy') & (bt.tp_price >= bt.open) & (bt.tp_price <= bt.high), True, bt.tp)
    if p.sell_tp and p.short:
        bt['tp'] = np.where((bt.signal == 'Sell') & (bt.tp_price <= bt.open) & (bt.tp_price >= bt.low), True, bt.tp)
    bt['new_trade'] = (bt.signal != bt.signal.shift(1)) | bt.sl.shift(1) | bt.tp.shift(1)
    bt['trade_id'] = np.where(bt.new_trade, bt.index, np.NaN)
    bt['open_price'] = np.where(bt.new_trade, bt.open, np.NaN)
    bt = bt.fillna(method='ffill')

    # SL takes precedence over TP if both are happening in same timeframe
    bt['close_price'] = np.where(bt.tp, bt.tp_price, bt.close)
    bt['close_price'] = np.where(bt.sl, bt.sl_price, bt.close_price)

    # Rolling Trade Return
    bt['ctr'] = np.where(bt.signal == 'Buy', bt.close_price / bt.open_price, 1)
    if p.short: bt['ctr'] = np.where(bt.signal == 'Sell', 2 - bt.close_price / bt.open_price, bt.ctr)
    # Breakout: Buy if SL is triggered for Sell trade
    if p.breakout: bt['ctr'] = np.where((bt.signal == 'Sell') & bt.sl, bt.ctr * (bt.close / bt.sl_price), bt.ctr)

    # Margin Calculation. Assuming margin is used for short trades only
    bt['margin'] = 0
    if p.short:
        bt['margin'] = np.where(bt['signal'] == 'Sell', p.margin, bt.margin)
        bt['margin'] = np.where(bt.new_trade & (bt['signal'] == 'Sell'), p.margin + p.margin_open, bt.margin)

    bt['summargin'] = bt.groupby('trade_id')['margin'].transform(pd.Series.cumsum)

    # Rolling Trade Open and Close Fees
    bt['fee'] = p.limit_fee + bt.ctr * p.limit_fee
    bt['fee'] = np.where(bt.sl, p.limit_fee + bt.ctr * p.market_fee, bt.fee)
    if p.short:
        if p.breakout:
            bt['fee'] = np.where((bt.signal == 'Sell') & bt.sl, bt.fee + bt.ctr * (p.market_fee + p.limit_fee), bt.fee)
    else:
        if not p.breakout: bt['fee'] = np.where(bt.signal == 'Sell', 0, bt.fee)

    # Rolling Trade Return minus fees and margin
    bt['ctrf'] = bt.ctr - bt.fee - bt.summargin

    # Daily Strategy Return
    bt['SR'] = np.where(bt.new_trade, bt.ctrf, bt.ctrf / bt.ctrf.shift(1))
    bt['DR'] = bt['close'] / bt['close'].shift(1)

    if p.position_sizing:
        bt['SR'] = (bt['SR'] - 1) * bt['size'] + 1

    # Adjust signal based on past performance
    if p.adjust_signal:
        bt['signal'] = np.where(bt.ADX.shift(1) < p.adx_lo_threshold, 'Cash', bt.signal)
        bt['signal'] = np.where(bt.ADX.shift(1) > p.adx_hi_threshold, 'Cash', bt.signal)
        bt['SR'] = np.where(bt.signal == 'Cash', 1, bt.SR)
        bt['ASR'] = bt.SR.rolling(p.asr_period).mean().shift(1)
        bt['signal'] = np.where(bt.ASR < p.asr_threshold, 'Cash', bt.signal)
        bt['SR'] = np.where(bt.signal == 'Cash', 1, bt.SR)

    bt['CSR'] = np.cumprod(bt.SR)
    bt['CMR'] = np.cumprod(bt.DR)

    return bt


def run_backtest(td, file):
    bt = run_pnl(td)

    bt['y_pred'] = td.y_pred
    bt['y_pred_val'] = td.y_pred_val
    bt['y_pred_id'] = td.y_pred_id
    bt['Price_Rise'] = np.where(bt['DR'] > 1, 1, 0)
    bt['date_to'] = bt['date'].shift(-1)
    bt.iloc[-1, bt.columns.get_loc('date_to')] = bt.iloc[-1, bt.columns.get_loc('date')] + dt.timedelta(
        minutes=p.trade_interval)

    stats, stats_mon = get_stats(bt)
    #    bt = bt.merge(stats, left_on='y_pred_id', right_index=True, how='left')

    tr = gen_trades(bt)

    if p.charts: plot_chart(bt, file, 'date')
    if p.stats: show_stats(bt, tr)

    stats.to_csv(p.cfgdir + '/stats.csv')
    stats_mon.to_csv(p.cfgdir + '/stats_mon.csv')
    tr.to_csv(p.cfgdir + '/tr.csv')
    bt.to_csv(p.cfgdir + '/bt.csv')

    return bt


def get_return_stats(ds):
    plt.style.use('ggplot')
    ds['mon'] = ds['date'].dt.month
    ds['month'] = ds['date'].dt.strftime('%b')
    st = ds.groupby(['mon', 'month'])[['DR']].prod()
    st['DR'] = 100*(st['DR'] - 1)
    st = st.reset_index()
    plt.bar(st.month, st.DR, color='green')
    plt.ylabel("Total Return %")
    plt.title("ETH Total Return by Month")
    plt.show()

    ds['weekday'] = ds['date'].dt.weekday
    ds['weekday_name'] = ds['date'].dt.weekday_name
    st = ds.groupby(['weekday', 'weekday_name'])[['DR']].prod()
    st['DR'] = 100 * (st['DR'] - 1)
    st = st.reset_index()
    plt.bar(st.weekday_name, st.DR, color='green')
    plt.ylabel("Total Return %")
    plt.title("ETH Total Return by Week Day")
    plt.show()
