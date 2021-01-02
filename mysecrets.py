#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:28:47 2018

@author: igor

This file should NOT be pushed to remote GIT repo
"""

import boto3
ssm = boto3.client('ssm')


def get_secret(param):
    p = ssm.get_parameter(Name=param, WithDecryption=True)
    return p['Parameter']['Value']


exchange_api_key = get_secret('/qtrader/prod/exchange_api_key')
exchange_sk = get_secret('/qtrader/prod/exchange_secret')
exchange_pass = ''
telegram_token = get_secret('/qtrader/prod/telegram_token')
# My Telegram Chat Id
telegram_chat_id = 498230493
# Rahul Telegram Chat Id
telegram_chat_id1 = 716893261
cryptocompare_key = get_secret('/qtrader/prod/cryptocompare_key')
quandl_key = get_secret('/qtrader/prod/quandl_key')
