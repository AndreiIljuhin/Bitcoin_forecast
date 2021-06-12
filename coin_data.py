import requests
import datetime
import time
import pytz
import numpy as np
import math

def getPrices():
    tz = pytz.timezone('UTC')
    date_prices = []
    end = int(math.floor(time.time() / 3600) * 3600)
    gap = 3600*24*10
    start = end - gap
    for i in range(0, 150):
        end_iso = datetime.datetime.fromtimestamp(end, tz).isoformat()
        start_iso = datetime.datetime.fromtimestamp(start, tz).isoformat()
        #datetime.datetime.fromtimestamp(start, tz).strftime('%H')
        print(end_iso)
        print(start_iso)
        
        #page_link = 'https://api.pro.coinbase.com/products/BTC-USD/candles?start=2021-06-05T12:00:00-00:00&end=2021-06-10T12:00:00-00:00&granularity=3600' # end
        # [ time, low, high, open, close, volume ],
        page_link = 'https://api.pro.coinbase.com/products/BTC-USD/candles?start='+start_iso+'&end='+end_iso+'&granularity=3600'
        page_link = page_link.replace('+', '-')
        #print(page_link)
        
        # adding the time of day field
        blocks = requests.get(page_link).text[2:-2].split('],[')
        for i in range(0, len(blocks)):
            block = blocks[i]
            fields = block.split(',')
            fields.insert(1, datetime.datetime.fromtimestamp(int(fields[0]), tz).strftime('%H'))
            blocks[i] = ','.join(fields)
            
        date_prices.append(blocks)
        
        timestamp_end = int(date_prices[-1][0].split(',')[0])
        timestamp_start = int(date_prices[-1][-1].split(',')[0])
        print(timestamp_end)
        print(timestamp_start)
        
        end = start - 3600
        start = end - gap
        
    return date_prices
    
prices = getPrices()
target = open('bitcoinprices.txt', 'w')
for strings in prices:
    for str in strings:
        target.write(str + '\n')