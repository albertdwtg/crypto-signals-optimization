import requests
import pandas as pd
from binance import Client
from binance.enums import HistoricalKlinesType
import datetime
from typing import List
import os

def get_futures_symbols(status: str, coin_pair: str) -> List[str]:
    """Function that returns a list of coins names

    Args:
        status (str): status of the coin on api binance. ex: TRADING
        coin_pair (str): pair of the coin we want to collect. ex: USDT

    Returns:
        List[str]: list of coins names
    """
    base = 'https://fapi.binance.com'
    endpoint = f'{base}/fapi/v1/exchangeInfo'
    
    params = {}
    result = requests.get(endpoint, params=params).json()

    usdt = []
    for symb in result['symbols']:
        if symb['status'] == 'TRADING':
            if symb['symbol'].endswith('USDT'):
                usdt.append(symb['symbol'])
                
    #-- we put BTCUSDT in the first place       
    usdt.remove("BTCUSDT")
    usdt.insert(0,"BTCUSDT")
    return usdt

def collect_historic_data(coins: List[str], start_date: str) -> dict:
    """Function that collects historical data of a list of coins

    Args:
        coins (List[str]): List of coins names. ex: BTCUSDT
        start_date(str): date where to start the historic
        
    Returns:
        dict: dict of dataframes containing data
    """
    historic={}
    historic["High"]=pd.DataFrame()
    historic["Volume"]=pd.DataFrame()
    historic["Low"]=pd.DataFrame()
    historic["Open"]=pd.DataFrame()
    historic["Close"]=pd.DataFrame()
    for i in coins:
        try: 
            klinesT = Client().get_historical_klines(i, Client.KLINE_INTERVAL_1DAY, start_date,
                                                    klines_type=HistoricalKlinesType.FUTURES)
            df = pd.DataFrame(klinesT, columns=[
                            'timestamp', 'Open', 'High', 'Low', 'Close',
                            'Volume', 'close_time', 'quote_av', 'trades',
                            'tb_base_av', 'tb_quote_av', 'ignore'])
            df.index = df['timestamp'].apply(
                            lambda x: datetime.datetime.fromtimestamp(x / 1000))
            df=df[["Open","Low","Close","High","Volume"]]
            cols=df.columns
            name=i[:-4]
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            historic["Open"][name]=df["Open"]
            historic["Close"][name]=df["Close"]
            historic["Volume"][name]=df["Volume"]
            historic["High"][name]=df["High"]
            historic["Low"][name]=df["Low"]
        except:
            continue
    return historic

def save_df_to_parquet(df: pd.DataFrame, path: str, filename: str) -> str:
    """
    Function that saves a dataframe to a parquet file

    Args:
        df (pd.DataFrame): dataframe to save in parquet file
        path (string): location of the parquet file
    Returns:
        str: name of the file
    """
    if not os.path.exists(path):
        os.makedirs(path)
    name = filename + ".parquet"
    target_path = os.path.join(path, name)  
    df.to_parquet(target_path)
    return name

def collect_coins_data(start_date: str, coin_pair: str, coin_status: str, data_folder_name: str) -> dict:
    """Function that collects historical data of coins

    Args:
        start_date (str): date where to start the historic
        coin_pair (str): pair of the coin we want to collect. ex: USDT
        coin_status (str): status of the coin on api binance. ex: TRADING
        data_folder_name (str): name of the folder wher we will save data
    Returns:
        dict: dict of dataframes containing data
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(root_dir)
    data_folder_path = os.path.join(root_dir, data_folder_name)
    print(data_folder_path)
    coins_names = get_futures_symbols(coin_status, coin_pair)
    
    historic = collect_historic_data(coins_names, start_date)
    for key, value in historic.items():
        print(key)
        save_df_to_parquet(value,data_folder_path, key)
    return historic