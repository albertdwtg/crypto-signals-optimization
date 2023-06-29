import ta
import pandas as pd
from typing import Tuple
from src.normalizations import apply_normalizations, convert_to_weights

def get_returns_data(historic_data: dict) -> pd.DataFrame:
    """Function that creates a returns matrix from historic data

    Args:
        historic_data (dict): dict of dataframes

    Returns:
        pd.DataFrame: returns dataframes
    """
    returns = historic_data["Close"].pct_change()
    return returns

def train_test_split(historic_data: dict, train_ratio: float) -> Tuple[dict,dict]:
    """Function that applies a train test split on all dataframes of a dict of dataframes

    Args:
        historic_data (dict): dict of dataframes
        train_ratio (float): ratio of the training sample (between 0 and 1)

    Returns:
        Tuple[dict,dict]: two dic of dataframes, one for training an the other for testing
    """
    train_data = {}
    test_data = {}
    for key in historic_data:
        nb_rows = int(train_ratio * len(historic_data[key]))
        train_data[key] = historic_data[key][:nb_rows]
        test_data[key] = historic_data[key][nb_rows:]
    return train_data, test_data

#-- Momentum Indicators
def get_rsi(df_records:dict, lag:int = 1, normalization_choice:int = 1, **params):
    signal_df=pd.DataFrame()
    for coin_pair in df_records["Open"].columns:
        coin_name=coin_pair.split("-")[0]
        signal_df[coin_name]=ta.momentum.rsi(close=df_records['Close'][coin_name], **params)
    signal_df = apply_normalizations(signal_df, normalization_choice)
    return signal_df.shift(lag)

def compute_signal(signal_name: str, historic_data: dict, **params) -> pd.DataFrame:
    lag = 1
    if "lag" in params:
        lag = params["lag"]
        del params["lag"]
    
    normalization_choice = 1
    if "normalization_choice" in params:
        normalization_choice = params["normalization_choice"]
        del params["normalization_choice"]
    
    signal = pd.DataFrame()
    if signal_name.lower() == "rsi":
        signal = get_rsi(historic_data, lag, normalization_choice, **params)
    
    signal_weighted = convert_to_weights(signal)
    return signal