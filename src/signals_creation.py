import ta
import pandas as pd

#-- Momentum Indicators

def get_rsi(df_records,lag=1,**params):
    signal_df=pd.DataFrame()
    for coin_pair in df_records["Open"].columns:
        coin_name=coin_pair.split("-")[0]
        signal_df[coin_name]=ta.momentum.rsi(close=df_records['Close'][coin_name], **params)
    return signal_df.shift(lag)