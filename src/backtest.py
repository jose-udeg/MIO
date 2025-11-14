# backtest.py
import numpy as np
import pandas as pd
import os
from src import config

def simple_backtest(model, tokenizer, X_text_seq, X_num, timestamps, prices_csv=config.PRICES_CSV, prob_thresh=0.6):
    # model: ya cargado; tokenizer disponible si usas texto tokenizado; aquí asumimos inputs listos
    # timestamps: array de timestamps correspondientes a las muestras
    prices = pd.read_csv(prices_csv, parse_dates=[config.TIME_COL])
    preds = model.predict({'text_input': X_text_seq, 'num_input': X_num}, batch_size=64)
    prob_up = preds[:, 2]
    prob_down = preds[:, 0]

    trades = []
    for i, t in enumerate(timestamps):
        t0 = pd.to_datetime(t)
        row0 = prices[prices[config.TIME_COL] >= t0]
        if row0.empty: 
            continue
        p0 = float(row0.iloc[0]['close'])
        if prob_up[i] > prob_thresh:
            t_exit = t0 + pd.Timedelta(hours=config.WINDOW_HOURS)
            row_exit = prices[prices[config.TIME_COL] >= t_exit]
            if row_exit.empty: continue
            p_exit = float(row_exit.iloc[0]['close'])
            ret = (p_exit - p0) / p0
            trades.append({'t0': t0, 'side':'long','ret':ret})
        elif prob_down[i] > prob_thresh:
            t_exit = t0 + pd.Timedelta(hours=config.WINDOW_HOURS)
            row_exit = prices[prices[config.TIME_COL] >= t_exit]
            if row_exit.empty: continue
            p_exit = float(row_exit.iloc[0]['close'])
            ret = (p0 - p_exit) / p0
            trades.append({'t0': t0, 'side':'short','ret':ret})
    if len(trades) == 0:
        print("No trades generados con thresh", prob_thresh)
        return None
    df = pd.DataFrame(trades)
    total = df['ret'].sum()
    avg = df['ret'].mean()
    win = (df['ret']>0).mean()
    print("Trades:", len(df), "Total ret:", total, "Avg:", avg, "Winrate:", win)
    return df

if __name__ == "__main__":
    print("Ejecuta backtest desde evaluate.py o un notebook tras cargar modelo y datos procesados.")
