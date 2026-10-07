import os
import io
import time
import requests
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import date, datetime, timezone
import numpy as np
import pandas as pd
import pandas_ta as ta
from tvDatafeed import TvDatafeed, Interval
from jugaad_data.nse import bhavcopy_save
from py_vollib.black_scholes.implied_volatility import implied_volatility
import yfinance as yf

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================
TARGET_DATE = date.today()        # Automatically uses the GitHub execution date
EXPIRY_DATE = date(2026, 10, 29)  # UPDATE THIS MONTHLY to the next F&O Expiry
RISK_FREE_RATE = 0.07             
BHAVCOPY_DIR = "/tmp"             

tv = TvDatafeed()

# =====================================================================
# CORE FUNCTIONS
# =====================================================================
def get_live_fo_watchlist() -> list[str]:
    """Downloads the official F&O Market Lots CSV directly from NSE."""
    url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(response.text), skipinitialspace=True)
        df = df.dropna(subset=['SYMBOL'])
        
        symbols = df['SYMBOL'].astype(str).str.strip().unique().tolist()
        indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
        
        return [sym for sym in symbols if sym not in indices and sym.upper() != 'SYMBOL']
    except Exception as e:
        print(f"Failed to fetch live F&O list: {e}")
        return []

def get_historical_volatility(close_prices: pd.Series, window: int = 20) -> pd.Series:
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252) * 100

def get_days_to_earnings(ticker: str) -> int:
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        cal = stock.get_earnings_dates(limit=8)
        if cal is not None and not cal.empty:
            now = datetime.now(timezone.utc)
            future_dates = cal[cal.index > now]
            if not future_dates.empty:
                return (future_dates.index.min() - now).days
        return 999
    except Exception:
        return 999

def calculate_max_pain(ticker_opts: pd.DataFrame) -> float:
    strikes = sorted(ticker_opts["STRIKE_PR"].unique())
    oi_table = ticker_opts.groupby(["STRIKE_PR", "OPTION_TYP"])["OPEN_INT"].sum().unstack(fill_value=0)
    if "CE" not in oi_table.columns: oi_table["CE"] = 0
    if "PE" not in oi_table.columns: oi_table["PE"] = 0

    pain_by_strike = {}
    for assumed_price in strikes:
        ce_loss = np.maximum(0, assumed_price - oi_table.index) * oi_table["CE"]
        pe_loss = np.maximum(0, oi_table.index - assumed_price) * oi_table["PE"]
        pain_by_strike[assumed_price] = ce_loss.sum() + pe_loss.sum()

    return min(pain_by_strike, key=pain_by_strike.get)

def calculate_iv_safe(price, spot, strike, t_years, r, flag):
    try:
        if price <= 0 or t_years <= 0: return np.nan
        return implied_volatility(price, spot, strike, t_years, r, flag.lower()) * 100
    except Exception:
        return np.nan

# =====================================================================
# MAIN SCREENER ENGINE
# =====================================================================
def run_short_straddle_screener(tickers: list[str]) -> pd.DataFrame:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Downloading Bhavcopy...")
    file_path = bhavcopy_save(TARGET_DATE, BHAVCOPY_DIR)
    bhavcopy = pd.read_csv(file_path)
    bhavcopy = bhavcopy[bhavcopy["INSTRUMENT"] == "OPTSTK"].copy()
    bhavcopy["EXPIRY_DT"] = pd.to_datetime(bhavcopy["EXPIRY_DT"]).dt.date

    t_years = (EXPIRY_DATE - TARGET_DATE).days / 365.0
    results = []

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(tickers)} stocks...")

    for ticker in tickers:
        try:
            df = tv.get_hist(symbol=ticker, exchange="NSE", interval=Interval.in_daily, n_bars=250)
            if df is None or df.empty or len(df) < 50: 
                time.sleep(0.5)
                continue

            df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
            latest_spot = df["Close"].iloc[-1]

            adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
            df["ADX"] = adx_df[[col for col in adx_df.columns if col.startswith("ADX")][0]]
            df["RSI"] = ta.rsi(df["Close"], length=14)
            df["HV_20"] = get_historical_volatility(df["Close"], window=20)
            latest = df.iloc[-1]

            if not (latest["ADX"] < 20 and 40 <= latest["RSI"] <= 60): 
                time.sleep(0.5)
                continue
            
            days_to_earnings = get_days_to_earnings(ticker)
            if days_to_earnings <= 30: 
                time.sleep(0.5)
                continue

            ticker_opts = bhavcopy[(bhavcopy["SYMBOL"] == ticker) & (bhavcopy["EXPIRY_DT"] == EXPIRY_DATE)].copy()
            if ticker_opts.empty: 
                time.sleep(0.5)
                continue

            max_pain_strike = calculate_max_pain(ticker_opts)
            pain_distance_pct = (abs(latest_spot - max_pain_strike) / latest_spot) * 100

            ticker_opts["Strike_Diff"] = abs(ticker_opts["STRIKE_PR"] - latest_spot)
            atm_strike = ticker_opts.loc[ticker_opts["Strike_Diff"].idxmin()]["STRIKE_PR"]

            atm_ce = ticker_opts[(ticker_opts["STRIKE_PR"] == atm_strike) & (ticker_opts["OPTION_TYP"] == "CE")]
            atm_pe = ticker_opts[(ticker_opts["STRIKE_PR"] == atm_strike) & (ticker_opts["OPTION_TYP"] == "PE")]
            
            if atm_ce.empty or atm_pe.empty: 
                time.sleep(0.5)
                continue

            ce_close, pe_close = atm_ce.iloc[0]["CLOSE"], atm_pe.iloc[0]["CLOSE"]
            avg_iv = (calculate_iv_safe(ce_close, latest_spot, atm_strike, t_years, RISK_FREE_RATE, "c") + 
                      calculate_iv_safe(pe_close, latest_spot, atm_strike, t_years, RISK_FREE_RATE, "p")) / 2.0
            
            iv_edge = avg_iv - latest["HV_20"]

            if iv_edge > 0:
                results.append({
                    "Ticker": ticker, "Spot": round(latest_spot, 2), "ATM Strike": atm_strike,
                    "Max Pain": max_pain_strike, "Pain Dist %": round(pain_distance_pct, 2),
                    "Days to Earnings": days_to_earnings if days_to_earnings != 999 else "N/A",
                    "Total Prem": round(ce_close + pe_close, 2), "IV Edge %": round(iv_edge, 2),
                })
                
            time.sleep(0.5) # Prevent TradingView API disconnects
            
        except Exception:
            time.sleep(0.5)
            continue

    df_out = pd.DataFrame(results)
    return df_out.sort_values(by="IV Edge %", ascending=False) if not df_out.empty else df_out

# =====================================================================
# EMAIL & EXECUTION
# =====================================================================
def send_email_alert(df_results):
    sender_email = os.environ.get("SENDER_EMAIL")
    app_password = os.environ.get("APP_PASSWORD")
    recipient_email = os.environ.get("RECIPIENT_EMAIL")
    
    msg = MIMEMultipart()
    msg['Subject'] = f"Straddle Screener Alert - {TARGET_DATE}"
    msg['From'] = sender_email
    msg['To'] = recipient_email
    
    if not df_results.empty:
        html_table = df_results.to_html(index=False, justify='center', border=1)
        body = f"<html><body><h2>Short Straddle Candidates</h2>{html_table}</body></html>"
    else:
        body = "<html><body><h3>No setups passed the filters today.</h3></body></html>"
        
    msg.attach(MIMEText(body, 'html'))
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, app_password)
        server.send_message(msg)

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Booting Screener...")
    
    # 1. Dynamically load the full 180+ F&O market
    full_market_watchlist = get_live_fo_watchlist()
    
    if not full_market_watchlist:
        print("Fatal Error: Could not fetch watchlist. Exiting.")
    else:
        # 2. Run the screener
        results = run_short_straddle_screener(full_market_watchlist)
        
        # 3. Print locally (visible in GitHub Actions logs)
        if not results.empty:
            print("\n=== SHORT STRADDLE CANDIDATES ===")
            print(results.to_string(index=False))
        else:
            print("\nNo setups passed the filters today.")
            
        # 4. Email the results
        send_email_alert(results)
        print("Workflow complete.")