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
from jugaad_data.nse import bhavcopy_fo_save
from py_vollib.black_scholes.implied_volatility import implied_volatility
import yfinance as yf

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================
TARGET_DATE = date.today()        # Automatically uses the execution date
EXPIRY_DATE = date(2026, 10, 29)  # UPDATE THIS MONTHLY to the next F&O Expiry
RISK_FREE_RATE = 0.07             
BHAVCOPY_DIR = "/tmp"             

tv = TvDatafeed()

# =====================================================================
# MASTER F&O FALLBACK UNIVERSE (213 CONSTITUENTS)
# =====================================================================
FALLBACK_FO_UNIVERSE = [
    "ADANIENT", "LODHA", "LT", "MANAPPURAM", "ONGC", "POLYCAB", "RADICO", "WIPRO", 
    "SRF", "ATHERENERG", "BOSCHLTD", "MFSL", "PRESTIGE", "TATAPOWER", "TIINDIA", 
    "HAL", "INDUSTOWER", "MARUTI", "PIDILITIND", "RECLTD", "SHREECEM", "BANDHANBNK", 
    "CIPLA", "FORTIS", "GVT&D", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "IEX", "INFY", 
    "KPITTECH", "LUPIN", "SAGILITY", "SAIL", "SBILIFE", "SIEMENS", "360ONE", 
    "ABCAPITAL", "ADANIPORTS", "AMBER", "ASHOKLEY", "AUBANK", "BAJFINANCE", 
    "BANKINDIA", "BHARTIARTL", "BHEL", "BLUESTARCO", "BRITANNIA", "BSE", "CAMS", 
    "COFORGE", "COLPAL", "CROMPTON", "FEDERALBNK", "HDFCAMC", "HINDZINC", "ICICIGI", 
    "ICICIPRULI", "IDEA", "INDIANB", "JUBLFOOD", "KALYANKJIL", "KEI", "MARICO", 
    "MOTILALOFS", "MPHASIS", "NTPC", "NYKAA", "OFSS", "PATANJALI", "RBLBANK", 
    "SHRIRAMFIN", "TCS", "TMPV", "UNITDSPR", "VBL", "VEDL", "VOLTAS", "YESBANK", 
    "ADANIENSOL", "ADANIPOWER", "ANANDRATHI", "ANGELONE", "ASTRAL", "AUROPHARMA", 
    "BAJAJFINSV", "BAJAJHLDNG", "BANKBARODA", "CHOLAFIN", "DLF", "EICHERMOT", 
    "ETERNAL", "GODFRYPHLP", "HINDUNILVR", "ICICIBANK", "IDFCFIRSTB", "INDHOTEL", 
    "IREDA", "ITC", "KAYNES", "KFINTECH", "M&M", "MAHABANK", "NAM-INDIA", 
    "NATIONALUM", "PAGEIND", "PAYTM", "PERSISTENT", "PFC", "PGEL", "PNB", 
    "PNBHOUSING", "POWERGRID", "PREMIERENE", "RELIANCE", "SOLARINDS", "SUPREMEIND", 
    "TATAELXSI", "TVSMOTOR", "UJJIVANSFB", "ULTRACEMCO", "APLAPOLLO", "APOLLOHOSP", 
    "ASIANPAINT", "BAJAJ-AUTO", "BHARATFORG", "BPCL", "DIVISLAB", "DMART", "DRREDDY", 
    "ENRIN", "FORCEMOT", "GAIL", "GLENMARK", "HCLTECH", "HINDPETRO", "HYUNDAI", 
    "INDUSINDBK", "INOXWIND", "KOTAKBANK", "LAURUSLABS", "LICI", "LTF", "MANKIND", 
    "MAXHEALTH", "MCX", "MOTHERSON", "NBCC", "NESTLEIND", "NHPC", "PIIND", 
    "POLICYBZR", "SUNPHARMA", "TATACONSUM", "UNOMINDA", "UPL", "VMM", "ABB", 
    "ADANIGREEN", "ALKEM", "AMBUJACEM", "AXISBANK", "BDL", "BEL", "BIOCON", 
    "CANBK", "CDSL", "CGPOWER", "COALINDIA", "COCHINSHIP", "CONCOR", "CUMMINSIND", 
    "DABUR", "DELHIVERY", "DIXON", "GMRAIRPORT", "GODREJCP", "GODREJPROP", "GRASIM", 
    "HAVELLS", "HINDALCO", "INDIGO", "IOC", "IRFC", "JINDALSTEL", "JIOFIN", 
    "JSWENERGY", "JSWSTEEL", "LICHSGFIN", "LTM", "MAZDOCK", "MUTHOOTFIN", "NAUKRI", 
    "NMDC", "OBEROIRLTY", "OIL", "PETRONET", "PHOENIXLTD", "POWERINDIA", "RVNL", 
    "SBICARD", "SBIN", "SONACOMS", "SUZLON", "SWIGGY", "TATASTEEL", "TECHM", 
    "TITAN", "TORNTPHARM", "TRENT", "UNIONBANK", "WAAREEEENER", "ZYDUSLIFE"
]

# =====================================================================
# CORE FUNCTIONS
# =====================================================================
def get_live_fo_watchlist() -> list[str]:
    """Downloads active symbols from NSE, handling header shifts and network blocks."""
    url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        raw_lines = [line.strip() for line in response.text.splitlines() if line.strip()]
        header_idx = None
        for idx, line in enumerate(raw_lines):
            if "SYMBOL" in line.upper():
                header_idx = idx
                break

        if header_idx is not None:
            clean_csv_text = "\n".join(raw_lines[header_idx:])
            df = pd.read_csv(io.StringIO(clean_csv_text), skipinitialspace=True, on_bad_lines='skip')
            df = df.dropna(subset=['SYMBOL'])
            symbols = df['SYMBOL'].astype(str).str.strip().unique().tolist()
            indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
            live_list = [sym for sym in symbols if sym not in indices and sym.upper() != 'SYMBOL']
            
            if live_list:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(live_list)} stocks from live NSE feed.")
                return live_list

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Live parse issue ({e}). Using embedded fallback universe.")

    return FALLBACK_FO_UNIVERSE

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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Downloading F&O Bhavcopy...")
    
    # 1. Download F&O Bhavcopy
    file_path = bhavcopy_fo_save(TARGET_DATE, BHAVCOPY_DIR)
    bhavcopy = pd.read_csv(file_path)
    
    # 2. Clean column headers and map UDiFF names to standard format
    bhavcopy.columns = bhavcopy.columns.str.strip()
    udiff_map = {
        "FinInstrmTp": "INSTRUMENT",
        "TckrSymb": "SYMBOL",
        "XpryDt": "EXPIRY_DT",
        "StrkPric": "STRIKE_PR",
        "OptnTp": "OPTION_TYP",
        "ClsPric": "CLOSE",
        "OpnIntrst": "OPEN_INT"
    }
    bhavcopy.rename(columns=udiff_map, inplace=True)

    # 3. Filter for Stock Options (supports legacy OPTSTK & UDiFF STO codes)
    bhavcopy = bhavcopy[bhavcopy["INSTRUMENT"].isin(["OPTSTK", "STO"])].copy()
    bhavcopy["EXPIRY_DT"] = pd.to_datetime(bhavcopy["EXPIRY_DT"]).dt.date
    bhavcopy["STRIKE_PR"] = pd.to_numeric(bhavcopy["STRIKE_PR"], errors='coerce')
    bhavcopy["CLOSE"] = pd.to_numeric(bhavcopy["CLOSE"], errors='coerce')
    bhavcopy["OPEN_INT"] = pd.to_numeric(bhavcopy["OPEN_INT"], errors='coerce').fillna(0)

    t_years = (EXPIRY_DATE - TARGET_DATE).days / 365.0
    results = []

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning universe of {len(tickers)} stocks...")

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
                
            time.sleep(0.5)
            
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
    
    watchlist = get_live_fo_watchlist()
    print(f"Targeting universe of {len(watchlist)} stocks...")
    
    results = run_short_straddle_screener(watchlist)
    
    if not results.empty:
        print("\n=== SHORT STRADDLE CANDIDATES ===")
        print(results.to_string(index=False))
    else:
        print("\nNo setups passed the filters today.")
        
    send_email_alert(results)
    print("Workflow complete. Email sent.")
