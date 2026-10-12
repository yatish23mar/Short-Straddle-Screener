import os
import io
import time
import requests
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import date, datetime, timedelta, timezone
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
EXPIRY_DATE = date(2026, 10, 29)  # Update monthly to the next F&O Expiry
RISK_FREE_RATE = 0.07             
BHAVCOPY_DIR = "/tmp"             

tv = TvDatafeed()

# =====================================================================
# MASTER F&O UNIVERSE (213 CONSTITUENTS)
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
# DATA EXTRACTION & RESILIENCE
# =====================================================================
def get_historical_data(ticker: str) -> pd.DataFrame:
    """Attempts TvDatafeed first; automatically falls back to Yahoo Finance on timeout."""
    try:
        df = tv.get_hist(symbol=ticker, exchange="NSE", interval=Interval.in_daily, n_bars=250)
        if df is not None and not df.empty and len(df) >= 50:
            df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
            return df
    except Exception:
        pass

    # Fallback to yfinance
    try:
        clean_sym = ticker.replace("&", "%26")
        stock = yf.Ticker(f"{clean_sym}.NS")
        df = stock.history(period="1y", interval="1d")
        if df is not None and not df.empty and len(df) >= 50:
            return df[["Open", "High", "Low", "Close"]]
    except Exception:
        pass

    return pd.DataFrame()

def download_latest_fo_bhavcopy() -> tuple[pd.DataFrame, date]:
    """Tries today's date, stepping back up to 5 days for weekends/holidays."""
    check_date = date.today()
    for _ in range(5):
        try:
            print(f"Checking F&O Bhavcopy for {check_date}...")
            file_path = bhavcopy_fo_save(check_date, BHAVCOPY_DIR)
            bhavcopy = pd.read_csv(file_path)
            bhavcopy.columns = bhavcopy.columns.str.strip()
            
            # Map UDiFF or legacy column headers
            udiff_map = {
                "FinInstrmTp": "INSTRUMENT", "TckrSymb": "SYMBOL", "XpryDt": "EXPIRY_DT",
                "StrkPric": "STRIKE_PR", "OptnTp": "OPTION_TYP", "ClsPric": "CLOSE", "OpnIntrst": "OPEN_INT"
            }
            bhavcopy.rename(columns=udiff_map, inplace=True)
            bhavcopy = bhavcopy[bhavcopy["INSTRUMENT"].isin(["OPTSTK", "STO"])].copy()
            bhavcopy["EXPIRY_DT"] = pd.to_datetime(bhavcopy["EXPIRY_DT"]).dt.date
            bhavcopy["STRIKE_PR"] = pd.to_numeric(bhavcopy["STRIKE_PR"], errors='coerce')
            bhavcopy["CLOSE"] = pd.to_numeric(bhavcopy["CLOSE"], errors='coerce')
            bhavcopy["OPEN_INT"] = pd.to_numeric(bhavcopy["OPEN_INT"], errors='coerce').fillna(0)
            
            print(f"Loaded valid F&O Bhavcopy for {check_date}")
            return bhavcopy, check_date
        except Exception:
            check_date -= timedelta(days=1)
            
    raise RuntimeError("Could not retrieve a valid F&O Bhavcopy within the last 5 days.")

def get_historical_volatility(close_prices: pd.Series, window: int = 20) -> pd.Series:
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252) * 100

def get_days_to_earnings(ticker: str) -> int:
    try:
        clean_sym = ticker.replace("&", "%26")
        stock = yf.Ticker(f"{clean_sym}.NS")
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
# SCREENER LOGIC
# =====================================================================
def run_short_straddle_screener(tickers: list[str]) -> tuple[pd.DataFrame, date]:
    bhavcopy, target_date = download_latest_fo_bhavcopy()
    t_years = max((EXPIRY_DATE - target_date).days / 365.0, 0.001)
    results = []

    print(f"Scanning {len(tickers)} stocks against settlements on {target_date}...")

    for ticker in tickers:
        try:
            df = get_historical_data(ticker)
            if df.empty or len(df) < 50:
                continue

            latest_spot = df["Close"].iloc[-1]
            adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
            df["ADX"] = adx_df[[col for col in adx_df.columns if col.startswith("ADX")][0]]
            df["RSI"] = ta.rsi(df["Close"], length=14)
            df["HV_20"] = get_historical_volatility(df["Close"], window=20)
            latest = df.iloc[-1]

            if not (latest["ADX"] < 20 and 40 <= latest["RSI"] <= 60):
                continue
            
            days_to_earnings = get_days_to_earnings(ticker)
            if days_to_earnings <= 30:
                continue

            ticker_opts = bhavcopy[(bhavcopy["SYMBOL"] == ticker) & (bhavcopy["EXPIRY_DT"] == EXPIRY_DATE)].copy()
            if ticker_opts.empty:
                continue

            max_pain_strike = calculate_max_pain(ticker_opts)
            pain_distance_pct = (abs(latest_spot - max_pain_strike) / latest_spot) * 100

            ticker_opts["Strike_Diff"] = abs(ticker_opts["STRIKE_PR"] - latest_spot)
            atm_strike = ticker_opts.loc[ticker_opts["Strike_Diff"].idxmin()]["STRIKE_PR"]

            atm_ce = ticker_opts[(ticker_opts["STRIKE_PR"] == atm_strike) & (ticker_opts["OPTION_TYP"] == "CE")]
            atm_pe = ticker_opts[(ticker_opts["STRIKE_PR"] == atm_strike) & (ticker_opts["OPTION_TYP"] == "PE")]
            
            if atm_ce.empty or atm_pe.empty:
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
        except Exception:
            continue

    df_out = pd.DataFrame(results)
    if not df_out.empty:
        df_out = df_out.sort_values(by="IV Edge %", ascending=False)
    return df_out, target_date

# =====================================================================
# EMAIL DELIVERY ENGINE (STARTTLS ON PORT 587)
# =====================================================================
def send_email_alert(df_results: pd.DataFrame, target_date: date):
    sender_email = os.environ.get("SENDER_EMAIL", "").strip()
    app_password = os.environ.get("APP_PASSWORD", "").replace(" ", "").strip()
    recipient_email = os.environ.get("RECIPIENT_EMAIL", "").strip()

    if not sender_email or not app_password or not recipient_email:
        print("ERROR: One or more email secrets are missing. Skipping delivery.")
        return

    msg = MIMEMultipart()
    msg['Subject'] = f"Straddle Screener Alert - {target_date}"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    if not df_results.empty:
        html_table = df_results.to_html(index=False, justify='center', border=1)
        body = f"<html><body><h2>Short Straddle Candidates ({target_date})</h2>{html_table}</body></html>"
    else:
        body = f"<html><body><h3>No setups passed the filters for {target_date}.</h3></body></html>"

    msg.attach(MIMEText(body, 'html'))

    # Reliable TLS delivery across Azure/GitHub runners
    try:
        print("Connecting via SMTP (Port 587 STARTTLS)...")
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=30)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
        print(f"Email successfully delivered to {recipient_email}")
    except Exception as e:
        print(f"Primary SMTP port 587 failed: {e}. Trying Port 465 fallback...")
        try:
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30)
            server.login(sender_email, app_password)
            server.send_message(msg)
            server.quit()
            print(f"Email successfully delivered via SSL fallback to {recipient_email}")
        except Exception as final_e:
            print(f"CRITICAL: Email delivery failed: {final_e}")

# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Screener Pipeline...")
    results, report_date = run_short_straddle_screener(FALLBACK_FO_UNIVERSE)

    if not results.empty:
        print("\n=== SHORT STRADDLE CANDIDATES ===")
        print(results.to_string(index=False))
    else:
        print("\nNo setups passed the filters today.")

    send_email_alert(results, report_date)
    print("Execution complete.")
