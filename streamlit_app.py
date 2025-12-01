import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# --- Streamlit 頁面設定 ---
st.set_page_config(
    page_title="個股估值與財報儀表板",
    page_icon="📈",
    layout="wide"
)

# --- 設定繪圖風格與字型 ---
plt.style.use('seaborn-v0_8-whitegrid') 
plt.rcParams['font.family'] = ['sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 共用工具函式 ---
def format_num(value, currency=False, percent=False, decimal=2):
    if value is None or pd.isna(value): return '-'
    if percent: return f'{value*100:+.{decimal}f}%'
    if currency:
        if abs(value) >= 1e12: return f'${value/1e12:.2f}T'
        if abs(value) >= 1e9: return f'${value/1e9:.0f}B'
        if abs(value) >= 1e6: return f'${value/1e6:.1f}M'
        return f'${value:,.0f}'
    return f'{value:,.{decimal}f}'

# --- 1. 資料獲取層 ---

def calculate_one_year_beta(ticker):
    """計算 1 年期 Beta (非快取，供獨立調用)"""
    period = "1y"
    try:
        stock_history = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        market_history = yf.download('^GSPC', period=period, progress=False, auto_adjust=True)
        
        if stock_history.empty or market_history.empty: return None

        stock_close = stock_history['Close'] if 'Close' in stock_history.columns else stock_history.iloc[:, 0]
        market_close = market_history['Close'] if 'Close' in market_history.columns else market_history.iloc[:, 0]
        if isinstance(stock_close, pd.DataFrame): stock_close = stock_close.iloc[:, 0]
        if isinstance(market_close, pd.DataFrame): market_close = market_close.iloc[:, 0]

        stock_returns = stock_close.pct_change().dropna()
        market_returns = market_close.pct_change().dropna()

        common_index = stock_returns.index.intersection(market_returns.index)
        stock_returns = stock_returns.loc[common_index]
        market_returns = market_returns.loc[common_index]

        covariance = stock_returns.cov(market_returns)
        market_variance = market_returns.var()

        if market_variance == 0: return None
        return (covariance / market_variance).round(2)
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        history = stock.history(period="3y", interval="1d", auto_adjust=True) 
        
        financials_q = stock.quarterly_financials
        balance_sheet_q = stock.quarterly_balance_sheet
        cashflow_q = stock.quarterly_cashflow
        
        if not info or history.empty or financials_q.empty:
            return None

        current_price = history['Close'].iloc[-1]
        
        beta_5y = info.get('beta')
        if beta_5y:
            info['beta_used'] = beta_5y
            info['beta_label'] = "Beta (5Y)"
        else:
            beta_1y = calculate_one_year_beta(ticker)
            info['beta_used'] = beta_1y if beta_1y else 1.0
            info['beta_label'] = "Beta (1Y)"

        return {
            'info': info,
            'financials_q': financials_q,
            'balance_sheet_q': balance_sheet_q,
            'cashflow_q': cashflow_q,
            'current_price': current_price,
            'history': history,
        }
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# --- 2. 數據處理與估值邏輯 ---

def get_key_indicators_df(data):
    info = data['info']
    revenue_ttm = info.get('totalRevenue', info.get('grossProfits'))
    market_cap = info.get('marketCap')

    indicators = {
        '52週區間': f"${format_num(info.get('fiftyTwoWeekLow'), decimal=2)} - ${format_num(info.get('fiftyTwoWeekHigh'), decimal=2)}",
        '營收 (TTM)': format_num(revenue_ttm, currency=True),
        '市值': format_num(market_cap, currency=True),
        '1年價格變化': format_num(info.get('52WeekChange'), percent=True),
        'P/E (TTM)': format_num(info.get('trailingPE'), decimal=1),
        'EV / EBITDA': format_num(info.get('enterpriseToEbitda'), decimal=1),
        '股息殖利率': format_num(info.get('dividendYield'), percent=True),
        'Fwd P/E': format_num(info.get('forwardPE'), decimal=1),
        '每股帳面價值': format_num(info.get('bookValue'), decimal=2),
        info.get('beta_label', 'Beta'): format_num(info.get('beta_used'), decimal=2),
        'EPS (TTM)': format_num(info.get('trailingEps'), decimal=2),
        'Fwd EPS': format_num(info.get('forwardEps'), decimal=2),
    }
    
    return pd.DataFrame(list(indicators.items()), columns=['指標', '數值'])

def get_quarterly_valuation_df(data):
    info = data['info']
    history = data['history']
    shares = info.get('sharesOutstanding', 1)
    fq = data['financials_q']
    
    if 'Net Income' not in fq.index or len(fq.columns) < 5:
        return None, {}, {}

    net_income_q = fq.loc['Net Income'].sort_index(ascending=True)
    rev_q = fq.loc['Total Revenue'].sort_index(ascending=True) if 'Total Revenue' in fq.index else pd.Series()
    
    ttm_net_income = net_income_q.rolling(window=4).sum().fillna(net_income_q * 4).sort_index(ascending=False)
    ttm_rev = rev_q.rolling(window=4).sum().fillna(rev_q * 4).sort_index(ascending=False)

    history_idx = pd.to_datetime(history.index).tz_localize(None)
    dates = sorted(ttm_net_income.index, reverse=True)[:5]
    
    metrics_list = []
    historical_multiples = {'PE': [], 'PB': [], 'PS': []}

    for date in dates:
        try:
            target_idx = history_idx.get_indexer([pd.to_datetime(date)], method='nearest')[0]
            quarter_price = history['Close'].iloc[target_idx] if target_idx != -1 else np.nan
        except: quarter_price = np.nan
            
        if pd.isna(quarter_price): continue

        def get_val(df, key_list):
            if isinstance(key_list, str): key_list = [key_list]
            for key in key_list:
                if key in df.index: return df.loc[key, date]
            return np.nan

        net_inc_val = ttm_net_income.loc[date]
        rev_val = ttm_rev.loc[date] if date in ttm_rev.index else np.nan
        
        equity = get_val(data['balance_sheet_q'], ['Stockholders Equity', 'Total Equity Gross Minority Interest'])
        ebitda = get_val(data['financials_q'], ['Ebitda', 'EBITDA'])
        debt = get_val(data['balance_sheet_q'], 'Total Debt')
        cash = get_val(data['balance_sheet_q'], ['Cash', 'Cash And Cash Equivalents'])
        
        mc = quarter_price * shares
        ev = mc + (debt if pd.notna(debt) else 0) - (cash if pd.notna(cash) else 0)
        
        pe = mc / net_inc_val if pd.notna(net_inc_val) and net_inc_val > 0 else np.nan
        ps = mc / rev_val if pd.notna(rev_val) and rev_val > 0 else np.nan
        pb = mc / equity if pd.notna(equity) and equity > 0 else np.nan
        ev_ebitda = ev / (ebitda * 4) if pd.notna(ebitda) and ebitda > 0 else np.nan

        if pd.notna(pe): historical_multiples['PE'].append(pe)
        if pd.notna(ps): historical_multiples['PS'].append(ps)
        if pd.notna(pb): historical_multiples['PB'].append(pb)

        metrics_list.append({
            '季度': date.strftime('%Y-%m-%d'),
            '市值': format_num(mc, currency=True),
            'P/E (TTM)': format_num(pe, decimal=1),
            'P/S (TTM)': format_num(ps, decimal=1),
            'P/B (MRQ)': format_num(pb, decimal=1),
            'EV/EBITDA': format_num(ev_ebitda, decimal=1),
        })
    
    return pd.DataFrame(metrics_list), historical_multiples, {'shares': shares}

def get_financial_summary_with_growth(data):
    fq = data['financials_q']
    bq = data['balance_sheet_q']
    
    if 'Total Revenue' not in fq.index: return None, None

    def calculate_growth(series):
        series = series.sort_index(ascending=True)
        qoq_growth = {}
        yoy_growth = {}
        for date in series.index:
            current_value = series.loc[date]
            # QoQ
            try:
                prev_loc = series.index.get_loc(date) - 1
                if prev_loc >= 0:
                    prev_val = series.iloc[prev_loc]
                    if pd.notna(current_value) and pd.notna(prev_val) and prev_val != 0:
                        qoq_growth[date] = (current_value - prev_val) / abs(prev_val)
            except: pass
            # YoY
            target_date = date - timedelta(days=360)
            try:
                nearest_idx = series.index.get_indexer([target_date], method='nearest')[0]
                if nearest_idx != -1 and abs((series.index[nearest_idx] - target_date).days) < 45:
                    prev_val = series.iloc[nearest_idx]
                    if pd.notna(current_value) and pd.notna(prev_val) and prev_val != 0:
                        yoy_growth[date] = (current_value - prev_val) / abs(prev_val)
            except: pass
        return pd.Series(qoq_growth), pd.Series(yoy_growth)

    revenue_q = fq.loc['Total Revenue']
    net_income_q = fq.loc['Net Income'] if 'Net Income' in fq.index else pd.Series(dtype=float)

    rev_qoq, rev_yoy = calculate_growth(revenue_q)
    ni_qoq, ni_yoy = calculate_growth(net_income_q)
    
    recent_cols = fq.columns[:5]
    
    income_rows = ['Total Revenue', 'Gross Profit', 'Cost Of Revenue', 'Operating Income', 'Net Income', 'Basic EPS']
    income_rows = [r for r in income_rows if r in fq.index]
    income_df = fq.loc[income_rows, fq.columns.intersection(recent_cols)].copy()
    
    if 'Gross Profit' not in income_df.index and 'Total Revenue' in income_df.index and 'Cost Of Revenue' in income_df.index:
        income_df.loc['Gross Profit'] = income_df.loc['Total Revenue'] - income_df.loc['Cost Of Revenue']

    income_df = income_df.T
    income_df['Revenue YoY'] = rev_yoy.reindex(income_df.index).values
    income_df['Revenue QoQ'] = rev_qoq.reindex(income_df.index).values
    income_df['Net Income YoY'] = ni_yoy.reindex(income_df.index).values
    income_df['Net Income QoQ'] = ni_qoq.reindex(income_df.index).values
    income_df = income_df.T

    bs_rows = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity', 'Total Debt']
    bs_rows = [r for r in bs_rows if r in bq.index]
    bs_df = bq.loc[bs_rows, bq.columns.intersection(recent_cols)]

    fmt_cols = [d.strftime('%Y-%m-%d') for d in recent_cols]
    if not income_df.empty: income_df.columns = fmt_cols
    if not bs_df.empty: bs_df.columns = fmt_cols

    return income_df, bs_df

def calculate_valuation_models(data, income_df, historical_multiples, extra_data, custom_g=None):
    """
    計算 P/E, P/S, P/B 估值區間。
    *** 核心修正：強制使用財報數據計算基礎指標 TTM EPS / RPS / BVPS ***
    """
    info = data['info']
    shares = extra_data.get('shares', 1)
    
    # --- 1. 計算基礎指標 (使用財報數據) ---
    
    # TTM EPS
    try:
        net_inc_ttm_total = data['financials_q'].loc['Net Income'].iloc[:4].sum()
        ttm_eps = net_inc_ttm_total / shares
    except: ttm_eps = np.nan
    
    # TTM RPS
    try:
        rev_ttm_total = data['financials_q'].loc['Total Revenue'].iloc[:4].sum()
        ttm_rps = rev_ttm_total / shares
    except: ttm_rps = np.nan
    
    # MRQ BVPS
    try:
        equity_mrq = data['balance_sheet_q'].loc['Stockholders Equity'].iloc[0]
        bvps = equity_mrq / shares
    except: bvps = np.nan


    def get_growth(row_name, info_key):
        rate = np.nan
        try:
            if row_name in income_df.index: rate = income_df.loc[row_name].iloc[0]
        except: pass
        if pd.isna(rate): rate = info.get(info_key)
        return rate

    rev_growth = get_growth('Revenue YoY', 'revenueGrowth')
    ni_growth = get_growth('Net Income YoY', 'earningsGrowth')

    results = []
    
    def add_model(name, base, growth, multiples, metric_name):
        if pd.isna(base) or not multiples: return
        valid_m = [m for m in multiples if m > 0]
        if not valid_m: return
        
        avg = np.mean(valid_m)
        std = np.std(valid_m)
        if pd.isna(std): std = avg * 0.1
        
        low_m = max(0, avg - std)
        high_m = avg + std
        
        g = growth if pd.notna(growth) else 0
        proj_val = base * (1 + g)
        price_low = proj_val * low_m
        price_high = proj_val * high_m
        
        results.append({
            '模型': name,
            '基礎指標': f"{metric_name}: ${base:.2f}",
            '成長率': f"{g:.1%}",
            '倍數區間 (Avg±SD)': f"{avg:.1f}x ± {std:.1f}",
            '預估股價區間': f"${price_low:.2f} - ${price_high:.2f}",
            'Low': price_low, 
        })

    add_model("P/E (本益比)", ttm_eps, ni_growth, historical_multiples.get('PE'), "EPS (TTM)")
    add_model("P/S (股價營收比)", ttm_rps, rev_growth, historical_multiples.get('PS'), "RPS (TTM)")
    
    pb_growth = rev_growth * 0.5 if pd.notna(rev_growth) else 0
    add_model("P/B (股價淨值比)", bvps, pb_growth, historical_multiples.get('PB'), "BVPS (MRQ)")
    
    if custom_g is not None and pd.notna(ttm_eps):
        add_model(f"自定義 ({custom_g:.1%})", ttm_eps, custom_g, historical_multiples.get('PE'), "EPS (TTM)")

    return pd.DataFrame(results)

def analyze_health(data, income_df):
    info = data['info']
    score = 0
    checks = []
    
    # ROE (使用 info 中的年度 ROE)
    roe = info.get('returnOnEquity', 0)
    if roe and roe > 0.15:
        score += 1
        checks.append(("✅", f"ROE 優秀: {roe:.1%} (>15%)"))
    else:
        checks.append(("⚠️", f"ROE 偏低: {roe:.1%}" if roe else "無 ROE 數據"))
        
    # 淨利率 (使用 info 中的 TTM 淨利率)
    pm = info.get('profitMargins', 0)
    if pm > 0.10:
        score += 1
        checks.append(("✅", f"淨利率健康: {pm:.1%} (>10%)"))
    else:
        checks.append(("⚠️", f"淨利率偏低: {pm:.1%}"))

    # 營收成長 (使用我們計算的最新 YoY)
    try:
        rev_yoy = income_df.loc['Revenue YoY'].iloc[0]
        if rev_yoy > 0:
            score += 1
            checks.append(("✅", f"營收正成長: {rev_yoy:.1%}"))
        else:
            checks.append(("⚠️", f"營收衰退: {rev_yoy:.1%}"))
    except:
        checks.append(("⚪", "無法判斷營收成長"))

    # 負債比
    de = info.get('debtToEquity', 0)
    if de and de < 200:
        score += 1
        checks.append(("✅", f"負債比率可控: {de}% (<200%)"))
    else:
        checks.append(("⚠️", f"負債比率偏高: {de}%"))
        
    # 自由現金流
    fcf = info.get('freeCashflow', 0)
    if fcf and fcf > 0:
        score += 1
        checks.append(("✅", "自由現金流為正"))
    else:
        checks.append(("⚠️", "自由現金流為負或無數據"))

    return score, checks

# --- 3. 繪圖函式 ---

def plot_financial_trends(data, income_df, bs_df, ticker):
    if income_df is None or income_df.empty: return None
    
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in income_df.columns][::-1]
    
    def get_d(name):
        return income_df.loc[name].values[::-1] if name in income_df.index else np.zeros(len(dates))

    rev = get_d('Total Revenue')
    gross = get_d('Gross Profit')
    op_inc = get_d('Operating Income')
    net_inc = get_d('Net Income')
    eps = get_d('Basic EPS')
    debt = bs_df.loc['Total Debt'].values[::-1] if not bs_df.empty and 'Total Debt' in bs_df.index else np.zeros(len(dates))
    equity = bs_df.loc['Stockholders Equity'].values[::-1] if not bs_df.empty and 'Stockholders Equity' in bs_df.index else np.zeros(len(dates))
    
    shares = data['info'].get('sharesOutstanding', 1)
    history = data['history']
    market_caps = []
    
    for d in dates:
        try:
            idx = history.index.get_indexer([d], method='nearest')[0]
            price = history['Close'].iloc[idx]
            market_caps.append(price * shares)
        except: market_caps.append(np.nan)
    market_caps = np.array(market_caps)
    
    rev_float = rev.astype(float)
    ps_ratio = np.divide(market_caps, (rev_float * 4), out=np.full_like(market_caps, np.nan), where=rev_float>0)

    gross_margin = np.divide(gross, rev_float, out=np.full_like(rev_float, np.nan), where=rev_float>0) * 100
    op_margin = np.divide(op_inc, rev_float, out=np.full_like(rev_float, np.nan), where=rev_float>0) * 100
    net_margin = np.divide(net_inc, rev_float, out=np.full_like(rev_float, np.nan), where=rev_float>0) * 100

    fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    plt.subplots_adjust(hspace=0.4)

    # Chart 1: Revenue & P/S (營收與P/S趨勢)
    ax1 = axes[0]
    ax1.bar(dates, rev/1e9, color='#A8D5BA', label='營收 (B)', width=20, alpha=0.8)
    ax1.set_ylabel('營收 ($B)', color='#2E8B57', fontweight='bold')
    
    if not np.isnan(ps_ratio).all() and np.nanmax(ps_ratio) > 0:
        ax1_r = ax1.twinx()
        ax1_r.plot(dates, ps_ratio, color='#3D405B', marker='o', linestyle='-', linewidth=2, label='P/S 比率')
        ax1_r.set_ylabel('P/S Ratio', color='#3D405B', fontweight='bold')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_r.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')
        
    ax1.set_title(f'{ticker} - 營收與 P/S 趨勢', fontsize=12, fontweight='bold')

    # Chart 2: Margins (獲利能力指標)
    ax2 = axes[1]
    ax2.plot(dates, gross_margin, marker='o', linestyle='-', label='毛利率')
    ax2.plot(dates, op_margin, marker='s', linestyle='--', label='營業利益率')
    ax2.plot(dates, net_margin, marker='^', linestyle='-.', label='淨利率')
    ax2.set_ylabel('百分比 (%)', fontweight='bold')
    ax2.set_title('獲利能力指標 (Margins)', fontsize=12, fontweight='bold')
    ax2.legend()

    # Chart 3: Net Income & EPS (淨利與 EPS 趨勢)
    ax3 = axes[2]
    ax3.bar(dates, net_inc/1e9, color='#87CEFA', label='淨利 (B)', width=20, alpha=0.8)
    ax3.set_ylabel('淨利 ($B)', color='#2E8B57', fontweight='bold')
    ax3_r = ax3.twinx()
    ax3_r.plot(dates, eps, color='#D4A373', marker='o', linewidth=2, label='EPS')
    ax3_r.set_ylabel('EPS ($)', color='#D4A373', fontweight='bold')
    ax3.set_title('淨利與 EPS 趨勢', fontsize=12, fontweight='bold')
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines + labels, lines2 + labels2, loc='upper left')

    # Chart 4: Capital (資本結構)
    ax4 = axes[3]
    if len(equity) > 0:
        ax4.stackplot(dates, equity/1e9, debt/1e9, labels=['股東權益', '總債務'], colors=['#A8D5BA', '#E07A5F'], alpha=0.7)
        ax4.set_ylabel('資本 ($B)', fontweight='bold')
        ax4.set_title('資本結構 (Debt vs Equity)', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper left')

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(True, linestyle='--', alpha=0.5)

    return fig

def plot_options_forecast(data, ticker):
    current_price = data['current_price']
    # 這裡需要重新獲取 Ticker 物件以避免 pickle 問題 (如果使用 cache)
    stock = yf.Ticker(ticker) 
    history = data['history'].iloc[-252:] 

    iv = None
    vol_source = "歷史波動率 (HV)"
    try:
        exp_dates = stock.options
        if exp_dates:
            chain = stock.option_chain(exp_dates[0])
            calls_iv = chain.calls[(chain.calls['strike'] >= current_price*0.9) & (chain.calls['strike'] <= current_price*1.1)]['impliedVolatility'].mean()
            puts_iv = chain.puts[(chain.puts['strike'] >= current_price*0.9) & (chain.puts['strike'] <= current_price*1.1)]['impliedVolatility'].mean()
            iv_market = (calls_iv + puts_iv) / 2
            if not pd.isna(iv_market) and iv_market > 0.05: 
                iv = iv_market
                vol_source = "期權市場 IV"
    except Exception: pass
    
    if iv is None or pd.isna(iv) or iv <= 0.05:
        try:
            returns = np.log(history['Close'] / history['Close'].shift(1))
            iv = returns.std() * np.sqrt(252)
            vol_source = "歷史波動率 (HV)"
            if iv < 0.05: iv = 0.3
        except:
            iv = 0.3
            vol_source = "預設波動率"

    days_forward = [7, 30, 60, 90, 120] 
    future_dates = [datetime.now() + timedelta(days=d) for d in days_forward]
    
    upper_prices = [current_price * np.exp(iv * np.sqrt(d/365)) for d in days_forward]
    lower_prices = [current_price * np.exp(-iv * np.sqrt(d/365)) for d in days_forward]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    recent_history = history.iloc[-90:]
    ax.plot(recent_history.index, recent_history['Close'], label='歷史股價', color='#3D405B', linewidth=2)
    
    last_date = recent_history.index[-1]
    all_dates = [last_date] + future_dates
    all_upper = [current_price] + upper_prices
    all_lower = [current_price] + lower_prices
    ax.plot(all_dates, all_upper, '--', color='#E07A5F', label=f'預估上界 (+1σ)')
    ax.plot(all_dates, all_lower, '--', color='#81B29A', label=f'預估下界 (-1σ)')
    ax.fill_between(all_dates, all_lower, all_upper, color='#F2CC8F', alpha=0.2)
    
    ax.scatter([last_date], [current_price], color='#E07A5F', s=80, zorder=5)
    
    for date, price in zip(future_dates, upper_prices):
        ax.text(date, price*1.01, f'${price:.0f}', ha='center', va='bottom', fontsize=8)
    for date, price in zip(future_dates, lower_prices):
        ax.text(date, price*0.99, f'${price:.0f}', ha='center', va='top', fontsize=8)

    ax.set_title(f'{ticker} - 波動率預估股價區間 (基於 {vol_source}: {iv:.1%})', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.autofmt_xdate()
    
    return fig

# --- 4. 輸出與格式化函式 ---

def format_financial_df(df, type='income'):
    """格式化財報 DataFrame：數字轉為 B 單位，並替換為中文欄位名"""
    if df is None or df.empty: return pd.DataFrame()

    df_disp = df.copy()
    
    col_map = {
        'Total Revenue': '營收', 'Gross Profit': '毛利', 'Cost Of Revenue': '營收成本', 
        'Operating Income': '營業利益', 'Net Income': '淨利', 'Basic EPS': '基本EPS',
        'Revenue YoY': '營收YoY', 'Revenue QoQ': '營收QoQ', 'Net Income YoY': '淨利YoY', 'Net Income QoQ': '淨利QoQ',
        'Total Assets': '總資產', 'Total Liabilities Net Minority Interest': '總負債', 
        'Stockholders Equity': '股東權益', 'Total Debt': '總借款',
    }
    
    df_disp.index = df_disp.index.map(lambda x: col_map.get(x, x))

    for col in df_disp.columns:
        if 'YoY' in col or 'QoQ' in col:
            df_disp[col] = df_disp[col].apply(lambda x: format_num(x, percent=True) if pd.notna(x) else '-')
        elif 'EPS' in col:
            df_disp[col] = df_disp[col].apply(lambda x: format_num(x, decimal=2) if pd.notna(x) else '-')
        else:
            # 轉換為 B (Billion) 單位並取整數，確保無小數點
            df_disp[col] = df_disp[col].apply(lambda x: format_num(x, currency=True) if pd.notna(x) else '-')
            
    return df_disp

# --- Main App ---

def main():
    st.sidebar.title("🔍 個股分析設定")
    ticker = st.sidebar.text_input("輸入股票代號 (如 GOOG, AAPL)", value="GOOG").upper()
    custom_g = st.sidebar.number_input("自定義 EPS 預估成長率 (%)", min_value=-100.0, max_value=500.0, value=15.0, step=0.5)
    run_btn = st.sidebar.button("開始分析", type="primary")
    
    st.sidebar.markdown("---")
    st.sidebar.info("本工具整合 Yahoo Finance 數據，提供估值模型、財務體質評分及技術面概覽。")

    if run_btn and ticker:
        with st.spinner(f'正在深入分析 {ticker} ... 請稍候'):
            data = get_stock_data(ticker)
            
            if not data:
                st.error(f"無法獲取 {ticker} 的數據，請檢查代號或稍後再試。")
                return

            current_price = data['current_price']
            key_df = get_key_indicators_df(data)
            q_df, hist_multiples, extra = get_quarterly_valuation_df(data)
            inc_df, bs_df = get_financial_summary_with_growth(data)
            
            g_decimal = custom_g / 100.0 if custom_g else None
            val_df = calculate_valuation_models(data, inc_df, hist_multiples, extra, g_decimal)
            
            score, checks = analyze_health(data, inc_df)
            
            st.title(f"{data['info'].get('shortName', ticker)} ({ticker})")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("當前股價", f"${current_price:.2f}")
            chg = data['info'].get('52WeekChange')
            col2.metric("52週漲跌幅", f"{chg:.2%}" if chg else "-", delta_color="normal" if chg and chg > 0 else "inverse")
            col3.metric("財務體質評分", f"{score} / 5")
            beta_val = data['info'].get('beta_used', 1.0)
            col4.metric("Beta 係數", f"{beta_val:.2f}")

            tab1, tab2, tab3, tab4 = st.tabs(["📊 總覽與估值", "💰 估值詳情", "📑 財務報表", "📈 趨勢圖表"])

            with tab1:
                st.subheader("1. 綜合估值模型 (一年目標價)")
                if not val_df.empty:
                    val_disp = val_df.copy()
                    st.dataframe(val_disp, use_container_width=True)
                    st.caption("註：估值區間基於近 5 季歷史倍數 (P/E, P/S, P/B) 平均值 ± 1 標準差推算。")
                else:
                    st.warning("數據不足，無法建立估值模型。")

                st.subheader("2. 財務體質診斷")
                cols_health = st.columns(2)
                for i, (icon, msg) in enumerate(checks):
                    with cols_health[i % 2]:
                        if icon == "✅": st.success(f"{icon} {msg}")
                        elif icon == "⚠️": st.warning(f"{icon} {msg}")
                        else: st.info(f"{icon} {msg}")

                st.subheader("3. 關鍵指標")
                st.dataframe(key_df, height=400, hide_index=True)

            with tab2:
                st.subheader("季度估值歷史 (TTM/MRQ)")
                if q_df is not None:
                    st.dataframe(q_df, use_container_width=True)
                else:
                    st.info("無足夠歷史數據。")

            with tab3:
                st.subheader("4. 損益表 (Income Statement) - 單位: Billion")
                inc_fmt_df = format_financial_df(inc_df, type='income')
                if not inc_fmt_df.empty:
                    st.dataframe(inc_fmt_df, use_container_width=True)
                else:
                    st.info("無法獲取損益表數據。")
                
                st.subheader("5. 資產負債表摘要 (Balance Sheet) - 單位: Billion")
                bs_fmt_df = format_financial_df(bs_df, type='balance')
                if not bs_fmt_df.empty:
                    st.dataframe(bs_fmt_df, use_container_width=True)

            with tab4:
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.markdown("##### 財務趨勢分析")
                    fig_trends = plot_financial_trends(data, inc_df, bs_df, ticker)
                    if fig_trends: st.pyplot(fig_trends)
                    else: st.write("無足夠數據繪製趨勢圖。")

                with col_chart2:
                    st.markdown("##### 期權/波動率股價預測")
                    fig_opt = plot_options_forecast(data, ticker)
                    st.pyplot(fig_opt)
                    st.caption("預測區間基於隱含波動率 (IV) 或歷史波動率 (HV) 計算之 68% 信賴區間 (±1σ)。")

if __name__ == "__main__":
    main()
