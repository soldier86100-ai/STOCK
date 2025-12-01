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

# --- 設定繪圖風格 (解決亂碼) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 共用工具函式 ---
def format_num(value, currency=False, percent=False, decimal=2):
    """通用數值格式化函式"""
    if value is None or pd.isna(value): return '-'
    if percent: return f'{value*100:+.{decimal}f}%'
    if currency:
        if abs(value) >= 1e12: return f'${value/1e12:.2f}T'
        if abs(value) >= 1e9: return f'${value/1e9:.2f}B'
        if abs(value) >= 1e6: return f'${value/1e6:.1f}M'
        return f'${value:,.0f}'
    return f'{value:,.{decimal}f}'

# --- 1. 資料獲取層 ---

def calculate_one_year_beta(ticker):
    """計算 1 年期 Beta (相對於 S&P 500)"""
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
        
        # 獲取 3 年歷史數據
        history = stock.history(period="3y", interval="1d", auto_adjust=True) 
        
        financials_q = stock.quarterly_financials
        balance_sheet_q = stock.quarterly_balance_sheet
        cashflow_q = stock.quarterly_cashflow
        
        if not info or history.empty: return None

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
        return None

# --- 2. 數據處理 ---

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
    """
    計算季度估值趨勢與歷史倍數。
    *** 核心邏輯：鎖定最近 5 季 ***
    """
    info = data['info']
    history = data['history']
    shares = info.get('sharesOutstanding', 1)
    fq = data['financials_q']
    
    if fq.empty or 'Net Income' not in fq.index:
        return pd.DataFrame(), {}, {'shares': shares}

    # 1. 準備數據 (由新到舊 -> 轉為舊到新以計算 Rolling)
    net_income_q = fq.loc['Net Income'].sort_index(ascending=True).fillna(0)
    rev_q = fq.loc['Total Revenue'].sort_index(ascending=True).fillna(0) if 'Total Revenue' in fq.index else pd.Series()
    
    # 2. 計算 TTM (滾動 4 季總和)
    # 智慧填補：若 TTM 數據不足(例如最早的幾季)，使用單季 x 4 作為近似，確保能湊滿 5 季
    ttm_net_income = net_income_q.rolling(window=4).sum().fillna(net_income_q * 4)
    ttm_rev = rev_q.rolling(window=4).sum().fillna(rev_q * 4)

    # 轉回新到舊
    ttm_net_income = ttm_net_income.sort_index(ascending=False)
    ttm_rev = ttm_rev.sort_index(ascending=False)

    history_idx = pd.to_datetime(history.index).tz_localize(None)
    
    # *** 鎖定最近 5 季 ***
    dates = sorted(ttm_net_income.index, reverse=True)[:5]
    
    metrics = []
    multiples = {'PE': [], 'PB': [], 'PS': []}

    for date in dates:
        # 找當季結束時的股價
        try:
            target_idx = history_idx.get_indexer([pd.to_datetime(date)], method='nearest')[0]
            price = history['Close'].iloc[target_idx] if target_idx != -1 else np.nan
        except: price = np.nan
            
        if pd.isna(price): continue

        # 獲取數據
        ni_val = ttm_net_income.loc[date]
        rev_val = ttm_rev.loc[date] if date in ttm_rev.index else np.nan
        
        # 股東權益
        try:
            equity = data['balance_sheet_q'].loc['Stockholders Equity', date]
        except: 
            try: equity = data['balance_sheet_q'].loc['Total Equity Gross Minority Interest', date]
            except: equity = np.nan

        # EBITDA
        try:
            ebitda = data['financials_q'].loc['EBITDA', date] * 4 # 年化
        except: 
            try: ebitda = data['financials_q'].loc['Ebitda', date] * 4
            except: ebitda = np.nan

        mc = price * shares
        
        # 計算倍數
        pe = mc / ni_val if ni_val > 0 else np.nan
        ps = mc / rev_val if rev_val > 0 else np.nan
        pb = mc / equity if equity > 0 else np.nan
        
        # 收集有效倍數
        if pd.notna(pe) and 0 < pe < 200: multiples['PE'].append(pe)
        if pd.notna(ps) and 0 < ps < 100: multiples['PS'].append(ps)
        if pd.notna(pb) and 0 < pb < 100: multiples['PB'].append(pb)

        metrics.append({
            '季度': date.strftime('%Y-%m-%d'),
            '市值': mc,
            'P/E (TTM)': pe,
            'P/S (TTM)': ps,
            'P/B (MRQ)': pb,
            'EV/EBITDA': mc / ebitda if ebitda > 0 else np.nan
        })
    
    return pd.DataFrame(metrics), multiples, {'shares': shares}

def get_income_statement(data):
    """獲取損益表並計算成長率"""
    fq = data['financials_q']
    if fq.empty: return pd.DataFrame()
    
    # 選取欄位
    target_rows = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'Basic EPS']
    rows = []
    for t in target_rows:
        found = [i for i in fq.index if t in i]
        if found: rows.append(found[0])
    
    df = fq.loc[rows].copy()
    
    # 計算成長率
    df_sorted = df.sort_index(axis=1, ascending=True)
    qoq = df_sorted.pct_change(axis=1)
    yoy = df_sorted.pct_change(axis=1, periods=4) # YoY 嚴格比較去年同季
    
    # 轉回新到舊
    df = df.sort_index(axis=1, ascending=False)
    qoq = qoq.sort_index(axis=1, ascending=False)
    yoy = yoy.sort_index(axis=1, ascending=False)
    
    final_df = df.T
    
    def get_growth(growth_df, row_name):
        try: return growth_df.loc[row_name]
        except: return pd.Series([np.nan]*len(final_df), index=final_df.index)

    rev_idx = [i for i in fq.index if 'Total Revenue' in i][0]
    ni_idx = [i for i in fq.index if 'Net Income' in i][0]

    final_df['營收 YoY'] = get_growth(yoy, rev_idx)
    final_df['營收 QoQ'] = get_growth(qoq, rev_idx)
    final_df['淨利 YoY'] = get_growth(yoy, ni_idx)
    final_df['淨利 QoQ'] = get_growth(qoq, ni_idx)
    
    col_map = {
        rev_idx: '營收', 
        [i for i in fq.index if 'Gross Profit' in i][0]: '毛利',
        [i for i in fq.index if 'Operating Income' in i][0]: '營業利益',
        ni_idx: '淨利',
        [i for i in fq.index if 'Basic EPS' in i][0]: 'EPS'
    }
    final_df = final_df.rename(columns=col_map)
    
    return final_df.head(5).T

def calculate_valuation(data, income_df, multiples, custom_g):
    info = data['info']
    shares = info.get('sharesOutstanding', 1)
    
    # 基礎數據 (優先用 TTM 計算)
    try:
        ttm_eps = data['financials_q'].loc['Basic EPS'].iloc[:4].sum()
    except: ttm_eps = info.get('trailingEps')
    
    try:
        ttm_rev = data['financials_q'].loc['Total Revenue'].iloc[:4].sum()
        ttm_rps = ttm_rev / shares
    except: ttm_rps = np.nan
    
    try:
        # 尋找最近一季的股東權益
        eq_rows = [i for i in data['balance_sheet_q'].index if 'Stockholders Equity' in i or 'Total Equity' in i]
        mrq_equity = data['balance_sheet_q'].loc[eq_rows[0]].iloc[0] if eq_rows else np.nan
        bvps = mrq_equity / shares
    except: bvps = info.get('bookValue')

    # 成長率
    try: rev_g = income_df.loc['營收 YoY'].iloc[0]
    except: rev_g = info.get('revenueGrowth', 0)
    
    try: ni_g = income_df.loc['淨利 YoY'].iloc[0]
    except: ni_g = info.get('earningsGrowth', 0)

    results = []
    
    def add_row(name, base, growth, hist_list):
        if pd.isna(base) or not hist_list: return
        
        # 計算近 5 季的平均與標準差
        valid_m = [m for m in hist_list if m > 0]
        if not valid_m: return
        
        avg = np.mean(valid_m)
        std = np.std(valid_m)
        if pd.isna(std): std = avg * 0.15
        
        g = growth if pd.notna(growth) else 0
        
        # 估值公式: 基礎 * (1+g) * 倍數
        target = base * (1 + g) * avg
        low = base * (1 + g) * max(0, avg - std)
        high = base * (1 + g) * (avg + std)
        
        results.append({
            '模型': name,
            '基礎指標': base,
            '成長率': g,
            '近5季倍數 (Avg±SD)': f"{avg:.1f}x ± {std:.1f}",
            '估值下限': low,
            '估值上限': high,
            '目標價': target
        })

    add_row("P/E (本益比)", ttm_eps, ni_g, multiples.get('PE'))
    add_row("P/S (營收比)", ttm_rps, rev_g, multiples.get('PS'))
    add_row("P/B (淨值比)", bvps, rev_g * 0.5, multiples.get('PB')) 
    
    if custom_g:
        add_row(f"自定義 ({custom_g}%)", ttm_eps, custom_g/100, multiples.get('PE'))

    return pd.DataFrame(results)

# --- 3. 繪圖函式 ---

def plot_charts(data, income_df, ticker):
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in income_df.columns][::-1]
    
    def get_series(row_name):
        if row_name in income_df.index:
            return income_df.loc[row_name].values[::-1]
        return np.zeros(len(dates))

    rev = get_series('營收')
    net_inc = get_series('淨利')
    eps = get_series('EPS')
    
    shares = data['info'].get('sharesOutstanding', 1)
    hist = data['history']
    market_caps = []
    for d in dates:
        try:
            idx = hist.index.get_indexer([d], method='nearest')[0]
            price = hist['Close'].iloc[idx]
            market_caps.append(price * shares)
        except: market_caps.append(np.nan)
    
    # 修正 P/S 計算: 市值 / (季營收 * 4)
    rev_float = rev.astype(float)
    ps_ratio = np.divide(market_caps, (rev_float * 4), out=np.full_like(market_caps, np.nan), where=rev_float>0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 圖 1: 營收與 P/S (雙軸)
    ax1.bar(dates, rev/1e9, color='#A8D5BA', width=20, label='Revenue (B)')
    ax1.set_ylabel('Revenue ($B)', color='green')
    
    # 智慧隱藏 P/S
    if not np.isnan(ps_ratio).all() and np.nanmax(ps_ratio) > 0:
        ax1_r = ax1.twinx()
        ax1_r.plot(dates, ps_ratio, color='purple', marker='o', label='P/S Ratio')
        ax1_r.set_ylabel('P/S Ratio', color='purple')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_r.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')
        
    ax1.set_title(f'{ticker} Revenue & P/S Trend')
    
    # 圖 2: 淨利與 EPS (雙軸)
    ax2.bar(dates, net_inc/1e9, color='#87CEFA', width=20, label='Net Income (B)')
    ax2.set_ylabel('Net Income ($B)', color='blue')
    ax2_r = ax2.twinx()
    ax2_r.plot(dates, eps, color='orange', marker='o', label='EPS')
    ax2_r.set_ylabel('EPS ($)', color='orange')
    ax2.set_title(f'{ticker} Net Income & EPS Trend')
    
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(True, linestyle='--', alpha=0.5)

    return fig

# --- Main ---

def main():
    st.sidebar.title("個股分析")
    ticker = st.sidebar.text_input("股票代號", "NVDA").upper()
    custom_g = st.sidebar.number_input("自定義 EPS 成長率 (%)", value=15.0)
    
    if st.sidebar.button("分析"):
        data = get_stock_data(ticker)
        if not data:
            st.error("無法獲取數據")
            return

        # 計算流程
        q_val_df, multiples, extra = get_quarterly_valuation_df(data)
        inc_df = get_income_statement(data)
        val_res = calculate_valuation(data, inc_df, multiples, custom_g)
        
        st.title(f"{ticker} 估值報告")
        st.metric("當前股價", f"${data['current_price']:.2f}")

        # 1. 估值表
        st.subheader("1. 綜合估值模型 (基於近5季倍數)")
        if not val_res.empty:
            st.dataframe(val_res.style.format({
                '基礎指標': '${:,.2f}',
                '成長率': '{:.2%}',
                '估值下限': '${:,.2f}',
                '估值上限': '${:,.2f}',
                '目標價': '${:,.2f}'
            }), use_container_width=True)
        else:
            st.warning("數據不足無法估值")

        # 2. 季度趨勢
        st.subheader("2. 季度估值歷史")
        if not q_val_df.empty:
            st.dataframe(q_val_df.style.format({
                '市值': format_large_num,
                'P/E (TTM)': '{:.1f}',
                'P/S (TTM)': '{:.1f}',
                'P/B (MRQ)': '{:.1f}',
                'EV/EBITDA': '{:.1f}'
            }), use_container_width=True)

        # 3. 損益表 (格式修復)
        st.subheader("3. 損益表 (單位: Billion)")
        if not inc_df.empty:
            format_dict = {
                'EPS': '{:.2f}',       # EPS 兩位小數
                '營收 YoY': '{:.2%}',  # 百分比
                '營收 QoQ': '{:.2%}',
                '淨利 YoY': '{:.2%}',
                '淨利 QoQ': '{:.2%}',
                '營收': format_large_num,
                '淨利': format_large_num,
                '毛利': format_large_num,
                '營業利益': format_large_num
            }
            st.dataframe(inc_df.style.format(format_dict, na_rep="-"), use_container_width=True)

        # 4. 圖表
        st.subheader("4. 趨勢圖表")
        fig = plot_charts(data, inc_df, ticker)
        if fig: st.pyplot(fig)

if __name__ == "__main__":
    main()
