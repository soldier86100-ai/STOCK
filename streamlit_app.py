import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# --- Streamlit 頁面設定 ---
st.set_page_config(
    page_title="個股估值儀表板 (Yahoo Style)",
    page_icon="📊",
    layout="wide"
)

# --- 設定繪圖風格 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 工具函式 ---
def format_large_num(value):
    """將大數字格式化為 T, B, M (仿 Yahoo 風格)"""
    if value is None or pd.isna(value): return '-'
    if abs(value) >= 1e12: return f'{value/1e12:.2f}T'
    if abs(value) >= 1e9: return f'{value/1e9:.2f}B'
    if abs(value) >= 1e6: return f'{value/1e6:.2f}M'
    return f'{value:,.2f}'

def format_float(value, decimal=2):
    """一般浮點數格式化"""
    if value is None or pd.isna(value): return '-'
    return f'{value:,.{decimal}f}'

# --- 1. 資料獲取層 ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 為了計算歷史 TTM，我們需要比顯示期間更長的數據，這裡抓 4 年
        history = stock.history(period="4y", interval="1d", auto_adjust=True) 
        info = stock.info
        
        # 獲取財報
        financials_q = stock.quarterly_financials
        balance_sheet_q = stock.quarterly_balance_sheet
        cashflow_q = stock.quarterly_cashflow
        
        if history.empty: return None

        return {
            'ticker': ticker,
            'info': info,
            'history': history,
            'financials_q': financials_q,
            'balance_sheet_q': balance_sheet_q,
            'cashflow_q': cashflow_q,
            'current_price': history['Close'].iloc[-1]
        }
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --- 2. 核心估值表計算 (仿 Yahoo Valuation Measures) ---

def get_yahoo_valuation_table(data):
    """
    生成與 Yahoo Finance 網站結構相同的 Valuation Measures 表格
    """
    info = data['info']
    hist = data['history']
    fq = data['financials_q']
    bs = data['balance_sheet_q']
    
    # 確保數據按日期排序 (舊 -> 新) 方便 rolling 計算
    fq = fq.sort_index(axis=1, ascending=True)
    bs = bs.sort_index(axis=1, ascending=True)

    # 準備 TTM 數據 (滾動 4 季總和)
    # yfinance 的 index 有時會有差異，這裡做一些容錯抓取
    def get_row_ttm(df, possible_names):
        for name in possible_names:
            if name in df.index:
                # Rolling sum over last 4 quarters
                return df.loc[name].rolling(window=4).sum()
        return pd.Series(dtype='float64')
    
    # 抓取原始數據列
    rev_ttm_series = get_row_ttm(fq, ['Total Revenue', 'Operating Revenue'])
    ebitda_ttm_series = get_row_ttm(fq, ['EBITDA', 'Normalized EBITDA'])
    ni_ttm_series = get_row_ttm(fq, ['Net Income', 'Net Income Common Stockholders'])
    
    # 對齊用的報告日期 (取最近 5 個季度)
    report_dates = fq.columns[-5:][::-1] # 新 -> 舊
    
    # 定義輸出的 DataFrame 結構
    # Rows 依照圖片順序
    metrics = [
        'Market Cap', 
        'Enterprise Value', 
        'Trailing P/E', 
        'Forward P/E', 
        'PEG Ratio (5yr expected)', 
        'Price/Sales', 
        'Price/Book', 
        'Enterprise Value/Revenue', 
        'Enterprise Value/EBITDA'
    ]
    
    # 建立結果字典，Key 為欄位名稱 (日期)
    result_data = {}
    
    # --- 1. 處理 "Current" (當前) 欄位 ---
    # Current 使用 info 中的數據最準確
    mkt_cap_curr = info.get('marketCap')
    ev_curr = info.get('enterpriseValue')
    pe_curr = info.get('trailingPE')
    fwd_pe_curr = info.get('forwardPE')
    peg_curr = info.get('pegRatio')
    ps_curr = info.get('priceToSalesTrailing12Months')
    pb_curr = info.get('priceToBook')
    ev_rev_curr = info.get('enterpriseToRevenue')
    ev_ebitda_curr = info.get('enterpriseToEbitda')

    result_data['Current'] = [
        mkt_cap_curr, ev_curr, pe_curr, fwd_pe_curr, peg_curr, 
        ps_curr, pb_curr, ev_rev_curr, ev_ebitda_curr
    ]

    # --- 2. 處理 "歷史季度" 欄位 ---
    shares_curr = info.get('sharesOutstanding', 0)
    
    for date in report_dates:
        col_name = date.strftime('%m/%d/%Y') # 格式: 10/31/2024
        
        # 1. 找該日期的股價 (若當天休市，找最近的前一天)
        try:
            target_idx = hist.index.get_indexer([date], method='pad')[0]
            price = hist['Close'].iloc[target_idx]
        except:
            price = np.nan
            
        if pd.isna(price):
            result_data[col_name] = [np.nan] * len(metrics)
            continue

        # 2. 獲取該時間點的財報數據
        # 注意: 我們假設在季報日當天已知該季報數據 (簡化處理)
        try:
            # 基本面數據 (TTM)
            rev_val = rev_ttm_series.get(date, np.nan)
            ebitda_val = ebitda_ttm_series.get(date, np.nan)
            ni_val = ni_ttm_series.get(date, np.nan)
            
            # 資產負債表數據 (Point in Time)
            total_debt = np.nan
            cash = np.nan
            equity = np.nan
            
            # 嘗試抓取各種可能的 Debt 欄位
            debt_keys = ['Total Debt', 'Long Term Debt And Capital Lease Obligation']
            for k in debt_keys:
                if k in bs.index and date in bs.columns:
                    total_debt = bs.loc[k, date]
                    break
            
            # 嘗試抓取 Cash
            cash_keys = ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments']
            for k in cash_keys:
                if k in bs.index and date in bs.columns:
                    cash = bs.loc[k, date]
                    break
            
            # 嘗試抓取 Equity (Book Value)
            eq_keys = ['Stockholders Equity', 'Total Equity Gross Minority Interest']
            for k in eq_keys:
                if k in bs.index and date in bs.columns:
                    equity = bs.loc[k, date]
                    break
            
            # 3. 計算各項指標
            # 假設股數變動不大，使用當前股數 (因歷史股數不易精確獲取)
            mkt_cap = price * shares_curr
            
            # Enterprise Value = Market Cap + Debt - Cash
            if pd.notna(total_debt) and pd.notna(cash):
                ev = mkt_cap + total_debt - cash
            else:
                ev = np.nan # 無法計算 EV
            
            # Ratios
            pe = mkt_cap / ni_val if ni_val and ni_val > 0 else np.nan
            ps = mkt_cap / rev_val if rev_val and rev_val > 0 else np.nan
            pb = mkt_cap / equity if equity and equity > 0 else np.nan
            ev_rev = ev / rev_val if pd.notna(ev) and rev_val > 0 else np.nan
            ev_ebitda = ev / ebitda_val if pd.notna(ev) and ebitda_val > 0 else np.nan

            # 歷史的 Forward PE 和 PEG 因為沒有歷史預估數據，設為 None
            fwd_pe = np.nan 
            peg = np.nan 

            result_data[col_name] = [
                mkt_cap, ev, pe, fwd_pe, peg, ps, pb, ev_rev, ev_ebitda
            ]

        except Exception as e:
            result_data[col_name] = [np.nan] * len(metrics)

    # 轉為 DataFrame
    df = pd.DataFrame(result_data, index=metrics)
    return df

# --- 3. 繪圖與其他資料處理 (保留原功能) ---

def get_income_statement(data):
    fq = data['financials_q']
    if fq.empty: return pd.DataFrame()
    
    target_rows = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'Basic EPS']
    rows = []
    for t in target_rows:
        found = [i for i in fq.index if t in i]
        if found: rows.append(found[0])
    
    df = fq.loc[rows].copy()
    
    # 轉置並排序 (最新在左)
    df = df.sort_index(axis=1, ascending=False).head(5)
    
    # 欄位重新命名
    col_map = {}
    for idx in df.index:
        if 'Revenue' in idx: col_map[idx] = '營收'
        elif 'Gross' in idx: col_map[idx] = '毛利'
        elif 'Operating' in idx: col_map[idx] = '營業利益'
        elif 'Net Income' in idx: col_map[idx] = '淨利'
        elif 'EPS' in idx: col_map[idx] = 'EPS'
    
    df = df.rename(index=col_map)
    return df

def plot_charts(data, ticker):
    fq = data['financials_q'].sort_index(axis=1, ascending=True) # 舊到新繪圖
    if fq.empty: return None

    dates = [pd.to_datetime(d) for d in fq.columns]
    
    # 獲取數據
    def get_data(keys):
        for k in keys:
            if k in fq.index: return fq.loc[k]
        return pd.Series([0]*len(dates))

    rev = get_data(['Total Revenue', 'Operating Revenue'])
    ni = get_data(['Net Income', 'Net Income Common Stockholders'])
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # 營收與淨利柱狀圖
    width = 20 # Bar width
    ax1.bar(dates, rev/1e9, width=width, label='Revenue (B)', color='#a8d5ba', alpha=0.7)
    ax1.bar(dates, ni/1e9, width=width, label='Net Income (B)', color='#87cefa', alpha=0.7)
    
    ax1.set_ylabel('Billions ($)', fontweight='bold')
    ax1.set_title(f'{ticker} Quarterly Revenue & Net Income', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    return fig

# --- Main ---

def main():
    st.sidebar.title("個股分析")
    ticker = st.sidebar.text_input("股票代號", "NVDA").upper()
    
    if st.sidebar.button("分析", type="primary"):
        with st.spinner(f'正在分析 {ticker} ...'):
            data = get_stock_data(ticker)
            
        if not data:
            return

        st.title(f"{ticker} 估值與財報儀表板")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("當前股價", f"${data['current_price']:.2f}")
        
        # --- 1. Valuation Measures (Yahoo Style) ---
        st.subheader("Valuation Measures")
        st.caption("數據來源：Yahoo Finance API (歷史 Forward P/E 與 PEG 因 API 限制僅顯示當前值)")
        
        val_df = get_yahoo_valuation_table(data)
        
        # 格式化 DataFrame 顯示
        # 我們將 DataFrame 轉為字串格式以便顯示 T/B/M
        formatted_val_df = val_df.copy()
        
        # 定義哪些列是大數字，哪些是倍數
        large_num_rows = ['Market Cap', 'Enterprise Value']
        
        for idx in val_df.index:
            for col in val_df.columns:
                val = val_df.loc[idx, col]
                if idx in large_num_rows:
                    formatted_val_df.loc[idx, col] = format_large_num(val)
                else:
                    formatted_val_df.loc[idx, col] = format_float(val, decimal=2)

        st.dataframe(formatted_val_df, use_container_width=True)

        # --- 2. 損益表概覽 ---
        st.subheader("Financial Highlights (Quarterly)")
        inc_df = get_income_statement(data)
        
        # 簡單轉置顯示
        if not inc_df.empty:
            # 格式化
            disp_df = inc_df.copy()
            for col in disp_df.columns:
                disp_df[col] = disp_df[col].apply(lambda x: format_large_num(x) if abs(x) > 1000 else f"{x:.2f}")
            st.dataframe(disp_df, use_container_width=True)
            
        # --- 3. 圖表 ---
        st.subheader("Revenue & Net Income Trend")
        fig = plot_charts(data, ticker)
        if fig: st.pyplot(fig)

if __name__ == "__main__":
    main()
