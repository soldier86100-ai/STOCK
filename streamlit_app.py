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

# --- 設定繪圖風格 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 共用工具函式 ---
def format_num(value, currency=False, percent=False, decimal=2):
    """通用數值格式化"""
    if value is None or pd.isna(value): return '-'
    if percent: return f'{value*100:+.{decimal}f}%'
    if currency:
        if abs(value) >= 1e12: return f'${value/1e12:.2f}T'
        if abs(value) >= 1e9: return f'${value/1e9:.2f}B'
        if abs(value) >= 1e6: return f'${value/1e6:.1f}M'
        return f'${value:,.0f}'
    return f'{value:,.{decimal}f}'

def format_large_num_chinese(value):
    """財報專用格式 (Billion, 無小數點)"""
    if value is None or pd.isna(value): return '-'
    # 轉換為 Billion (B) 並取整
    val_in_b = value / 1e9
    return f'{val_in_b:,.0f}B'

# --- 1. 資料獲取層 ---

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="3y", interval="1d", auto_adjust=True)
        
        financials_q = stock.quarterly_financials
        balance_sheet_q = stock.quarterly_balance_sheet
        cashflow_q = stock.quarterly_cashflow
        
        if not info or history.empty: return None
        
        return {
            'info': info,
            'financials_q': financials_q,
            'balance_sheet_q': balance_sheet_q,
            'cashflow_q': cashflow_q,
            'current_price': history['Close'].iloc[-1],
            'history': history,
        }
    except Exception as e:
        return None

# --- 2. 數據計算層 ---

def get_quarterly_valuation_df(data):
    """
    計算季度估值趨勢 (基於 Yahoo 直接提供的 Diluted EPS)
    """
    info = data['info']
    history = data['history']
    fq = data['financials_q']
    
    # 檢查是否有 EPS 欄位
    # Yahoo 欄位名稱可能是 'Diluted EPS' 或 'Basic EPS'
    eps_row_name = None
    for name in ['Diluted EPS', 'Basic EPS']:
        if name in fq.index:
            eps_row_name = name
            break
            
    if not eps_row_name or len(fq.columns) < 5:
        return pd.DataFrame(), {}, {}

    # 1. 準備數據 (由新到舊 -> 轉為舊到新以計算 Rolling)
    eps_q = fq.loc[eps_row_name].sort_index(ascending=True).fillna(0)
    rev_q = fq.loc['Total Revenue'].sort_index(ascending=True).fillna(0)
    
    # 2. 計算 TTM (滾動 4 季總和)
    # 這是最標準的 TTM EPS 計算方式：直接加總最近 4 季的 EPS
    ttm_eps = eps_q.rolling(window=4).sum()
    ttm_rev = rev_q.rolling(window=4).sum()
    
    # 智慧填補：若數據不足 4 季 (例如最早的那幾筆)，用 (單季 * 4) 近似，避免開頭數據空白
    ttm_eps = ttm_eps.fillna(eps_q * 4)
    ttm_rev = ttm_rev.fillna(rev_q * 4)

    # 轉回 (新到舊)
    ttm_eps = ttm_eps.sort_index(ascending=False)
    ttm_rev = ttm_rev.sort_index(ascending=False)

    # 準備股價索引
    history_idx = pd.to_datetime(history.index).tz_localize(None)
    
    # 鎖定最近 5 季
    dates = sorted(ttm_eps.index, reverse=True)[:5]
    
    metrics = []
    multiples = {'PE': [], 'PB': [], 'PS': []}
    
    shares = info.get('sharesOutstanding', 1)

    for date in dates:
        # 找季末股價
        try:
            target_idx = history_idx.get_indexer([pd.to_datetime(date)], method='nearest')[0]
            price = history['Close'].iloc[target_idx] if target_idx != -1 else np.nan
        except: price = np.nan
            
        if pd.isna(price): continue

        # 獲取 TTM 數據
        eps_val = ttm_eps.loc[date]
        rev_val = ttm_rev.loc[date]
        
        # 股東權益
        try: equity = data['balance_sheet_q'].loc['Stockholders Equity', date]
        except: 
            try: equity = data['balance_sheet_q'].loc['Total Equity Gross Minority Interest', date]
            except: equity = np.nan
            
        # EBITDA (年化)
        try: ebitda = data['financials_q'].loc['EBITDA', date] * 4
        except: 
            try: ebitda = data['financials_q'].loc['Ebitda', date] * 4
            except: ebitda = np.nan

        mc = price * shares
        
        # --- 計算倍數 ---
        # P/E = 股價 / TTM EPS (直接使用 Yahoo 提供的 EPS 數據)
        pe = price / eps_val if eps_val > 0 else np.nan
        
        # P/S = 市值 / TTM Revenue
        ps = mc / rev_val if rev_val > 0 else np.nan
        
        # P/B = 市值 / 股東權益
        pb = mc / equity if equity > 0 else np.nan
        
        # 收集有效數據 (用於計算平均值)
        if pd.notna(pe) and pe > 0: multiples['PE'].append(pe)
        if pd.notna(ps) and ps > 0: multiples['PS'].append(ps)
        if pd.notna(pb) and pb > 0: multiples['PB'].append(pb)

        metrics.append({
            '季度': date.strftime('%Y-%m-%d'),
            '市值': format_num(mc, currency=True),
            'P/E (TTM)': format_num(pe, decimal=2),
            'P/S (TTM)': format_num(ps, decimal=2),
            'P/B (MRQ)': format_num(pb, decimal=2),
            'EV/EBITDA': format_num(mc/ebitda, decimal=2) if ebitda > 0 else '-'
        })
    
    return pd.DataFrame(metrics), multiples

def get_financial_summary_with_growth(data):
    """獲取損益表並計算 YoY/QoQ (百分比格式)"""
    fq = data['financials_q']
    bq = data['balance_sheet_q']
    
    if fq.empty: return pd.DataFrame(), pd.DataFrame()

    # 選取關鍵欄位 (支援不同命名)
    target_rows = ['Total Revenue', 'Gross Profit', 'Cost Of Revenue', 'Operating Income', 'Net Income', 'Diluted EPS', 'Basic EPS']
    # 模糊搜尋
    found_rows = []
    for t in target_rows:
        matches = [i for i in fq.index if t in i]
        if matches: found_rows.append(matches[0]) # 取第一個匹配的
    
    income_df = fq.loc[found_rows].copy()
    
    # 補 Gross Profit
    if 'Gross Profit' not in income_df.index:
        rev_idx = [i for i in income_df.index if 'Revenue' in i]
        cost_idx = [i for i in income_df.index if 'Cost' in i]
        if rev_idx and cost_idx:
            income_df.loc['Gross Profit'] = income_df.loc[rev_idx[0]] - income_df.loc[cost_idx[0]]

    # 計算成長率 (舊到新)
    df_sorted = income_df.sort_index(axis=1, ascending=True)
    qoq = df_sorted.pct_change(axis=1)
    yoy = df_sorted.pct_change(axis=1, periods=4)
    
    # 轉回 (新到舊)
    income_df = income_df.sort_index(axis=1, ascending=False)
    qoq = qoq.sort_index(axis=1, ascending=False)
    yoy = yoy.sort_index(axis=1, ascending=False)
    
    final_df = income_df.T
    
    # 安全添加成長率
    def get_growth(growth_df, keyword):
        matches = [i for i in growth_df.index if keyword in i]
        if matches: return growth_df.loc[matches[0]]
        return pd.Series([np.nan]*len(final_df), index=final_df.index)

    final_df['營收 YoY'] = get_growth(yoy, 'Revenue')
    final_df['營收 QoQ'] = get_growth(qoq, 'Revenue')
    final_df['淨利 YoY'] = get_growth(yoy, 'Net Income')
    final_df['淨利 QoQ'] = get_growth(qoq, 'Net Income')
    
    final_df = final_df.T
    
    # 資產負債表
    bs_rows = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity', 'Total Debt']
    valid_bs_rows = [r for r in bs_rows if r in bq.index]
    bs_df = bq.loc[valid_bs_rows].copy() if valid_bs_rows else pd.DataFrame()

    # 截取前 5 季
    final_df = final_df.iloc[:, :5]
    bs_df = bs_df.iloc[:, :5] if not bs_df.empty else bs_df
    
    # 格式化日期
    final_df.columns = [d.strftime('%Y-%m-%d') for d in final_df.columns]
    if not bs_df.empty: bs_df.columns = [d.strftime('%Y-%m-%d') for d in bs_df.columns]

    return final_df, bs_df

def calculate_valuation(data, income_df, multiples, custom_g):
    info = data['info']
    
    # 1. 基礎指標：TTM EPS (優先使用財報計算值，以確保與 P/E 邏輯一致)
    try:
        eps_row = [i for i in income_df.index if 'EPS' in i][0]
        # income_df 的前 4 列是最近 4 季 (因為已經轉為新到舊並截取了)
        # 但 income_df 裡面的 EPS 是單季的，我們需要 sum(最近4季)
        # 注意: income_df 可能只有 5 列，這裡我們取前 4 列相加
        ttm_eps = income_df.loc[eps_row].iloc[:4].sum()
    except: 
        ttm_eps = info.get('trailingEps') # Fallback

    # TTM RPS
    shares = info.get('sharesOutstanding', 1)
    try:
        rev_row = [i for i in income_df.index if 'Revenue' in i and 'YoY' not in i and 'QoQ' not in i][0]
        ttm_rev = income_df.loc[rev_row].iloc[:4].sum()
        ttm_rps = ttm_rev / shares
    except: ttm_rps = np.nan
    
    # BVPS
    try:
        equity = data['balance_sheet_q'].iloc[0,0] # 近一季
        bvps = equity / shares
    except: bvps = info.get('bookValue')

    # 2. 成長率
    try: rev_g = income_df.loc['營收 YoY'].iloc[0]
    except: rev_g = info.get('revenueGrowth', 0)
    
    try: ni_g = income_df.loc['淨利 YoY'].iloc[0]
    except: ni_g = info.get('earningsGrowth', 0)

    results = []
    
    def add_row(name, base, growth, hist_list):
        if pd.isna(base) or not hist_list: return
        avg = np.mean(hist_list)
        std = np.std(hist_list)
        if pd.isna(std) or len(hist_list) < 2: std = avg * 0.1
        
        g = growth if pd.notna(growth) else 0
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

# --- 3. 繪圖 (保持不變，已修復) ---
def plot_charts(data, income_df, ticker):
    # 略 (使用前面的 plot_charts 邏輯即可，這裡為節省篇幅簡略，實際應包含完整繪圖代碼)
    # 為確保完整性，這裡重寫一次關鍵部分
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in income_df.columns][::-1]
    
    # 找出對應列名
    rev_row = [i for i in income_df.index if 'Revenue' in i and 'YoY' not in i][0]
    ni_row = [i for i in income_df.index if 'Net Income' in i and 'YoY' not in i][0]
    eps_row = [i for i in income_df.index if 'EPS' in i][0]

    rev = income_df.loc[rev_row].values[::-1]
    net_inc = income_df.loc[ni_row].values[::-1]
    eps = income_df.loc[eps_row].values[::-1]
    
    shares = data['info'].get('sharesOutstanding', 1)
    hist = data['history']
    market_caps = []
    for d in dates:
        try:
            idx = hist.index.get_indexer([d], method='nearest')[0]
            price = hist['Close'].iloc[idx]
            market_caps.append(price * shares)
        except: market_caps.append(np.nan)
    
    ps_ratio = np.array(market_caps) / (rev * 4) 

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 圖 1
    ax1.bar(dates, rev/1e9, color='#A8D5BA', width=20, label='Revenue (B)')
    ax1.set_ylabel('Revenue ($B)', color='green')
    ax1_r = ax1.twinx()
    ax1_r.plot(dates, ps_ratio, color='purple', marker='o', label='P/S Ratio')
    ax1_r.set_ylabel('P/S Ratio', color='purple')
    ax1.set_title(f'{ticker} Revenue & P/S Trend')
    
    # 圖 2
    ax2.bar(dates, net_inc/1e9, color='#87CEFA', width=20, label='Net Income (B)')
    ax2.set_ylabel('Net Income ($B)', color='blue')
    ax2_r = ax2.twinx()
    ax2_r.plot(dates, eps, color='orange', marker='o', label='EPS')
    ax2_r.set_ylabel('EPS ($)', color='orange')
    ax2.set_title(f'{ticker} Net Income & EPS Trend')

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
        q_val_df, multiples = get_quarterly_valuation_df(data)
        inc_df, bs_df = get_financial_summary_with_growth(data)
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
                '市值': format_num, # 預設格式
                'P/E (TTM)': '{:.2f}',
                'P/S (TTM)': '{:.2f}',
                'P/B (MRQ)': '{:.2f}',
                'EV/EBITDA': '{:.2f}'
            }), use_container_width=True)

        # 3. 損益表 (格式修復: 百分比、EPS 兩位、Billion)
        st.subheader("3. 損益表 (單位: Billion)")
        if not inc_df.empty:
            # 動態建立格式字典
            fmt_dict = {}
            for idx in inc_df.index:
                if 'YoY' in idx or 'QoQ' in idx:
                    fmt_dict[idx] = '{:.2%}'
                elif 'EPS' in idx:
                    fmt_dict[idx] = '{:.2f}'
                else:
                    fmt_dict[idx] = format_large_num_chinese # 使用 Billion 格式函式
            
            # 轉置後顯示比較直觀 (日期在上方，項目在左側)
            # 但使用者通常習慣日期在左側 (Row) 或是上方 (Col)? 
            # 之前的圖是日期在上方 (Col)。
            # style.format 需要 index 對應
            
            st.dataframe(inc_df.style.format(fmt_dict, na_rep="-"), use_container_width=True)

        # 4. 圖表
        st.subheader("4. 趨勢圖表")
        fig = plot_charts(data, inc_df, ticker)
        if fig: st.pyplot(fig)

        # 5. 資產負債表
        st.subheader("5. 資產負債表 (單位: Billion)")
        if not bs_df.empty:
             st.dataframe(bs_df.style.format(format_large_num_chinese, na_rep="-"), use_container_width=True)

if __name__ == "__main__":
    main()
