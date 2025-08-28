#!/usr/bin/env python3
"""
JuusoTrader - Kaupankäynti Dashboard
Real-time monitoring dashboard for paper trading accounts
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
import csv
import yfinance as yf
from typing import Dict, List, Tuple
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="JuusoTrader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_benchmark_data(symbols: List[str] = ["^IXIC", "^GSPC", "SPY", "QQQ"], days: int = 90) -> pd.DataFrame:
    """Load benchmark data (NASDAQ, S&P 500, etc.) for comparison"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        benchmark_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    # Calculate normalized returns (starting from 100)
                    returns = (data['Close'] / data['Close'].iloc[0]) * 100
                    benchmark_data[symbol] = returns
            except Exception as e:
                st.warning(f"Ei voitu ladata dataa symbolille {symbol}: {e}")
                continue
        
        if benchmark_data:
            df = pd.DataFrame(benchmark_data)
            df.index.name = 'Date'
            return df
        else:
            # Fallback: create dummy data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            dummy_data = {
                'NASDAQ': np.cumsum(np.random.randn(len(dates))) + 100,
                'S&P500': np.cumsum(np.random.randn(len(dates))) + 100
            }
            return pd.DataFrame(dummy_data, index=dates)
    except Exception as e:
        st.error(f"Virhe benchmark-datan lataamisessa: {e}")
        return pd.DataFrame()

def load_trade_logs() -> Dict[str, pd.DataFrame]:
    """Load trade logs for all strategies"""
    logs = {}
    log_dir = Path('storage/logs')
    
    if not log_dir.exists():
        return logs
    
    for log_file in log_dir.glob('trades_*.csv'):
        strategy_name = log_file.stem.replace('trades_', '')
        try:
            df = pd.read_csv(log_file)
            if not df.empty:
                df['ts'] = pd.to_datetime(df['ts'])
                df['strategy'] = strategy_name
                logs[strategy_name] = df
        except Exception as e:
            st.warning(f"Virhe kauppalokien lataamisessa ({strategy_name}): {e}")
    
    return logs

def calculate_account_performance(trade_logs: Dict[str, pd.DataFrame], starting_nav: float = 100000) -> Dict[str, pd.DataFrame]:
    """Calculate individual account performance"""
    account_performance = {}
    
    # Map strategies to accounts
    strategy_to_account = {
        'EMA': 'Account A - EMA',
        'XGB': 'Account B - XGB', 
        'ACCOUNT_C_ML': 'Account C - Enhanced ML'
    }
    
    for strategy, trades in trade_logs.items():
        if trades.empty:
            continue
            
        account_name = strategy_to_account.get(strategy, f'Account {strategy}')
        
        # Calculate running P&L for this account
        portfolio_value = starting_nav
        portfolio_history = []
        current_positions = {}  # symbol -> (qty, avg_price)
        
        for _, trade in trades.iterrows():
            symbol = trade['symbol']
            side = trade['side']
            qty = trade['qty']
            price = trade['price']
            
            if side == 'buy':
                if symbol in current_positions:
                    old_qty, old_price = current_positions[symbol]
                    new_qty = old_qty + qty
                    new_avg_price = ((old_qty * old_price) + (qty * price)) / new_qty
                    current_positions[symbol] = (new_qty, new_avg_price)
                else:
                    current_positions[symbol] = (qty, price)
                portfolio_value -= qty * price
                
            elif side == 'sell':
                if symbol in current_positions:
                    old_qty, avg_price = current_positions[symbol]
                    if old_qty >= qty:
                        realized_pnl = qty * (price - avg_price)
                        portfolio_value += qty * price
                        new_qty = old_qty - qty
                        if new_qty > 0:
                            current_positions[symbol] = (new_qty, avg_price)
                        else:
                            del current_positions[symbol]
            
            portfolio_history.append({
                'date': trade['ts'],
                'portfolio_value': portfolio_value,
                'total_pnl': portfolio_value - starting_nav,
                'daily_return': ((portfolio_value - starting_nav) / starting_nav) * 100
            })
        
        if portfolio_history:
            account_performance[account_name] = pd.DataFrame(portfolio_history)
    
    # Return accounts in A, B, C order
    ordered_performance = {}
    account_order = [
        'Account A - EMA',
        'Account B - XGB', 
        'Account C - Enhanced ML'
    ]
    
    for account_name in account_order:
        if account_name in account_performance:
            ordered_performance[account_name] = account_performance[account_name]
    
    return ordered_performance

def calculate_portfolio_performance(trade_logs: Dict[str, pd.DataFrame], starting_nav: float = 100000) -> pd.DataFrame:
    """Calculate portfolio performance over time"""
    all_trades = []
    
    for strategy, trades in trade_logs.items():
        if not trades.empty:
            trades_copy = trades.copy()
            trades_copy['strategy'] = strategy
            all_trades.append(trades_copy)
    
    if not all_trades:
        # Return dummy data if no trades
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        return pd.DataFrame({
            'date': dates,
            'portfolio_value': [starting_nav] * len(dates),
            'total_pnl': [0] * len(dates),
            'daily_return': [0] * len(dates)
        })
    
    # Combine all trades
    combined_trades = pd.concat(all_trades, ignore_index=True)
    combined_trades = combined_trades.sort_values('ts')
    
    # Calculate running P&L
    portfolio_value = starting_nav
    portfolio_history = []
    
    current_positions = {}  # symbol -> (qty, avg_price)
    
    for _, trade in combined_trades.iterrows():
        symbol = trade['symbol']
        side = trade['side']
        qty = trade['qty']
        price = trade['price']
        
        if side == 'buy':
            if symbol in current_positions:
                old_qty, old_price = current_positions[symbol]
                new_qty = old_qty + qty
                new_avg_price = ((old_qty * old_price) + (qty * price)) / new_qty
                current_positions[symbol] = (new_qty, new_avg_price)
            else:
                current_positions[symbol] = (qty, price)
            
            portfolio_value -= qty * price  # Cash out
            
        elif side == 'sell':
            if symbol in current_positions:
                old_qty, avg_price = current_positions[symbol]
                if old_qty >= qty:
                    # Calculate realized P&L
                    realized_pnl = qty * (price - avg_price)
                    portfolio_value += qty * price  # Cash in
                    
                    # Update position
                    new_qty = old_qty - qty
                    if new_qty > 0:
                        current_positions[symbol] = (new_qty, avg_price)
                    else:
                        del current_positions[symbol]
        
        portfolio_history.append({
            'date': trade['ts'],
            'portfolio_value': portfolio_value,
            'trade_type': f"{side} {symbol}",
            'trade_value': qty * price
        })
    
    if portfolio_history:
        df = pd.DataFrame(portfolio_history)
        df['total_pnl'] = df['portfolio_value'] - starting_nav
        df['daily_return'] = df['portfolio_value'].pct_change() * 100
        return df
    else:
        # Return starting value if no trades
        return pd.DataFrame({
            'date': [datetime.now()],
            'portfolio_value': [starting_nav],
            'total_pnl': [0],
            'daily_return': [0]
        })

def create_performance_chart(portfolio_df: pd.DataFrame, benchmark_df: pd.DataFrame):
    """Create performance comparison chart with benchmarks"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Portfolion Arvo vs. Vertailuindeksit', 'Päivittäinen Tuotto %'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Portfolio performance (normalized to 100 at start)
    if not portfolio_df.empty:
        portfolio_norm = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0]) * 100
        fig.add_trace(
            go.Scatter(
                x=portfolio_df['date'],
                y=portfolio_norm,
                name='JuusoTrader Portfolio',
                line=dict(color='#00ff00', width=3),
                hovertemplate='%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add benchmark indices
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    benchmark_names = {
        '^IXIC': 'NASDAQ',
        '^GSPC': 'S&P 500', 
        'SPY': 'SPY ETF',
        'QQQ': 'QQQ ETF'
    }
    
    for i, (symbol, data) in enumerate(benchmark_df.items()):
        if len(data.dropna()) > 0:
            display_name = benchmark_names.get(str(symbol), str(symbol))
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    name=f'Vertailu: {display_name}',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    hovertemplate='%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Daily returns
    if not portfolio_df.empty and 'daily_return' in portfolio_df.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_df['date'],
                y=portfolio_df['daily_return'],
                name='Päivittäinen Tuotto',
                line=dict(color='#ffa500'),
                fill='tozeroy',
                hovertemplate='%{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title="JuusoTrader Suorituskyky",
        height=700,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Päivämäärä", row=2, col=1)
    fig.update_yaxes(title_text="Normalisoitu Arvo (100 = aloitusarvo)", row=1, col=1)
    fig.update_yaxes(title_text="Tuotto %", row=2, col=1)
    
    return fig

def create_trades_table(trade_logs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a comprehensive trades table"""
    all_trades = []
    
    for strategy, trades in trade_logs.items():
        if not trades.empty:
            trades_copy = trades.copy()
            trades_copy['strategia'] = strategy
            all_trades.append(trades_copy)
    
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined = combined.sort_values('ts', ascending=False)
        
        # Rename columns to Finnish
        combined_display = combined.rename(columns={
            'ts': 'Aika',
            'symbol': 'Symboli', 
            'side': 'Toiminto',
            'qty': 'Määrä',
            'price': 'Hinta',
            'strategia': 'Strategia'
        })
        
        return combined_display
    else:
        return pd.DataFrame(columns=['Aika', 'Symboli', 'Toiminto', 'Määrä', 'Hinta', 'Strategia'])

def main():
    # Header
    st.title("📈 JuusoTrader - Kaupankäynti Dashboard")
    
    # System status check
    import os
    trade_logs_exist = os.path.exists("storage/logs") and len(os.listdir("storage/logs")) > 0
    
    if trade_logs_exist:
        st.success("✅ Kaupankäyntijärjestelmä on aktiivinen")
    else:
        st.warning("⚠️ Live engine ei ole käynnissä - näytetään demo-data")
        st.info("💡 Käynnistä kaupankäynti: `python launch_nonblocking.py`")
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Asetukset")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Automaattinen päivitys (30s)", value=False)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Päivitä nyt"):
        st.cache_data.clear()
        st.rerun()
    
    # Benchmark selection
    st.sidebar.subheader("Vertailuindeksit")
    show_nasdaq = st.sidebar.checkbox("NASDAQ", value=True)
    show_sp500 = st.sidebar.checkbox("S&P 500", value=True) 
    show_spy = st.sidebar.checkbox("SPY ETF", value=False)
    show_qqq = st.sidebar.checkbox("QQQ ETF", value=False)
    
    selected_benchmarks = []
    if show_nasdaq: selected_benchmarks.append("^IXIC")
    if show_sp500: selected_benchmarks.append("^GSPC")
    if show_spy: selected_benchmarks.append("SPY")
    if show_qqq: selected_benchmarks.append("QQQ")
    
    # Time range
    st.sidebar.subheader("Aikaväli")
    days_back = st.sidebar.selectbox(
        "Näytä viimeiset:",
        [30, 60, 90, 180, 365],
        index=2,
        format_func=lambda x: f"{x} päivää"
    )
    
    # Load data
    with st.spinner("Ladataan tietoja..."):
        trade_logs = load_trade_logs()
        benchmark_df = load_benchmark_data(selected_benchmarks, days_back) if selected_benchmarks else pd.DataFrame()
        portfolio_df = calculate_portfolio_performance(trade_logs)
        account_performance = calculate_account_performance(trade_logs)
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if not portfolio_df.empty:
        current_value = portfolio_df['portfolio_value'].iloc[-1]
        total_pnl = portfolio_df['total_pnl'].iloc[-1]
        starting_value = 100000  # From config
        total_return_pct = (total_pnl / starting_value) * 100
        
        # Count trades
        total_trades = sum(len(trades) for trades in trade_logs.values())
        
        with col1:
            st.metric("💰 Portfolion Arvo", f"€{current_value:,.2f}")
        
        with col2:
            st.metric("📊 Kokonais P&L", f"€{total_pnl:,.2f}", f"{total_return_pct:+.2f}%")
        
        with col3:
            if len(portfolio_df) > 1:
                recent_return = portfolio_df['daily_return'].iloc[-1]
                st.metric("📈 Viimeisin Päivätuotto", f"{recent_return:.2f}%")
            else:
                st.metric("📈 Viimeisin Päivätuotto", "0.00%")
        
        with col4:
            st.metric("🔄 Kauppoja Yhteensä", f"{total_trades}")
    else:
        with col1:
            st.metric("💰 Portfolion Arvo", "€100,000.00")
        with col2:
            st.metric("📊 Kokonais P&L", "€0.00", "0.00%")
        with col3:
            st.metric("📈 Viimeisin Päivätuotto", "0.00%")
        with col4:
            st.metric("🔄 Kauppoja Yhteensä", "0")
    
    st.markdown("---")
    
    # Performance chart
    st.subheader("📈 Suorituskykykuvaaja")
    
    if not portfolio_df.empty or not benchmark_df.empty:
        fig = create_performance_chart(portfolio_df, benchmark_df)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Ei vielä kaupankäyntidataa näytettäväksi.")
    
    # Individual Account Performance
    st.markdown("---")
    st.subheader("🏦 Tilien Erillinen Suorituskyky")
    
    if account_performance:
        # Create tabs for each account
        account_names = list(account_performance.keys())
        tabs = st.tabs(account_names)
        
        for i, (account_name, perf_data) in enumerate(account_performance.items()):
            with tabs[i]:
                if not perf_data.empty:
                    # Account metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    current_value = perf_data['portfolio_value'].iloc[-1]
                    total_pnl = perf_data['total_pnl'].iloc[-1]
                    total_return_pct = (total_pnl / 100000) * 100
                    trade_count = len(perf_data)
                    
                    with col1:
                        st.metric("💰 Tilin Arvo", f"€{current_value:,.2f}")
                    with col2:
                        st.metric("📊 P&L", f"€{total_pnl:,.2f}", f"{total_return_pct:+.2f}%")
                    with col3:
                        if len(perf_data) > 1:
                            recent_return = perf_data['daily_return'].iloc[-1]
                            st.metric("📈 Tuotto %", f"{recent_return:.2f}%")
                        else:
                            st.metric("📈 Tuotto %", "0.00%")
                    with col4:
                        st.metric("🔄 Kauppoja", f"{trade_count}")
                    
                    # Account performance chart
                    fig_account = go.Figure()
                    
                    # Portfolio value line
                    fig_account.add_trace(go.Scatter(
                        x=perf_data['date'],
                        y=perf_data['portfolio_value'],
                        mode='lines',
                        name=f'{account_name} Arvo',
                        line=dict(color='#1f77b4', width=3)
                    ))
                    
                    # Starting value reference line
                    fig_account.add_hline(y=100000, line_dash="dash", line_color="gray", 
                                        annotation_text="Aloitusarvo (100k)")
                    
                    fig_account.update_layout(
                        title=f"{account_name} - Portfolion Kehitys",
                        xaxis_title="Päivämäärä",
                        yaxis_title="Portfolion Arvo (€)",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_account, width='stretch')
                else:
                    st.info(f"Ei vielä kaupankäyntidataa tilille {account_name}")
    else:
        # Show placeholder for accounts
        col1, col2, col3 = st.columns(3)
        
        accounts_info = [
            ("Account A - EMA", "EMA Trend Strategy", "100k"),
            ("Account B - XGB", "XGBoost ML Strategy", "100k"), 
            ("Account C - Enhanced ML", "ML + News Sentiment", "100k")
        ]
        
        for i, (name, strategy, capital) in enumerate(accounts_info):
            with [col1, col2, col3][i]:
                st.write(f"**{name}**")
                st.write(f"Strategia: {strategy}")
                st.write(f"Pääoma: €{capital}")
                st.metric("Arvo", f"€100,000.00")
                st.metric("P&L", "€0.00", "0.00%")
                st.info("Ei vielä kaupankäyntidataa")
    
    st.markdown("---")
    
    # Recent trades table
    st.subheader("📋 Viimeisimmät Kaupat")
    
    trades_df = create_trades_table(trade_logs)
    if not trades_df.empty:
        # Show last 20 trades
        st.dataframe(
            trades_df.head(20),
            width='stretch',
            hide_index=True
        )
        
        # Download button for full trade history
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Lataa Kaikki Kaupat (CSV)",
            data=csv,
            file_name=f"juusotrader_kaupat_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Ei vielä kauppoja näytettäväksi.")
    
    # Account status
    st.markdown("---")
    st.subheader("🏦 Tilit ja Strategiat")
    
    account_col1, account_col2, account_col3 = st.columns(3)
    
    with account_col1:
        st.write("**Tili A - Klassinen**")
        st.write("✅ EMA Trend (100k pääoma)")
        st.write("Status: Aktiivinen")
    
    with account_col2:
        st.write("**Tili B - ML**") 
        st.write("✅ XGBoost Classifier (100k pääoma)")
        st.write("Status: Aktiivinen")
    
    with account_col3:
        st.write("**Tili C - Enhanced ML**")
        st.write("✅ ML + News Sentiment (100k pääoma)")
        st.write("✅ Pattern Recognition")
        st.write("✅ Ensemble Methods")
        st.write("Status: Aktiivinen")
    
    # Footer
    st.markdown("---")
    st.markdown(f"*Viimeksi päivitetty: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Auto-refresh timer
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
