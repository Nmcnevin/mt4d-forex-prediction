import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
API_KEY = "9098809f5f804dbeba36dfdbda22fe81"
BASE_URL = "https://api.twelvedata.com"

# Currency pairs - TwelveData format
CURRENCY_PAIRS = {
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD"
}

# Timeframes
TIMEFRAMES = {
    "M1": "1min",
    "M5": "5min"
}

# ===========================
# DATA COLLECTION & CSV FUNCTIONS
# ===========================
def collect_historical_data(symbol, interval="1min", outputsize=5000):
    """Collect large historical dataset from TwelveData API"""
    try:
        st.info(f"🔄 Collecting historical data for {symbol}...")
        
        url = f"{BASE_URL}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": API_KEY
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
            
            # Add volume as 0 for Forex (no volume data available)
            if 'volume' not in df.columns:
                df['volume'] = 0
            else:
                df['volume'] = pd.to_numeric(df['volume'])
            
            st.success(f"✅ Collected {len(df)} candles")
            return df
        else:
            st.error(f"API Error: {data}")
            return None
            
    except Exception as e:
        st.error(f"Error collecting data: {e}")
        return None

def save_to_csv(df, symbol, interval):
    """Save dataframe to CSV file"""
    try:
        # Clean filename - remove slashes
        clean_symbol = symbol.replace("/", "")
        filename = f"{clean_symbol}_{interval}_data.csv"
        df.to_csv(filename, index=False)
        st.success(f"✅ Data saved to {filename}")
        return filename
    except Exception as e:
        st.error(f"Error saving CSV: {e}")
        return None

def load_from_csv(filename):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        st.success(f"✅ Loaded {len(df)} candles from {filename}")
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

# ===========================
# DATA FETCHING FUNCTIONS
# ===========================
def fetch_live_data(symbol, interval="1min", outputsize=100):
    """Fetch live data from TwelveData API"""
    try:
        url = f"{BASE_URL}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": API_KEY
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
            
            # Add volume as 0 for Forex (no volume data available)
            if 'volume' not in df.columns:
                df['volume'] = 0
            else:
                df['volume'] = pd.to_numeric(df['volume'])
            
            return df
        else:
            st.error(f"API Error: {data}")
            return None
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ===========================
# FEATURE ENGINEERING
# ===========================
def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    if df is None or len(df) < 30:
        return df
    
    # Simple Moving Averages (reduce window for smaller datasets)
    df['SMA_5'] = df['close'].rolling(window=5, min_periods=3).mean()
    df['SMA_10'] = df['close'].rolling(window=10, min_periods=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20, min_periods=10).mean()
    
    # Exponential Moving Average
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False, min_periods=3).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False, min_periods=5).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=7).mean()
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False, min_periods=6).mean()
    exp2 = df['close'].ewm(span=26, adjust=False, min_periods=13).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=5).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20, min_periods=10).mean()
    bb_std = df['close'].rolling(window=20, min_periods=10).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14, min_periods=7).mean()
    
    # Price momentum
    df['momentum'] = df['close'] - df['close'].shift(4)
    
    # Price change percentage
    df['price_change'] = df['close'].pct_change()
    
    # Candle body size (useful for Forex without volume)
    df['candle_size'] = abs(df['close'] - df['open'])
    
    # High-Low range
    df['hl_range'] = df['high'] - df['low']
    
    # Fill any remaining NaN with forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def prepare_features(df):
    """Prepare features for ML model"""
    if df is None or len(df) < 30:
        return None, None, None
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop only infinite values, keep as much data as possible
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    if len(df) < 20:
        return None, None, None
    
    # Create target: 1 if next candle closes higher, 0 otherwise
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Select features (updated for Forex - added candle_size and hl_range instead of volume_change)
    feature_columns = ['open', 'high', 'low', 'close', 'volume',
                      'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10',
                      'RSI', 'MACD', 'Signal_Line', 'BB_upper', 'BB_lower',
                      'ATR', 'momentum', 'price_change', 'candle_size', 'hl_range']
    
    # Remove last row (no target)
    df = df[:-1]
    
    X = df[feature_columns].values
    y = df['target'].values
    
    return X, y, df

# ===========================
# ML MODEL TRAINING
# ===========================
def train_model(X, y):
    """Train Random Forest model"""
    if X is None or len(X) < 20:
        return None, None
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# ===========================
# MODEL EVALUATION METRICS
# ===========================
def calculate_metrics(model, scaler, X, y, df_processed):
    """Calculate comprehensive model performance metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Make predictions
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)
    
    # Classification Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    
    # Trading Performance Simulation
    trades = simulate_trades(df_processed, y_pred, y)
    
    metrics = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'confusion_matrix': cm,
        'trades': trades
    }
    
    return metrics

def simulate_trades(df, predictions, actual):
    """Simulate trading performance based on predictions"""
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    total_loss = 0
    wins = []
    losses = []
    
    for i in range(len(predictions)):
        if i >= len(df):
            break
            
        # Get ATR for TP/SL calculation
        if 'ATR' in df.columns:
            atr = df.iloc[i]['ATR']
        else:
            atr = 0.0015  # Default ATR
        
        # Calculate profit/loss based on prediction vs actual
        if predictions[i] == actual[i]:
            # Correct prediction
            if predictions[i] == 1:  # BUY prediction was correct
                profit = atr * 2  # TP hit (2x ATR)
                total_profit += profit
                winning_trades += 1
                wins.append(profit * 100000)  # Convert to pips for standard lot
            else:  # SELL prediction was correct
                profit = atr * 2
                total_profit += profit
                winning_trades += 1
                wins.append(profit * 100000)
        else:
            # Wrong prediction
            loss = atr * 1  # SL hit (1x ATR)
            total_loss += loss
            losing_trades += 1
            losses.append(loss * 100000)
    
    total_trades = winning_trades + losing_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    net_profit = total_profit - total_loss
    profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
    
    # Convert to dollars (assuming 1 standard lot = $10 per pip)
    net_profit_dollars = net_profit * 100000 * 10
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'net_profit': net_profit_dollars,
        'total_profit': total_profit * 100000 * 10,
        'total_loss': total_loss * 100000 * 10,
        'profit_factor': profit_factor,
        'avg_win_pips': avg_win,
        'avg_loss_pips': avg_loss
    }

# ===========================
# PREDICTION FUNCTIONS
# ===========================
def make_prediction(model, scaler, df):
    """Make buy/sell prediction"""
    if model is None or df is None or len(df) < 30:
        return None
    
    # Add indicators
    df = add_technical_indicators(df)
    df = df.dropna()
    
    if len(df) < 1:
        return None
    
    # Get latest data
    latest = df.iloc[-1]
    
    feature_columns = ['open', 'high', 'low', 'close', 'volume',
                      'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10',
                      'RSI', 'MACD', 'Signal_Line', 'BB_upper', 'BB_lower',
                      'ATR', 'momentum', 'price_change', 'candle_size', 'hl_range']
    
    X_latest = latest[feature_columns].values.reshape(1, -1)
    X_scaled = scaler.transform(X_latest)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    # Calculate TP, SL based on ATR
    current_price = latest['close']
    atr = latest['ATR']
    
    if prediction == 1:  # BUY signal
        signal = "BUY"
        entry_price = current_price
        take_profit = entry_price + (atr * 2)
        stop_loss = entry_price - (atr * 1)
        confidence = probability[1] * 100
    else:  # SELL signal
        signal = "SELL"
        entry_price = current_price
        take_profit = entry_price - (atr * 2)
        stop_loss = entry_price + (atr * 1)
        confidence = probability[0] * 100
    
    # Calculate risk-reward ratio
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    rr_ratio = reward / risk if risk > 0 else 0
    
    return {
        'signal': signal,
        'entry_price': round(entry_price, 5),
        'take_profit': round(take_profit, 5),
        'stop_loss': round(stop_loss, 5),
        'current_price': round(current_price, 5),
        'confidence': round(confidence, 2),
        'rr_ratio': round(rr_ratio, 2),
        'atr': round(atr, 5),
        'rsi': round(latest['RSI'], 2)
    }

# ===========================
# STREAMLIT UI
# ===========================
def main():
    st.set_page_config(page_title="MT4D Forex Prediction", layout="wide")
    
    st.title("🚀 MT4D – Forex Prediction & Live Signal System")
    st.markdown("Real-time Buy/Sell predictions for ultra-short-term trading")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    selected_pair = st.sidebar.selectbox("Select Currency Pair", list(CURRENCY_PAIRS.keys()))
    selected_timeframe = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()))
    
    # Get correct symbol format for API
    symbol = CURRENCY_PAIRS[selected_pair]
    interval = TIMEFRAMES[selected_timeframe]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**API Status:** Connected ✅\n**Pair:** {selected_pair}\n**Timeframe:** {selected_timeframe}")
    
    # Tabs for different operations
    tab1, tab2, tab3 = st.tabs(["📥 Data Collection", "🧠 Model Training", "🎯 Live Predictions"])
    
    # ===========================
    # TAB 1: DATA COLLECTION
    # ===========================
    with tab1:
        st.header("📥 Collect Historical Data & Save CSV")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Download from API")
            data_size = st.selectbox("Dataset Size", [200, 500, 1000, 2000, 5000], index=0)
            
            st.info("💡 Recommended: Use at least **200 candles** for better model training")
            
            if st.button("📊 Collect & Save Historical Data", type="primary"):
                with st.spinner(f"Collecting {data_size} candles..."):
                    df = collect_historical_data(symbol, interval, data_size)
                    
                    if df is not None:
                        filename = save_to_csv(df, symbol, interval)
                        
                        if filename:
                            st.session_state['csv_filename'] = filename
                            st.session_state['collected_df'] = df
                            
                            # Show stats
                            st.markdown("### 📊 Dataset Statistics")
                            st.write(f"**Total Candles:** {len(df)}")
                            st.write(f"**Date Range:** {df['datetime'].min()} to {df['datetime'].max()}")
                            st.write(f"**Columns:** {', '.join(df.columns)}")
                            
                            # Show preview
                            st.markdown("### 👀 Data Preview")
                            st.dataframe(df.head(20), use_container_width=True)
        
        with col2:
            st.subheader("Upload Existing CSV")
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
                
                st.success(f"✅ Uploaded {len(df)} candles")
                st.session_state['collected_df'] = df
                st.session_state['csv_filename'] = uploaded_file.name
                
                # Show preview
                st.markdown("### 👀 Data Preview")
                st.dataframe(df.head(20), use_container_width=True)
    
    # ===========================
    # TAB 2: MODEL TRAINING
    # ===========================
    with tab2:
        st.header("🧠 Train Machine Learning Model")
        
        if 'collected_df' in st.session_state:
            df = st.session_state['collected_df']
            
            st.info(f"📊 Ready to train with **{len(df)} candles** from `{st.session_state.get('csv_filename', 'uploaded file')}`")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("🚀 Train Model from CSV", type="primary", use_container_width=True):
                    with st.spinner("Training model..."):
                        # Show initial data info
                        st.info(f"📊 Starting with {len(df)} candles from CSV")
                        
                        # Prepare features
                        X, y, df_processed = prepare_features(df)
                        
                        if X is not None:
                            st.success(f"✅ Prepared {len(X)} training samples with 20 features")
                            
                            # Train model
                            model, scaler = train_model(X, y)
                            
                            if model is not None:
                                # Store in session state
                                st.session_state['model'] = model
                                st.session_state['scaler'] = scaler
                                st.session_state['symbol'] = symbol
                                st.session_state['interval'] = interval
                                
                                # Calculate comprehensive metrics
                                with st.spinner("Calculating performance metrics..."):
                                    metrics = calculate_metrics(model, scaler, X, y, df_processed)
                                
                                # Show results
                                st.success(f"✅ **Model Trained Successfully!**")
                                
                                # ===========================
                                # CLASSIFICATION METRICS (Simple Display)
                                # ===========================
                                st.markdown("### 📊 Model Performance Metrics")
                                
                                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                
                                with col_m1:
                                    st.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
                                
                                with col_m2:
                                    st.metric("Precision", f"{metrics['precision']:.2f}%")
                                
                                with col_m3:
                                    st.metric("Recall", f"{metrics['recall']:.2f}%")
                                
                                with col_m4:
                                    st.metric("F1-Score", f"{metrics['f1_score']:.2f}%")
                            else:
                                st.error("Failed to train model")
                        else:
                            # Debug: Show what happened during processing
                            st.error(f"❌ Processing failed!")
                            
                            # Check if df is valid
                            if df is None:
                                st.warning("CSV data is None")
                            elif len(df) < 30:
                                st.warning(f"CSV has only {len(df)} rows - need at least 30")
                            else:
                                st.warning(f"CSV has {len(df)} rows but processing removed too many")
                                
                                # Try to show what columns are available
                                st.write("CSV Columns:", df.columns.tolist())
                                st.write("First few rows:")
                                st.dataframe(df.head())
                                
                            st.info("💡 Try collecting **1000 candles** from Tab 1 for better results")
            
            with col2:
                st.markdown("### 📊 Training Info")
                st.write(f"**Dataset:** {len(df)} candles")
                st.write(f"**Features:** 20 indicators")
                st.write(f"**Model:** Random Forest")
                st.write(f"**Trees:** 100")
                st.write(f"**Max Depth:** 10")
                st.info("ℹ️ Forex data has no volume, using candle patterns instead")
        else:
            st.warning("⚠️ Please collect or upload data first (Tab 1)")
    
    # ===========================
    # TAB 3: LIVE PREDICTIONS
    # ===========================
    with tab3:
        st.header("🎯 Live Trading Signals")
        
        if 'model' in st.session_state:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📊 Live Market Data - {selected_pair}")
                
                if st.button("🔄 Refresh Market Data", type="primary"):
                    with st.spinner("Fetching latest data..."):
                        df_latest = fetch_live_data(symbol, interval, outputsize=100)
                        
                        if df_latest is not None:
                            st.session_state['latest_df'] = df_latest
                            st.success(f"✅ Fetched {len(df_latest)} latest candles")
                            st.dataframe(df_latest.tail(10), use_container_width=True)
            
            with col2:
                st.subheader("⚡ Generate Signal")
                
                if st.button("🎯 Get Prediction", type="primary", use_container_width=True):
                    with st.spinner("Analyzing market..."):
                        # Fetch latest data if not already loaded
                        if 'latest_df' not in st.session_state:
                            df_latest = fetch_live_data(symbol, interval, outputsize=100)
                        else:
                            df_latest = st.session_state['latest_df']
                        
                        if df_latest is not None:
                            prediction = make_prediction(
                                st.session_state['model'],
                                st.session_state['scaler'],
                                df_latest
                            )
                            
                            if prediction:
                                # Display signal
                                signal_color = "🟢" if prediction['signal'] == "BUY" else "🔴"
                                st.markdown(f"## {signal_color} {prediction['signal']}")
                                
                                st.metric("Current Price", f"{prediction['current_price']:.5f}")
                                st.metric("Confidence", f"{prediction['confidence']:.2f}%")
                                
                                st.markdown("---")
                                st.markdown("### 📍 Trade Levels")
                                st.write(f"**Entry:** {prediction['entry_price']:.5f}")
                                st.write(f"**Take Profit:** {prediction['take_profit']:.5f}")
                                st.write(f"**Stop Loss:** {prediction['stop_loss']:.5f}")
                                
                                st.markdown("---")
                                st.markdown("### 📊 Indicators")
                                st.write(f"**Risk/Reward:** {prediction['rr_ratio']:.2f}")
                                st.write(f"**ATR:** {prediction['atr']:.5f}")
                                st.write(f"**RSI:** {prediction['rsi']:.2f}")
                            else:
                                st.error("Could not generate prediction")
                        else:
                            st.error("Failed to fetch latest data")
        else:
            st.warning("⚠️ Please train the model first (Tab 2)")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Workflow:** Tab 1: Collect Data → Tab 2: Train Model → Tab 3: Get Predictions")

if __name__ == "__main__":
    main()