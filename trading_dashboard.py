import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify
from datetime import datetime
import threading
import time
from collections import deque
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT']

latest_signals = {}
active_trades = {}
trade_history = deque(maxlen=200)
performance_metrics = {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                       'total_pnl': 0, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0,
                       'profit_factor': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
data_lock = threading.Lock()
TRADING_FEE = 0.001

def fetch_binance_klines(symbol, interval='1h', limit=250):
    for attempt in range(3):
        try:
            url = 'https://api.binance.com/api/v3/klines'
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                 'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                                 'taker_buy_quote', 'ignore'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.error(f"Fetch error {symbol}: {e}")
            if attempt < 2:
                time.sleep(1)
    return None

def calculate_bollinger_bands(prices, length=20, std_dev=1):
    if len(prices) < length:
        return None, None, None
    sma = prices.rolling(window=length).mean()
    std = prices.rolling(window=length).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def bollinger_signal(symbol, prices_df, upper_band, lower_band, sma):
    global active_trades
    
    current_price = prices_df['close'].iloc[-1]
    previous_price = prices_df['close'].iloc[-2]
    current_upper, current_lower = upper_band.iloc[-1], lower_band.iloc[-1]
    current_sma = sma.iloc[-1]
    previous_upper, previous_lower = upper_band.iloc[-2], lower_band.iloc[-2]
    
    # DIRECTIONAL BREAKOUT STRATEGY
    # BUY (go long) when price crosses ABOVE upper band
    # SELL (go short) when price crosses BELOW lower band
    new_buy_signal = previous_price <= previous_upper and current_price > current_upper
    new_sell_signal = previous_price >= previous_lower and current_price < current_lower
    at_sma = abs(current_price - current_sma) / current_sma < 0.002
    
    # Debug logging
    if new_buy_signal or new_sell_signal:
        logger.info(f"{symbol}: BUY={new_buy_signal}, SELL={new_sell_signal}, Price={current_price:.4f}, Upper={current_upper:.4f}, Lower={current_lower:.4f}")
    
    if symbol in active_trades:
        entry_data = active_trades[symbol]
        signal_type = entry_data['signal']
        entry_price = entry_data['entry_price']
        
        entry_data['highest_since_entry'] = max(entry_data.get('highest_since_entry', entry_price), current_price)
        entry_data['lowest_since_entry'] = min(entry_data.get('lowest_since_entry', entry_price), current_price)
        
        if signal_type == 'BUY':
            pnl = ((current_price - entry_price) / entry_price) * 100
            highest = entry_data['highest_since_entry']
            
            if pnl >= 1:
                trailing_stop = highest * 0.98
                entry_data['stop_loss'] = max(entry_data.get('stop_loss', 0), trailing_stop)
                entry_data['stop_type'] = 'Trailing'
                
                if current_price <= trailing_stop:
                    net_pnl = pnl - (TRADING_FEE * 200)
                    record_trade(symbol, entry_price, current_price, 'BUY', net_pnl, 'Trailing')
                    del active_trades[symbol]
                    return 'NEUTRAL', f'Closed (P&L: {net_pnl:.2f}%)', {}
            else:
                entry_data['stop_loss'] = entry_price * 0.98
                entry_data['stop_type'] = 'Initial'
                
                if current_price <= entry_data['stop_loss']:
                    net_pnl = pnl - (TRADING_FEE * 200)
                    record_trade(symbol, entry_price, current_price, 'BUY', net_pnl, 'Stop')
                    del active_trades[symbol]
                    return 'NEUTRAL', f'Stop hit (P&L: {net_pnl:.2f}%)', {}
        else:
            pnl = ((entry_price - current_price) / entry_price) * 100
            lowest = entry_data['lowest_since_entry']
            
            if pnl >= 1:
                trailing_stop = lowest * 1.02
                entry_data['stop_loss'] = min(entry_data.get('stop_loss', float('inf')), trailing_stop)
                entry_data['stop_type'] = 'Trailing'
                
                if current_price >= trailing_stop:
                    net_pnl = pnl - (TRADING_FEE * 200)
                    record_trade(symbol, entry_price, current_price, 'SELL', net_pnl, 'Trailing')
                    del active_trades[symbol]
                    return 'NEUTRAL', f'Closed (P&L: {net_pnl:.2f}%)', {}
            else:
                entry_data['stop_loss'] = entry_price * 1.02
                entry_data['stop_type'] = 'Initial'
                
                if current_price >= entry_data['stop_loss']:
                    net_pnl = pnl - (TRADING_FEE * 200)
                    record_trade(symbol, entry_price, current_price, 'SELL', net_pnl, 'Stop')
                    del active_trades[symbol]
                    return 'NEUTRAL', f'Stop hit (P&L: {net_pnl:.2f}%)', {}
        
        if at_sma:
            net_pnl = pnl - (TRADING_FEE * 200)
            record_trade(symbol, entry_price, current_price, signal_type, net_pnl, 'SMA')
            del active_trades[symbol]
            return 'NEUTRAL', f'Exit at SMA (P&L: {net_pnl:.2f}%)', {}
        
        if (signal_type == 'BUY' and new_sell_signal) or (signal_type == 'SELL' and new_buy_signal):
            net_pnl = pnl - (TRADING_FEE * 200)
            record_trade(symbol, entry_price, current_price, signal_type, net_pnl, 'Reversal')
            del active_trades[symbol]
            
            new_signal = 'SELL' if new_sell_signal else 'BUY'
            active_trades[symbol] = {'entry_price': current_price, 'signal': new_signal, 
                                    'entry_time': datetime.now(), 'highest_since_entry': current_price,
                                    'lowest_since_entry': current_price,
                                    'stop_loss': current_price * 0.98 if new_signal == 'BUY' else current_price * 1.02,
                                    'stop_type': 'Initial'}
            
            position_info = {'in_position': True, 'entry_price': current_price, 'pnl': 0,
                           'current_stop': current_price * 0.98 if new_signal == 'BUY' else current_price * 1.02,
                           'stop_type': 'Initial', 'position_type': new_signal}
            return new_signal, f'Reversed (Prev: {net_pnl:.2f}%)', position_info
        
        position_info = {'in_position': True, 'entry_price': entry_price, 'pnl': round(pnl, 2),
                        'current_stop': entry_data.get('stop_loss', 0), 'stop_type': entry_data.get('stop_type', 'Initial'),
                        'position_type': signal_type}
        return signal_type, f'{entry_data["stop_type"]} stop active', position_info
    
    if new_buy_signal:
        active_trades[symbol] = {'entry_price': current_price, 'signal': 'BUY', 'entry_time': datetime.now(),
                                'highest_since_entry': current_price, 'lowest_since_entry': current_price,
                                'stop_loss': current_price * 0.98, 'stop_type': 'Initial'}
        return 'BUY', 'NEW LONG', {'in_position': True, 'entry_price': current_price, 'pnl': 0,
                                   'current_stop': current_price * 0.98, 'stop_type': 'Initial', 'position_type': 'BUY'}
    
    if new_sell_signal:
        active_trades[symbol] = {'entry_price': current_price, 'signal': 'SELL', 'entry_time': datetime.now(),
                                'highest_since_entry': current_price, 'lowest_since_entry': current_price,
                                'stop_loss': current_price * 1.02, 'stop_type': 'Initial'}
        return 'SELL', 'NEW SHORT', {'in_position': True, 'entry_price': current_price, 'pnl': 0,
                                    'current_stop': current_price * 1.02, 'stop_type': 'Initial', 'position_type': 'SELL'}
    
    return 'NEUTRAL', 'Between bands', {}

def record_trade(symbol, entry_price, exit_price, signal_type, pnl, reason):
    trade_history.append({'symbol': symbol, 'entry_price': entry_price, 'exit_price': exit_price,
                         'signal_type': signal_type, 'pnl': pnl, 'reason': reason,
                         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    update_performance_metrics()

def update_performance_metrics():
    global performance_metrics
    if not trade_history:
        return
    
    pnls = [t['pnl'] for t in trade_history]
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p < 0]
    
    performance_metrics.update({
        'total_trades': len(pnls), 'winning_trades': len(wins), 'losing_trades': len(losses),
        'total_pnl': sum(pnls), 'win_rate': (len(wins) / len(pnls) * 100) if pnls else 0,
        'avg_win': (sum(wins) / len(wins)) if wins else 0,
        'avg_loss': (sum(losses) / len(losses)) if losses else 0,
        'profit_factor': (sum(wins) / abs(sum(losses))) if losses and sum(losses) else 0
    })
    
    if len(pnls) > 1:
        performance_metrics['sharpe_ratio'] = (np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))) if np.std(pnls) > 0 else 0
    
    cumulative = np.cumsum(pnls)
    drawdown = np.maximum.accumulate(cumulative) - cumulative
    performance_metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0

def fetch_asset_data(symbol, interval='1h'):
    try:
        df = fetch_binance_klines(symbol, interval=interval, limit=250)
        if df is None or len(df) < 50:
            return {'symbol': symbol.replace('USDT', ''), 'price': 0, 'signal': 'ERROR',
                   'reason': 'No data', 'change': 0, 'rsi': 0, 'atr': 0, 'position': {}}
        
        close_prices = df['close']
        upper_band, sma_line, lower_band = calculate_bollinger_bands(close_prices, 20, 1)
        
        if upper_band is None:
            return {'symbol': symbol.replace('USDT', ''), 'price': 0, 'signal': 'ERROR',
                   'reason': 'Calc error', 'change': 0, 'rsi': 0, 'atr': 0, 'position': {}}
        
        signal, reason, position = bollinger_signal(symbol, df, upper_band, lower_band, sma_line)
        
        rsi = calculate_rsi(close_prices)
        atr = calculate_atr(df)
        current_price = close_prices.iloc[-1]
        change_pct = ((current_price - close_prices.iloc[-2]) / close_prices.iloc[-2]) * 100
        
        return {'symbol': symbol.replace('USDT', ''), 'price': round(current_price, 8), 'change': round(change_pct, 2),
                'signal': signal, 'reason': reason, 'position': position,
                'rsi': round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else 50,
                'atr': round(atr.iloc[-1], 4) if not pd.isna(atr.iloc[-1]) else 0}
    except Exception as e:
        logger.error(f"Error {symbol}: {e}")
        return {'symbol': symbol.replace('USDT', ''), 'price': 0, 'signal': 'ERROR',
               'reason': str(e)[:20], 'change': 0, 'rsi': 0, 'atr': 0, 'position': {}}

def update_all_signals(interval='1h'):
    global latest_signals
    logger.info(f"Updating {len(ASSETS)} assets ({interval})...")
    signals = {}
    for symbol in ASSETS:
        signals[symbol] = fetch_asset_data(symbol, interval)
        time.sleep(0.1)
    with data_lock:
        latest_signals = signals
    logger.info("Update complete")

def background_updater(interval_min=2, tf='1h'):
    while True:
        try:
            update_all_signals(tf)
        except Exception as e:
            logger.error(f"Background error: {e}")
        time.sleep(interval_min * 60)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/signals')
def get_signals():
    with data_lock:
        return jsonify(latest_signals)

@app.route('/api/performance')
def get_performance():
    with data_lock:
        return jsonify(performance_metrics)

@app.route('/api/trades')
def get_trades():
    with data_lock:
        return jsonify(list(trade_history)[-50:])

@app.route('/api/update/<timeframe>')
def update_timeframe(timeframe):
    interval_map = {'1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
    update_all_signals(interval=interval_map.get(timeframe, '1h'))
    with data_lock:
        return jsonify(latest_signals)

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f5f5f5;color:#333;padding:20px}
.container{max-width:1400px;margin:0 auto}
.header{background:#fff;padding:20px;margin-bottom:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);display:flex;justify-content:space-between;align-items:center}
.header h1{font-size:24px;font-weight:600;color:#333}
.header .time{font-size:14px;color:#666}
.controls{background:#fff;padding:15px;margin-bottom:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);display:flex;gap:10px;flex-wrap:wrap}
.btn{padding:8px 16px;background:#fff;border:1px solid #ddd;border-radius:4px;cursor:pointer;font-size:14px;transition:all 0.2s}
.btn:hover{background:#f8f8f8}
.btn.active{background:#2196F3;color:#fff;border-color:#2196F3}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;margin-bottom:20px}
.metric{background:#fff;padding:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);text-align:center}
.metric-label{font-size:12px;color:#666;text-transform:uppercase;margin-bottom:8px}
.metric-value{font-size:28px;font-weight:600}
.green{color:#4CAF50}
.red{color:#f44336}
.orange{color:#FF9800}
.table-container{background:#fff;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);overflow:hidden}
table{width:100%;border-collapse:collapse}
thead{background:#fafafa;border-bottom:2px solid #eee}
th{text-align:left;padding:15px;font-size:13px;font-weight:600;color:#666;text-transform:uppercase}
tbody tr{border-bottom:1px solid #eee;transition:background 0.2s}
tbody tr:hover{background:#fafafa}
td{padding:15px;font-size:14px}
.symbol{font-weight:600;color:#333}
.badge{display:inline-block;padding:4px 12px;border-radius:4px;font-size:12px;font-weight:600}
.badge-buy{background:#e8f5e9;color:#2e7d32}
.badge-sell{background:#ffebee;color:#c62828}
.badge-neutral{background:#f5f5f5;color:#666}
.badge-error{background:#fafafa;color:#999}
.position-info{font-size:12px;color:#666;margin-top:4px}
.loading{text-align:center;padding:40px;color:#999}
@media(max-width:768px){
.header h1{font-size:20px}
.metrics{grid-template-columns:repeat(2,1fr)}
th,td{padding:10px;font-size:12px}
}
</style>
</head>
<body>
<div class="container">
<div class="header"><h1>Standard Deviation Dashboard</h1><div class="time" id="time"></div></div>
<div class="controls">
<button class="btn" onclick="updateTF('1m')">1M</button>
<button class="btn" onclick="updateTF('5m')">5M</button>
<button class="btn" onclick="updateTF('15m')">15M</button>
<button class="btn active" onclick="updateTF('1h')">1H</button>
<button class="btn" onclick="updateTF('4h')">4H</button>
<button class="btn" onclick="updateTF('1d')">1D</button>
</div>
<div class="metrics">
<div class="metric"><div class="metric-label">Buy Signals</div><div class="metric-value green" id="buy">0</div></div>
<div class="metric"><div class="metric-label">Sell Signals</div><div class="metric-value red" id="sell">0</div></div>
<div class="metric"><div class="metric-label">Neutral</div><div class="metric-value orange" id="neutral">0</div></div>
<div class="metric"><div class="metric-label">Total Trades</div><div class="metric-value" id="trades">0</div></div>
<div class="metric"><div class="metric-label">Win Rate</div><div class="metric-value green" id="winrate">0%</div></div>
<div class="metric"><div class="metric-label">Total P&L</div><div class="metric-value" id="pnl">0%</div></div>
</div>
<div class="table-container">
<table><thead><tr>
<th>Symbol</th><th>Price</th><th>Change</th><th>RSI</th><th>ATR</th><th>Signal</th><th>Status</th>
</tr></thead><tbody id="tbody">
<tr><td colspan="7" class="loading">Loading data...</td></tr>
</tbody></table>
</div></div>
<script>
function updateTime(){document.getElementById('time').textContent=new Date().toLocaleString()}
setInterval(updateTime,1000);updateTime();
function updateTF(tf){
document.querySelectorAll('.btn').forEach(b=>b.classList.remove('active'));
event.target.classList.add('active');
document.getElementById('tbody').innerHTML='<tr><td colspan="7" class="loading">Updating...</td></tr>';
fetch('/api/update/'+tf).then(r=>r.json()).then(render).catch(err=>{
console.error('Update error:',err);
document.getElementById('tbody').innerHTML='<tr><td colspan="7" class="loading">Error updating. Please try again.</td></tr>';
})
}
function render(data){
const tbody=document.getElementById('tbody');
if(!data||Object.keys(data).length===0){
tbody.innerHTML='<tr><td colspan="7" class="loading">No data available</td></tr>';
return;
}
const assets=Object.values(data);
let buyC=0,sellC=0,neuC=0;
assets.forEach(a=>{
if(a.signal==='BUY')buyC++;
else if(a.signal==='SELL')sellC++;
else if(a.signal==='NEUTRAL')neuC++;
});
document.getElementById('buy').textContent=buyC;
document.getElementById('sell').textContent=sellC;
document.getElementById('neutral').textContent=neuC;
fetchPerformance();
tbody.innerHTML=assets.map(a=>{
const pColor=a.change>=0?'green':'red';
let rColor='orange';
if(a.rsi>70)rColor='red';
else if(a.rsi<30)rColor='green';
const badgeClass=a.signal==='BUY'?'badge-buy':a.signal==='SELL'?'badge-sell':a.signal==='NEUTRAL'?'badge-neutral':'badge-error';
return'<tr><td class="symbol">'+a.symbol+'</td><td class="'+pColor+'">'+a.price.toFixed(4)+'</td><td class="'+pColor+'">'+(a.change>=0?'+':'')+a.change+'%</td><td class="'+rColor+'">'+a.rsi+'</td><td>'+a.atr+'</td><td><span class="badge '+badgeClass+'">'+a.signal+'</span></td><td>'+a.reason+(a.position&&a.position.in_position?'<div class="position-info">Entry: '+a.position.entry_price.toFixed(4)+' | P&L: '+(a.position.pnl>=0?'+':'')+a.position.pnl+'% | Stop: '+a.position.current_stop.toFixed(4)+'</div>':'')+'</td></tr>'
}).join('')
}
function fetchPerformance(){
fetch('/api/performance').then(r=>r.json()).then(perf=>{
document.getElementById('trades').textContent=perf.total_trades;
document.getElementById('winrate').textContent=perf.win_rate.toFixed(1)+'%';
const pnlVal=perf.total_pnl.toFixed(2)+'%';
const pnlEl=document.getElementById('pnl');
pnlEl.textContent=pnlVal;
pnlEl.className=perf.total_pnl>=0?'metric-value green':'metric-value red';
}).catch(e=>console.error('Performance fetch error:',e))
}
function fetchData(){
fetch('/api/signals').then(r=>r.json()).then(data=>{
if(data&&Object.keys(data).length>0){
render(data);
}
}).catch(e=>{
console.error('Fetch error:',e);
document.getElementById('tbody').innerHTML='<tr><td colspan="7" class="loading">Error loading data. Retrying...</td></tr>';
})
}
fetchData();
setInterval(fetchData,60000);
setInterval(fetchPerformance,5000);
</script>
</body>
</html>'''

if __name__ == '__main__':
    import os
    os.makedirs('templates', exist_ok=True)
    with open('templates/dashboard.html', 'w') as f:
        f.write(HTML_TEMPLATE)
    
    print("\n" + "="*50)
    print("Trading Dashboard Starting...")
    print("="*50)
    update_all_signals()
    
    updater_thread = threading.Thread(target=background_updater, args=(2, '1h'), daemon=True)
    updater_thread.start()
    
    print("\nServer running at: http://127.0.0.1:5000")
    print("Strategy: Directional Breakout (20 SMA, 1 StdDev)")
    print("BUY = Price crosses ABOVE upper band")
    print("SELL = Price crosses BELOW lower band")
    print("="*50 + "\n")
    
    app.run(debug=True, use_reloader=False)