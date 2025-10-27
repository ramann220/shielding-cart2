import base64
import binascii
import requests
import urllib3
import json
import time
from datetime import datetime, timezone, timedelta
import os
import subprocess
import configparser
import logging
from collections import defaultdict, deque
from threading import Lock, Thread, Event
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from waitress import serve
from flask import Flask, jsonify, request, Response, render_template_string, send_file
from flask_cors import CORS
from urllib.parse import urlparse, quote_plus
import cv2
# from ultralytics import YOLO # Keep commented unless vision is actively used
import warnings
import io
import random
import re

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# --- FIX: Suppress Prophet/CmdStanPy logging spam ---
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
# --- END FIX ---

# === FIX START: Custom JSON Encoder ===
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32)):
            return None if np.isnan(obj) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, deque):
            return list(obj)
        elif isinstance(obj, set):
            return list(obj)
        if pd.isna(obj):
            return None
        return super(CustomJSONEncoder, self).default(obj)
# === FIX END ===

# --- Logging & Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

def get_config(section, key, default=None):
    return config.get(section, key, fallback=default)

# --- Global Configuration ---
# --- RENDER FIX (COMMENTED OUT): We are NOT using a persistent disk ---
# This is the mount path you would set up in the Render dashboard.
# RENDER_DISK_MOUNT_PATH = '/var/render/data' 
# --- END RENDER FIX ---

LOCAL_DATA_FOLDER = get_config('FILES', 'local_data_folder', default='local_data_storage')
# --- RENDER FIX (COMMENTED OUT): We are NOT using a persistent disk ---
# if LOCAL_DATA_FOLDER == 'local_data_storage':
#     LOCAL_DATA_FOLDER = os.path.join(RENDER_DISK_MOUNT_PATH, 'local_data_storage')
# --- END RENDER FIX ---

MODEL_VERSIONS_FILE = config.get('FILES', 'model_versions_file')
# The app will now create 'local_data_storage' in its own directory
SHIELDING_SESSIONS_LOG_FILE = os.path.join(LOCAL_DATA_FOLDER, 'shielding_cart_sessions.csv')
MASTER_INVENTORY_FILE = os.path.join(LOCAL_DATA_FOLDER, 'master_inventory.csv')
MODEL_VERSIONS_FILE = config.get('FILES', 'model_versions_file')
SHIELDING_SESSIONS_LOG_FILE = os.path.join(LOCAL_DATA_FOLDER, 'shielding_cart_sessions.csv')
MASTER_INVENTORY_FILE = os.path.join(LOCAL_DATA_FOLDER, 'master_inventory.csv')
VIDEO_SOURCE_CONFIG = get_config('DEFAULT', 'video_source_app2', default='0') # Keep for potential vision use
try:
    VIDEO_SOURCE = int(VIDEO_SOURCE_CONFIG)
except ValueError:
    VIDEO_SOURCE = VIDEO_SOURCE_CONFIG

TARGET_SCANS_PER_MINUTE = 600 # Performance target

if not os.path.exists(LOCAL_DATA_FOLDER):
    os.makedirs(LOCAL_DATA_FOLDER)
    logging.info(f"Created local data storage directory at: {LOCAL_DATA_FOLDER}")

# --- Global State & Concurrency ---
DATA_LOCK = Lock()
BACKGROUND_WORKERS = {}
MODELS = {}
PROPHET_MODELS = {}
MODEL_VERSIONS = defaultdict(lambda: '1.0.0')

# --- RFID Reader & System State ---
READER_STATE = {'config': None, 'thread': None, 'stop_event': None, 'status': 'Disconnected'}
SYSTEM_HEALTH = {
    'reader': {'status': 'Disconnected', 'last_seen': 0, 'internal_temp': 0},
    'antennas': {},
    'maintenance_prediction': 'Nominal', # 'Nominal', 'Warning', 'Critical'
}

# --- Vision System State (Retained for potential future use) ---
VISION_WORKER_STATE = {'thread': None, 'stop_event': None, 'status': 'Stopped'}
VISION_DATA = {'object_count': 0, 'image_analysis': None}
LATEST_FRAME = None

# --- Shielding Cart Application State ---
SHIELDING_CART_STATE = {
    'status': 'idle',
    'inventory_mode': 'scan_only',
    'session_id': None,
    'start_time': None,
    'ai_feedback': {'message': 'System ready. Start a session to begin scanning.', 'type': 'info'},
    'inventory_predictions': {}, # This is now for 'live' check-in/out only
    'session_summary': {},
    'historical_stats': {},
    'read_rate_anomaly': False,
    'sc_inventory_view': pd.DataFrame().to_dict('records'),
    'power_boost_active': False,
    'scanned_items': defaultdict(lambda: {'count': 0, 'reads': deque(maxlen=50)}),
    'audit_target': None,
    'audit_results': {},
    'asn_status': 'idle',
    'asn_data': None,
    'current_asn_box': None,
    'asn_scan_results': {},
}

# --- Master Data Storage ---
MASTER_INVENTORY = pd.DataFrame()

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)
app.json_encoder = CustomJSONEncoder

# ==============================================================================
# === HELPER & UTILITY FUNCTIONS ===
# ==============================================================================

def load_master_inventory():
    global MASTER_INVENTORY
    if os.path.exists(MASTER_INVENTORY_FILE):
        try:
            MASTER_INVENTORY = pd.read_csv(MASTER_INVENTORY_FILE, dtype={'epc': str, 'sku': str})
            for col in ['epc', 'sku', 'name']:
                if col in MASTER_INVENTORY.columns:
                    MASTER_INVENTORY[col] = MASTER_INVENTORY[col].astype(str).str.upper().str.strip()
                else:
                    logging.error(f"CRITICAL: Master inventory file is missing required '{col}' column.")
                    MASTER_INVENTORY = pd.DataFrame()
                    return
            MASTER_INVENTORY.dropna(subset=['epc', 'sku', 'name'], inplace=True)
            MASTER_INVENTORY.set_index('epc', inplace=True, drop=False)
            if 'image_url' not in MASTER_INVENTORY.columns:
                MASTER_INVENTORY['image_url'] = 'https://via.placeholder.com/150'
            else:
                 MASTER_INVENTORY['image_url'] = MASTER_INVENTORY['image_url'].fillna('https://via.placeholder.com/150')
            if 'stock_level' not in MASTER_INVENTORY.columns: MASTER_INVENTORY['stock_level'] = 0
            if 'avg_daily_usage' not in MASTER_INVENTORY.columns: MASTER_INVENTORY['avg_daily_usage'] = 1.0
            if 'inventory_status' not in MASTER_INVENTORY.columns: MASTER_INVENTORY['inventory_status'] = 'Available' # For audit
            if 'sales_class' not in MASTER_INVENTORY.columns: MASTER_INVENTORY['sales_class'] = 'Normal' # For viewing
            if 'price' not in MASTER_INVENTORY.columns: MASTER_INVENTORY['price'] = 0.0 # Needed if reusing depletion model
            if 'prophet_days_left' not in MASTER_INVENTORY.columns: MASTER_INVENTORY['prophet_days_left'] = 0.0

            MASTER_INVENTORY['stock_level'] = pd.to_numeric(MASTER_INVENTORY['stock_level'], errors='coerce').fillna(0).astype(int)
            MASTER_INVENTORY['avg_daily_usage'] = pd.to_numeric(MASTER_INVENTORY['avg_daily_usage'], errors='coerce').fillna(1.0)
            MASTER_INVENTORY['prophet_days_left'] = pd.to_numeric(MASTER_INVENTORY['prophet_days_left'], errors='coerce').fillna(0.0)

            logging.info(f"Loaded and standardized {len(MASTER_INVENTORY)} items from {MASTER_INVENTORY_FILE}")
        except Exception as e:
            logging.error(f"Failed to load master inventory file: {e}")
            MASTER_INVENTORY = pd.DataFrame()

def save_master_inventory():
    global MASTER_INVENTORY
    if not MASTER_INVENTORY.empty:
        try:
            df_to_save = MASTER_INVENTORY.copy()
            if 'epc' not in df_to_save.columns: df_to_save.reset_index(inplace=True)
            if 'stock_level' in df_to_save.columns: df_to_save['stock_level'] = df_to_save['stock_level'].astype(int)
            df_to_save.to_csv(MASTER_INVENTORY_FILE, index=False)
            logging.info(f"Saved updated master inventory with {len(MASTER_INVENTORY)} items.")
            # Restore index after saving
            if 'epc' in MASTER_INVENTORY.columns and MASTER_INVENTORY.index.name != 'epc':
                MASTER_INVENTORY.set_index('epc', inplace=True, drop=False)
        except Exception as e:
            logging.error(f"Failed to save master inventory file: {e}")

def process_sales_data_for_usage(sales_df):
    global MASTER_INVENTORY, PROPHET_MODELS
    with DATA_LOCK:
        if MASTER_INVENTORY.empty:
            logging.warning("Cannot process sales data: Master Inventory is not loaded.")
            raise ValueError("Master inventory must be loaded before processing sales data.")
        
        current_index_name = MASTER_INVENTORY.index.name # Save current index
        try:
            sales_df['timestamp'] = pd.to_datetime(sales_df['timestamp'], errors='coerce')
            sales_df.dropna(subset=['timestamp'], inplace=True)
            sales_df['epc'] = sales_df['epc'].astype(str).str.upper()

            # --- 1. Calculate Average Daily Usage (Original Logic) ---
            total_days = (sales_df['timestamp'].max() - sales_df['timestamp'].min()).days
            if total_days < 1: total_days = 1

            sales_counts = sales_df['epc'].value_counts().reset_index()
            sales_counts.columns = ['epc', 'total_sales']
            sales_counts['avg_daily_usage'] = sales_counts['total_sales'] / total_days

            update_df = sales_counts.set_index('epc')
            if MASTER_INVENTORY.index.name != 'epc':
                 MASTER_INVENTORY.reset_index(inplace=True)
                 MASTER_INVENTORY.set_index('epc', inplace=True, drop=False)

            MASTER_INVENTORY.update(update_df)
            MASTER_INVENTORY['avg_daily_usage'] = MASTER_INVENTORY['avg_daily_usage'].fillna(0.1)
            logging.info(f"Updated average daily usage for {len(sales_counts)} items based on sales data spanning {total_days} days.")

            # --- 2. Train Prophet Models & Predict Depletion ---
            logging.info("--- 🧠 Starting Prophet Model Training & Prediction ---")
            # Need SKU on the sales_df to train by SKU
            sales_df_with_sku = sales_df.merge(MASTER_INVENTORY[['sku']], left_on='epc', right_index=True, how='left')
            sales_df_with_sku.dropna(subset=['sku'], inplace=True)
            all_skus = sales_df_with_sku['sku'].unique()
            today = pd.to_datetime(datetime.now(timezone.utc).date())
            
            new_predictions = {}
            temp_prophet_models = {}

            for sku in all_skus:
                sku_sales_df = sales_df_with_sku[sales_df_with_sku['sku'] == sku]
                sales_by_day = sku_sales_df.set_index('timestamp').resample('D').size().reset_index()
                sales_by_day.columns = ['ds', 'y']
                
                # Ensure we have enough data and not just zeros
                if len(sales_by_day) < 5 or sales_by_day['y'].sum() == 0:
                    logging.info(f"Skipping Prophet for SKU {sku}: Insufficient data points.")
                    continue

                try:
                    # Train model
                    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
                    m.fit(sales_by_day)
                    temp_prophet_models[sku] = m
                    
                    # Make prediction
                    future = m.make_future_dataframe(periods=365 * 2) # 2 years
                    forecast = m.predict(future)
                    
                    # Get current total stock for this SKU
                    current_stock = MASTER_INVENTORY[MASTER_INVENTORY['sku'] == sku]['stock_level'].sum()
                    
                    forecast_from_today = forecast[forecast['ds'] >= today].copy() # Use .copy() to avoid SettingWithCopyWarning
                    forecast_from_today['cumulative_sales'] = forecast_from_today['yhat'].clip(lower=0).cumsum()
                    
                    depletion_days = forecast_from_today[forecast_from_today['cumulative_sales'] >= current_stock]
                    
                    days_left = 730 # Default to 2 years if never depletes in forecast
                    if not depletion_days.empty:
                        depletion_date = depletion_days['ds'].min()
                        days_left = (depletion_date - today).days
                    
                    new_predictions[sku] = round(float(days_left), 1)

                except Exception as e:
                    logging.error(f"Prophet training/prediction for SKU {sku} failed: {e}", exc_info=False) # Keep log tidy

            if new_predictions:
                logging.info(f"Prophet Predictions: {new_predictions}")
                pred_df = pd.DataFrame.from_dict(new_predictions, orient='index', columns=['prophet_days_left'])
                pred_df.index.name = 'sku'
                
                # Use temporary SKU index to update
                MASTER_INVENTORY.reset_index(inplace=True)
                MASTER_INVENTORY.set_index('sku', inplace=True, drop=False)
                MASTER_INVENTORY.update(pred_df)
                
            PROPHET_MODELS = temp_prophet_models # Store trained models
            logging.info("--- 🧠 Prophet Model Training Complete ---")

            # --- 3. Save Inventory ---
            save_master_inventory()

        except Exception as e:
            logging.error(f"Failed to process sales data for usage calculation: {e}", exc_info=True)
            raise e
        finally:
            # Restore original index
            try:
                if current_index_name in MASTER_INVENTORY.columns and MASTER_INVENTORY.index.name != current_index_name:
                     MASTER_INVENTORY.reset_index(inplace=True)
                     MASTER_INVENTORY.set_index(current_index_name, inplace=True, drop=False)
                elif 'epc' in MASTER_INVENTORY.columns and MASTER_INVENTORY.index.name != 'epc':
                     MASTER_INVENTORY.reset_index(inplace=True)
                     MASTER_INVENTORY.set_index('epc', inplace=True, drop=False)
            except Exception as e:
                logging.error(f"Failed to restore master inventory index: {e}")
                MASTER_INVENTORY.set_index('epc', inplace=True, drop=False) # Fallback

def load_model_versions():
    global MODEL_VERSIONS
    if os.path.exists(MODEL_VERSIONS_FILE):
        try:
            with open(MODEL_VERSIONS_FILE, 'r') as f:
                MODEL_VERSIONS.update(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Could not read model versions file: {e}")

def load_shielding_cart_models():
    """Loads only the models needed for the Shielding Cart."""
    global MODELS
    logging.info("--- 🧠 Loading Shielding Cart AI Models ---")
    with DATA_LOCK:
        model_files = {
            'operator_guidance_model': 'operator_guidance_model.joblib',
            'maintenance_predictor': 'maintenance_predictor.joblib', # Shared
        }
        
        MODELS['inventory_depletion_model'] = 'prophet' # Use Prophet
        logging.info("✅ Inventory Depletion Model set to use 'Prophet'.")

        for model_name, filename in model_files.items():
            try:
                if not os.path.exists(filename):
                    logging.warning(f"⚠️ Model file not found: '{filename}'. Feature disabled.")
                    MODELS[model_name] = None
                    continue
                MODELS[model_name] = joblib.load(filename)
                logging.info(f"✅ {model_name.replace('_', ' ').title()} loaded.")
            except Exception as e:
                MODELS[model_name] = None
                logging.error(f"‼️ Error loading {model_name} from '{filename}': {e}")
    load_model_versions()
    logging.info(f"--- Model versions: {json.dumps(dict(MODEL_VERSIONS))} ---")

def decode_epc(epc_base64url):
    epc_base64 = epc_base64url.replace('-', '+').replace('_', '/') + '=' * (-len(epc_base64url) % 4)
    try:
        return binascii.hexlify(base64.b64decode(epc_base64)).decode('utf-8').upper()
    except (binascii.Error, ValueError):
        return None

def log_to_csv(filepath, data_dict):
    try:
        clean_data_dict = json.loads(json.dumps(data_dict, cls=CustomJSONEncoder))
        df = pd.DataFrame([clean_data_dict])
        df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)
    except Exception as e:
        logging.error(f"Failed to log data to {filepath}: {e}")

# ==============================================================================
# === CORE WORKER THREADS ===
# ==============================================================================

def process_shielding_cart_read(epc, read_data):
    """Processes an RFID read specifically for the shielding cart context."""
    with DATA_LOCK:
        if SHIELDING_CART_STATE['status'] == 'scanning':
            SHIELDING_CART_STATE['scanned_items'][epc]['count'] += 1
            rssi = float(read_data.get('peakRssiCdbm', -10000) / 100.0)
            SHIELDING_CART_STATE['scanned_items'][epc]['reads'].append(rssi)
        elif SHIELDING_CART_STATE['status'] == 'auditing':
            audit_target = SHIELDING_CART_STATE.get('audit_target')
            target_sku = audit_target.get('sku') if audit_target else None
            if not target_sku: return

            if not MASTER_INVENTORY.empty and epc in MASTER_INVENTORY.index:
                item_sku = MASTER_INVENTORY.loc[epc].get('sku')
                if item_sku == target_sku:
                    audit_sku_data = SHIELDING_CART_STATE['audit_results'].get(target_sku)
                    if audit_sku_data and epc not in audit_sku_data['scanned_epcs']:
                        audit_sku_data['scanned_epcs'].add(epc)
        elif SHIELDING_CART_STATE['asn_status'] == 'scanning_box':
            current_box = SHIELDING_CART_STATE.get('current_asn_box')
            if current_box:
                current_box['scanned_epcs'].add(epc)

def rfid_stream_worker(config, stop_event):
    """Standalone RFID worker for the Shielding Cart app."""
    global READER_STATE
    ip, user, password = config.get('ip'), config.get('user'), config.get('pass')
    reconnect_delay, max_reconnect_delay = 5, 60

    with DATA_LOCK: READER_STATE['status'] = 'Connecting'

    base_url = f"https://{ip}/api/v1"
    profile_name = "Raman01" # Or configure this if needed
    start_url = f"{base_url}/profiles/inventory/presets/{profile_name}/start"
    stop_url = f"{base_url}/profiles/stop"
    stream_url = f"{base_url}/data/stream"

    s = requests.Session()
    s.auth = (user, password)
    s.verify = False

    while not stop_event.is_set():
        try:
            logging.info(f"Attempting to connect to reader at {ip} for Shielding Cart...")
            try: # Pre-emptive stop
                s.post(stop_url, timeout=5)
                time.sleep(0.5)
            except requests.RequestException: pass

            start_response = s.post(start_url, timeout=5)
            if start_response.status_code not in [200, 204, 409]:
                start_response.raise_for_status()

            r = s.get(stream_url, stream=True, timeout=10)
            r.raise_for_status()

            with DATA_LOCK: READER_STATE['status'] = 'Connected'
            logging.info(f"✅ Shielding Cart Reader at {ip} connected and streaming.")
            reconnect_delay = 5

            for line in r.iter_lines():
                if stop_event.is_set(): break
                if not line: continue
                try:
                    raw_line = line.decode('utf-8').strip()
                    if not raw_line: continue
                    json_data = json.loads(raw_line)
                    data = json_data.get('tagInventoryEvent', json_data)

                    if data.get('epc'):
                        epc = decode_epc(data['epc'])
                        if not epc: continue
                        process_shielding_cart_read(epc, data) # Directly call the relevant function

                        with DATA_LOCK: # Update general system health
                            SYSTEM_HEALTH['reader']['last_seen'] = time.time()
                            SYSTEM_HEALTH['reader']['internal_temp'] = round(random.uniform(45.0, 55.0), 1)
                            port = data.get('antennaPort')
                            if port and port not in SYSTEM_HEALTH['antennas']:
                                SYSTEM_HEALTH['antennas'][port] = {'power_level': 20.0, 'manual_override': False, 'read_latency': deque(maxlen=20)}
                            # Add latency if needed by maintenance model
                            if port and port in SYSTEM_HEALTH['antennas']:
                                reader_ts_us = data.get('firstSeenTimestampUsec', 0)
                                latency = (time.time() - (reader_ts_us / 1_000_000.0)) if reader_ts_us > 0 else -1
                                if latency != -1:
                                    SYSTEM_HEALTH['antennas'][port]['read_latency'].append(latency)


                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logging.warning(f"Failed to parse stream line: {e}")

        except requests.exceptions.RequestException as e:
            logging.error(f"⚠️ Shielding Cart Reader connection error: {e}. Retrying in {reconnect_delay}s.")
            with DATA_LOCK: READER_STATE['status'] = 'Error'
            stop_event.wait(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
        except Exception as e:
             logging.critical(f"‼️ Unexpected error in reader worker: {e}", exc_info=True)
             with DATA_LOCK: READER_STATE['status'] = 'Error'
             stop_event.wait(reconnect_delay)
             reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)


    logging.info(f"🛑 Stopping inventory on reader at {ip}...")
    try: s.post(stop_url, timeout=5)
    except requests.exceptions.RequestException as e: logging.error(f"‼️ Failed to send STOP command: {e}")
    s.close()
    with DATA_LOCK:
        READER_STATE = {'config': None, 'thread': None, 'stop_event': None, 'status': 'Disconnected'}
        SYSTEM_HEALTH['antennas'].clear()
    logging.info(f"🛑 Shielding Cart Reader worker has fully stopped.")

# Vision worker (kept minimal for potential future use)
def vision_worker(stop_event):
    global LATEST_FRAME, VISION_DATA, VISION_WORKER_STATE
    logging.info("✅ Vision worker starting...")
    cap = None
    try:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            logging.error(f"‼️ Could not open video source {VIDEO_SOURCE}.")
            with DATA_LOCK: VISION_WORKER_STATE['status'] = 'Error'
            return

        with DATA_LOCK: VISION_WORKER_STATE['status'] = 'Running'
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from video source.")
                time.sleep(1); continue

            # Placeholder - add detection if needed later
            cv2.putText(frame, "Shielding Cart Vision (Inactive)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                with DATA_LOCK: LATEST_FRAME = buffer.tobytes()
            time.sleep(0.1)
    except Exception as e:
        logging.critical(f"‼️ Vision worker error: {e}", exc_info=True)
        with DATA_LOCK: VISION_WORKER_STATE['status'] = 'Error'
    finally:
        if cap: cap.release()
        logging.info("🛑 Vision worker stopped.")
        with DATA_LOCK:
            LATEST_FRAME = None
            VISION_WORKER_STATE = {'thread': None, 'stop_event': None, 'status': 'Stopped'}
            VISION_DATA = {'object_count': 0, 'image_analysis': None}

# ==============================================================================
# === AI & ANALYTICS WORKERS (Shielding Cart Specific) ===
# ==============================================================================

def shielding_cart_ai_worker():
    logging.info("✅ Shielding Cart AI Worker started.")
    last_read_count = 0
    last_check_time = time.time()

    while True:
        time.sleep(2)
        with DATA_LOCK:
            if SHIELDING_CART_STATE['status'] not in ['scanning', 'auditing']:
                 # Update SC live inventory view when not scanning
                 if not MASTER_INVENTORY.empty:
                    try: # Add try-except for safety during state transitions
                        SHIELDING_CART_STATE['sc_inventory_view'] = MASTER_INVENTORY.to_dict('records')
                    except Exception as e:
                         logging.warning(f"Error updating SC inventory view in idle state: {e}")
                 continue

            is_audit = SHIELDING_CART_STATE['status'] == 'auditing'
            if is_audit:
                scanned_items = SHIELDING_CART_STATE['audit_results']
                total_reads = sum(len(item_data.get('scanned_epcs', set())) for item_data in scanned_items.values())
            else:
                scanned_items = SHIELDING_CART_STATE['scanned_items']
                total_reads = sum(item['count'] for item in scanned_items.values())

            if not scanned_items: continue

            # Anomaly Detection (non-audit)
            if not is_audit:
                time_now = time.time()
                duration_since_last_check = time_now - last_check_time
                current_read_rate = (total_reads - last_read_count) / duration_since_last_check if duration_since_last_check > 0 else 0

                if duration_since_last_check > 5 and current_read_rate < 1 and total_reads > 0:
                    if not SHIELDING_CART_STATE['read_rate_anomaly']:
                        SHIELDING_CART_STATE['read_rate_anomaly'] = True
                        SHIELDING_CART_STATE['ai_feedback'] = {'message': 'Anomaly: Low read rate detected.', 'type': 'error'}
                else:
                    SHIELDING_CART_STATE['read_rate_anomaly'] = False
                last_read_count = total_reads
                last_check_time = time_now

            # Operator Guidance (non-audit)
            if not is_audit and MODELS.get('operator_guidance_model') and not SHIELDING_CART_STATE['read_rate_anomaly']:
                try:
                    all_rssi_reads = [rssi for item in scanned_items.values() for rssi in item['reads']]
                    if not all_rssi_reads: continue
                    all_rssi_reads = [float(r) for r in all_rssi_reads]
                    avg_rssi = np.mean(all_rssi_reads)
                    rssi_std = np.std(all_rssi_reads)

                    features = np.array([[len(scanned_items), current_read_rate, avg_rssi, rssi_std]])
                    prediction = MODELS['operator_guidance_model'].predict(features)[0]

                    feedback = {'message': 'System nominal. Read performance is optimal.', 'type': 'success'}
                    if prediction == 1:
                        feedback = {'message': 'High tag density detected. Scan items in smaller batches.', 'type': 'warning'}
                        if not SHIELDING_CART_STATE['power_boost_active']:
                            SHIELDING_CART_STATE['power_boost_active'] = True
                            logging.info("AI detected high tag density. Triggering antenna power boost.")
                            Thread(target=adjust_power_for_density, daemon=True).start()
                    elif prediction == 2:
                        feedback = {'message': 'Low average signal strength. Ensure items are centered.', 'type': 'error'}

                    # Don't overwrite error/warning with success immediately
                    current_feedback_type = SHIELDING_CART_STATE['ai_feedback']['type']
                    if not (current_feedback_type in ['error', 'warning'] and feedback['type'] == 'success'):
                         SHIELDING_CART_STATE['ai_feedback'] = feedback

                except Exception as e:
                    logging.error(f"Shielding Cart Guidance Model Error: {e}")

            # --- PREDICTION LOGIC (NOW ONLY FOR LIVE CHECK-IN/OUT) ---
            if not is_audit and MODELS.get('inventory_depletion_model') == 'prophet' and SHIELDING_CART_STATE['inventory_mode'] != 'scan_only':
                if MASTER_INVENTORY.empty or 'stock_level' not in MASTER_INVENTORY.columns or 'avg_daily_usage' not in MASTER_INVENTORY.columns:
                    SHIELDING_CART_STATE['inventory_predictions'] = {}
                else:
                    try:
                        predictions = {}
                        current_mode = SHIELDING_CART_STATE['inventory_mode']
                        for epc, data in scanned_items.items():
                            if epc in MASTER_INVENTORY.index:
                                usage_row = MASTER_INVENTORY.loc[epc]
                                avg_usage = float(usage_row.get('avg_daily_usage', 1.0))
                                stock_level = float(usage_row.get('stock_level', 0))

                                # Use simple linear model for check-in/out
                                current_stock_for_pred = stock_level
                                if current_mode == 'check_out': current_stock_for_pred -= 1
                                elif current_mode == 'check_in': current_stock_for_pred += 1
                                
                                days_left = max(0, current_stock_for_pred) / max(0.1, avg_usage)
                                predictions[epc] = {"name": usage_row.get('name', 'Unknown'), "days_left": max(0, round(float(days_left), 1))}
                        
                        SHIELDING_CART_STATE['inventory_predictions'] = predictions
                    except Exception as e:
                        logging.error(f"Shielding Cart Depletion Model Error (Live): {e}", exc_info=False)


# Helper function for power adjustment
def adjust_power_for_density():
    """Increases power for all antennas by 2 dBm."""
    with DATA_LOCK:
        antennas = list(SYSTEM_HEALTH['antennas'].keys())
        if not antennas: return
        first_ant_id = antennas[0]
        current_power = SYSTEM_HEALTH['antennas'][first_ant_id].get('power_level', 20.0)

    new_power = min(30.0, current_power + 2.0)
    logging.info(f"AI POWER BOOST: Adjusting power from {current_power:.1f} to {new_power:.1f} dBm.")

    for ant_id in antennas:
        success = set_antenna_power_on_reader(ant_id, new_power) # set_antenna_power function needs to exist
        if success:
            with DATA_LOCK:
                if ant_id in SYSTEM_HEALTH['antennas']:
                    SYSTEM_HEALTH['antennas'][ant_id]['power_level'] = new_power
                    SYSTEM_HEALTH['antennas'][ant_id]['manual_override'] = False

def predictive_maintenance_worker():
    """Monitors system health to predict maintenance needs."""
    logging.info("✅ Predictive Maintenance Worker started.")
    while True:
        time.sleep(60)
        with DATA_LOCK:
            if READER_STATE['status'] != 'Connected' or not MODELS.get('maintenance_predictor'):
                SYSTEM_HEALTH['maintenance_prediction'] = 'Nominal'
                continue
            try:
                first_ant_id = next(iter(SYSTEM_HEALTH['antennas']), None)
                ant_data = SYSTEM_HEALTH['antennas'].get(first_ant_id, {})

                temp = SYSTEM_HEALTH['reader'].get('internal_temp', 50)
                vibration = random.uniform(0.01, 0.08) # Placeholder if no sensor
                latency = 100.0 # Default

                if ant_data.get('read_latency'):
                     # Ensure all latencies are floats before calculating mean
                     valid_latencies = [float(l) for l in ant_data['read_latency'] if isinstance(l, (int, float))]
                     if valid_latencies:
                        latency = float(np.mean(valid_latencies) * 1000)

                features = np.array([[temp, vibration, latency]])
                prediction = MODELS['maintenance_predictor'].predict(features.astype(np.float32))[0]

                SYSTEM_HEALTH['maintenance_prediction'] = 'Warning' if prediction == 1 else 'Nominal' # Simplified

            except Exception as e:
                logging.error(f"Maintenance prediction failed: {e}")

def rl_power_tuning_worker():
    """RL Agent for auto-calibrating antenna power."""
    logging.info("✅ RL Power Tuning Agent started.")
    q_table = np.zeros((1, 3, 3)) # State 0 = Shielding Cart, 3 density levels, 3 actions
    last_action_time = 0
    COOLDOWN_PERIOD = 15
    MANUAL_OVERRIDE_COOLDOWN = 60 # Cooldown in seconds after manual set

    while True:
        time.sleep(5)
        now = time.time()
        with DATA_LOCK:
            reader_connected = READER_STATE['status'] == 'Connected'
            session_active = SHIELDING_CART_STATE['status'] == 'scanning' # Only tune during active scanning
            power_boost_active = SHIELDING_CART_STATE['power_boost_active']
            manual_override_active = any(ant.get('manual_override', False) for ant in SYSTEM_HEALTH['antennas'].values())

        if not (reader_connected and session_active and not power_boost_active and (now - last_action_time > COOLDOWN_PERIOD) and not manual_override_active):
            if manual_override_active and (now - last_action_time > MANUAL_OVERRIDE_COOLDOWN):
                with DATA_LOCK:
                    logging.info(f"RL AGENT (SC): Manual override cooldown ({MANUAL_OVERRIDE_COOLDOWN}s) expired. Resuming auto-tuning.")
                    for ant_id in SYSTEM_HEALTH['antennas']:
                         if ant_id in SYSTEM_HEALTH['antennas']: # Check again just in case
                            SYSTEM_HEALTH['antennas'][ant_id]['manual_override'] = False
                last_action_time = now # Reset time to avoid immediate re-trigger
            continue

        with DATA_LOCK:
            item_count = len(SHIELDING_CART_STATE['scanned_items'])
            if item_count < 20: density_level = 0
            elif item_count < 100: density_level = 1
            else: density_level = 2
            total_reads = sum(item['count'] for item in SHIELDING_CART_STATE['scanned_items'].values())
            duration = (now - (SHIELDING_CART_STATE.get('start_time') or now))
            read_rate_before = total_reads / max(1, duration) if duration > 0 else 0

        action = np.argmax(q_table[0, density_level]) # State 0 for shielding cart
        if random.random() < 0.15: action = random.randint(0, 2)

        with DATA_LOCK:
             current_power = next((ant['power_level'] for ant in SYSTEM_HEALTH['antennas'].values()), 20.0)

        new_power = current_power
        if action == 0: new_power = max(10.0, current_power - 1.0)
        elif action == 2: new_power = min(30.0, current_power + 1.0)

        if abs(new_power - current_power) > 0.5:
            logging.info(f"RL AGENT (SC): Density '{density_level}', Action -> Set Power to {new_power:.1f} dBm")
            all_success = True
            with DATA_LOCK: antenna_ids = list(SYSTEM_HEALTH['antennas'].keys())

            for ant_id in antenna_ids:
                 success = set_antenna_power_on_reader(ant_id, new_power)
                 if success:
                     with DATA_LOCK:
                         if ant_id in SYSTEM_HEALTH['antennas']:
                             SYSTEM_HEALTH['antennas'][ant_id]['power_level'] = new_power
                             SYSTEM_HEALTH['antennas'][ant_id]['manual_override'] = False # RL sets its own flag to false
                 else:
                     all_success = False

            if all_success:
                last_action_time = now
                time.sleep(2) # Let power change take effect

                with DATA_LOCK:
                    total_reads_after = sum(item['count'] for item in SHIELDING_CART_STATE['scanned_items'].values())
                    duration_after = (time.time() - (SHIELDING_CART_STATE.get('start_time') or time.time()))
                    read_rate_after = total_reads_after / max(1, duration_after) if duration_after > 0 else 0

                reward = 1.0 if read_rate_after > read_rate_before else -1.0
                q_table[0, density_level, action] = float(q_table[0, density_level, action] * 0.9 + 0.1 * reward)
                logging.info(f"RL AGENT (SC): Reward for action was {reward}")
            else:
                 logging.warning("RL AGENT (SC): Failed to set power on one or more antennas. Skipping Q-table update.")

def shielding_cart_analytics_worker():
    """Periodically analyzes shielding cart history."""
    logging.info("✅ Shielding Cart Analytics Worker started.")
    while True:
        time.sleep(60 * 10)
        if os.path.exists(SHIELDING_SESSIONS_LOG_FILE):
            try:
                sessions_df = pd.read_csv(SHIELDING_SESSIONS_LOG_FILE, on_bad_lines='skip')
                # Filter only for non-audit sessions with a score
                score_sessions = sessions_df.dropna(subset=['timestamp', 'performance_score'])
                if not score_sessions.empty:
                    score_sessions['timestamp'] = pd.to_datetime(score_sessions['timestamp'])
                    recent_sessions = score_sessions.sort_values('timestamp', ascending=False).head(20)
                    with DATA_LOCK:
                        SHIELDING_CART_STATE['historical_stats']['recent_scores'] = recent_sessions[['timestamp', 'performance_score']].to_dict('records')
            except Exception as e:
                logging.error(f"Error in Shielding Cart analytics worker: {e}")

# ==============================================================================
# === FLASK API ENDPOINTS ===
# ==============================================================================

@app.route('/')
def index():
 return app.send_static_file('index.html')

@app.route('/api/app_data')
def get_shielding_cart_data():
    """Simplified data endpoint for the Shielding Cart app."""
    with DATA_LOCK:
        reader_state_copy = {k: v for k, v in READER_STATE.items() if k not in ['thread', 'stop_event']}
        vision_state_copy = {k: v for k, v in VISION_WORKER_STATE.items() if k not in ['thread', 'stop_event']}
        vision_data_copy = VISION_DATA.copy()
        sc_state_copy = json.loads(json.dumps(SHIELDING_CART_STATE, cls=CustomJSONEncoder))
        health_copy = json.loads(json.dumps(SYSTEM_HEALTH, cls=CustomJSONEncoder))

        # --- Post-process audit results for frontend ---
        if sc_state_copy.get('status') in ['auditing', 'awaiting_resolution', 'complete'] and sc_state_copy.get('audit_results'):
            for sku, data in sc_state_copy['audit_results'].items():
                data['scanned_count'] = len(data.get('scanned_epcs', []))

        # Remove detailed reads from main poll
        if 'scanned_items' in sc_state_copy:
            sc_state_copy['scanned_items'] = {
                epc: {'count': data.get('count', 0)} for epc, data in sc_state_copy['scanned_items'].items()
            }
        # Remove live inventory from main poll (use specific endpoint)
        if 'sc_inventory_view' in sc_state_copy: del sc_state_copy['sc_inventory_view']
        
        # --- REMOVE: Predictions no longer sent in main poll ---
        if 'inventory_predictions' in sc_state_copy: del sc_state_copy['inventory_predictions']
        # --- END REMOVE ---
        
        # Create ASN summary but keep full data
        if 'asn_data' in sc_state_copy and sc_state_copy['asn_data']:
             sc_state_copy['asn_summary'] = {
                 'po_number': sc_state_copy['asn_data'].get('po_number'),
                 'total_boxes': len(sc_state_copy['asn_data'].get('boxes', {}))
             }

        response = {
            'reader_status': reader_state_copy.get('status', 'Disconnected'),
            'vision_worker_status': vision_state_copy,
            'vision_data': vision_data_copy,
            'shielding_cart': sc_state_copy,
            'model_versions': dict(MODEL_VERSIONS),
            'system_health': health_copy,
        }
        return jsonify(response)

# --- Reader and Antenna Control Endpoints ---
@app.route('/api/reader', methods=['POST', 'DELETE'])
def handle_reader():
    global READER_STATE
    if request.method == 'POST':
        config_data = request.json
        if not all(k in config_data for k in ['ip', 'user', 'pass']):
            return jsonify({'status': 'error', 'message': 'Missing reader IP, user, or pass'}), 400
        with DATA_LOCK:
            if READER_STATE.get('thread') and READER_STATE['thread'].is_alive():
                return jsonify({'status': 'error', 'message': 'Reader is already running.'}), 409
            stop_event = Event()
            thread = Thread(target=rfid_stream_worker, args=(config_data, stop_event), name="SC-RFID-Stream-Worker", daemon=True)
            READER_STATE = {'config': config_data, 'thread': thread, 'stop_event': stop_event, 'status': 'Connecting'}
            thread.start()
        return jsonify({'status': 'success', 'message': 'Reader connection initiated.'})
    elif request.method == 'DELETE':
        with DATA_LOCK:
            if READER_STATE.get('stop_event'): READER_STATE['stop_event'].set()
            READER_STATE['status'] = 'Disconnecting'
        return jsonify({'status': 'success', 'message': 'Reader disconnection initiated.'})

# Centralized function from app2.py
def set_antenna_power_on_reader(physical_port, power_level):
    with DATA_LOCK:
        if READER_STATE.get('status') != 'Connected' or not READER_STATE.get('config'):
            logging.warning("Cannot set antenna power: Reader not connected.")
            return False
        config_data = READER_STATE['config']
        ip, user, password = config_data['ip'], config_data['user'], config_data['pass']

    base_url = f"https://{ip}/api/v1"
    profile_name = "Raman01" # Or get from config if dynamic
    profile_url = f"{base_url}/profiles/inventory/presets/{profile_name}"

    session = requests.Session()
    session.auth = (user, password)
    session.verify = False

    try:
        stop_url = f"{base_url}/profiles/stop"
        logging.info(f"Stopping profile for power update (Antenna {physical_port})...")
        try:
            stop_response = session.post(stop_url, timeout=5)
            if stop_response.status_code not in [204, 409]: stop_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to stop reader profile: {e}"); return False
        time.sleep(1)

        response = session.get(profile_url, timeout=5); response.raise_for_status()
        current_profile_config = response.json()

        found_antenna = False
        for antenna in current_profile_config.get('antennaConfigs', []):
            if antenna.get('antennaPort') == physical_port:
                antenna['transmitPowerCdbm'] = int(power_level * 100); found_antenna = True; break

        if not found_antenna:
            if 'antennaConfigs' not in current_profile_config: current_profile_config['antennaConfigs'] = []
            current_profile_config['antennaConfigs'].append({"antennaPort": physical_port, "transmitPowerCdbm": int(power_level * 100)})

        response = session.put(profile_url, json=current_profile_config, timeout=5); response.raise_for_status()
        logging.info(f"✅ Updated profile with new power for antenna {physical_port}.")

        start_url = f"{profile_url}/start"; logging.info(f"Restarting inventory profile...")
        start_response = session.post(start_url, timeout=5); start_response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"‼️ Failed to set power for antenna {physical_port}: {e}")
        try: session.post(f"{profile_url}/start", timeout=5) # Best effort restart
        except: pass
        return False
    finally:
        session.close()

@app.route('/api/antenna/power', methods=['POST'])
def handle_antenna_power():
    data = request.json
    antenna_id, power = data.get('antenna_id'), data.get('power')
    try:
        antenna_id, power = int(antenna_id), float(power)
        if not (10.0 <= power <= 30.0): raise ValueError("Power out of range")
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid antenna ID or power level.'}), 400

    success = set_antenna_power_on_reader(antenna_id, power)
    
    if success:
        with DATA_LOCK:
            if antenna_id in SYSTEM_HEALTH['antennas']:
                SYSTEM_HEALTH['antennas'][antenna_id]['power_level'] = power
                SYSTEM_HEALTH['antennas'][antenna_id]['manual_override'] = True
            for ant_id in SYSTEM_HEALTH['antennas']:
                 SYSTEM_HEALTH['antennas'][ant_id]['manual_override'] = True
        logging.info(f"Manual power set for Antenna {antenna_id}. RL auto-tuning paused.")
        return jsonify({'status': 'success', 'message': f'Antenna {antenna_id} power set to {power} dBm.'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to set power on reader hardware.'}), 500


# --- Shielding Cart Specific Endpoints ---
@app.route('/api/shielding_cart/inventory_mode', methods=['POST'])
def set_shielding_cart_inventory_mode():
    mode = request.json.get('mode')
    valid_modes = ['scan_only', 'check_in', 'check_out', 'inventory_audit', 'asn_check_in', 'asn_check_out']
    if mode not in valid_modes:
        return jsonify({'status': 'error', 'message': 'Invalid inventory mode'}), 400
    with DATA_LOCK:
        SHIELDING_CART_STATE['inventory_mode'] = mode
    return jsonify({'status': 'success', 'message': f'Inventory mode set to {mode.replace("_", " ")}'})

@app.route('/api/shielding_cart/session', methods=['POST'])
def control_shielding_cart_session():
    global MASTER_INVENTORY
    action = request.json.get('action')
    with DATA_LOCK:
        current_mode = SHIELDING_CART_STATE['inventory_mode']
        if current_mode == 'inventory_audit' or 'asn' in current_mode:
             return jsonify({'status': 'error', 'message': f'Use specific /audit or /asn endpoints for mode {current_mode}'}), 400

        if action == 'start':
            SHIELDING_CART_STATE.update({
                'status': 'scanning', 'session_id': f"session_{int(time.time())}",
                'scanned_items': defaultdict(lambda: {'count': 0, 'reads': deque(maxlen=50)}),
                'start_time': time.time(), 'session_summary': {}, 'inventory_predictions': {},
                'power_boost_active': False
            })
            message = "Shielding cart scan started."
        elif action == 'stop':
            if SHIELDING_CART_STATE['status'] != 'scanning':
                return jsonify({'status': 'error', 'message': 'No active scan to stop.'}), 400

            duration = time.time() - SHIELDING_CART_STATE.get('start_time', time.time())
            scanned_items = SHIELDING_CART_STATE['scanned_items']
            unique_items = len(scanned_items)
            total_reads = sum(d['count'] for d in scanned_items.values())
            mode = SHIELDING_CART_STATE['inventory_mode']

            scans_per_minute = (total_reads / duration * 60) if duration > 0 else 0
            scan_rate_score = min(100, (scans_per_minute / TARGET_SCANS_PER_MINUTE) * 100)
            all_rssi = [float(r) for d in scanned_items.values() for r in d['reads'] if isinstance(r, (int, float))]
            rssi_std = np.std(all_rssi) if len(all_rssi) > 1 else 0
            stability_score = max(0, 100 - (float(rssi_std) * 10))
            performance_score = np.nan_to_num(round((scan_rate_score * 0.4) + (stability_score * 0.6)))

            summary = {
                'unique_items': unique_items, 'total_reads': total_reads, 'duration_sec': round(float(duration), 2),
                'scans_per_minute': round(float(scans_per_minute), 2),
                'performance_score': performance_score
            }

            updates_made = 0
            if mode != 'scan_only' and not MASTER_INVENTORY.empty:
                 for epc, data in scanned_items.items():
                     if epc in MASTER_INVENTORY.index:
                         item_count = 1 # Each unique EPC is one item in standard modes
                         current_stock = MASTER_INVENTORY.loc[epc, 'stock_level']
                         if mode == 'check_in':
                             MASTER_INVENTORY.loc[epc, 'stock_level'] = current_stock + item_count
                             updates_made += 1
                         elif mode == 'check_out':
                             MASTER_INVENTORY.loc[epc, 'stock_level'] = max(0, current_stock - item_count)
                             updates_made += 1
                 if updates_made > 0:
                     save_master_inventory()
                 message = f"Session complete. Stock levels updated ({mode}) for {updates_made} unique items."
            else:
                 message = "Shielding cart scan complete. Scorecard generated."

            SHIELDING_CART_STATE['status'] = 'complete'
            SHIELDING_CART_STATE['session_summary'] = summary
            log_to_csv(SHIELDING_SESSIONS_LOG_FILE, {'timestamp': datetime.now(timezone.utc).isoformat(), 'session_type': mode, **summary})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid action'}), 400
    return jsonify({'status': 'success', 'message': message})

@app.route('/api/shielding_cart/audit', methods=['POST'])
def control_shielding_cart_audit():
    global MASTER_INVENTORY
    action = request.json.get('action')
    with DATA_LOCK:
        if action == 'start':
            SHIELDING_CART_STATE.update({
                'status': 'auditing', 'session_id': f"audit_{int(time.time())}",
                'start_time': time.time(), 'audit_target': None, 'audit_results': {}, 'session_summary': {},
            })
            return jsonify({'status': 'success', 'message': 'Audit session started. Set a SKU to begin scanning.'})
        elif action == 'set_sku':
            if SHIELDING_CART_STATE['status'] != 'auditing': return jsonify({'status': 'error', 'message': 'No active audit session.'}), 400
            sku = request.json.get('sku', '').upper().strip()
            if not sku: return jsonify({'status': 'error', 'message': 'SKU is required'}), 400
            if MASTER_INVENTORY.empty or 'sku' not in MASTER_INVENTORY.columns: return jsonify({'status': 'error', 'message': 'Master Inventory with SKU column is not loaded.'}), 400

            sku_info = MASTER_INVENTORY[MASTER_INVENTORY['sku'] == sku]
            if sku_info.empty: return jsonify({'status': 'error', 'message': f'SKU "{sku}" not found.'}), 404

            item_name = sku_info['name'].iloc[0]
            expected_count = int(sku_info['stock_level'].iloc[0]) # Use stock level for expected

            SHIELDING_CART_STATE['audit_target'] = {'sku': sku, 'name': item_name}
            if sku not in SHIELDING_CART_STATE['audit_results']:
                SHIELDING_CART_STATE['audit_results'][sku] = {
                    'name': item_name, 'expected_count': expected_count, 'scanned_epcs': set(), 'status': 'pending'
                }
            return jsonify({'status': 'success', 'message': f'Audit target: {sku} ({item_name}). Expected: {expected_count}. Ready.'})
        elif action == 'pause_scan': # Pause current SKU scan
            if SHIELDING_CART_STATE['status'] != 'auditing' or not SHIELDING_CART_STATE['audit_target']: return jsonify({'status': 'error', 'message': 'No active SKU is being scanned.'}), 400
            current_sku = SHIELDING_CART_STATE['audit_target']['sku']
            if current_sku in SHIELDING_CART_STATE['audit_results']:
                SHIELDING_CART_STATE['audit_results'][current_sku]['status'] = 'scanned'
            SHIELDING_CART_STATE['audit_target'] = None
            return jsonify({'status': 'success', 'message': f'Scan for SKU {current_sku} paused. Set a new SKU or finish.'})
        elif action == 'finish_scan': # Move to resolution phase
            if SHIELDING_CART_STATE['status'] != 'auditing': return jsonify({'status': 'error', 'message': 'No active audit session to finish.'}), 400
            if SHIELDING_CART_STATE.get('audit_target'): # If user didn't explicitly pause last item
                current_sku = SHIELDING_CART_STATE['audit_target']['sku']
                if current_sku in SHIELDING_CART_STATE['audit_results']:
                     SHIELDING_CART_STATE['audit_results'][current_sku]['status'] = 'scanned'

            for sku, data in SHIELDING_CART_STATE['audit_results'].items():
                data['scanned_count'] = len(data['scanned_epcs'])
                data['discrepancy'] = data['scanned_count'] - data['expected_count']
                data['status'] = 'pending_resolution'
            SHIELDING_CART_STATE['status'] = 'awaiting_resolution'
            SHIELDING_CART_STATE['audit_target'] = None
            return jsonify({'status': 'success', 'message': 'Audit scanning finished. Awaiting resolution.'})
        elif action == 'rescan':
             sku = request.json.get('sku', '').upper().strip()
             if not sku or sku not in SHIELDING_CART_STATE['audit_results']: return jsonify({'status': 'error', 'message': 'Valid SKU required for rescan.'}), 400
             sku_data = SHIELDING_CART_STATE['audit_results'][sku]
             sku_data['scanned_epcs'] = set()
             sku_data['status'] = 'pending'
             SHIELDING_CART_STATE['audit_target'] = {'sku': sku, 'name': sku_data['name']}
             SHIELDING_CART_STATE['status'] = 'auditing'
             return jsonify({'status': 'success', 'message': f'Ready to rescan SKU: {sku}.'})
        elif action == 'finalize':
            if SHIELDING_CART_STATE['status'] != 'awaiting_resolution': return jsonify({'status': 'error', 'message': 'Audit must be in resolution state.'}), 400
            resolutions = request.json.get('resolutions', {})
            updates_made = 0
            for sku, resolution in resolutions.items():
                if sku not in SHIELDING_CART_STATE['audit_results']: continue
                data = SHIELDING_CART_STATE['audit_results'][sku]
                scanned_count = data['scanned_count']
                if resolution == 'accept':
                    if data['discrepancy'] != 0:
                        MASTER_INVENTORY.loc[MASTER_INVENTORY['sku'] == sku, 'stock_level'] = scanned_count
                        logging.info(f"AUDIT ACCEPT: SKU {sku} stock updated to {scanned_count}.")
                        updates_made += 1
                elif resolution == 'log_missing':
                    if data['discrepancy'] < 0:
                        expected_epcs = set(MASTER_INVENTORY[MASTER_INVENTORY['sku'] == sku].index)
                        scanned_epcs = data['scanned_epcs']
                        missing_epcs = list(expected_epcs - scanned_epcs)
                        if 'inventory_status' not in MASTER_INVENTORY.columns: MASTER_INVENTORY['inventory_status'] = 'Available'
                        MASTER_INVENTORY.loc[missing_epcs, 'inventory_status'] = 'Missing'
                        MASTER_INVENTORY.loc[MASTER_INVENTORY['sku'] == sku, 'stock_level'] = scanned_count
                        logging.info(f"AUDIT LOG MISSING: SKU {sku} stock updated. Marked {len(missing_epcs)} EPCs as Missing.")
                        updates_made += 1
                elif resolution == 'reject':
                     logging.info(f"AUDIT REJECT: Discrepancy for SKU {sku} rejected.")

            if updates_made > 0: save_master_inventory()
            SHIELDING_CART_STATE['status'] = 'complete'
            log_to_csv(SHIELDING_SESSIONS_LOG_FILE, {'timestamp': datetime.now(timezone.utc).isoformat(), 'session_type': 'audit', 'results': json.dumps(SHIELDING_CART_STATE['audit_results'], cls=CustomJSONEncoder)})
            return jsonify({'status': 'success', 'message': f'Audit finalized. Inventory updated for {updates_made} SKUs.'})
        elif action == 'cancel':
             SHIELDING_CART_STATE.update({'status': 'idle', 'session_id': None, 'start_time': None, 'audit_target': None, 'audit_results': {}})
             return jsonify({'status': 'success', 'message': 'Audit session cancelled.'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid audit action'}), 400

@app.route('/api/shielding_cart/upload_asn', methods=['POST'])
def upload_asn():
    if 'file' not in request.files: return jsonify({'status': 'error', 'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    try:
        df = pd.read_csv(file) if file.filename.lower().endswith('.csv') else pd.read_excel(file)
        required_cols = ['po_number', 'sku', 'box_id', 'epc']
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f'ASN file must contain columns: {", ".join(required_cols)}')
        df = df.astype(str).apply(lambda x: x.str.strip().str.upper())
        po_number = df['po_number'].iloc[0]
        boxes = defaultdict(lambda: {'expected_epcs': set(), 'sku_counts': defaultdict(int), 'skus': set()})
        for _, row in df.iterrows():
            boxes[row['box_id']]['expected_epcs'].add(row['epc'])
            boxes[row['box_id']]['sku_counts'][row['sku']] += 1
            boxes[row['box_id']]['skus'].add(row['sku'])

        asn_data = {
            'po_number': po_number,
            'boxes': {box_id: { 'expected_epcs': list(data['expected_epcs']), 'sku_counts': dict(data['sku_counts']), 'skus': list(data['skus']) } for box_id, data in boxes.items()}
        }
        with DATA_LOCK:
            SHIELDING_CART_STATE.update({
                'asn_status': 'awaiting_box_scan', 'asn_data': asn_data, 'asn_scan_results': {}, 'current_asn_box': None, 'status': 'idle'
            })
        return jsonify({'status': 'success', 'message': f'ASN for PO "{po_number}" ({len(boxes)} boxes) loaded.', 'asn_data': asn_data})
    except Exception as e:
        logging.error(f"Error processing ASN file: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/shielding_cart/asn_action', methods=['POST'])
def asn_action():
    data = request.json
    action = data.get('action')
    with DATA_LOCK:
        mode = SHIELDING_CART_STATE['inventory_mode']
        if mode not in ['asn_check_in', 'asn_check_out']: return jsonify({'status': 'error', 'message': 'Action only available in ASN modes.'}), 400

        if action == 'start_box_scan':
            box_id = data.get('box_id')
            if not box_id or box_id not in SHIELDING_CART_STATE.get('asn_data', {}).get('boxes', {}): return jsonify({'status': 'error', 'message': 'Invalid Box ID.'}), 400
            SHIELDING_CART_STATE['asn_status'] = 'scanning_box'
            SHIELDING_CART_STATE['current_asn_box'] = {'box_id': box_id, 'scanned_epcs': set()}
            return jsonify({'status': 'success', 'message': f'Scanning started for Box {box_id}.'})
        elif action == 'rescan_box':
             box_id = data.get('box_id')
             if not box_id or box_id not in SHIELDING_CART_STATE.get('asn_data', {}).get('boxes', {}): return jsonify({'status': 'error', 'message': 'Invalid Box ID for rescan.'}), 400
             SHIELDING_CART_STATE['asn_scan_results'].pop(box_id, None)
             SHIELDING_CART_STATE['asn_status'] = 'scanning_box'
             SHIELDING_CART_STATE['current_asn_box'] = {'box_id': box_id, 'scanned_epcs': set()}
             return jsonify({'status': 'success', 'message': f'Ready to rescan Box {box_id}.'})
        elif action == 'finalize_box_scan':
            if SHIELDING_CART_STATE['asn_status'] != 'scanning_box': return jsonify({'status': 'error', 'message': 'No active box scan to finalize.'}), 400
            current_box = SHIELDING_CART_STATE['current_asn_box']
            box_id = current_box['box_id']
            scanned_epcs = current_box['scanned_epcs']
            expected_epcs = set(SHIELDING_CART_STATE['asn_data']['boxes'][box_id]['expected_epcs'])
            matched, missing, extra = scanned_epcs.intersection(expected_epcs), expected_epcs - scanned_epcs, scanned_epcs - expected_epcs
            result = {'matched': list(matched), 'missing': list(missing), 'extra': list(extra)}
            SHIELDING_CART_STATE['asn_scan_results'][box_id] = result
            SHIELDING_CART_STATE['asn_status'] = 'awaiting_box_scan'
            SHIELDING_CART_STATE['current_asn_box'] = None
            return jsonify({'status': 'success', 'message': f'Box {box_id} scan finalized. M:{len(matched)}, Ms:{len(missing)}, E:{len(extra)}.', 'result': result})
        elif action == 'complete_asn':
            resolution = data.get('resolution', 'accept')
            if SHIELDING_CART_STATE['asn_status'] != 'awaiting_box_scan': return jsonify({'status': 'error', 'message': f"Cannot complete ASN. Status is '{SHIELDING_CART_STATE['asn_status']}'."}), 400

            updates_made = 0
            if resolution != 'reject':
                change = 1 if 'check_in' in mode else -1
                missing_epcs_total = []
                for box_id, result in SHIELDING_CART_STATE['asn_scan_results'].items():
                    items_to_update = result['matched']
                    if mode == 'asn_check_in': items_to_update += result['extra']
                    for epc in items_to_update:
                        if epc in MASTER_INVENTORY.index:
                            current_stock = MASTER_INVENTORY.loc[epc, 'stock_level']
                            MASTER_INVENTORY.loc[epc, 'stock_level'] = max(0, current_stock + change)
                            updates_made += 1
                    missing_epcs_total.extend(result['missing'])

                if resolution == 'log_missing' and missing_epcs_total:
                    if 'inventory_status' not in MASTER_INVENTORY.columns: MASTER_INVENTORY['inventory_status'] = 'Available'
                    valid_missing_epcs = [epc for epc in missing_epcs_total if epc in MASTER_INVENTORY.index]
                    if valid_missing_epcs:
                         MASTER_INVENTORY.loc[valid_missing_epcs, 'inventory_status'] = 'Missing'
                         logging.info(f"ASN LOG MISSING: Marked {len(valid_missing_epcs)} EPCs as 'Missing'.")

                if updates_made > 0: save_master_inventory()

            message = {'accept': f'ASN accepted. Inventory updated for {updates_made} items.',
                       'log_missing': f'ASN processed. Inventory updated for {updates_made} items, missing logged.',
                       'reject': 'ASN Rejected. No inventory changes.'}.get(resolution, 'ASN finalized.')
            SHIELDING_CART_STATE['asn_status'] = 'complete'
            return jsonify({'status': 'success', 'message': message})
        elif action == 'cancel_asn':
            SHIELDING_CART_STATE.update({'status': 'idle', 'inventory_mode': 'scan_only', 'asn_status': 'idle', 'asn_data': None, 'asn_scan_results': {}, 'current_asn_box': None})
            return jsonify({'status': 'success', 'message': 'Returned to main dashboard.'})
    return jsonify({'status': 'error', 'message': 'Invalid ASN action.'}), 400

@app.route('/api/shielding_cart/history')
def get_shielding_cart_history():
    if not os.path.exists(SHIELDING_SESSIONS_LOG_FILE):
        return jsonify({'status': 'success', 'history': []})
    try:
        df = pd.read_csv(SHIELDING_SESSIONS_LOG_FILE, on_bad_lines='skip').tail(30)
        if 'timestamp' not in df.columns: return jsonify({'status': 'success', 'history': []})
        df.dropna(subset=['timestamp'], inplace=True)
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                 df[col] = df[col].astype(float if np.issubdtype(df[col].dtype, np.floating) else int)
        return jsonify({'status': 'success', 'history': df.to_dict('records')})
    except Exception as e:
        logging.error(f"Error reading shielding cart history: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- Inventory Management Endpoints ---
@app.route('/api/sc/upload_inventory_doc', methods=['POST'])
def sc_upload_inventory_doc():
    global MASTER_INVENTORY
    if 'file' not in request.files: return jsonify({'status': 'error', 'message': 'No file part'}), 400
    try:
        df = pd.read_csv(request.files['file'], dtype={'epc': str, 'sku': str}) if request.files['file'].filename.lower().endswith('.csv') else pd.read_excel(request.files['file'], dtype={'epc': str, 'sku': str})
        
        required_cols = ['epc', 'sku', 'name']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f'File must contain at least: {", ".join(required_cols)}')

        df['epc'] = df['epc'].astype(str).str.upper().str.strip()
        df['sku'] = df['sku'].astype(str).str.upper().str.strip()

        if 'stock_level' not in df.columns: df['stock_level'] = 100
        if 'avg_daily_usage' not in df.columns: df['avg_daily_usage'] = 1.0
        if 'prophet_days_left' not in df.columns: df['prophet_days_left'] = 0.0

        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0).round(2)
        else:
            df['price'] = 0.0

        df['avg_daily_usage'] = pd.to_numeric(df['avg_daily_usage'], errors='coerce').fillna(1.0).round(2)
        df['stock_level'] = pd.to_numeric(df['stock_level'], errors='coerce').fillna(0).astype(int)
        df['prophet_days_left'] = pd.to_numeric(df['prophet_days_left'], errors='coerce').fillna(0.0)

        with DATA_LOCK:
            MASTER_INVENTORY = df.set_index('epc', drop=False)
            save_master_inventory()
        logging.info(f"Successfully loaded master inventory for Shielding Cart with {len(df)} items.")
        return jsonify({'status': 'success', 'message': f'Loaded {len(df)} items.'})
    except Exception as e:
        logging.error(f"Error processing inventory file: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sc/upload_sales_doc', methods=['POST'])
def sc_upload_sales_doc():
    if 'file' not in request.files: return jsonify({'status': 'error', 'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    try:
        df = pd.read_csv(file) if file.filename.lower().endswith('.csv') else pd.read_excel(file)
        required_cols = ['timestamp', 'epc']
        if not all(col in df.columns for col in required_cols): raise ValueError(f'Sales file must contain: {required_cols}')
        # This function now triggers Prophet training
        process_sales_data_for_usage(df)
        return jsonify({'status': 'success', 'message': f'Processed {len(df)} sales records and triggered AI model training.'})
    except Exception as e:
        logging.error(f"Error processing sales file: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/download_inventory')
def download_inventory():
    with DATA_LOCK:
        if MASTER_INVENTORY.empty: return jsonify({"status": "error", "message": "No inventory data."}), 404
        df_to_save = MASTER_INVENTORY.copy()
        if 'epc' not in df_to_save.columns: df_to_save.reset_index(inplace=True)
        for col in ['stock_level', 'price', 'avg_daily_usage', 'prophet_days_left']:
            if col in df_to_save.columns: df_to_save[col] = df_to_save[col].apply(lambda x: float(x) if isinstance(x, (np.floating, float)) else (int(x) if isinstance(x, (np.integer, int)) else x))
        buffer = io.StringIO()
        df_to_save.to_csv(buffer, index=False)
        mem = io.BytesIO()
        mem.write(buffer.getvalue().encode('utf-8'))
        mem.seek(0); buffer.close()
        return send_file(mem, as_attachment=True, download_name=f'inventory_snapshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', mimetype='text/csv')

@app.route('/api/inventory_view')
def get_inventory_view():
    with DATA_LOCK:
        if MASTER_INVENTORY.empty: return jsonify({'status': 'success', 'inventory': []})
        df = MASTER_INVENTORY.copy().replace({np.nan: None}) # CRITICAL: Replace NaN
        if 'last_checked' not in df.columns: df['last_checked'] = [(datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d') for _ in range(len(df))]
        query = request.args.get('query', '').lower()
        if query:
            df = df[df['name'].astype(str).str.lower().str.contains(query, na=False) |
                    df['epc'].astype(str).str.lower().str.contains(query, na=False) |
                    df['sku'].astype(str).str.lower().str.contains(query, na=False) | # Add SKU search
                    df['sales_class'].astype(str).str.lower().str.contains(query, na=False)]
        df.sort_values('stock_level', ascending=True, inplace=True)
        inventory_list = df.to_dict('records')
    return jsonify({'status': 'success', 'inventory': inventory_list})

# --- VISION & AI ENDPOINTS ---

# === NEW ENDPOINT START ===
@app.route('/api/ai/all_depletion_predictions')
def get_all_depletion_predictions():
    """Fetches all calculated Prophet predictions."""
    with DATA_LOCK:
        if MASTER_INVENTORY.empty or 'prophet_days_left' not in MASTER_INVENTORY.columns:
            return jsonify({'status': 'success', 'predictions': []})
        
        try:
            predictions_df = MASTER_INVENTORY[MASTER_INVENTORY['prophet_days_left'] > 0].copy()
            if predictions_df.empty:
                return jsonify({'status': 'success', 'predictions': []})
                
            predictions_df = predictions_df[['name', 'sku', 'prophet_days_left']]
            predictions_df.sort_values('prophet_days_left', ascending=True, inplace=True)
            predictions_list = predictions_df.to_dict('records')
            return jsonify({'status': 'success', 'predictions': predictions_list})
        except Exception as e:
            logging.error(f"Error fetching all predictions: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
# === NEW ENDPOINT END ===


@app.route('/api/vision_control', methods=['POST', 'DELETE'])
def handle_vision():
    global VISION_WORKER_STATE
    if request.method == 'POST':
        with DATA_LOCK:
            if VISION_WORKER_STATE.get('thread') and VISION_WORKER_STATE['thread'].is_alive(): return jsonify({'status': 'error', 'message': 'Vision worker running.'}), 409
            stop_event = Event(); thread = Thread(target=vision_worker, args=(stop_event,), name="SC-Vision-Worker", daemon=True)
            VISION_WORKER_STATE = {'thread': thread, 'stop_event': stop_event, 'status': 'Starting'}; thread.start()
        return jsonify({'status': 'success', 'message': 'Vision worker started.'})
    elif request.method == 'DELETE':
        with DATA_LOCK:
            if VISION_WORKER_STATE.get('stop_event'): VISION_WORKER_STATE['stop_event'].set()
        return jsonify({'status': 'success', 'message': 'Vision worker stopped.'})

@app.route('/video_feed_sc') # Renamed endpoint for clarity
def video_feed_sc():
    def gen():
        while True:
            with DATA_LOCK: frame_bytes = LATEST_FRAME
            if frame_bytes: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera Off", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', placeholder)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==============================================================================
# === MAIN EXECUTION ===
# ==============================================================================

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        app.json_encoder = CustomJSONEncoder

    load_shielding_cart_models()
    load_master_inventory()

    worker_threads_to_start = {
        'shielding_cart_ai': Thread(target=shielding_cart_ai_worker, name="Shielding-Cart-AI-Worker", daemon=True),
        'predictive_maintenance': Thread(target=predictive_maintenance_worker, name="Maint-Predict-Worker", daemon=True),
        'rl_power_tuning': Thread(target=rl_power_tuning_worker, name="RL-Power-Tuning-Agent", daemon=True),
        'shielding_cart_analytics': Thread(target=shielding_cart_analytics_worker, name="SC-Analytics-Worker", daemon=True),
        # Vision worker is started via API if needed
    }

    for name, thread in worker_threads_to_start.items():
        thread.start()
        BACKGROUND_WORKERS[name] = thread

    logging.info("\n--- 🚀 Shielding Cart Application (Standalone) is LIVE on port 5002 ---")
   # --- RENDER FIX: Bind to 0.0.0.0 and use Render's PORT environment variable ---
port = int(os.environ.get("PORT", 5002))
serve(app, host='0.0.0.0', port=port)
# --- END RENDER FIX ---