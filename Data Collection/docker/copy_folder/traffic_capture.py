#!/usr/bin/python3
import argparse
import asyncio
import multiprocessing
import nest_asyncio
import os
import pyshark
import subprocess
import time
import signal
import resource
import gc
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

nest_asyncio.apply()
multiprocessing.set_start_method('spawn', force=True)

# --- CONFIGURATION ---
INTERFACE = "eth0"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 1080
PROXY_ENDPOINT = "162.159.198.1:443"
INPUT_WEBSITES_FILE = "./target_websites.txt"
# ---------------------

parser = argparse.ArgumentParser(description="Capture network traffic.")
parser.add_argument("--trace_file_dir", dest="trace_file_dir", default="./traces/")
parser.add_argument("--target_websites", dest="target_websites", default=INPUT_WEBSITES_FILE)
parser.add_argument("--websites_count", dest="websites_count", default=10, type=int)
parser.add_argument("--gap_count", dest="gap_count", default=0, type=int)
parser.add_argument("--access_count", dest="access_count", default=2, type=int)
parser.add_argument('--filter', action='store_true', default=False)
args = parser.parse_args()


# ─── CHROME ───────────────────────────────────────────────────────────────────

def open_website(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--incognito")
    options.add_argument("--disk-cache-size=0")
    options.add_argument(f"--proxy-server=socks5://{PROXY_HOST}:{PROXY_PORT}")
    options.add_argument("--ignore-certificate-errors")
    
    options.page_load_strategy = 'none'
    
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(10)
        driver.get(url)
        print(f"-----Loaded: {driver.title}")
        time.sleep(10)
    except Exception as e:
        print(f"-----Browser Error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


# ─── TCPDUMP ──────────────────────────────────────────────────────────────────

def start_tcpdump(trace_file):
    print("-----Running tcpdump")
    return subprocess.Popen([
        "/usr/bin/tcpdump", "-i", INTERFACE,
        "-w", trace_file,
        "port", "80", "or", "port", "443", "or", "port", "853"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_tcpdump(proc):
    print("-----Stopping tcpdump")
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─── MASQUE PROXY ─────────────────────────────────────────────────────────────

def start_proxy():
    print("-----Starting Masque Proxy")
    os.system("fuser -k 1080/tcp 2>/dev/null")
    os.system("pkill -9 -f masque-plus 2>/dev/null")
    os.system("pkill -9 -f usque 2>/dev/null")
    time.sleep(1)
    
    proc = subprocess.Popen(
        ["./masque-plus", "--endpoint", PROXY_ENDPOINT],
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    time.sleep(3)
    return proc

def stop_proxy(proc):
    print("-----Stopping Masque Proxy")
    if proc:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass

    os.system("fuser -k 1080/tcp 2>/dev/null")
    os.system("pkill -9 -f masque-plus 2>/dev/null")
    os.system("pkill -9 -f usque 2>/dev/null")
    time.sleep(1)


# ─── PYSHARK PROCESSING ───────────────────────────────────────────────────────

def contains_quic(trace_file):
    try:
        # Dùng 'with' để Pyshark tự đóng luồng an toàn, không sinh lỗi Event Loop
        with pyshark.FileCapture(trace_file, display_filter="quic") as cap:
            for packet in cap:
                if hasattr(packet, 'udp') and (packet.udp.srcport == "443" or packet.udp.dstport == "443"):
                    return True
        return False
    except Exception as e:
        print(f"Pyshark read error: {e}")
        return False
    finally:
        os.system("pkill -9 -f tshark 2>/dev/null")
        time.sleep(0.5)

def parse_pcap(trace_file_path):
    features = []
    try:
        host_ip = os.popen("hostname -i").read().strip().split()[0]
        # Dùng 'with' quản lý vòng đời Capture
        with pyshark.FileCapture(trace_file_path, keep_packets=False) as cap:
            for packet in cap:
                try:
                    if not hasattr(packet, 'ip') or not hasattr(packet, 'transport_layer') or packet.transport_layer is None:
                        continue
                    
                    t_layer = packet.transport_layer
                    src_ip, dst_ip = str(packet.ip.src_host), str(packet.ip.dst_host)
                    
                    features.append({
                        "protocol": "1" if t_layer in ("QUIC", "UDP") else "0",
                        "length": str(packet.length),
                        "relative_time": f"{float(getattr(packet.frame_info, 'time_delta')):.9f}",
                        "direction": "1" if src_ip == host_ip else "0",
                        "src_ip": src_ip,
                        "src_port": str(packet[t_layer].srcport),
                        "dst_ip": dst_ip,
                        "dst_port": str(packet[t_layer].dstport)
                    })
                except Exception:
                    continue
    except Exception as e:
        print(f"-----Error parsing PCAP: {e}")
    finally:
        os.system("pkill -9 -f tshark 2>/dev/null")
        time.sleep(0.5)
        return features


# ─── PACKET CAPTURE ───────────────────────────────────────────────────────────

def packet_capture(url, index, pcap_base_dir):
    tcp_proc = proxy_proc = web_proc = None
    domain = url.split("//")[1]
    dir_name = "_".join(domain.split(".")).replace("/", "_").rstrip("_")
    
    website_pcap_dir = os.path.join(pcap_base_dir, dir_name)
    os.makedirs(website_pcap_dir, exist_ok=True)
    trace_file = os.path.join(website_pcap_dir, f"{index + 1}_{dir_name}.pcap")

    try:
        tcp_proc = start_tcpdump(trace_file)
        time.sleep(1)

        proxy_proc = start_proxy()

        web_proc = multiprocessing.Process(target=open_website, args=(url,))
        web_proc.start()
        web_proc.join(timeout=20)
        
        if web_proc.is_alive():
            print("-----Chrome timeout, terminating...")
            web_proc.terminate()
            web_proc.join(timeout=5)
            if web_proc.is_alive(): web_proc.kill()

    except Exception as e:
        print(f"Error during capture: {e}")
        if os.path.exists(trace_file): os.remove(trace_file)
    finally:
        if web_proc and web_proc.is_alive():
            web_proc.terminate()
            web_proc.join(timeout=3)
        
        stop_proxy(proxy_proc)
        stop_tcpdump(tcp_proc)
        
        os.system("pkill -9 -f chromedriver 2>/dev/null")
        os.system("pkill -9 -f chrome 2>/dev/null")
        time.sleep(1)

    return trace_file, dir_name


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

def generate_traces(target_websites_file):
    pcap_dir = os.path.join(args.trace_file_dir, "pcap")
    csv_dir = os.path.join(args.trace_file_dir, "csv")
    os.makedirs(pcap_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Đọc và phân loại danh sách URL đúng 1 lần duy nhất
    if not os.path.exists(target_websites_file):
        print(f"File {target_websites_file} không tồn tại!")
        return

    with open(target_websites_file, 'r') as f:
        all_urls = [line.strip() for line in f if line.strip()]
    
    # Cắt danh sách (slice) theo cấu hình gap_count và websites_count
    target_urls = all_urls[args.gap_count : args.gap_count + args.websites_count]

    for i in range(args.access_count):
        for url in target_urls:
            print(f"\n[+] Processing {url} (Attempt {i+1}/{args.access_count})")

            trace_file_path, dir_name = packet_capture(url, i, pcap_dir)
            if not os.path.exists(trace_file_path):
                continue

            # Phân tích Pyshark
            has_quic = contains_quic(trace_file_path) if args.filter else True
            
            if has_quic:
                features = parse_pcap(trace_file_path)
                if features:
                    website_csv_dir = os.path.join(csv_dir, dir_name)
                    os.makedirs(website_csv_dir, exist_ok=True)
                    dataset_file_path = os.path.join(website_csv_dir, f"{i + 1}_{dir_name}.csv")
                    
                    with open(dataset_file_path, mode='w') as df:
                        df.write(";".join(features[0].keys()) + '\n')
                        for feature in features:
                            df.write(";".join(feature.values()) + '\n')
                    
                    print(f"--> Saved PCAP to: {trace_file_path}")
                    print(f"--> Saved CSV to: {dataset_file_path}")
            
            # Xóa rác RAM lập tức
            gc.collect()


if __name__ == '__main__':
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
        _, hard_proc = resource.getrlimit(resource.RLIMIT_NPROC)
        resource.setrlimit(resource.RLIMIT_NPROC, (65536, hard_proc))
    except Exception as e:
        pass

    start_time = time.time()
    generate_traces(args.target_websites)
    print(f"\nElapsed time: {time.time() - start_time:.2f} seconds")
    
    # Trả lại quyền file cho host
    subprocess.run(["chmod", "-R", "777", args.trace_file_dir], stderr=subprocess.DEVNULL)