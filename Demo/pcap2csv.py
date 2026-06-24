#!/usr/bin/env python3
"""
pcap2csv.py — Convert PCAP → CSV cho DeepMASQUE
=================================================
Format output (separator ";"):
  protocol;length;relative_time;direction;src_ip;src_port;dst_ip;dst_port

Trong đó:
  protocol  : 1 = QUIC/UDP, 0 = khác
  direction : 1 = outgoing (client → proxy), 0 = incoming (proxy → client)

Usage:
  python3 pcap2csv.py <pcap_file> <output_csv> --client <victim_ip>

Examples:
  python3 pcap2csv.py /tmp/victim_traffic.pcap /tmp/trace.csv --client 192.168.0.120
  python3 pcap2csv.py capture.pcap out.csv --client 192.168.0.120 --proxy 162.159.198.1
  python3 pcap2csv.py capture.pcap out.csv --client 192.168.0.120 --port 443
"""

import argparse
import sys
import os

try:
    from scapy.all import rdpcap, IP, IPv6, UDP, TCP
except ImportError:
    print("[!] scapy not found. Install: pip install scapy --break-system-packages")
    sys.exit(1)


def is_quic(pkt):
    """
    Heuristic nhận biết QUIC:
    - UDP port 443 (QUIC thường chạy trên UDP/443)
    - Hoặc UDP port 80 (HTTP/3)
    """
    if UDP in pkt:
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
        if 443 in (sport, dport) or 80 in (sport, dport):
            return True
    return False


def get_ip_layer(pkt):
    if IP in pkt:
        return pkt[IP]
    if IPv6 in pkt:
        return pkt[IPv6]
    return None


def get_ports(pkt):
    if UDP in pkt:
        return pkt[UDP].sport, pkt[UDP].dport
    if TCP in pkt:
        return pkt[TCP].sport, pkt[TCP].dport
    return 0, 0


def pcap_to_csv(pcap_path, output_path, client_ip,
                proxy_ip=None, filter_port=None,
                quic_only=False, max_packets=None,
                verbose=True):
    """
    Đọc pcap, filter theo client_ip, xuất CSV.

    Args:
        pcap_path   : đường dẫn file .pcap
        output_path : đường dẫn file .csv output
        client_ip   : IP của victim/client để xác định direction
        proxy_ip    : nếu có, chỉ lấy traffic giữa client và proxy này
        filter_port : chỉ lấy packet có port này (ví dụ 443)
        quic_only   : nếu True, chỉ lấy QUIC/UDP packet
        max_packets : giới hạn số packet (None = không giới hạn)
        verbose     : in progress
    """
    if not os.path.exists(pcap_path):
        print(f"[!] File not found: {pcap_path}")
        sys.exit(1)

    if verbose:
        print(f"[*] Reading: {pcap_path}")

    try:
        pkts = rdpcap(pcap_path)
    except Exception as e:
        print(f"[!] Cannot read pcap: {e}")
        sys.exit(1)

    if verbose:
        print(f"[*] Total packets in pcap: {len(pkts)}")

    rows = []
    t0_abs = None
    skipped = 0
    total_seen = 0

    for pkt in pkts:
        ip = get_ip_layer(pkt)
        if ip is None:
            skipped += 1
            continue

        src_ip = ip.src
        dst_ip = ip.dst

        # Filter: chỉ lấy packet liên quan đến client
        if src_ip != client_ip and dst_ip != client_ip:
            skipped += 1
            continue

        # Filter theo proxy nếu có
        if proxy_ip:
            if src_ip != proxy_ip and dst_ip != proxy_ip:
                skipped += 1
                continue

        # Filter theo port nếu có
        sport, dport = get_ports(pkt)
        if filter_port:
            if sport != filter_port and dport != filter_port:
                skipped += 1
                continue

        # Filter QUIC only
        if quic_only and not is_quic(pkt):
            skipped += 1
            continue

        # Timestamp
        ts = float(pkt.time)
        if t0_abs is None:
            t0_abs = ts
        relative_time = ts - t0_abs

        # Direction: 1 = outgoing (client→server), 0 = incoming
        direction = 1 if src_ip == client_ip else 0

        # Protocol: 1 = QUIC/UDP, 0 = other
        protocol = 1 if is_quic(pkt) else 0

        # Packet length (layer 3 trở lên)
        length = len(ip)

        rows.append([
            protocol,
            length,
            f"{relative_time:.9f}",
            direction,
            src_ip,
            sport,
            dst_ip,
            dport,
        ])

        total_seen += 1
        if max_packets and total_seen >= max_packets:
            if verbose:
                print(f"[*] Reached max_packets limit: {max_packets}")
            break

    if verbose:
        print(f"[*] Packets matched:  {len(rows)}")
        print(f"[*] Packets skipped:  {skipped}")

    if len(rows) == 0:
        print(f"[!] No packets matched client_ip={client_ip}.")
        print(f"    Make sure ARP poisoning was active during capture.")
        sys.exit(1)

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w') as f:
        # Header
        f.write("protocol;length;relative_time;direction;src_ip;src_port;dst_ip;dst_port\n")
        for row in rows:
            f.write(";".join(str(x) for x in row) + "\n")

    if verbose:
        print(f"[+] Saved: {output_path}  ({len(rows)} rows)")

    return len(rows)


def auto_detect_client(pcap_path, gateway_ip=None, exclude_ips=None):
    """
    Tự detect IP của client từ pcap:
    - Đếm số packet mỗi IP gửi/nhận
    - Loại bỏ gateway và các IP trong exclude_ips
    - IP có nhiều traffic nhất (2 chiều) → likely là victim
    """
    pkts = rdpcap(pcap_path)
    exclude = set(exclude_ips or [])
    if gateway_ip:
        exclude.add(gateway_ip)

    traffic = {}  # ip → packet count
    for pkt in pkts:
        ip = get_ip_layer(pkt)
        if ip is None: continue
        for addr in (ip.src, ip.dst):
            if addr not in exclude:
                traffic[addr] = traffic.get(addr, 0) + 1

    if not traffic:
        return None

    # Sort by count, lấy IP có nhiều traffic nhất
    sorted_ips = sorted(traffic.items(), key=lambda x: x[1], reverse=True)
    print(f"[*] Auto-detect — top IPs by packet count:")
    for ip, cnt in sorted_ips[:5]:
        print(f"    {ip:20s} → {cnt} packets")

    best_ip = sorted_ips[0][0]
    print(f"[*] Detected client IP: {best_ip}")
    return best_ip


def print_stats(pcap_path, client_ip):
    """In thống kê nhanh về pcap mà không convert."""
    pkts = rdpcap(pcap_path)
    total = len(pkts)
    matched = 0
    quic_count = 0
    tcp_count = 0
    out_count = 0
    in_count = 0

    for pkt in pkts:
        ip = get_ip_layer(pkt)
        if ip is None: continue
        if ip.src != client_ip and ip.dst != client_ip: continue
        matched += 1
        if is_quic(pkt): quic_count += 1
        if TCP in pkt: tcp_count += 1
        if ip.src == client_ip: out_count += 1
        else: in_count += 1

    print(f"\n{'='*45}")
    print(f"  PCAP Stats: {os.path.basename(pcap_path)}")
    print(f"{'='*45}")
    print(f"  Total packets     : {total}")
    print(f"  Client packets    : {matched}")
    print(f"  ├─ Outgoing (→)   : {out_count}")
    print(f"  └─ Incoming (←)   : {in_count}")
    print(f"  QUIC/UDP          : {quic_count}")
    print(f"  TCP               : {tcp_count}")
    print(f"{'='*45}\n")


# ══════════════════════════════════════════
# CLI
# ══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Convert PCAP → CSV (DeepMASQUE format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic convert
  python3 pcap2csv.py /tmp/victim_traffic.pcap /tmp/trace.csv --client 192.168.0.120

  # Chỉ lấy traffic đến proxy MASQUE cụ thể
  python3 pcap2csv.py capture.pcap out.csv --client 192.168.0.120 --proxy 162.159.198.1

  # Chỉ lấy QUIC/UDP port 443
  python3 pcap2csv.py capture.pcap out.csv --client 192.168.0.120 --quic-only --port 443

  # Xem stats trước khi convert
  python3 pcap2csv.py capture.pcap --client 192.168.0.120 --stats
        """
    )

    parser.add_argument('pcap',        help='Input PCAP file')
    parser.add_argument('output',      nargs='?', default='/tmp/trace.csv',
                        help='Output CSV file (default: /tmp/trace.csv)')
    parser.add_argument('--client',    default=None,
                        help='Client/victim IP. Nếu không truyền → auto-detect từ pcap')
    parser.add_argument('--gateway',   default=None,
                        help='Gateway IP để loại khỏi auto-detect (e.g. 192.168.0.1)')
    parser.add_argument('--exclude',   nargs='+', default=[],
                        help='Danh sách IP loại khỏi auto-detect (e.g. IP của attacker)')
    parser.add_argument('--proxy',     default=None,
                        help='Proxy server IP (optional filter)')
    parser.add_argument('--port',      type=int, default=None,
                        help='Filter by port number (e.g. 443)')
    parser.add_argument('--quic-only', action='store_true',
                        help='Only include QUIC/UDP packets')
    parser.add_argument('--max',       type=int, default=None,
                        help='Max number of packets to process')
    parser.add_argument('--stats',     action='store_true',
                        help='Print stats only, do not convert')
    parser.add_argument('--quiet',     action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    if args.stats:
        client = args.client or auto_detect_client(args.pcap, args.gateway, args.exclude)
        if not client:
            print("[!] Could not detect client IP. Use --client to specify manually.")
            sys.exit(1)
        print_stats(args.pcap, client)
        return

    # Resolve client IP
    client_ip = args.client
    if not client_ip:
        print("[*] --client not specified, auto-detecting from pcap...")
        client_ip = auto_detect_client(args.pcap, args.gateway, args.exclude)
        if not client_ip:
            print("[!] Auto-detect failed. Specify --client <ip> manually.")
            sys.exit(1)

    n = pcap_to_csv(
        pcap_path=args.pcap,
        output_path=args.output,
        client_ip=client_ip,
        proxy_ip=args.proxy,
        filter_port=args.port,
        quic_only=args.quic_only,
        max_packets=args.max,
        verbose=not args.quiet,
    )

    print(f"[✓] Done. {n} packets → {args.output}")


if __name__ == '__main__':
    main()
