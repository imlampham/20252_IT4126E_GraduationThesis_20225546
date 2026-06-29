import sys, os, argparse

try:
    import pyshark
except ImportError:
    print("[!] pyshark not found. Install: pip install pyshark --break-system-packages")
    sys.exit(1)


def pcap_to_csv(pcap_path, output_path, client_ip, verbose=True):
    if not os.path.exists(pcap_path):
        print(f"[!] File not found: {pcap_path}")
        sys.exit(1)

    if verbose:
        print(f"[*] Reading: {pcap_path}")
        print(f"[*] Client IP: {client_ip}")

    features = []

    try:
        # Khớp với script gốc: dùng pyshark, keep_packets=False để tiết kiệm RAM
        with pyshark.FileCapture(pcap_path, keep_packets=False) as cap:
            for packet in cap:
                try:
                    if not hasattr(packet, 'ip'):
                        continue
                    if not hasattr(packet, 'transport_layer') or packet.transport_layer is None:
                        continue

                    t_layer = packet.transport_layer
                    src_ip  = str(packet.ip.src_host)
                    dst_ip  = str(packet.ip.dst_host)

                    # Filter: chỉ lấy packet liên quan đến client
                    if src_ip != client_ip and dst_ip != client_ip:
                        continue

                    # relative_time = time_delta (IAT) 
                    rel_time = float(getattr(packet.frame_info, 'time_delta', 0))

                    # direction: 1 = outgoing (client→proxy), 0 = incoming
                    direction = "1" if src_ip == client_ip else "0"

                    # protocol: 1 = QUIC/UDP, 0 = other 
                    protocol = "1" if t_layer in ("QUIC", "UDP") else "0"

                    # length = packet.length (toàn bộ frame)
                    length = str(packet.length)

                    src_port = str(packet[t_layer].srcport)
                    dst_port = str(packet[t_layer].dstport)

                    features.append({
                        "protocol":      protocol,
                        "length":        length,
                        "relative_time": f"{rel_time:.9f}",
                        "direction":     direction,
                        "src_ip":        src_ip,
                        "src_port":      src_port,
                        "dst_ip":        dst_ip,
                        "dst_port":      dst_port,
                    })

                except Exception:
                    continue

    except Exception as e:
        print(f"[!] Error reading pcap: {e}")
        sys.exit(1)
    finally:
        os.system("pkill -9 -f tshark 2>/dev/null")

    if verbose:
        print(f"[*] Packets matched: {len(features)}")

    if len(features) == 0:
        print(f"[!] No packets matched client_ip={client_ip}")
        sys.exit(1)

    # Reset packet đầu tiên về 0 
    features[0]['relative_time'] = '0.000000000'

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(";".join(features[0].keys()) + "\n")
        for feat in features:
            f.write(";".join(feat.values()) + "\n")

    if verbose:
        print(f"[+] Saved: {output_path}  ({len(features)} rows)")

    return len(features)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PCAP → CSV (DeepMASQUE format, khớp script gốc)"
    )
    parser.add_argument('pcap',   help='Input PCAP file')
    parser.add_argument('output', nargs='?', default='/tmp/trace.csv',
                        help='Output CSV (default: /tmp/trace.csv)')
    parser.add_argument('--client', required=True,
                        help='Victim/client IP address')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    n = pcap_to_csv(args.pcap, args.output, args.client, verbose=not args.quiet)
    print(f"[✓] Done. {n} packets → {args.output}")


if __name__ == '__main__':
    main()