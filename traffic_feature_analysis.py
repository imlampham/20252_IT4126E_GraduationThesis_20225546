import subprocess, os, sys, random, glob, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter
from pathlib import Path


ROOT_PATH   = r"D:\Documents\ĐATN\pcap_41_45"
PROXY_IP    = "auto"           # "auto" = tự detect | hoặc "162.159.198.1"
SAMPLE_N    = 20               # số file lấy mẫu mỗi folder
RANDOM_SEED = 42
OUTPUT_DIR  = r"D:\Documents\ĐATN\quic_output\final5"

SAMPLE_MODE = "random"         # "random" | "first" | "last"
TSHARK_PATH = r"D:\Wireshark\tshark.exe"             

# ╚══════════════════════════════════════════════════════════════╝

BG    = "#0D1117"
PANEL = "#161B22"
BORDER= "#30363D"
TEXT  = "#C9D1D9"
MUTED = "#8B949E"

PHASE_COLORS = {
    "Keep-alive":    "#4A90D9",
    "Handshake":     "#E67E22",
    "Data Transfer": "#27AE60",
    "Post-Data":     "#8E44AD",
}
PHASE_LIST = list(PHASE_COLORS.keys())
C_UP   = "#2ECC71"
C_DOWN = "#E74C3C"



def find_tshark():
    if TSHARK_PATH and Path(TSHARK_PATH).exists():
        return TSHARK_PATH
    found = shutil.which("tshark")
    if found:
        return found
    for p in [r"C:\Program Files\Wireshark\tshark.exe",
              r"C:\Program Files (x86)\Wireshark\tshark.exe"]:
        if Path(p).exists():
            return p
    return None

TSHARK_BIN = find_tshark()


def log(msg, indent=0):
    print("  " * indent + msg)


def run_tshark(args, timeout=30):
    if not TSHARK_BIN:
        raise RuntimeError("tshark not found. Install Wireshark or set TSHARK_PATH.")
    cmd = [TSHARK_BIN] + args
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return r.stdout
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        raise RuntimeError(f"tshark not found at: {TSHARK_BIN}")


def detect_proxy_ip(pcap_path):
    stdout = run_tshark([
        "-r", str(pcap_path), "-Y", "udp.port == 443",
        "-T", "fields", "-e", "ip.src", "-e", "ip.dst", "-E", "separator=,",
    ], timeout=15)
    if not stdout:
        return None

    ip_counter = Counter()
    SKIP = ("224.", "239.", "255.", "127.", "0.", "::1", "ff0")
    PRIV = ("192.168.", "10.", "172.")

    for line in stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 2:
            continue
        src, dst = parts[0].strip(), parts[1].strip()
        if any(src.startswith(p) for p in SKIP) or any(dst.startswith(p) for p in SKIP):
            continue
        ip_counter[src] += 1
        ip_counter[dst] += 1

    candidates = [(ip, c) for ip, c in ip_counter.most_common(20)
                  if not any(ip.startswith(p) for p in PRIV)]
    if candidates:
        return candidates[0][0]
    for ip, _ in ip_counter.most_common():
        if not any(ip.startswith(p) for p in ("127.", "0.")):
            return ip
    return None


def detect_proxy_ip_for_folder(file_list, n_probe=3):
    votes = Counter()
    for f in file_list[:n_probe]:
        ip = detect_proxy_ip(f)
        if ip:
            votes[ip] += 1
            log(f"probe {Path(f).name} → {ip}", 2)
    return votes.most_common(1)[0][0] if votes else None


def scan_folders(root):
    root = Path(root)
    folders = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        pcaps = sorted(glob.glob(str(d / "*.pcap")) + glob.glob(str(d / "*.pcapng")))
        if pcaps:
            folders[d.name] = pcaps
    return folders


def sample_files(file_list, n, mode, seed):
    if len(file_list) <= n:
        return file_list
    if mode == "random":
        return sorted(random.Random(seed).sample(file_list, n))
    return file_list[:n] if mode == "first" else file_list[-n:]


def extract_packets(pcap_path, proxy_ip):
    stdout = run_tshark([
        "-r", str(pcap_path),
        "-Y", f"udp and ip.addr == {proxy_ip}",
        "-T", "fields",
        "-e", "frame.number", "-e", "frame.time_relative",
        "-e", "ip.src", "-e", "ip.dst", "-e", "frame.len",
        "-E", "separator=,",
    ])
    if not stdout:
        return []
    packets = []
    for line in stdout.strip().split("\n"):
        p = line.split(",")
        if len(p) < 5:
            continue
        try:
            packets.append({
                "no":        int(p[0]),
                "time":      float(p[1]),
                "src":       p[2].strip(),
                "size":      int(p[4]),
                "direction": +1 if p[2].strip() != proxy_ip else -1,
            })
        except:
            continue
    return packets


def detect_phases(packets):
    phases = []
    n = len(packets)
    if n == 0:
        return phases

    hs_start = None
    for i, p in enumerate(packets):
        # Initial packet được pad ≥1200B theo RFC 9000 
        if p["direction"] == +1 and p["size"] >= 1200:
            hs_start = i
            break
    # Fallback nếu không tìm được ≥1200B
    if hs_start is None:
        for i, p in enumerate(packets):
            if p["size"] >= 145:
                hs_start = i
                break

    # Data Transfer start: 3 packets liên tiếp <300B (Short Header transition)
    data_start = None
    if hs_start is not None:
        small_run = 0
        for i in range(hs_start, n):
            if packets[i]["size"] < 300:
                small_run += 1
                if small_run >= 3:
                    data_start = i - small_run + 1
                    break
            else:
                small_run = 0
        if data_start is None:
            data_start = hs_start + 1

    # Post-Data start
    # gap >200ms sau data_start (client processing time)
    # HOẶC packet lớn đầu tiên (adaptive threshold)
    burst_start = None
    if data_start is not None:
        all_sizes = [p["size"] for p in packets]
        adaptive_thresh = max(400, int(np.percentile(all_sizes, 60)))

        for i in range(data_start + 1, n):
            dt_ms = (packets[i]["time"] - packets[i-1]["time"]) * 1000
            if dt_ms > 200:
                burst_start = i
                break

        if burst_start is None:
            for i in range(data_start, n):
                if packets[i]["size"] >= adaptive_thresh:
                    burst_start = i
                    break

    for i in range(n):
        if hs_start is None or i < hs_start:
            phases.append("Keep-alive")
        elif data_start is None or i < data_start:
            phases.append("Handshake")
        elif burst_start is None or i < burst_start:
            phases.append("Data Transfer")
        else:
            phases.append("Post-Data")
    return phases


def compute_ipt(packets):
    ipt = [0.0]
    for i in range(1, len(packets)):
        ipt.append((packets[i]["time"] - packets[i-1]["time"]) * 1000)
    return ipt


def aggregate_stats(all_packets_list):
    up, dn, ipts = [], [], []
    phase_sizes  = defaultdict(list)
    phase_ipt    = defaultdict(list)
    phase_dir    = defaultdict(list)   # direction counts per phase per file
    for packets in all_packets_list:
        if not packets:
            continue
        phases = detect_phases(packets)
        ipt    = compute_ipt(packets)
        for p, ph, it in zip(packets, phases, ipt):
            phase_sizes[ph].append(p["size"])
            phase_ipt[ph].append(it)
            phase_dir[ph].append(p["direction"])
            (up if p["direction"] == +1 else dn).append(p["size"])
        ipts.extend(ipt[1:])
    return {
        "up_sizes":   up,
        "down_sizes": dn,
        "ipts":       ipts,
        "phase_sizes": dict(phase_sizes),
        "phase_ipt":   dict(phase_ipt),
        "phase_dir":   dict(phase_dir),
    }


def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, axis="y", color="#21262D", lw=0.5, ls="--", alpha=0.7)


def draw_boxplot(ax, data_groups, labels, colors, title, ylabel,
                 showfliers=False, vert=True, clip_pct=99, log_scale=False):
    """
    Vẽ boxplot cho nhiều nhóm dữ liệu với tùy chọn:
    clip_pct: clip tại percentile này để zoom vào phần box chính
    log_scale: dùng log scale cho trục giá trị
    showmeans: hiện diamond vàng = mean
    """
    valid = [(d, l, c) for d, l, c in zip(data_groups, labels, colors)
             if len(d) >= 2]
    if not valid:
        ax.set_title(title, fontsize=9)
        style_ax(ax)
        return

    data_v, labels_v, colors_v = zip(*valid)

    # Clip extreme outliers để zoom box
    all_vals = [x for d in data_v for x in d if x != 0]
    if all_vals and clip_pct < 100:
        clip_val = np.percentile(all_vals, clip_pct)
        data_plot = [np.clip(d, None, clip_val) for d in data_v]
    else:
        data_plot = list(data_v)

    bp = ax.boxplot(
        data_plot,
        patch_artist=True,
        vert=vert,
        notch=False,
        showfliers=showfliers,
        flierprops=dict(marker=".", markersize=2, alpha=0.2,
                        markerfacecolor=MUTED, markeredgecolor="none"),
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(color=MUTED, linewidth=1.2, linestyle="--"),
        capprops=dict(color=MUTED, linewidth=1.5),
        boxprops=dict(linewidth=1.2),
        meanprops=dict(marker="D", markersize=4,
                       markerfacecolor="#FFD700", markeredgecolor="#FFD700"),
        showmeans=True,
    )
    for patch, color in zip(bp["boxes"], colors_v):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    if vert:
        ax.set_xticklabels(labels_v, fontsize=7, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=8)
        if log_scale and all_vals and min(x for x in all_vals if x > 0) > 0:
            ax.set_yscale("log")
    else:
        ax.set_yticklabels(labels_v, fontsize=7)
        ax.set_xlabel(ylabel, fontsize=8)

    ax.set_title(title, fontsize=9)
    style_ax(ax)

    # Annotate median AFTER style_ax — fixed top row, uniform height
    ylim   = ax.get_ylim()
    yrange = ylim[1] - ylim[0]
    n_boxes = len(data_v)
    for i in range(n_boxes):
        med = np.median(data_v[i])
        pos = i + 1
        if vert:
            # All labels at same y = top 2% of axes → clean uniform row
            y_label = ylim[1] - yrange * 0.01
            ax.text(pos, y_label,
                    f"md={med:.0f}",
                    ha="center", va="top",
                    fontsize=5.5, color="#FFD700", alpha=0.95,
                    bbox=dict(boxstyle="round,pad=0.1", fc=PANEL,
                              ec="none", alpha=0.65))


# PER-FILE CHART
def plot_single(packets, phases, ipt, title, out_path):
    n      = len(packets)
    sizes  = [p["size"] for p in packets]
    dirs   = [p["direction"] for p in packets]
    times  = [p["time"] - packets[0]["time"] for p in packets]

    ph_order   = [ph for ph in PHASE_LIST if ph in set(phases)]
    ph_colors  = [PHASE_COLORS[ph] for ph in ph_order]

    # Collect data per phase
    ph_sizes_up   = [[sizes[i] for i in range(n)
                      if phases[i] == ph and dirs[i] == +1] for ph in ph_order]
    ph_sizes_down = [[sizes[i] for i in range(n)
                      if phases[i] == ph and dirs[i] == -1] for ph in ph_order]
    ph_ipt        = [[ipt[i] for i in range(1, n)
                      if phases[i] == ph] for ph in ph_order]

    # Signed size per phase (all directions)
    ph_signed = [[dirs[i] * sizes[i] for i in range(n)
                  if phases[i] == ph] for ph in ph_order]

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(title, fontsize=10, color="white", fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.5, wspace=0.38,
                           left=0.07, right=0.97, top=0.94, bottom=0.08)

    # Packet size ↑ per phase
    ax1 = fig.add_subplot(gs[0, 0])
    draw_boxplot(ax1, ph_sizes_up, ph_order, ph_colors,
                 "① Packet Size ↑ C→S per Phase", "Size (bytes)")

    # Packet size ↓ per phase
    ax2 = fig.add_subplot(gs[0, 1])
    draw_boxplot(ax2, ph_sizes_down, ph_order, ph_colors,
                 "② Packet Size ↓ S→C per Phase", "Size (bytes)")

    # Signed size per phase
    ax3 = fig.add_subplot(gs[0, 2])
    draw_boxplot(ax3, ph_signed, ph_order, ph_colors,
                 "③ Signed Size (±) per Phase", "±bytes")
    ax3.axhline(0, color=MUTED, lw=0.8, ls="--")

    # IPT per phase
    ax4 = fig.add_subplot(gs[1, 0])
    ph_ipt_clipped = [[min(x, 500) for x in d] for d in ph_ipt]
    draw_boxplot(ax4, ph_ipt_clipped, ph_order, ph_colors,
                 "④ IPT per Phase (clip 500ms)", "IPT (ms)")

    # Direction ratio per phase (stacked bar)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(PANEL)
    ph_up_pct  = []
    ph_dn_pct  = []
    for ph in ph_order:
        d = [dirs[i] for i in range(n) if phases[i] == ph]
        tot = len(d) or 1
        ph_up_pct.append(sum(1 for x in d if x == +1) / tot * 100)
        ph_dn_pct.append(sum(1 for x in d if x == -1) / tot * 100)
    x = np.arange(len(ph_order))
    ax5.bar(x, ph_up_pct, color=C_UP,   alpha=0.8, label="C→S %")
    ax5.bar(x, ph_dn_pct, bottom=ph_up_pct, color=C_DOWN, alpha=0.8, label="S→C %")
    ax5.set_xticks(x)
    ax5.set_xticklabels(ph_order, fontsize=7, rotation=15, ha="right")
    ax5.set_ylabel("Direction %", fontsize=8)
    ax5.set_title("⑤ Direction Ratio per Phase", fontsize=9)
    ax5.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor="white")
    style_ax(ax5)
    ax5.grid(True, axis="y", color="#21262D", lw=0.5, ls="--", alpha=0.7)

    # Overall size boxplot: ↑ vs ↓
    ax6 = fig.add_subplot(gs[1, 2])
    up_all = [sizes[i] for i in range(n) if dirs[i] == +1]
    dn_all = [sizes[i] for i in range(n) if dirs[i] == -1]
    draw_boxplot(ax6,
                 [up_all, dn_all],
                 ["↑ C→S", "↓ S→C"],
                 [C_UP, C_DOWN],
                 "⑥ Overall Size: C→S vs S→C", "Size (bytes)")

    # Footer
    up_c  = sum(1 for d in dirs if d == +1)
    dn_c  = sum(1 for d in dirs if d == -1)
    ipt_r = ipt[1:]
    txt = (f"Total: {n} pkts  |  ↑{up_c}  ↓{dn_c}  |  "
           f"IPT mean={np.mean(ipt_r):.1f}ms  max={max(ipt_r):.0f}ms"
           if ipt_r else f"Total: {n}")
    fig.text(0.5, 0.005, txt, ha="center", fontsize=7,
             color=MUTED, fontfamily="monospace")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()


# FOLDER SUMMARY
def plot_folder_summary(folder_name, stats, n_files, proxy_ip, out_path):
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle(
        f"Folder Summary: {folder_name}  ({n_files} files)  |  Proxy: {proxy_ip}",
        fontsize=12, color="white", fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.5, wspace=0.38,
                           left=0.07, right=0.97, top=0.93, bottom=0.08)

    up     = stats["up_sizes"]
    dn     = stats["down_sizes"]
    ipts   = stats["ipts"]
    psizes = stats["phase_sizes"]
    pipt   = stats["phase_ipt"]

    ph_order  = [ph for ph in PHASE_LIST if ph in psizes and len(psizes[ph]) >= 2]
    ph_colors = [PHASE_COLORS[ph] for ph in ph_order]

    # A. Size ↑ per phase
    ax = fig.add_subplot(gs[0, 0])
    draw_boxplot(ax,
                 [psizes.get(ph, []) for ph in ph_order],
                 ph_order, ph_colors,
                 "A. Packet Size ↑ C→S per Phase", "Size (bytes)")

    # B. Size overall: ↑ vs ↓
    ax = fig.add_subplot(gs[0, 1])
    draw_boxplot(ax,
                 [up, dn],
                 ["↑ C→S", "↓ S→C"],
                 [C_UP, C_DOWN],
                 "B. Overall Size: C→S vs S→C", "Size (bytes)")

    # C. Size ↓ per phase
    ax = fig.add_subplot(gs[0, 2])
    ph_dn = {ph: [s for s, d in zip(
                    stats["phase_sizes"].get(ph, []),
                    stats["phase_dir"].get(ph, []))
                  if d == -1]
             for ph in ph_order}
    draw_boxplot(ax,
                 [ph_dn.get(ph, []) for ph in ph_order],
                 ph_order, ph_colors,
                 "C. Packet Size ↓ S→C per Phase", "Size (bytes)")

    # D. IPT per phase
    ax = fig.add_subplot(gs[1, 0])
    ph_ipt_data    = [pipt.get(ph, []) for ph in ph_order]
    ph_ipt_clipped = [[min(x, 300) for x in d] for d in ph_ipt_data]
    draw_boxplot(ax,
                 ph_ipt_clipped, ph_order, ph_colors,
                 "D. IPT per Phase (clip 300ms)", "IPT (ms)")

    # E. IPT overall
    ax = fig.add_subplot(gs[1, 1])
    draw_boxplot(ax,
                 [[min(x, 300) for x in ipts]],
                 ["All phases"],
                 ["#58A6FF"],
                 "E. IPT Overall Distribution", "IPT (ms)")

    # F. Stats table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off"); ax.set_facecolor(PANEL)
    def _f(v): return f"{v:.1f}" if v else "—"
    rows = [
        ["Metric",    "↑ C→S",                               "↓ S→C"],
        ["Count",     str(len(up)),                           str(len(dn))],
        ["Total (B)", f"{sum(up):,}",                         f"{sum(dn):,}"],
        ["Mean (B)",  _f(np.mean(up) if up else 0),           _f(np.mean(dn) if dn else 0)],
        ["Median(B)", _f(np.median(up) if up else 0),         _f(np.median(dn) if dn else 0)],
        ["Q3 (B)",    _f(np.percentile(up,75) if up else 0),  _f(np.percentile(dn,75) if dn else 0)],
        ["Max (B)",   str(max(up)) if up else "—",            str(max(dn)) if dn else "—"],
        ["IPT mean",  f"{np.mean(ipts):.2f}ms" if ipts else "—", ""],
        ["IPT median",f"{np.median(ipts):.2f}ms" if ipts else "—", ""],
        ["IPT Q3",    f"{np.percentile(ipts,75):.1f}ms" if ipts else "—", ""],
        ["Proxy IP",  proxy_ip, ""],
        ["Files",     str(n_files), ""],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1C2128" if r % 2 == 0 else PANEL)
        cell.set_text_props(color=TEXT if r > 0 else "#58A6FF",
                            fontweight="bold" if r == 0 else "normal")
        cell.set_edgecolor(BORDER)
    ax.set_title("F. Summary Statistics", fontsize=9, color="white", pad=4)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()


# GLOBAL COMPARISON (boxplot) 
def plot_global_comparison(folder_stats, out_path):
    folders = list(folder_stats.keys())
    n       = len(folders)
    palette = [plt.cm.tab10(i / max(n-1, 1)) for i in range(n)]
    labels  = [f[:18] for f in folders]

    fig = plt.figure(figsize=(20, 12), facecolor=BG)
    fig.suptitle("Global Comparison across Folders — Boxplot",
                 fontsize=13, color="white", fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.5, wspace=0.38,
                           left=0.07, right=0.97, top=0.94, bottom=0.08)

    def get_s(f, key): return folder_stats[f]["stats"][key]

    # A. Packet size ↑ per folder — clip at p95 to zoom into IQR
    ax = fig.add_subplot(gs[0, 0])
    draw_boxplot(ax,
                 [get_s(f, "up_sizes") for f in folders],
                 labels, palette,
                 "A. Packet Size ↑ C→S per Website", "Size (bytes)",
                 clip_pct=95)

    # B. Packet size ↓ per folder — broken axis: 0-500 bottom, 1100-1450 top
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_visible(False)  # hide placeholder

    # Manual broken-axis using two inset axes
    pos = ax_b.get_position()
    fig_b_bot = fig.add_axes([pos.x0, pos.y0,         pos.width, pos.height * 0.40])
    fig_b_top = fig.add_axes([pos.x0, pos.y0 + pos.height * 0.42, pos.width, pos.height * 0.55])

    dn_data = [get_s(f, "down_sizes") for f in folders]

    for ax_sub, ylim_sub, show_xlabel in [
        (fig_b_top, (1050, 1420), False),
        (fig_b_bot, (0,    220),  True),
    ]:
        bp2 = ax_sub.boxplot(
            dn_data, patch_artist=True, notch=False, showfliers=False,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color=MUTED, linewidth=1.2, linestyle="--"),
            capprops=dict(color=MUTED, linewidth=1.5),
            boxprops=dict(linewidth=1.2),
            meanprops=dict(marker="D", markersize=4,
                           markerfacecolor="#FFD700", markeredgecolor="#FFD700"),
            showmeans=True,
        )
        for patch, color in zip(bp2["boxes"], palette):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax_sub.set_ylim(*ylim_sub)
        ax_sub.set_facecolor(PANEL)
        ax_sub.tick_params(colors=MUTED, labelsize=7)
        ax_sub.yaxis.label.set_color(TEXT)
        for sp in ax_sub.spines.values(): sp.set_color(BORDER)
        ax_sub.grid(True, axis="y", color="#21262D", lw=0.5, ls="--", alpha=0.7)
        if show_xlabel:
            ax_sub.set_xticks(range(1, len(labels)+1))
            ax_sub.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
            ax_sub.spines["top"].set_visible(False)
            ax_sub.set_ylabel("Size (bytes)", fontsize=8)
        else:
            ax_sub.set_xticks([])
            ax_sub.spines["bottom"].set_visible(False)
            ax_sub.set_title("B. Packet Size ↓ S→C  (broken axis)", fontsize=9, color="white")
            # median annotation on top panel
            for i, d in enumerate(dn_data):
                med = np.median(d)
                ax_sub.text(i+1, 1420 - 20, f"md={med:.0f}",
                            ha="center", va="top", fontsize=5.5,
                            color="#FFD700", alpha=0.95,
                            bbox=dict(boxstyle="round,pad=0.1", fc=PANEL, ec="none", alpha=0.6))
        # break marks
        d_val, d_size = 0.015, 0.6
        kwargs = dict(transform=ax_sub.transAxes, color=MUTED, clip_on=False, lw=1.2)
        if show_xlabel:
            ax_sub.plot((-d_val, +d_val), (1-d_size*0.04, 1+d_size*0.04), **kwargs)
            ax_sub.plot((1-d_val, 1+d_val), (1-d_size*0.04, 1+d_size*0.04), **kwargs)
        else:
            ax_sub.plot((-d_val, +d_val), (-d_size*0.04, +d_size*0.04), **kwargs)
            ax_sub.plot((1-d_val, 1+d_val), (-d_size*0.04, +d_size*0.04), **kwargs)

    # C. Signed size per folder (↑ = positive, ↓ = negative)
    ax = fig.add_subplot(gs[0, 2])
    signed_data = []
    for f in folders:
        s = folder_stats[f]["stats"]
        signed = [+x for x in s["up_sizes"]] + [-x for x in s["down_sizes"]]
        signed_data.append(signed)
    draw_boxplot(ax, signed_data, labels, palette,
                 "C. Signed Size (±) — above 0 = net upload bias", "±bytes",
                 clip_pct=98, showfliers=False)
    ax.axhline(0, color="#FFD700", lw=1.2, ls="--", alpha=0.9, zorder=5)
    # Labels anchored to right edge, clear of all boxes
    ax.text(1.01, 0.53, "▲ upload",
            transform=ax.transAxes,
            fontsize=6, color=C_UP, alpha=0.9, va="bottom", ha="left",
            clip_on=False)
    ax.text(1.01, 0.47, "▼ download",
            transform=ax.transAxes,
            fontsize=6, color=C_DOWN, alpha=0.9, va="top", ha="left",
            clip_on=False)

    # D. IPT per folder
    ax = fig.add_subplot(gs[1, 0])
    # IPT: most values are sub-ms, convert to µs for readability, clip at p99
    ipt_raw  = [get_s(f, "ipts") for f in folders]
    ipt_us   = [[x * 1000 for x in d if x > 0] for d in ipt_raw]  # ms → µs
    draw_boxplot(ax, ipt_us, labels, palette,
                 "D. IPT per Website (µs, clip p99)", "IPT (µs)", clip_pct=99)

    # E. Phase size boxplot — Data Transfer vs Post-Data across folders
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(PANEL)
    ax.set_title("E. Phase Size Comparison (Data Transfer vs Post-Data)", fontsize=9)
    ax.title.set_color("white")

    x       = np.arange(n)
    width   = 0.35
    dt_med  = []
    pd_med  = []
    dt_q1, dt_q3 = [], []
    pd_q1, pd_q3 = [], []

    for f in folders:
        ps = folder_stats[f]["stats"]["phase_sizes"]
        dt = ps.get("Data Transfer", [0])
        pd = ps.get("Post-Data",     [0])
        dt_med.append(np.median(dt)); dt_q1.append(np.percentile(dt, 25)); dt_q3.append(np.percentile(dt, 75))
        pd_med.append(np.median(pd)); pd_q1.append(np.percentile(pd, 25)); pd_q3.append(np.percentile(pd, 75))

    dt_err = [np.array(dt_med) - np.array(dt_q1), np.array(dt_q3) - np.array(dt_med)]
    pd_err = [np.array(pd_med) - np.array(pd_q1), np.array(pd_q3) - np.array(pd_med)]

    C_DT = PHASE_COLORS["Data Transfer"]   # #27AE60 green
    C_PD = PHASE_COLORS["Post-Data"]       # #8E44AD purple

    ax.bar(x - width/2, dt_med, width, yerr=dt_err, color=C_DT,
           alpha=0.85, capsize=4, label="Data Transfer",
           error_kw=dict(ecolor="white", lw=1, capsize=4))
    ax.bar(x + width/2, pd_med, width, yerr=pd_err, color=C_PD,
           alpha=0.85, capsize=4, label="Post-Data",
           error_kw=dict(ecolor="white", lw=1, capsize=4))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Median Size ± IQR (bytes)", fontsize=8)
    style_ax(ax)
    # Add legend AFTER style_ax to keep correct colours
    ax.legend(
        handles=[
            mpatches.Patch(color=C_DT, alpha=0.85, label="Data Transfer (median ± IQR)"),
            mpatches.Patch(color=C_PD, alpha=0.85, label="Post-Data (median ± IQR)"),
        ],
        fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor="white",
        loc="upper right",
    )

    # F. Summary comparison table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off"); ax.set_facecolor(PANEL)
    header = ["Website", "↑Mean", "↓Mean", "IPT med", "IPT Q3"]
    rows   = [header]
    for f in folders:
        s   = folder_stats[f]["stats"]
        up  = s["up_sizes"];  dn = s["down_sizes"]; it = s["ipts"]
        def fmt_ipt(v):
            if v < 1: return f"{v*1000:.0f}µs"
            return f"{v:.2f}ms"
        rows.append([
            f[:16],
            f"{np.mean(up):.0f}B"  if up else "—",
            f"{np.mean(dn):.0f}B"  if dn else "—",
            fmt_ipt(np.median(it)) if it else "—",
            fmt_ipt(np.percentile(it, 75)) if it else "—",
        ])
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1C2128" if r % 2 == 0 else PANEL)
        cell.set_text_props(color=TEXT if r > 0 else "#58A6FF",
                            fontweight="bold" if r == 0 else "normal")
        cell.set_edgecolor(BORDER)
    ax.set_title("F. Cross-folder Summary", fontsize=9, color="white", pad=4)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    log(f"Global comparison saved: {out_path}", 1)


def main():
    print("=" * 65)
    print("  QUIC Batch Analyzer — Boxplot Edition")
    print(f"  tshark    : {TSHARK_BIN or '*** NOT FOUND ***'}")
    print(f"  Root      : {ROOT_PATH}")
    print(f"  Proxy IP  : {PROXY_IP}")
    print(f"  Sample    : {SAMPLE_N} files/folder ({SAMPLE_MODE})")
    print(f"  Output    : {OUTPUT_DIR}")
    print("=" * 65)

    if not TSHARK_BIN:
        print("\n[ERROR] tshark không tìm thấy!")
        print("Cài Wireshark: https://www.wireshark.org/download.html")
        sys.exit(1)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    folders = scan_folders(ROOT_PATH)
    if not folders:
        log(f"[!] Không tìm thấy folder nào trong: {ROOT_PATH}")
        sys.exit(1)

    log(f"\nFound {len(folders)} folder(s):")
    for name, files in folders.items():
        log(f"  {name}/  →  {len(files)} pcap files", 1)

    all_folder_stats = {}

    for folder_name, all_files in folders.items():
        log(f"\n{'─'*55}")
        log(f"Folder: {folder_name}  ({len(all_files)} files total)")

        # Detect proxy IP
        if PROXY_IP == "auto":
            log("  Detecting proxy IP...", 1)
            proxy_ip = detect_proxy_ip_for_folder(all_files, n_probe=3)
            if proxy_ip:
                log(f"  → Detected: {proxy_ip}", 1)
            else:
                log("  [!] Không detect được proxy IP, skip", 1)
                continue
        else:
            proxy_ip = PROXY_IP
            log(f"  Proxy IP: {proxy_ip}", 1)

        sampled    = sample_files(all_files, SAMPLE_N, SAMPLE_MODE, RANDOM_SEED)
        folder_out = str(Path(OUTPUT_DIR) / folder_name)
        Path(folder_out).mkdir(parents=True, exist_ok=True)

        all_packets_list = []
        ok_count = 0

        for pcap_path in sampled:
            fname = Path(pcap_path).stem
            log(f"  [{ok_count+1:02d}/{len(sampled)}] {Path(pcap_path).name}", 1)

            packets = extract_packets(pcap_path, proxy_ip)
            if not packets:
                log("       → no QUIC packets, skip", 2)
                continue

            phases = detect_phases(packets)
            ipt    = compute_ipt(packets)
            all_packets_list.append(packets)
            ok_count += 1

            out_png = str(Path(folder_out) / f"{fname}.png")
            try:
                plot_single(packets, phases, ipt,
                            f"{folder_name} / {Path(pcap_path).name} [proxy: {proxy_ip}]",
                            out_png)
                log(f"       → {len(packets)} pkts  ✓", 2)
            except Exception as e:
                log(f"       → plot error: {e}", 2)

        if all_packets_list:
            stats = aggregate_stats(all_packets_list)
            all_folder_stats[folder_name] = {"stats": stats, "proxy_ip": proxy_ip}
            summary_png = str(Path(OUTPUT_DIR) / f"_summary_{folder_name}.png")
            try:
                plot_folder_summary(folder_name, stats, ok_count, proxy_ip, summary_png)
                log(f"  Summary → {summary_png}", 1)
            except Exception as e:
                log(f"  Summary error: {e}", 1)
        else:
            log("  No valid files in this folder.", 1)

    if len(all_folder_stats) > 1:
        log(f"\n{'═'*55}")
        log("Global cross-folder comparison...")
        plot_global_comparison(
            all_folder_stats,
            str(Path(OUTPUT_DIR) / "_GLOBAL_comparison.png")
        )

    log(f"\n[✓] Done. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()