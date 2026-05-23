#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import binascii
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import scapy.all as scapy


@dataclass
class PacketView:
    tokens: list
    delta: float
    length: float


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def packet_protocol(packet):
    if packet.haslayer(scapy.TCP):
        return "tcp"
    if packet.haslayer(scapy.UDP):
        return "udp"
    if packet.haslayer(scapy.ICMP):
        return "icmp"
    if packet.haslayer(scapy.IPv6):
        return "ipv6"
    if packet.haslayer(scapy.IP):
        return "ip"
    return "eth"


def packet_endpoints(packet):
    src, dst = "", ""
    if packet.haslayer(scapy.IP):
        src = getattr(packet[scapy.IP], "src", "")
        dst = getattr(packet[scapy.IP], "dst", "")
    elif packet.haslayer(scapy.IPv6):
        src = getattr(packet[scapy.IPv6], "src", "")
        dst = getattr(packet[scapy.IPv6], "dst", "")
    elif packet.haslayer(scapy.Ether):
        src = getattr(packet[scapy.Ether], "src", "")
        dst = getattr(packet[scapy.Ether], "dst", "")

    sport, dport = 0, 0
    if packet.haslayer(scapy.TCP):
        sport = safe_int(getattr(packet[scapy.TCP], "sport", 0))
        dport = safe_int(getattr(packet[scapy.TCP], "dport", 0))
    elif packet.haslayer(scapy.UDP):
        sport = safe_int(getattr(packet[scapy.UDP], "sport", 0))
        dport = safe_int(getattr(packet[scapy.UDP], "dport", 0))
    return src, dst, sport, dport


def canonical_flow_key(packet):
    protocol = packet_protocol(packet)
    src, dst, sport, dport = packet_endpoints(packet)
    left = (src, sport)
    right = (dst, dport)
    if right < left:
        left, right = right, left
    return protocol, left[0], left[1], right[0], right[1]


def direction_code(packet, origin):
    src, dst, sport, dport = packet_endpoints(packet)
    endpoint = (src, sport, dst, dport)
    if endpoint == origin:
        return "0"
    reverse = (origin[2], origin[3], origin[0], origin[1])
    if endpoint == reverse:
        return "1"
    return "2"


def protocol_code(packet):
    protocol = packet_protocol(packet)
    if protocol == "tcp":
        return "6"
    if protocol == "udp":
        return "17"
    if protocol == "icmp":
        return "1"
    return "0"


def normalized_packet_bytes(packet):
    pkt = packet.copy()
    try:
        if pkt.haslayer(scapy.Ether):
            pkt[scapy.Ether].src = "00:00:00:00:00:00"
            pkt[scapy.Ether].dst = "00:00:00:00:00:00"
        if pkt.haslayer(scapy.IP):
            pkt[scapy.IP].src = "0.0.0.0"
            pkt[scapy.IP].dst = "0.0.0.0"
            if hasattr(pkt[scapy.IP], "chksum"):
                del pkt[scapy.IP].chksum
        if pkt.haslayer(scapy.IPv6):
            pkt[scapy.IPv6].src = "::"
            pkt[scapy.IPv6].dst = "::"
        if pkt.haslayer(scapy.TCP):
            pkt[scapy.TCP].sport = 0
            pkt[scapy.TCP].dport = 0
            if hasattr(pkt[scapy.TCP], "chksum"):
                del pkt[scapy.TCP].chksum
        if pkt.haslayer(scapy.UDP):
            pkt[scapy.UDP].sport = 0
            pkt[scapy.UDP].dport = 0
            if hasattr(pkt[scapy.UDP], "chksum"):
                del pkt[scapy.UDP].chksum
        return bytes(pkt)
    except Exception:
        return bytes(packet)


def packet_bigram_tokens(packet, max_bigrams, normalize_headers=False):
    raw_bytes = normalized_packet_bytes(packet) if normalize_headers else bytes(packet)
    raw_hex = binascii.hexlify(raw_bytes).decode()
    byte_tokens = [raw_hex[i:i + 2] for i in range(0, len(raw_hex), 2) if len(raw_hex[i:i + 2]) == 2]
    bigrams = [byte_tokens[i] + byte_tokens[i + 1] for i in range(max(len(byte_tokens) - 1, 0))]
    return bigrams[:max_bigrams]


def make_packet_view(packet, origin, prev_ts, args):
    ts = safe_float(getattr(packet, "time", 0.0))
    delta = 0.0 if prev_ts is None else max(ts - prev_ts, 0.0)
    length = float(len(bytes(packet)))

    tokens = []
    if args.include_meta_tokens:
        tokens.extend([direction_code(packet, origin), protocol_code(packet)])
    tokens.extend(packet_bigram_tokens(packet, args.max_bigrams_per_packet, args.normalize_headers))

    return PacketView(tokens=tokens, delta=delta, length=length), ts


def sample_from_window(packet_window):
    text_tokens = []
    delta_values = []
    length_values = []
    packet_start = []

    for packet_view in packet_window:
        for token_idx, token in enumerate(packet_view.tokens):
            text_tokens.append(token)
            delta_values.append(packet_view.delta)
            length_values.append(packet_view.length)
            packet_start.append(1 if token_idx == 0 else 0)

    if not text_tokens:
        return None
    return {
        "text_a": " ".join(text_tokens),
        "delta_ts": " ".join(f"{value:.6f}" for value in delta_values),
        "pkt_len": " ".join(f"{value:.1f}" for value in length_values),
        "packet_start": " ".join(str(value) for value in packet_start),
    }


def split_for_flow(label_id, source_file, flow_key, seed, train_ratio, valid_ratio):
    flow_repr = "\t".join(str(part) for part in flow_key)
    key = f"{seed}\t{label_id}\t{source_file}\t{flow_repr}".encode("utf-8")
    score = int.from_bytes(hashlib.sha1(key).digest()[:8], "big") / float(1 << 64)
    if score < train_ratio:
        return "train"
    if score < train_ratio + valid_ratio:
        return "valid"
    return "test"


def split_for_window(label_id, source_file, flow_key, window_start, seed, train_ratio, valid_ratio):
    flow_repr = "\t".join(str(part) for part in flow_key)
    key = f"{seed}\t{label_id}\t{source_file}\t{flow_repr}\t{window_start}".encode("utf-8")
    score = int.from_bytes(hashlib.sha1(key).digest()[:8], "big") / float(1 << 64)
    if score < train_ratio:
        return "train"
    if score < train_ratio + valid_ratio:
        return "valid"
    return "test"


def open_writers(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    handles = {}
    writers = {}
    header = ["label", "text_a", "delta_ts", "pkt_len", "packet_start", "source_file", "flow_id", "window_start"]
    for split in ("train", "valid", "test"):
        path = os.path.join(out_dir, f"{split}_dataset.tsv")
        handle = open(path, "w", newline="", encoding="utf-8")
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        handles[split] = handle
        writers[split] = writer
    return handles, writers


def close_writers(handles):
    for handle in handles.values():
        handle.close()


def collect_pcap_files(root):
    pcap_files = []
    for path, _, files in os.walk(root):
        for filename in files:
            lower = filename.lower()
            if lower.endswith(".pcap") or lower.endswith(".pcapng"):
                pcap_files.append(os.path.join(path, filename))
    return sorted(pcap_files)


def infer_flat_label(path):
    name = os.path.splitext(os.path.basename(path))[0]
    upper = name.upper()
    for prefix in ("FILE-TRANSFER", "BROWSING", "AUDIO", "CHAT", "MAIL", "P2P", "VIDEO", "VOIP"):
        if upper == prefix or upper.startswith(prefix + "_") or upper.startswith(prefix + "-"):
            return prefix

    lower = name.lower()
    if "p2p" in lower or "torrent" in lower or "vuze" in lower:
        return "P2P"
    if "vimeo" in lower or "youtube" in lower or "video" in lower:
        return "VIDEO"
    if "spotify" in lower:
        return "AUDIO"
    if "voip" in lower or "voice" in lower:
        return "VOIP"
    if "skype_audio" in lower or "facebook_audio" in lower or "hangout_audio" in lower:
        return "VOIP"
    if "chat" in lower or "aim" in lower or "icq" in lower or "hangout" in lower:
        return "CHAT"
    if "mail" in lower or "email" in lower or "imap" in lower or "pop" in lower:
        return "MAIL"
    if "file" in lower or "ftp" in lower or "sftp" in lower or "transfer" in lower:
        return "FILE-TRANSFER"
    if "brows" in lower or "facebook" in lower or "google" in lower or "twitter" in lower:
        return "BROWSING"
    return "OTHER"


def collect_labeled_pcap_files(pcap_root):
    label_dirs = [
        name for name in sorted(os.listdir(pcap_root))
        if os.path.isdir(os.path.join(pcap_root, name))
    ]
    if label_dirs:
        return {
            label_name: files
            for label_name in label_dirs
            for files in [collect_pcap_files(os.path.join(pcap_root, label_name))]
            if files
        }

    labeled_files = defaultdict(list)
    for pcap_file in collect_pcap_files(pcap_root):
        labeled_files[infer_flat_label(pcap_file)].append(pcap_file)
    return dict(sorted(labeled_files.items()))


def flush_flow(flow_state, label_id, source_file, flow_key, split, writers, counts, args, final=False):
    buffer = flow_state["buffer"]
    next_start = flow_state["next_start"]
    produced = 0

    while len(buffer) >= args.window_packets:
        window = list(buffer)[: args.window_packets]
        sample = sample_from_window(window)
        if sample is not None:
            row_split = split
            if args.split_by_window:
                row_split = split_for_window(
                    label_id, source_file, flow_key, next_start, args.seed, args.train_ratio, args.valid_ratio
                )
            writers[row_split].writerow([
                label_id,
                sample["text_a"],
                sample["delta_ts"],
                sample["pkt_len"],
                sample["packet_start"],
                source_file,
                hashlib.sha1(repr(flow_key).encode("utf-8")).hexdigest()[:12],
                next_start,
            ])
            counts[row_split] += 1
            produced += 1
        for _ in range(min(args.stride_packets, len(buffer))):
            buffer.popleft()
        flow_state["next_start"] += args.stride_packets
        next_start = flow_state["next_start"]
        if args.max_windows_per_flow > 0 and produced >= args.max_windows_per_flow:
            break

    if final and len(buffer) >= args.min_packets:
        sample = sample_from_window(list(buffer))
        if sample is not None:
            row_split = split
            if args.split_by_window:
                row_split = split_for_window(
                    label_id, source_file, flow_key, next_start, args.seed, args.train_ratio, args.valid_ratio
                )
            writers[row_split].writerow([
                label_id,
                sample["text_a"],
                sample["delta_ts"],
                sample["pkt_len"],
                sample["packet_start"],
                source_file,
                hashlib.sha1(repr(flow_key).encode("utf-8")).hexdigest()[:12],
                next_start,
            ])
            counts[row_split] += 1
            produced += 1
        buffer.clear()

    return produced


def process_pcap(pcap_file, label_id, writers, counts, args):
    try:
        reader = scapy.PcapReader(pcap_file)
    except Exception as exc:
        print(f"warn: failed to open {pcap_file}: {exc}")
        return 0

    flows = {}
    produced = 0

    packets_seen = 0

    try:
        for packet in reader:
            packets_seen += 1
            if args.max_packets_per_file > 0 and packets_seen > args.max_packets_per_file:
                break

            flow_key = canonical_flow_key(packet)
            ts = safe_float(getattr(packet, "time", 0.0))

            flow_state = flows.get(flow_key)
            if flow_state is None:
                src, dst, sport, dport = packet_endpoints(packet)
                origin = (src, sport, dst, dport)
                flow_state = {
                    "origin": origin,
                    "prev_ts": None,
                    "last_ts": ts,
                    "buffer": deque(),
                    "next_start": 0,
                    "split": split_for_flow(label_id, pcap_file, flow_key, args.seed, args.train_ratio, args.valid_ratio),
                }
                flows[flow_key] = flow_state

            if ts - flow_state["last_ts"] > args.flow_timeout and len(flow_state["buffer"]) >= args.min_packets:
                produced += flush_flow(
                    flow_state, label_id, pcap_file, flow_key, flow_state["split"], writers, counts, args, final=True
                )
                flow_state["prev_ts"] = None
                flow_state["next_start"] = 0

            packet_view, flow_state["prev_ts"] = make_packet_view(packet, flow_state["origin"], flow_state["prev_ts"], args)
            flow_state["last_ts"] = ts
            flow_state["buffer"].append(packet_view)

            produced += flush_flow(
                flow_state, label_id, pcap_file, flow_key, flow_state["split"], writers, counts, args, final=False
            )

            stale_keys = [
                key for key, state in flows.items()
                if ts - state["last_ts"] > args.flow_timeout and len(state["buffer"]) < args.min_packets
            ]
            for key in stale_keys:
                del flows[key]
    except Exception as exc:
        print(f"warn: failed while reading {pcap_file}: {exc}")
    finally:
        reader.close()

    for flow_key, flow_state in list(flows.items()):
        produced += flush_flow(
            flow_state, label_id, pcap_file, flow_key, flow_state["split"], writers, counts, args, final=True
        )

    return produced


def main():
    parser = argparse.ArgumentParser(description="Flow-aware PCAP preprocessing for multimodal ET-BERT TSV files.")
    parser.add_argument("--pcap_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--window_packets", type=int, default=5)
    parser.add_argument("--stride_packets", type=int, default=2)
    parser.add_argument("--min_packets", type=int, default=2)
    parser.add_argument("--max_bigrams_per_packet", type=int, default=24)
    parser.add_argument("--flow_timeout", type=float, default=120.0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_label", type=int, default=0)
    parser.add_argument("--max_packets_per_file", type=int, default=0)
    parser.add_argument("--max_windows_per_flow", type=int, default=0)
    parser.add_argument("--split_by_window", action="store_true")
    parser.add_argument("--normalize_headers", action="store_true")
    parser.add_argument("--no_meta_tokens", action="store_true")
    args = parser.parse_args()
    args.include_meta_tokens = not args.no_meta_tokens

    if args.train_ratio <= 0 or args.valid_ratio < 0 or args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("Invalid split ratios. Need train_ratio>0, valid_ratio>=0, and train+valid<1.")

    label_files = collect_labeled_pcap_files(args.pcap_root)
    if not label_files:
        raise ValueError("No pcap files found under pcap_root.")

    label2id = {name: idx for idx, name in enumerate(sorted(label_files))}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    counts = Counter()
    handles, writers = open_writers(args.out_dir)
    try:
        for label_name in sorted(label_files):
            label_id = label2id[label_name]
            files = list(label_files[label_name])
            if args.max_per_label > 0:
                random.Random(args.seed).shuffle(files)
                files = files[: args.max_per_label]

            label_total = 0
            for file_index, pcap_file in enumerate(files, start=1):
                produced = process_pcap(pcap_file, label_id, writers, counts, args)
                label_total += produced
                for handle in handles.values():
                    handle.flush()
                print(
                    f"label={label_name} id={label_id} file={file_index}/{len(files)} "
                    f"path={pcap_file} generated_samples={produced}"
                )
            print(f"label={label_name} id={label_id} files={len(files)} generated_samples={label_total}")
    finally:
        close_writers(handles)

    total = counts["train"] + counts["valid"] + counts["test"]
    if total == 0:
        raise ValueError("No usable samples generated from pcap files.")
    print(f"generated: train={counts['train']}, valid={counts['valid']}, test={counts['test']}")
    print(f"saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
