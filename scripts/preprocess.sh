#!/usr/bin/env bash
set -e

python -m hptf.data.pcap_to_flow \
  --pcap_root data/raw \
  --out_dir data/processed \
  --window_packets 5 \
  --stride_packets 2
