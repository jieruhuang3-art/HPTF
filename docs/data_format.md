# Data Format

The migrated preprocessing and training code uses TSV files with the following columns:

```text
label	text_a	delta_ts	pkt_len	packet_start
```

- `label`: integer class label.
- `text_a`: space-separated traffic tokens.
- `delta_ts`: inter-arrival time sequence aligned with `text_a`.
- `pkt_len`: packet-length sequence aligned with `text_a`.
- `packet_start`: packet-boundary marker sequence, where `1` denotes the first token of a packet.

Additional source-tracking columns such as `source_file`, `flow_id`, and `window_start` may be emitted by the migrated PCAP-to-flow converter and are ignored by the classifier.

Evidence source: `/root/ET-BERT-main/data_process/pcap_to_flow_tsv.py` and `/root/ET-BERT-main/fine-tuning/run_classifier.py`.
