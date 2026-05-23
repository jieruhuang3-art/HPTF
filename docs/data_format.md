# Data Format

HPTF uses packet-aware TSV samples for training and evaluation.

```text
label	text_a	delta_ts	pkt_len	packet_start
```

- `label`: traffic class label.
- `text_a`: space-separated traffic tokens.
- `delta_ts`: inter-arrival time sequence aligned with the token sequence.
- `pkt_len`: packet length sequence aligned with the token sequence.
- `packet_start`: packet boundary indicator sequence.
