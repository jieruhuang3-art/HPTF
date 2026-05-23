# Data Format

HPTF uses packet-aware TSV samples for training and evaluation.

```text
label	text_a	delta_ts	pkt_len	packet_start
```

| Field | Description |
|---|---|
| `label` | Traffic class label. |
| `text_a` | Space-separated traffic tokens. |
| `delta_ts` | Inter-arrival time sequence aligned with the token sequence. |
| `pkt_len` | Packet length sequence aligned with the token sequence. |
| `packet_start` | Packet boundary indicator sequence. |
