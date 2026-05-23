# Internal Maintenance Notes

This file records repository maintenance notes for developers. It is not required for the main paper reproduction workflow.

## Current Public Entry Points

- `README.md`: public project overview.
- `docs/model_overview.md`: HPTF method summary.
- `docs/reproduction.md`: reproduction guide.
- `docs/ablation.md`: ablation-study description.
- `docs/code_mapping.md`: concise mapping from paper modules to implementation components.

## Implementation Notes

- The HPTF model implementation is centered on `FlowSemMAEClassifier`.
- Model-facing helper files under `hptf/models/` provide lightweight modular interfaces.
- Training, pre-training, and evaluation entry points are under `hptf/training/` and `hptf/evaluation/`.

## Release Notes

- Public documentation is written from the HPTF method perspective.
- Detailed engineering migration history is intentionally omitted from the README.
