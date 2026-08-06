# inference-sim

A deterministic simulator for LLM inference placement, memory protocols, and
resource scheduling.

Inference Sim combines exact protocol checks with calibrated performance
estimates. It can verify state transitions and resource accounting, but does
not claim hardware-accurate latency without calibration data.

![Inference Sim workbench](docs/images/inference-sim-workbench.jpg)

## Capabilities

- Model dense, MoE, multimodal, and image-generation architectures using
  built-in presets or local ONNX model packages.
- Configure CPU, GPU, NPU, unified-memory, multi-device, and multi-node
  topologies with explicit compute, memory, storage, and link constraints.
- Simulate continuous batching, chunked prefill, speculative decoding,
  parallelism strategies, expert caching, model co-residency, and node faults.
- Estimate latency, TTFT, ITL, throughput, memory pressure, FLOPs, bandwidth,
  utilization, and context capacity.
- Explore long-context workloads up to 1M input tokens and 32K output tokens.
- Export deterministic results for replay and verification.

The browser workbench is static. Imported models stay local and are processed
in Web Workers; no inference server or model upload is required.

## Quick Start

Requires Node.js 22 and pnpm 9.15.

```bash
pnpm install --frozen-lockfile
pnpm build
pnpm test
pnpm dev:web
```

## CLI

```bash
# List presets and inspect a scenario
pnpm sim presets
pnpm sim scenario multi-gpu-ring-4

# Analyze an ONNX model or static scenario
pnpm sim onnx-inspect /path/to/model.onnx /path/to/manifest.json
pnpm sim static examples/mixtral-dgx-h100.yaml

# Run serving and failure simulations
pnpm sim serving multi-gpu examples/serving.yaml
pnpm sim speculative examples/speculative-mtp.yaml
pnpm sim node-failover multi-node single-gpu-cpu examples/node-failover.yaml
```

Run `pnpm sim --help` for all commands. Commands accept built-in scenarios or
validated YAML/JSON configurations. Performance commands can also use
calibration data; without it, results use the bundled heuristic cost model.

## Formal Verification

The TLA+ specifications in [`specs/tla`](specs/tla) check pressure handling,
collective ordering, buffer ownership, KV admission, node failure, and model
co-residency invariants.

```bash
TLA2TOOLS_JAR=/path/to/tla2tools.jar ./specs/tla/check.sh
```

## Documentation

- [Design and execution contracts](docs/DESIGN.md)
- [Computer presets](docs/COMPUTER_PRESETS.md)
- [Hardware compute registry](docs/HARDWARE_COMPUTE_REGISTRY.md)
- [ONNX Runtime GenAI capture format](docs/ONNX_GENAI_CAPTURE.md)
- [Representative model validation](docs/REPRESENTATIVE_MODEL_VALIDATION.md)

## License

[MIT](LICENSE)
