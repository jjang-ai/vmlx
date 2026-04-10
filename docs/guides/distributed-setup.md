# Distributed Inference — Setup Guide

> **Status: work in progress.** Pipeline parallelism works for text LLMs but
> this feature is still under active development. Expect rough edges around
> worker crash recovery, protocol version handshake, and the sleep/wake
> lifecycle. Not recommended for production workloads. Tensor parallelism
> is stubbed for a future release.

Distributed inference lets you run a large language model across multiple
Macs connected via Thunderbolt 5, Ethernet, or WiFi. Each Mac holds a
contiguous slice of transformer layers and passes hidden states to the next
Mac in the pipeline. The "coordinator" Mac tokenizes input, runs the first
layer slice, and drives the generation loop; each "worker" runs its own
layer slice and returns hidden states over the network.

## When to use it

- You want to run a model that's too large to fit in any single Mac's RAM
- You have 2+ Apple Silicon Macs on the same network
- You can tolerate experimental software and have a way to debug issues

## When NOT to use it

- You're shipping to end users (feature is experimental)
- Your workload is latency-sensitive — pipeline parallelism has inherent
  per-token bubbles; expect 60-70% of single-node throughput
- You're using features that aren't distributed-aware yet: Flash MoE, JIT
  compilation, Smelt mode, speculative decoding, embedding/rerank/audio
  endpoints, VLM models, continuous batching (multi-request), L2/block
  disk cache, image generation

## Hardware + network

| Link | Bandwidth | Latency | Verdict |
|---|---|---|---|
| Thunderbolt 5 cable | ~120 Gbps | ~0.1ms | Best — nearly compute-limited |
| Thunderbolt 4 cable | ~40 Gbps | ~0.1ms | Excellent |
| 10 GbE | ~10 Gbps | ~1ms | Excellent |
| 1 GbE | ~1 Gbps | ~1ms | Fine for pipeline parallelism |
| WiFi 6E (5GHz) | ~1 Gbps | ~5-20ms | Works, latency hurts |
| Tailscale (WAN) | ~100 Mbps | ~20-50ms | Works for testing, slow |

Any network that can ping between Macs will function. Pipeline parallelism
is bandwidth-tolerant because only hidden states cross the network, not
weights.

## Prerequisites

- Two or more Apple Silicon Macs (M1 or newer)
- Same version of vMLX installed on every Mac (coordinator **and** workers).
  Version mismatch has undefined behavior — there is no protocol version
  handshake yet.
- A shared cluster secret — pick a random string. Example:
  `python3 -c "import secrets; print(secrets.token_urlsafe(24))"`
- The same model path available on every Mac, OR hosted on a shared mount.
  Workers load their layer slice independently.

## Setup

### 1. Install vMLX on every Mac

```bash
pip install vmlx
```

Or use the Electron app (vMLX.app). Workers can run as pip-installed
command-line processes; you don't need the GUI on workers.

### 2. Start workers on every non-coordinator Mac

On each worker Mac, open Terminal and run:

```bash
vmlx-worker --port 9100 --secret YOUR_CLUSTER_SECRET
```

The worker will:
1. Advertise itself via Bonjour / mDNS on the local network
2. Listen on TCP port 9100 for coordinator connections
3. Accept a layer-range assignment from the coordinator
4. Load only its assigned slice of the model weights

Optional worker flags:
- `--bind 127.0.0.1` — bind to localhost only (default binds to all
  interfaces; lock down for production)
- `--name my-mac-studio` — set a friendly hostname for the UI
- `--log-level DEBUG` — verbose logging for troubleshooting

**The worker process must stay running** — if you close Terminal, the
worker dies. Use `nohup`, `screen`, `tmux`, or a `launchd` plist for
persistent workers.

### 3. Start the coordinator

On the coordinator Mac (the one your client will talk to), start vMLX
normally with the `--distributed` flag:

```bash
vmlx serve \
  --model mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit \
  --distributed \
  --distributed-mode pipeline \
  --cluster-secret YOUR_CLUSTER_SECRET \
  --port 8000
```

Or from the Electron UI:
1. Create a new session
2. Open **Distributed Compute** in the session config
3. Toggle **Enable Distributed Inference** on
4. Enter the same cluster secret you gave the workers
5. Start the session

### 4. Verify mesh discovery

The coordinator will:
1. Probe the local network for advertised workers (Bonjour)
2. Add any manually-specified nodes (`--worker-nodes 192.168.1.50:9100,...`)
3. Elect itself as coordinator based on capability score
4. Assign layer ranges to each worker based on RAM
5. Ship model weights / layer assignments to each worker
6. Begin serving inference requests

Watch the coordinator log for lines like:

```
Distributed mesh ready: 3 nodes
DistributedEngine created (deferred start — 3 nodes)
Distributed inference active: 3 nodes, skipping local engine
```

Or query the REST API:

```bash
curl http://localhost:8000/v1/cluster/status
curl http://localhost:8000/v1/cluster/nodes
curl http://localhost:8000/health
```

## Manual node discovery (when Bonjour fails)

Bonjour / mDNS is blocked on many corporate networks. If the coordinator
can't see your workers:

**From the Electron UI:** open the Distributed Compute section and click
"Add Manual" in the node list. Enter the worker's IP and port.

**From the CLI:** pass `--worker-nodes` explicitly:

```bash
vmlx serve \
  --model <model-path> \
  --distributed \
  --cluster-secret YOUR_SECRET \
  --worker-nodes 192.168.1.50:9100,192.168.1.51:9100
```

**From the REST API:**

```bash
curl -X POST http://localhost:8000/v1/cluster/nodes \
  -H 'Content-Type: application/json' \
  -d '{"address":"192.168.1.50","port":9100}'
```

## Troubleshooting

### "Distributed mesh setup deferred to event loop" but no workers appear
Check that workers are running (`ps aux | grep vmlx-worker`) and that the
cluster secret matches on both sides. Bonjour broadcasts may be blocked by
your router or firewall — fall back to `--worker-nodes`.

### Workers connect but layer assignment looks weird
The layer-assign algorithm is RAM-proportional. A 128GB Mac Studio will get
more layers than a 16GB MacBook Air. Check each worker's reported RAM in
`/v1/cluster/nodes` — if a Mac reports wrong RAM, investigate `mesh_node.py`.

### Generation hangs forever
Most common cause: a worker died mid-request. There is no crash detection
yet (Phase 2). Restart the coordinator to reset the mesh.

### "Version mismatch" errors
All Macs must run the same vMLX version. Version handshake is Phase 2.

### Generation works but output is garbage
- Check that all Macs are the same Apple Silicon generation (mixing M1 +
  M4 should work but is untested)
- Check that the same compute dtype is used on every worker
- Check that the model was loaded identically on every worker (same
  quantization, same JANG config)

### `--distributed` + `--enable-jit` errors out
This is expected. JIT (mx.compile) traces local layer objects; the
coordinator doesn't own the layers that distributed workers hold, so any
compiled graph would be wrong for workers. Run distributed without JIT.

### Same error for `--distributed` + `--smelt` / `--speculative-model`
Also expected. These combinations are guarded off in Phase 1 until
validated in Phase 2.

## What doesn't work yet (known limitations)

- **Continuous batching across the mesh** — each request currently serializes
  through the pipeline. Multi-request parallelism is Phase 2.
- **Prefix cache across the mesh** — coordinator-only cache; workers don't
  share cached KV blocks.
- **L2 / block disk cache** — not distributed-aware.
- **Speculative decoding** — draft model would have to be co-located with
  target, negating the speedup.
- **VLM models** — vision encoder and language tower split is untested.
- **Embeddings / rerank / audio endpoints** — run on the coordinator only;
  workers have no embedding model.
- **Image generation (mflux)** — automatically single-node even with
  `--distributed` set.
- **Tensor parallelism** — the `--distributed-mode tensor` option exists but
  is a stub; pipeline is the only working mode today.
- **Worker crash recovery** — the coordinator does not detect worker
  disappearance mid-request.
- **Heterogeneous RAM with MoE models** — the layer assigner is RAM-aware
  but the MoE-aware layer grouping has not been stress-tested across
  very asymmetric meshes.

## Security notes

- The cluster secret is passed in plaintext over the initial handshake
  (HMAC per-message is Phase 2). Do **not** run distributed over untrusted
  WiFi without a VPN or Tailscale.
- Workers bind to all interfaces by default. Use `--bind 127.0.0.1` for
  localhost-only testing, or put the workers behind a firewall.
- The coordinator's OpenAI API is protected by your normal `--api-key` flag;
  the cluster secret only authenticates workers to the coordinator, not
  API clients to the coordinator.

## Next steps

- If distributed works for your workload, file a success report with your
  mesh topology (Mac models, link type, model size, tokens/sec) so we can
  add it to the compatibility matrix.
- If it doesn't work, please file an issue at
  https://github.com/jjang-ai/vmlx/issues with logs from both coordinator
  and at least one worker.
