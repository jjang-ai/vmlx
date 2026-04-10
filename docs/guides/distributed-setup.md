# Distributed Inference — Setup Guide

> **⚠ Pre-alpha — localhost loopback testing only.**
>
> This feature is under active development. The pipeline-parallel text
> inference path produces correct output for single-node and for
> localhost coordinator + localhost worker combos, but several security
> and resilience gaps make it **unsafe to expose on any network you
> don't fully control**:
>
> - Cluster secret is sent plaintext over TCP (no TLS, no per-message HMAC)
> - Worker crash detection and recovery are not implemented
> - Coordinator-loss re-election recovery is a stub
> - Protocol has no version handshake
> - Tensor parallelism is stubbed for a future release
>
> The guide below walks through the **recommended** usage: running a
> coordinator and one worker on the same Mac for correctness smoke
> testing. Multi-Mac deployment is covered in a "Not yet supported"
> section at the bottom so you know what's coming.

## What pipeline parallelism does

A text model has N transformer layers. Pipeline parallelism splits the
layer list into contiguous ranges, puts each range on a different
process (the "coordinator" holds the embedding, some layers, and the
LM head; each "worker" holds a middle slice of layers), and passes
hidden-state tensors through the pipeline via TCP per token.

For **localhost smoke testing** this is still useful because it
exercises the entire mesh setup, discovery, authentication, layer
assignment, KV cache lifecycle, and per-token protocol round-trip
code paths — just without the network hop.

## Recommended: single-Mac loopback smoke test

### 1. Install vMLX

```bash
pip install vmlx
```

The Electron app bundles the same `vmlx-worker` binary under
`vMLX.app/Contents/Resources/bundled-python`, but for testing it's
easier to use a pip-installed version and a Terminal window.

### 2. Generate a cluster secret

Do **not** pass the secret on the command line — it will land in
`ps aux` where any logged-in user on the Mac can read it. Use an
environment variable:

```bash
export VMLX_CLUSTER_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
echo $VMLX_CLUSTER_SECRET  # copy this — you'll need it for the coordinator
```

Keep this terminal open for the worker. In a separate terminal for
the coordinator, `export` the same value.

### 3. Start the worker bound to localhost

```bash
vmlx-worker --port 9100 --bind 127.0.0.1
```

What this does:

- Binds only to `127.0.0.1` (the default — pass `--allow-public`
  to bind `0.0.0.0` and reach the network, not recommended right now)
- Refuses to start if `$VMLX_CLUSTER_SECRET` is empty
- Uses constant-time secret comparison (`hmac.compare_digest`)
- Validates any `model_path` the coordinator tries to load against
  an allowlist (`~/mlx`, `~/.cache/huggingface`, `~/.cache/vmlx`,
  and the current working directory by default; override with
  `--allowed-model-root PATH`)

You should see log output like:

```
============================================================
vMLX Worker (EXPERIMENTAL — localhost testing recommended)
============================================================
  Host: mac-studio.local
  Chip: Apple M2 Ultra
  RAM:  192 GB (150 GB available)
  Bind: 127.0.0.1:9100
  Bonjour: ON
  Auth: cluster secret set (33 bytes)
============================================================
Worker listening on 127.0.0.1:9100 (binary protocol)
HTTP identity endpoint on 127.0.0.1:9101
Worker started, waiting for coordinator to assign layers...
```

### 4. Start the coordinator in a second terminal

```bash
export VMLX_CLUSTER_SECRET=<same-value-as-worker>
vmlx serve \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/... \
  --distributed \
  --distributed-mode pipeline \
  --worker-nodes 127.0.0.1:9100 \
  --port 8000
```

The `--model` path must be something small enough that splitting
its layers across two processes is worth doing (any text LLM, even
0.5B, works for correctness testing). The path must be absolute
and must exist on disk — workers will validate it against their
allowlist before loading.

Watch the coordinator log for:

```
Distributed mode enabled (pipeline parallelism)
Distributed mesh setup deferred to event loop
Cluster ready: coordinator[L0-15] → worker[L16-31]
Distributed inference active: 2 nodes, skipping local engine
```

### 5. Hit it with a real inference request

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 32
  }' | python3 -m json.tool
```

Then compare against the same model running single-node on the
same Mac. The text output should match (or differ only by
sampling noise if temperature > 0).

### 6. Check the cluster REST endpoints

With your API key set via the `Authorization` header if you configured one:

```bash
curl http://localhost:8000/health | python3 -m json.tool
curl http://localhost:8000/v1/cluster/status | python3 -m json.tool
curl http://localhost:8000/v1/cluster/nodes | python3 -m json.tool
```

The `/health` response should include a `distributed` block showing
the mode, worker count, and pipeline layer assignments.

### 7. Clean shutdown

Ctrl-C the coordinator first, then Ctrl-C the worker. Both should
cleanly close their TCP listeners. Worker logs should end with:

```
Received signal 2, shutting down...
UDP responder cancelled
Worker shut down
```

## Using the Electron UI instead

Same idea, but driven from the vMLX.app session config:

1. Open **Settings → Session Config → Distributed Compute**
2. The prominent amber banner at the top warns you about the pre-alpha state
3. Toggle **Enable Distributed Inference**
4. Leave mode on **Pipeline Parallelism**
5. Enter the same cluster secret you exported in the worker's terminal
6. Start the session

The **Node List** inside the panel will poll `/v1/cluster/nodes` every
5 seconds and show any workers it finds. Use **Add Manual** to enter
`127.0.0.1:9100` if Bonjour discovery is blocked.

## What to look for when testing

- **Multi-token generation matches single-node output** for a fixed seed
  (the "wrong seq_pos" bug that made all outputs after the first token
  garbage is fixed in v1.3.37; verify on your own model)
- **Concurrent requests don't corrupt each other** — fire two requests at
  the same time and confirm each gets sensible output (per-request
  `request_id` now isolates KV cache slots on workers)
- **Deep sleep + wake keeps working** — trigger JIT sleep, make a request
  to wake the coordinator, verify the distributed mesh rebuilds
- **Abort / client disconnect releases worker KV cache** — start a long
  generation, disconnect the client, check that `/v1/cluster/nodes`
  shows the worker's cache dropping the slot (requires future telemetry)

## Not yet supported — don't try these in production

These paths exist in the code but are not hardened enough to rely on:

### Multi-Mac deployment
Running `vmlx-worker` on a second Mac and connecting the coordinator
over a real network. You *can* do it with `--allow-public` on the
worker and `--worker-nodes IP:PORT` on the coordinator, but:

- Cluster secret travels plaintext across the network — anyone on the
  same segment can sniff it
- No TLS, no per-message HMAC, no replay protection
- No protocol version handshake, so a coordinator and worker on
  different vMLX versions can silently misbehave
- If a worker crashes or its network drops mid-generation the
  coordinator will currently hang until its 120-second per-worker
  read timeout fires, then raise an exception

Wait for Phase 2.

### Tensor parallelism
`--distributed-mode tensor` is stubbed — the code compiles and the CLI
flag is accepted, but there's no working tensor-parallel forward pass.
Use `pipeline` only.

### Coordinator loss recovery
If the coordinator dies mid-session, surviving workers will try to
elect a new coordinator, but layer redistribution is a `# TODO`. The
re-elected coordinator has no loaded model. Planned for Phase 2.

### Distributed continuous batching
Multi-request parallelism across the mesh. Today the coordinator
serializes requests with an asyncio lock — correct, but no throughput
benefit over single-node for concurrent load.

### Distributed prefix cache
Each worker caches its own layer slice per-request; coordinator has
no global view. Prefix cache hits work only on the coordinator's
local layer slice.

### Audio, embeddings, rerank, image generation, VLM
Run on the coordinator only. `--distributed` is silently ignored for
image models; audio/embed/rerank routes the request through the
coordinator's local model.

## Troubleshooting

### `Refusing to start: cluster secret is required`
You didn't `export VMLX_CLUSTER_SECRET`. Set the env var, then relaunch.

### `model_path not under any allowed root`
The worker's path allowlist rejected the coordinator's model path.
Either move the model under `~/mlx` / `~/.cache/huggingface` /
`~/.cache/vmlx` / the worker's CWD, or pass
`--allowed-model-root /path/to/models` when starting the worker.

### `Join rejected: invalid cluster secret`
Coordinator and worker have different secrets. Both must read
`$VMLX_CLUSTER_SECRET` from the same value.

### Coordinator log shows `Distributed mesh startup failed: ...`
Check the worker log for the matching error. Common causes: worker
isn't listening yet, worker port conflict, wrong `--worker-nodes`
address.

### Generation hangs forever
A worker died mid-request. The coordinator's per-request read has a
120-second timeout, so you'll get an exception eventually. Crash
detection is Phase 2.

### `/v1/cluster/status` returns 401
Your session has an API key set but the IPC layer isn't sending it.
This was fixed in v1.3.37. If you're still seeing it, rebuild the
Electron app.

### Output is correct for one token but garbage after that
This was the `seq_pos=0` bug. It's fixed in v1.3.37. If you're still
seeing it, verify you're running the post-fix version with:
```
python3 -c "import vmlx_engine.distributed.coordinator as c; import inspect; print('seq_pos' in inspect.getsource(c.Coordinator.forward))"
```
It should print `True`.

## What's in a follow-up release

- TLS option for inter-node traffic
- HMAC + nonce handshake (proper replay protection)
- Worker crash detection via heartbeat timeouts
- Coordinator fail-over with layer redistribution
- Protocol version handshake
- Multi-Mac deployment removed from the "not supported" list
- Tensor parallelism wire-up
- Distributed continuous batching
- Per-worker telemetry in `/v1/cluster/nodes` (tokens/sec, latency, memory)
- SSH-based worker auto-launch helper (`vmlx cluster start --hosts ...`)

If you hit a bug doing localhost smoke testing, please file an issue
with logs from both coordinator and worker at
https://github.com/jjang-ai/vmlx/issues.
