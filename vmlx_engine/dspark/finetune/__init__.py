"""Fine-tune a bundled DSpark drafter on the bundle's OWN target (self-distillation).

A DSpark drafter ships trained against its source checkpoint. A quantized bundle agrees with that source's top-1 at
most positions, not all, so drafts aimed at the source are rejected more often. This package retrains the drafter to
predict the bundle's own next tokens from the bundle's own tap features:

1. ``datagen``  -- run the target (the bundle) over prompts and record, per position, exactly what the drafter
   consumes: the residual stream after each of the drafter's ``target_layer_ids`` (its dtype, bf16) and the token
   ids; plus the label the drafter must predict (the target's greedy token) and which positions the target generated.
   Two modes: ``generate`` (the target's own greedy or sampled continuation of a prompt, then one teacher-forced pass
   for the taps) and ``teacher`` (an existing conversation teacher-forced in one pass; labels = the target's argmax at
   every assistant position). Sharded to disk, bounded memory, resumable (``records``).
2. ``train``  -- MLX training of the drafter from the shipped weights: the block-drafting objective of the runtime
   forward (``objective``): blocks ``[anchor, mask ...]`` at many anchors of a sequence in one pass, each block row
   attending its sliding window of context taps plus its own block, per-row cross-entropy of the drafter's logits
   (target head + Markov bias with teacher-forced previous tokens) against the target's tokens, a BCE term for the
   confidence head. fp32 master weights, bf16 compute, MXFP8 fake quantization (straight-through) of the linears the
   release stores as MXFP8, checkpointing, held-out evaluation.
3. ``evaluate``  -- offline acceptance: (a) teacher-forced accepted-prefix lengths per anchor with round simulation
   at fixed widths, from the training forward; (b) a replay of the serving rounds through the runtime's own
   ``DSparkDrafter`` (context cache, sliding window, Markov candidates) with Adaptive widths (the runtime's
   ``ARFallback`` width policy) or fixed widths, against held-out generated data.
4. ``export``  -- write the result as a ``dspark/`` directory in the release format (same file name, tensor names,
   dtypes: MXFP8 e4m3 + e8m0 ``weight_scale_inv`` for the decoder linears, BF16 elsewhere; config.json byte-identical),
   loadable by ``load_drafter`` / ``load_dspark_runtime`` and vmlx-swift without code changes.

``python -m vmlx_engine.dspark.finetune {gen,train,eval,export,synth}``; see ``__main__``.
"""

FORMAT = "dspark-ft-v1"
