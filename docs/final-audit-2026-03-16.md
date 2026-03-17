# Final Consolidated Audit — 2026-03-16

## ALREADY FIXED (Rounds 1+2)

1. server.py: `import base64` added
2. server.py: `from pathlib import Path` in edit endpoint
3. server.py: `images = []` moved before try block
4. server.py: Top-level exception handler added
5. server.py: `load_edit_model()` now receives model_path + quantize
6. server.py: Rate limit added to /v1/images/edits
7. server.py: Dimension validation (64-4096) added to edits
8. server.py: Middleware message updated for edits endpoint
9. server.py: `_image_quantize` declared as global
10. cli.py: EDIT_MODELS added to MFLUX_NAMED_MODELS
11. cli.py: `--image-mode` flag added, dispatches to load_edit_model
12. cli.py: `--served-model-name` used for model name
13. cli.py: Quantize detection checks both model name and path dir
14. cli.py: `server._image_quantize` stored globally
15. image_gen.py: load_edit_model quantize auto-detection from config.json
16. image_gen.py: z-image vs z-image-turbo detection improved
17. image.ts: startServer passes imageMode, imageQuantize, servedModelName
18. image.ts: getRunningServer reads imageMode from config (no regex)
19. image.ts: Weight file verification checks .safetensors in transformer/
20. image.ts: getRunningServers (plural) handler added
21. sessions.ts: buildArgs passes --image-mode, --image-quantize, --served-model-name
22. sessions.ts: findAvailablePort fixed (reverted to all ports for UNIQUE constraint)
23. models.ts: checkImageModelLocal verifies weight files
24. models.ts: checkImageModelInHFCache verifies weight files
25. models.ts: snapshot_download uses local_dir_use_symlinks=False
26. models.ts: HF search includes image-to-image (not just text-to-image)
27. ImageTab.tsx: Uses explicit category from model picker (no regex)
28. ImageModelPicker.tsx: Passes category (edit/gen) to onSelect
29. SessionView.tsx: Reads config.imageMode (no regex)
30. ApiDashboard.tsx: Reads config.imageMode (no regex)
31. EndpointList.tsx: Accepts isEdit, filters endpoints
32. SessionConfigForm.tsx: imageMode/imageQuantize fields added
33. ServerSettingsDrawer.tsx: Passes imageMode to form
34. CreateSession.tsx: Image Mode + Quantize dropdowns added
35. App.tsx: isImageSession filter excludes image from chat
36. ChatModeToolbar.tsx: isImageSession filter excludes image from picker
37. SessionsContext.tsx: config field added to SessionSummary
38. env.d.ts: Updated types for startServer, getRunningServer, getRunningServers
39. preload/index.ts: Updated bridge for all new IPC calls
40. server.ts: imageMode, imageQuantize added to ServerConfig
41. Zombie test code cleaned up (image-system.test.ts, session-port-ui.test.ts)
42. GGUF detection added to convert command
43. pyproject.toml: Alpha → Beta classifier
44. package.json: Author standardized
45. README: Image editing docs, screenshot, CLI options added

## REMAINING ISSUES TO FIX (prioritized)

### HIGH
- [ ] H1: image.ts saveFile has no source path validation (readFile restricts to ~/.mlxstudio but saveFile doesn't)
- [ ] H2: Custom model path always gets 'generate' category — needs imageMode selector for custom models
- [ ] H3: PerformancePanel hardcodes "JANG" prefix for all quantization display
- [ ] H4: CodeSnippets image edit/gen curl omit `model` field
- [ ] H5: DownloadStatusBar: download errors silently clear bar with no error shown
- [ ] H6: getSizeEstimate missing all edit models (returns wrong ~12GB for 37GB qwen)
- [ ] H7: image.ts getModelStatus always returns downloaded:false (dead handler)

### MEDIUM
- [ ] M1: Edit size parsing silently defaults to 1024x1024 (gen endpoint raises 400)
- [ ] M2: EmbeddingsPanel hardcoded model list, no custom model input
- [ ] M3: Temperature 0.0 unreachable via server settings slider
- [ ] M4: DownloadStatusBar: filesProgress and downloaded/total never rendered
- [ ] M5: DownloadTab: "No MLX models found" shows in Image mode too
- [ ] M6: Session/server mode mismatch when clicking edit session with gen server running

### LOW/COSMETIC
- [ ] L1: Error state shows as "Stopped" in ChatModeToolbar dropdown
- [ ] L2: flux2-klein-edit in getDefaultSteps but not in NAMED_MODELS
- [ ] L3: ImageHistory: no session rename, long paths display poorly
- [ ] L4: Scroll dead zone between 100-200px from bottom
- [ ] L5: Cancel endpoints lack auth badge in EndpointList
- [ ] L6: convert.py suggests 'convert-gguf-to-hf' which may not exist
- [ ] L7: Redundant require('os')/require('path') in readFile handler
- [ ] L8: _embedding_lock lazy init (theoretical race, practically impossible)
