# Release automation

vMLX uses three manual GitHub Actions workflows. None of them runs on `push`,
and public promotion requires protected GitHub environments.

| Workflow | Purpose | Public mutation |
| --- | --- | --- |
| `Dev DMG build` | Build unsigned Tahoe, Sequoia, or both DMGs for review | None |
| `Signed release candidate` | Build both production DMGs, Developer ID sign, notarize, staple, verify, and upload an immutable candidate | None |
| `Publish exact release candidate` | Promote the exact candidate bytes to GitHub, PyPI, updater manifests, JANGQ, and the website | Yes, after environment approval |

Development builds are diagnostics, not release evidence. The signed candidate
continues to use `panel/scripts/build-release-dmgs.sh`,
`panel/scripts/notarize-release-dmgs.sh`, and the existing private
`r20_production` evidence gate. Automation does not weaken or replace those
owners.

## One-time GitHub setup

### Release runner

Register the Apple Silicon signing/build Mac as a self-hosted runner with these
labels:

```text
self-hosted, macOS, ARM64, vmlx-release
```

The runner needs Node 22.12 or newer at `/opt/homebrew/bin/node`, matching
`npm` and `npx`, a current full Xcode selected with `xcode-select`, and `uv`
on `PATH` with managed CPython 3.13 available. Candidate setup uses `uv venv`
so `sys.executable` remains bound to the isolated checkout's `.venv`, rather
than Homebrew's framework interpreter. It also needs enough disk for both app
bundles and the private evidence root used by the production gate. Keep the
private evidence root outside the Git
checkout. The candidate workflow requires its absolute path as an explicit
dispatch input; no runner username or machine-specific path is stored in the
public workflow.

The runner does not need a persistent release keychain. The candidate workflow
creates a per-run keychain, imports the certificate and App Store Connect API
key, then removes both in an `always()` cleanup step.

On a dedicated runner that already has a validated notarytool profile in its
login Keychain, set the `apple-release-signing` environment variable
`APPLE_NOTARY_KEYCHAIN_PROFILE` to that profile name. The workflow still
creates and removes an ephemeral Developer ID signing keychain, but validates
and uses the existing notary profile without exporting its private key.

If that dedicated runner also owns the exact Developer ID Application identity,
set `APPLE_USE_EXISTING_SIGNING_IDENTITY=1`. The workflow requires exactly one
matching ShieldStack team identity and leaves it in the login Keychain; it does
not export the private key or use the PKCS#12 secrets in that mode.

### Protected environments

Create all four environments with Eric as a required reviewer. While Eric is
the only reviewer, allow self-review so a run he manually dispatches can reach
the explicit approval gate. Enable GitHub's prevent-self-review option only
after adding a second independent release reviewer; otherwise every release is
permanently deadlocked.

#### `apple-release-signing`

Secrets:

- `APPLE_DEVELOPER_ID_P12_BASE64`
- `APPLE_DEVELOPER_ID_P12_PASSWORD`
- `APPLE_NOTARY_KEY_P8_BASE64`
- `APPLE_NOTARY_KEY_ID`
- `APPLE_NOTARY_ISSUER_ID`

The certificate must contain `Developer ID Application: ShieldStack LLC
(55KGF2S5AY)`. The App Store Connect key needs only the access required by
Apple's notarization service.

#### `production-release`

Create a GitHub App installed only on `jjang-ai/vmlx`,
`jjang-ai/mlxstudio`, and `jjang-ai/jangq`. Grant:

- Contents: read/write on vMLX and MLX Studio.
- Actions: read/write on JANGQ, for the optional exact-tag publisher dispatch.
- Metadata: read.

Store the App ID as environment variable `RELEASE_APP_ID` and its private key
as secret `RELEASE_APP_PRIVATE_KEY`. This avoids a broad personal access token.

For emergency continuity while the App is being provisioned, the workflow also
accepts `RELEASE_PAT` as a protected-environment secret. It must be a
fine-grained credential limited to the three named repositories and removed as
soon as the GitHub App is available. The publication preflight proves access to
all required repositories before any public release is created.

#### `pypi`

Do not add a PyPI API token. Configure a PyPI Trusted Publisher for project
`vmlx` with:

```text
Owner: jjang-ai
Repository: vmlx
Workflow: publish-release.yml
Environment: pypi
```

After the publisher is registered and reviewed, set the `pypi` environment
variable `TRUSTED_PUBLISHER_CONFIGURED=true`. The workflow refuses to create a
GitHub release until this explicit setup receipt is present. PyPI remains the
authority during publication; no API credential is stored by the workflow.

The publish job has only `id-token: write` plus read permissions and uses the
official PyPA publisher action pinned to an immutable commit.

JANGQ's `publish-pypi.yml` remains the owner of the `jang` package. When the
vMLX promotion input `publish_jangq` is enabled, the selected JANGQ tag must
already resolve to the exact JANG commit embedded in the candidate. The vMLX
workflow dispatches that owner and waits for it.

#### `website-release`

Secrets:

- `WEBSITE_SSH_PRIVATE_KEY`
- `CLOUDFLARE_API_TOKEN` (optional when all Cloudflare values are omitted)

Variables:

- `WEBSITE_SSH_HOST`
- `WEBSITE_SSH_USER`
- `WEBSITE_SSH_HOST_KEY` (a pinned `known_hosts` line, never a runtime scan)
- `CLOUDFLARE_MLX_STUDIO_ZONE_ID`
- `CLOUDFLARE_VMLX_NET_ZONE_ID`

The SSH principal needs write permission to `/var/www/mlx.studio/update`,
`/var/www/mlx.studio/download`, and `/var/backups/vmlx-releases`. Deployment is
atomic and takes an outside-web-root backup before changing the two pages.
The preflight proves that access before any GitHub or PyPI mutation. Configure
all three Cloudflare values to enable an explicit purge, or leave all three
unset; cache-busting, browser-compatible live verification is mandatory either
way.

## Operating sequence

### 1. Development artifact

Run `Dev DMG build` with a vMLX ref, JANGQ ref, and desired flavor. The result
is explicitly marked `unsigned_development_build`, retained for 14 days, and
is never notarized or published.

### 2. Signed candidate

Before dispatch:

1. Bump and commit every vMLX version stamp and release changelog.
2. Push the exact vMLX release commit to `jjang-ai/vmlx` `main`.
3. Push the exact JANG runtime commit to `jjang-ai/jangq` `main`.
4. Ensure the private evidence gate is current for that exact source and
   executable provenance.

Run `Signed release candidate` with full 40-character source and JANG SHAs.
The workflow refuses anything other than the live public `origin/main` heads.
It builds both flavors in one production invocation, notarizes the exact bytes,
and uploads:

- Tahoe and Sequoia DMGs plus blockmaps;
- Python wheel and source distribution;
- `release-info.json` with public source/JANG/artifact provenance;
- `release-notes.md` containing every non-merge commit since the previous tag.

Private manifests, proof logs, local paths, Apple submissions, and credentials
are not uploaded.

The private prepackage reader supports versioned acceptance scopes. Schema v3
retains every v2 Qwen acceptance owner and adds Flash-Next 4S, GLM DFlash2,
MiniCPM 35B DSpark, and MiniCPM 1B without a drafter. DSpark defaults, cache,
API, non-regression, GLM integration, and native-EOS classification each require
separate source-bound receipts with visual, streaming API, and runtime-identity
artifacts. Missing evidence remains a build failure; schema validation does not
replace review of the artifacts or the later installed-build checks.

Schema v4 is the explicitly scoped DeepSeek-V4.1 checkpoint: JANG affine and
JANGHT have separate source-bound visual, API, runtime and turn records. It
distinguishes retained JANGHT API-result captures from current affine raw SSE,
and keeps the accepted history-conditioned image-color error visible as an
output-quality limitation. Neither limitation becomes a passing universal
model-quality or fresh all-protocol claim. Owning source hashes, changed-path
coverage and artifact closure remain required; both-flavor packaging,
signing/notarization, installed smoke and exact-byte promotion remain pending.
Pass its private envelope through the candidate workflow’s optional
`scoped_prepackage_receipt` input. The same production build driver validates
it; omitting the input retains the original archived-build evidence path.

### 3. Exact-byte promotion

Run `Publish exact release candidate` with the successful candidate run ID and
the same version/source/JANG inputs. The typed confirmation must be exact:

```text
PUBLISH vX.Y.Z EXACT CANDIDATE
```

After protected approval, the workflow:

1. rehashes both DMGs and blockmaps against `release-info.json`;
2. rechecks both public `main` SHAs;
3. publishes the same files and release notes to `jjang-ai/vmlx` and
   `jjang-ai/mlxstudio`;
4. publishes the prebuilt `vmlx` distributions through PyPI Trusted
   Publishing;
5. writes one Tahoe-first updater manifest and commits byte-identical copies to
   both repositories;
6. optionally dispatches and waits for JANGQ's exact tagged publisher;
7. atomically updates the website, purges both Cloudflare zones, and verifies
   GitHub, `mlx.studio`, and `vmlx.net` public surfaces.

Publication across GitHub, PyPI, Git, and the website cannot be globally
transactional. A later-stage failure must be treated as a partial public
release and resumed at the failed owning surface; never rebuild different DMG
bytes under the same version.

`pypi_recovery=true` is the exact-byte continuation path after GitHub release
publication. It verifies the public wheel and source archive when they already
exist (and publishes only when absent), then resumes the byte-identical updater
manifests and idempotent website deployment. A same-version mismatch on PyPI or
the website fails closed. Re-running the website owner against an already exact
version succeeds without creating another backup.

## Proof boundary

GitHub Actions proves source identity, packaging, signing, notarization,
artifact identity, and public-surface agreement. It does not replace installed
Electron proof on the runtime Mac. User-visible model, cache, settings,
sleep/wake, tool, reasoning, and API claims remain blocked until the exact
candidate has the required paired Electron and raw-API receipts.
