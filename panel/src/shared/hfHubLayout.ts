/**
 * Reverse HuggingFace hub cache layout to its repo id.
 * Hub layout: `<base>/models--<org>--<name>/snapshots/<sha>/`.
 * Returns `"<org>/<name>"` when `currentPath` is the snapshot leaf, else null.
 * Matches `huggingface_hub`'s forward mapping (`/` → `--`); see
 * `model-config-registry.ts:resolveHuggingFaceRepoToLocalPath` for the inverse
 * direction the engine already relied on.
 * mlxstudio#113: scanner previously displayed snapshot sha as the model name.
 */
export function hubRepoIdFromSnapshotPath(currentPath: string): string | null {
  const parts = currentPath.split("/");
  if (parts.length < 3) return null;
  const sha = parts[parts.length - 1];
  const snapshots = parts[parts.length - 2];
  const folder = parts[parts.length - 3];
  if (snapshots !== "snapshots") return null;
  if (!folder.startsWith("models--")) return null;
  if (!sha || sha === "snapshots") return null;
  const repoId = folder.slice("models--".length).replace(/--/g, "/");
  if (!repoId.includes("/")) return null;
  return repoId;
}
