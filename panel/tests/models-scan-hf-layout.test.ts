import { describe, expect, it } from 'vitest'
import { hubRepoIdFromSnapshotPath } from '../src/shared/hfHubLayout'

// mlxstudio#113: scanner used to display HF hub snapshot sha as the model
// name (e.g. `snapshots/891dc849cfba…`). Display id must instead reverse
// HF's `models--<org>--<name>/snapshots/<sha>/` layout back to `<org>/<name>`.

describe('hubRepoIdFromSnapshotPath', () => {
  it('resolves standard HF hub layout', () => {
    expect(
      hubRepoIdFromSnapshotPath(
        '/Users/example/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/abc123',
      ),
    ).toBe('mlx-community/Qwen3-4B-4bit')
  })

  it('resolves nested org names with hyphens', () => {
    expect(
      hubRepoIdFromSnapshotPath(
        '/x/models--Qwen--Qwen3-30B-A3B-Instruct/snapshots/deadbeef',
      ),
    ).toBe('Qwen/Qwen3-30B-A3B-Instruct')
  })

  it('returns null for non-hub paths', () => {
    expect(
      hubRepoIdFromSnapshotPath('/Users/example/.mlxstudio/models/MyModel'),
    ).toBeNull()
    expect(
      hubRepoIdFromSnapshotPath('/Users/example/models/org/repo'),
    ).toBeNull()
  })

  it('returns null when snapshots segment is missing', () => {
    expect(
      hubRepoIdFromSnapshotPath(
        '/x/models--mlx-community--Qwen3-4B-4bit/refs/abc',
      ),
    ).toBeNull()
  })

  it('returns null when models-- prefix is missing', () => {
    expect(
      hubRepoIdFromSnapshotPath('/x/mlx-community--Qwen3/snapshots/abc'),
    ).toBeNull()
  })

  it('returns null when the folder lacks an org separator', () => {
    expect(
      hubRepoIdFromSnapshotPath('/x/models--only-one-segment/snapshots/abc'),
    ).toBeNull()
  })

  it('returns null when sha is empty', () => {
    expect(
      hubRepoIdFromSnapshotPath('/x/models--org--repo/snapshots/'),
    ).toBeNull()
  })
})
