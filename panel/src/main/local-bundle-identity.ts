import { realpathSync } from 'node:fs'

/** Session creation must never substitute a different same-basename bundle.
 * Chat history's display-name aliases are not filesystem/load identities.
 */
export function sameLocalBundlePath(left: string, right: string): boolean {
  if (!left || !right) return false
  if (left === right) return true
  try {
    return realpathSync(left) === realpathSync(right)
  } catch {
    return false
  }
}
