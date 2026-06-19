export const metalWiredLimitCommand = 'sudo sysctl iogpu.wired_limit_mb=120000'

export const metalWiredLimitHelpText =
  `If a large model hits Metal OOM, SIGKILL, or kernel-panic risk, macOS may need a tuned Metal wired-memory limit. Example for large-memory Macs: ${metalWiredLimitCommand}. Do not set it equal to physical RAM; leave OS/WindowServer/app headroom. On 128GB Macs, 115000-120000 MB is usually safer than 128000 MB. The command requires an admin password and resets after reboot.`

const METAL_WIRED_LIMIT_RE =
  /(?:Command buffer execution failed|Insufficient Memory|kIOGPUCommandBufferCallbackErrorOutOfMemory|Metal OOM|kernel-panic risk|SIGKILL|likely out of memory|out of memory)/i

export function appendMetalWiredLimitGuidance(message: string): string {
  if (!METAL_WIRED_LIMIT_RE.test(message)) return message
  if (message.includes(metalWiredLimitCommand)) return message
  return `${message}\n\nMetal wired-memory limit help: ${metalWiredLimitHelpText}`
}
