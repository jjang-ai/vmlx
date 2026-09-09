/** Inspection belongs to one input; a stale result must never launch a folder. */
export function canLaunchInspectedImageFolder(input: {
  input: string
  inspectedInput: string
  inspecting: boolean
  success: boolean
  detected: boolean
  explicitClass: string
  explicitTask: string
}): boolean {
  return !!input.input.trim() && input.input.trim() === input.inspectedInput
    && !input.inspecting && input.success
    && (input.detected || (!!input.explicitClass && ['generate', 'edit'].includes(input.explicitTask)))
}
