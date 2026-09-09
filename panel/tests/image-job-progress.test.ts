import { beforeEach, describe, expect, it } from 'vitest'
import { beginImageGeneration, bindImageGenerationRequest, finishImageGeneration, getImageGenerationStatus, recordImageGenerationLog, resetImageGenerationStateForTests } from '../src/main/ipc/imageGenerationState'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
const log = (fields: object) => 'INFO:engine:IMAGEJOB '+JSON.stringify({request_id:'owned',job_id:'job1',phase:'denoise_checkpoint',step_index:2,requested_steps:20,...fields})+'\n'
beforeEach(resetImageGenerationStateForTests)
describe('request exact image progress', () => {
  it('ignores other requests and sessions, including stale jobs', () => {
    const c=beginImageGeneration('history')
    bindImageGenerationRequest(c,'server','owned')
    recordImageGenerationLog('other',log({}))
    recordImageGenerationLog('server',log({request_id:'foreign'}))
    expect(getImageGenerationStatus().progress).toEqual({requestId:'owned',phase:'waiting'})
    recordImageGenerationLog('server',log({}))
    expect(getImageGenerationStatus().progress).toEqual({requestId:'owned',jobId:'job1',phase:'denoising',stepIndex:2,totalSteps:20})
    finishImageGeneration(c)
    recordImageGenerationLog('server',log({step_index:18}))
    expect(getImageGenerationStatus().progress).toBeNull()
    bindImageGenerationRequest(beginImageGeneration('history'),'server','new')
    recordImageGenerationLog('server',log({}))
    expect(getImageGenerationStatus().progress?.phase).toBe('waiting')
  })
  it('assembles fragmented lines and clears step state for a new image', () => {
    bindImageGenerationRequest(beginImageGeneration('history'),'server','owned')
    const line=log({})
    recordImageGenerationLog('server',line.slice(0,40))
    expect(getImageGenerationStatus().progress?.phase).toBe('waiting')
    recordImageGenerationLog('server',line.slice(40))
    expect(getImageGenerationStatus().progress?.stepIndex).toBe(2)
    recordImageGenerationLog('server',log({job_id:'job2',phase:'model_call_started'}))
    expect(getImageGenerationStatus().progress?.jobId).toBe('job2')
    expect(getImageGenerationStatus().progress?.stepIndex).toBeUndefined()
    recordImageGenerationLog('server','IMAGECLIENT {"client_job_id":"owned","phase":"saving_outputs"}\n')
    expect(getImageGenerationStatus().progress).toEqual({requestId:'owned',phase:'saving'})
  })
  it('keeps stderr fragments separate from stdout and client records', () => {
    bindImageGenerationRequest(beginImageGeneration('history'),'server','owned')
    const line=log({})
    recordImageGenerationLog('server',line.slice(0,40),'stderr')
    recordImageGenerationLog('server','unrelated stdout\n','stdout')
    recordImageGenerationLog('server','IMAGECLIENT {"client_job_id":"owned","phase":"request_submitted"}\n','client')
    recordImageGenerationLog('server',line.slice(40),'stderr')
    expect(getImageGenerationStatus().progress?.stepIndex).toBe(2)
    const source=readFileSync(join(__dirname,'../src/main/sessions.ts'),'utf8')
    expect(source).toContain("recordImageGenerationLog(sessionId, text, 'stderr')")
    expect(source).toContain("recordImageGenerationLog(sessionId, text, 'stdout')")
    expect(source).not.toContain('recordImageGenerationLog(sessionId, data)')
  })
  it('does not derive progress from malformed, missing, or impossible step data', () => {
    bindImageGenerationRequest(beginImageGeneration('history'),'server','owned')
    recordImageGenerationLog('server','IMAGEJOB {broken}\n')
    recordImageGenerationLog('server',log({job_id:null}))
    expect(getImageGenerationStatus().progress?.phase).toBe('waiting')
    for(const step_index of [-1,20,Infinity,'3']) {
      recordImageGenerationLog('server',log({step_index}))
      expect(getImageGenerationStatus().progress?.stepIndex).toBeUndefined()
    }
    recordImageGenerationLog('server',log({phase:'encoding_png'}))
    expect(getImageGenerationStatus().progress?.phase).toBe('encoding')
    const snapshot=getImageGenerationStatus().progress!
    snapshot.phase='waiting'
    expect(getImageGenerationStatus().progress?.phase).toBe('encoding')
  })
  it('keeps every phase label in every shipped locale', () => {
    for(const locale of ['en','zh','ko','ja','es']) {
      const data=JSON.parse(readFileSync(join(__dirname,'../src/renderer/src/i18n/locales',locale+'.json'),'utf8'))
      for(const key of ['waiting','preparing','denoising','rendering','encoding','saving','cancelling','checkpoint','checkpointHint']) expect(data.image.jobProgress[key]).toBeTruthy()
    }
  })
})
