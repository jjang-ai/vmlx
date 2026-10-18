import { describe, expect, it } from 'vitest'
import { imageEditBody, imagePromptForRequest, isNativeJsonPrompt, parseUpscaleResolution } from '../src/shared/imageNativeInputs'
import { loadImageRuntimeSettings, saveImageRuntimeSettings } from '../src/shared/imageRuntimeSettings'
const edit = { prompt: 'Make it blue', model: 'existing', imageBase64: 'data:image/png;base64,AAAA',
  width: 512, height: 768, steps: 20, guidance: 4, strength: .8, seed: 7, negativePrompt: 'blur', maskBase64: 'data:image/png;base64,BBBB' }
describe('native image request contract', () => {
  it('preserves generic editing parameters', () => {
    expect(imageEditBody(edit)).toEqual({ model: 'existing', prompt: 'Make it blue', image: 'AAAA', n: 1, response_format: 'b64_json',
      size: '512x768', steps: 20, guidance: 4, strength: .8, seed: 7, negative_prompt: 'blur', mask: 'BBBB' })
  })
  it('sends only native SeedVR fields even if stale generic controls remain in saved settings', () => {
    expect(imageEditBody({ ...edit, operation: 'upscale', resolution: '2x', softness: 0 })).toEqual({
      model: 'existing', image: 'AAAA', n: 1, response_format: 'b64_json', seed: 7, resolution: '2x', softness: 0,
    })
    expect(imageEditBody({ ...edit, operation: 'upscale' })).toMatchObject({resolution:384,softness:0})
  })
  it('matches the native scale grammar and rejects bad values instead of clamping', () => {
    for (const x of [2,384,'384','1.5x','2X','0.5x']) expect(parseUpscaleResolution(x)).toBeDefined()
    for (const x of [0,1,-2,2.5,Infinity,'','1.5','.5x','0x','2xjunk','Infinityx']) expect(parseUpscaleResolution(x)).toBeUndefined()
    expect(() => imageEditBody({...edit,operation:'upscale',resolution:'.5x'})).toThrow()
    for (const softness of [-1,1.1,NaN,Infinity]) expect(() => imageEditBody({...edit,operation:'upscale',softness})).toThrow()
  })
  it('preserves FIBO JSON exactly on first and variation requests, without inventing a schema', () => {
    const json = '  {"short_description": "A blue mug"}\n'
    expect(isNativeJsonPrompt(json)).toBe(true)
    expect(isNativeJsonPrompt('A blue mug')).toBe(false)
    expect(isNativeJsonPrompt('[]')).toBe(true) // native json.loads has no imposed object schema
    expect(imagePromptForRequest(json,null,true)).toBe(json)
    expect(imagePromptForRequest(json,'{"short_description":"A red mug"}',true)).toBe(json)
    expect(imagePromptForRequest(' blue ', 'red mug',false)).toBe('red mug, blue')
  })
  it('persists native options per server session and rejects malformed stored values', () => {
    const values = new Map<string,string>()
    const db = {getSetting:(k:string)=>values.get(k),setSetting:(k:string,v:string)=>{values.set(k,v)}}
    const owner = {sessionId:'upscale',modelId:'seedvr2-3b',quantize:0}
    saveImageRuntimeSettings(db,owner,{resolution:'1.5x',softness:.25})
    expect(loadImageRuntimeSettings(db,owner)).toMatchObject({resolution:'1.5x',softness:.25})
    saveImageRuntimeSettings(db,owner,{resolution:'.5x',softness:2})
    expect(loadImageRuntimeSettings(db,owner)).toMatchObject({resolution:384,softness:0})
    expect(loadImageRuntimeSettings(db,{...owner,sessionId:'other'})).toMatchObject({resolution:384,softness:0})
  })
})

import { getImageModel, imageModelForTask } from '../src/shared/imageModels'
describe('native model identity and explicit task selection', () => {
  it.each(['4b','9b'])('preserves Klein %s generation by default and selects its separate edit class explicitly', size => {
    const model = getImageModel(`flux2-klein-${size}`)!
    expect(imageModelForTask(model)).toBe(model)
    expect(model.mfluxClass).toBe('Flux2Klein')
    expect(model.supportedModes).toBeUndefined()
    for (const selected of [imageModelForTask(model,'edit'),imageModelForTask(model,undefined,'Flux2KleinEdit')]) {
      expect(selected).toMatchObject({mfluxName:`flux2-klein-${size}`,mfluxClass:'Flux2KleinEdit',category:'edit',steps:20,guidance:3.5})
    }
  })
  it('never invents a Klein variant for an unknown folder or changes unrelated adapters', () => {
    expect(imageModelForTask(undefined,'edit','Flux2KleinEdit')).toBeUndefined()
    const fibo = getImageModel('fibo')
    expect(imageModelForTask(fibo)).toBe(fibo)
    expect(fibo).toMatchObject({steps:20,guidance:3.5,supportsImg2Img:true})
  })
})
