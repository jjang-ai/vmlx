import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

const source = readFileSync('src/main/database.ts','utf8')
const ast = ts.createSourceFile('database.ts',source,ts.ScriptTarget.Latest,true)
const cls = ast.statements.find(node => ts.isClassDeclaration(node) && node.name?.text === 'DatabaseManager') as ts.ClassDeclaration
const methods = cls.members.filter(node => ts.isMethodDeclaration(node) && ['addImageGeneration','getImageGenerations','getImageGeneration'].includes(node.name.getText(ast)))
const code = ts.transpileModule(`class History { db: any; ensureOpen() {} updateImageSession() {} ${methods.map(node=>node.getText(ast)).join('\n')} } return History`,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText
const History = new Function(code)()

describe('upscale history uses the production database accessors', () => {
  it('binds native options to named nullable columns and reads them on list/single lookup', () => {
    let row: Record<string,unknown> = {}
    const history = new History()
    history.db = {prepare:(sql:string)=>({
      run:(...values:unknown[])=>{
        const columns = sql.match(/INSERT INTO image_generations \(([^)]+)\)/)![1].split(',').map(c=>c.trim())
        expect(columns).toHaveLength(values.length)
        row = Object.fromEntries(columns.map((c,i)=>[c,values[i]]))
      },
      get:()=>row, all:()=>[row],
    })}
    const generation = {id:'u',sessionId:'s',prompt:'',modelName:'seedvr2-3b',width:768,height:512,steps:1,guidance:0,seed:123,
      upscaleResolution:'2x',upscaleSoftness:0,elapsedSeconds:2,imagePath:'/output.png',sourceImagePath:'/source.png',createdAt:7}
    history.addImageGeneration(generation)
    expect(row.upscale_resolution).toBe('2x')
    expect(row.upscale_softness).toBe(0)
    for (const restored of [history.getImageGeneration('u'),history.getImageGenerations('s')[0]]) {
      expect(restored).toMatchObject(generation)
    }
    history.addImageGeneration({...generation,upscaleResolution:undefined,upscaleSoftness:undefined})
    expect(row.upscale_resolution).toBeNull()
    expect(row.upscale_softness).toBeNull()
    expect(history.getImageGeneration('u').upscaleResolution).toBeUndefined()
  })
})
