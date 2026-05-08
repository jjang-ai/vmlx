const { spawnSync } = require('node:child_process')
const { existsSync } = require('node:fs')
const { join } = require('node:path')

function run(cmd, args, cwd) {
  const proc = spawnSync(cmd, args, {
    cwd,
    stdio: 'inherit',
    env: process.env,
  })
  if (proc.error) {
    throw proc.error
  }
  if (proc.status !== 0) {
    throw new Error(`${cmd} ${args.join(' ')} failed with exit ${proc.status}`)
  }
}

async function beforePack(context) {
  const panelDir = context && context.packager && context.packager.projectDir
    ? context.packager.projectDir
    : process.cwd()

  const verifyScript = join(panelDir, 'scripts', 'verify-bundled-python.sh')
  if (!existsSync(verifyScript)) {
    throw new Error(`Missing bundled-python verifier: ${verifyScript}`)
  }

  run('bash', [verifyScript], panelDir)

  if (process.env.VMLX_BEFORE_PACK_SKIP_VITE === '1') {
    console.log('VMLX_BEFORE_PACK_SKIP_VITE=1: skipped electron-vite build')
    return
  }

  run('npx', ['electron-vite', 'build'], panelDir)
}

module.exports = beforePack

if (require.main === module) {
  beforePack().catch((error) => {
    console.error(error && error.stack ? error.stack : error)
    process.exit(1)
  })
}
