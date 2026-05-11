const { spawnSync } = require('node:child_process')
const { existsSync, readdirSync } = require('node:fs')
const { join } = require('node:path')

function walk(dir, out = []) {
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, ent.name)
    if (ent.isDirectory()) {
      walk(path, out)
    } else if (ent.isFile() && (path.endsWith('.so') || path.endsWith('.dylib'))) {
      out.push(path)
    }
  }
  return out
}

function removeSignature(path) {
  const proc = spawnSync('codesign', ['--remove-signature', path], {
    stdio: 'pipe',
    encoding: 'utf8',
  })
  // Unsigned files are fine; this hook only normalizes wheels that already
  // carry upstream signatures before electron-builder signs the whole app.
  const output = `${proc.stdout || ''}${proc.stderr || ''}`
  if (proc.status !== 0 && !output.includes('code object is not signed at all')) {
    throw new Error(`codesign --remove-signature failed for ${path}\n${output}`)
  }
}

async function afterPack(context) {
  const appOutDir = context && context.appOutDir
  const appName = context && context.packager && context.packager.appInfo
    ? context.packager.appInfo.productFilename
    : 'vMLX'
  if (!appOutDir) {
    throw new Error('electron-builder afterPack hook missing appOutDir')
  }

  const bundledPython = join(
    appOutDir,
    `${appName}.app`,
    'Contents',
    'Resources',
    'bundled-python',
    'python',
  )
  if (!existsSync(bundledPython)) {
    console.log(`[afterPack] bundled Python not found, skipping signature normalization: ${bundledPython}`)
    return
  }

  const nativeFiles = walk(bundledPython)
  for (const file of nativeFiles) {
    removeSignature(file)
  }
  console.log(`[afterPack] normalized signatures for ${nativeFiles.length} bundled Python native files`)
}

module.exports = afterPack

if (require.main === module) {
  afterPack({
    appOutDir: join(process.cwd(), 'release', 'mac-arm64'),
    packager: { appInfo: { productFilename: 'vMLX' } },
  }).catch((error) => {
    console.error(error && error.stack ? error.stack : error)
    process.exit(1)
  })
}
