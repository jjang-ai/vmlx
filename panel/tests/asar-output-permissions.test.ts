import { expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { resolve } from 'node:path'

it('normalizes copied assets to a byte-and-mode exact ASAR roundtrip without weakening parity', () => {
  const script = String.raw`
    const fs = require("node:fs");
    const path = require("node:path");
    const assert = require("node:assert/strict");
    const asar = require("@electron/asar");
    const hook = require("./scripts/electron-builder-before-pack.cjs");
    const root = fs.mkdtempSync(path.join(fs.realpathSync(require("node:os").tmpdir()), "vmlx-asar-mode-"));
    process.umask(0o077);
    (async () => {
      try {
        const source = path.join(root, "source"); fs.mkdirSync(source);
        for (const [name, mode] of [["icon.png", 0o644], ["font.woff2", 0o644], ["index.js", 0o600], ["run", 0o700]]) {
          fs.writeFileSync(path.join(source, name), name); fs.chmodSync(path.join(source, name), mode);
        }
        const mode = p => fs.statSync(p).mode & 0o7777;
        await asar.createPackage(source, path.join(root, "before.asar"));
        asar.extractAll(path.join(root, "before.asar"), path.join(root, "before"));
        assert.equal(mode(path.join(source, "icon.png")), 0o644);
        assert.equal(mode(path.join(root, "before/icon.png")), 0o600);
        hook.normalizeAsarOutputPermissions(source);
        await asar.createPackage(source, path.join(root, "after.asar"));
        asar.extractAll(path.join(root, "after.asar"), path.join(root, "after"));
        for (const name of fs.readdirSync(source)) {
          const a = path.join(source, name), b = path.join(root, "after", name);
          assert.deepEqual(fs.readFileSync(a), fs.readFileSync(b));
          assert.equal(mode(a), mode(b));
        }
        fs.symlinkSync(path.join(source, "icon.png"), path.join(source, "link"));
        assert.throws(() => hook.normalizeAsarOutputPermissions(source), /symlink/);
        fs.unlinkSync(path.join(source, "link"));
        fs.linkSync(path.join(source, "icon.png"), path.join(source, "hardlink"));
        assert.throws(() => hook.normalizeAsarOutputPermissions(source), /single-link/);
        assert.match(hook.toString(), /normalizeAsarOutputPermissions\(join\(panelDir, "out"\)\)/);
        console.log("ASAR copied-mode defect reproduced; normalized bytes/modes exact; unsafe links refused");
      } finally { fs.rmSync(root, { recursive: true, force: true }); }
    })().catch(error => { console.error(error); process.exitCode = 1 });
  `
  const result = spawnSync(process.execPath, ['-e', script], {
    cwd: resolve(__dirname, '..'), encoding: 'utf8',
  })
  expect(result.status, result.stderr + result.stdout).toBe(0)
})
