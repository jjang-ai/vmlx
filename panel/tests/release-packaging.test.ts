import {
  chmodSync,
  copyFileSync,
  linkSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "fs";
import { spawnSync } from "child_process";
import { tmpdir } from "os";
import { dirname, join } from "path";
import { createRequire } from "module";
import { createHash } from "crypto";
import { describe, expect, it } from "vitest";

const repo = join(__dirname, "..");
const requireCjs = createRequire(import.meta.url);

function read(rel: string): string {
  return readFileSync(join(repo, rel), "utf8");
}

function r20Toolchain() {
  const invocationPaths = {
    git: "/usr/bin/git",
    node: "/opt/homebrew/bin/node",
    npm: "/opt/homebrew/bin/npm",
    npx: "/opt/homebrew/bin/npx",
    shasum: "/usr/bin/shasum",
    awk: "/usr/bin/awk",
    file: "/usr/bin/file",
    find: "/usr/bin/find",
    asar: join(repo, "node_modules/@electron/asar/bin/asar.js"),
    app_builder: join(
      repo,
      "node_modules/app-builder-bin/mac/app-builder_arm64",
    ),
    electron_builder: join(repo, "node_modules/electron-builder/cli.js"),
  };
  return Object.fromEntries(
    Object.entries(invocationPaths).map(([name, path]) => {
      const realpath = realpathSync(path);
      return [
        name,
        {
          path,
          realpath,
          sha256: createHash("sha256")
            .update(readFileSync(realpath))
            .digest("hex"),
        },
      ];
    }),
  );
}

const mlxRuntimeSpec = JSON.parse(read("scripts/mlx-runtime.json"));
const r20RuntimeContracts = {
  sequoia: {
    mlx_wheel_platform: "macosx_14_0_arm64",
    minimum_system_version: mlxRuntimeSpec.platforms.macosx_14_0_arm64.minimum_system_version,
  },
  tahoe: {
    mlx_wheel_platform: "macosx_26_0_arm64",
    minimum_system_version: mlxRuntimeSpec.platforms.macosx_26_0_arm64.minimum_system_version,
  },
};

function writeR20RuntimeFixture(
  beforePack: any,
  app: string,
  flavor: keyof typeof r20RuntimeContracts,
  sourceCommit: string,
  sourceTree: string,
  attestationPath: string,
) {
  const contract = r20RuntimeContracts[flavor];
  const bundleRoot = join(app, "Contents", "Resources", "bundled-python");
  const sitePackages = join(
    bundleRoot,
    "python",
    "lib",
    "python3.12",
    "site-packages",
  );
  for (const [distribution, normalized, tagPrefix] of [
    ["mlx", "mlx", "cp312-cp312"],
    ["mlx-metal", "mlx_metal", "py3-none"],
  ]) {
    const distInfo = join(sitePackages, `${normalized}-0.31.2.dist-info`);
    mkdirSync(distInfo, { recursive: true });
    writeFileSync(
      join(distInfo, "METADATA"),
      `Metadata-Version: 2.1\nName: ${distribution}\nVersion: 0.31.2\n`,
    );
    writeFileSync(
      join(distInfo, "WHEEL"),
      `Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: ${tagPrefix}-${contract.mlx_wheel_platform}\n`,
    );
  }
  writeFileSync(
    join(bundleRoot, "vmlx-bundle-provenance.json"),
    `${JSON.stringify(
      {
        schema_version: 1,
        vmlx: { commit: sourceCommit, version: "1.6.20" },
        jang: { commit: "f".repeat(40), version: "2.5.36" },
        mlx_wheel_platform: contract.mlx_wheel_platform,
      },
      null,
      2,
    )}\n`,
  );
  mkdirSync(join(app, "Contents"), { recursive: true });
  writeFileSync(
    join(app, "Contents", "Info.plist"),
    `<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>net.vmlx.app</string>
<key>CFBundleShortVersionString</key><string>1.6.20</string>
<key>CFBundleVersion</key><string>1.6.20</string>
<key>LSMinimumSystemVersion</key><string>${contract.minimum_system_version}</string>
</dict></plist>
`,
  );
  const runtimeContract = beforePack.inspectBundleRuntimeContract(
    bundleRoot,
    flavor,
  );
  const payload = {
    schema_version: 1,
    scope: "r20_production",
    stage: "bundle_runtime",
    version: "1.6.20",
    flavor,
    source: { commit: sourceCommit, tree: sourceTree },
    runtime_contract: runtimeContract,
  };
  const encoded = `${JSON.stringify(payload, null, 2)}\n`;
  writeFileSync(attestationPath, encoded, { mode: 0o400 });
  return {
    reference: {
      path: attestationPath,
      sha256: createHash("sha256").update(encoded).digest("hex"),
    },
    runtimeContract,
  };
}

describe("release packaging", () => {
  it("removes bundled pip distlib Windows launcher stubs before app signing", async () => {
    const afterPack = requireCjs(
      join(repo, "scripts/electron-builder-after-pack.cjs"),
    );
    const temp = mkdtempSync(join(tmpdir(), "vmlx-after-pack-"));
    try {
      const bundledPython = join(
        temp,
        "vMLX.app",
        "Contents",
        "Resources",
        "bundled-python",
        "python",
        "lib",
        "python3.12",
        "site-packages",
        "pip",
        "_vendor",
        "distlib",
      );
      mkdirSync(bundledPython, { recursive: true });
      const launcher = join(bundledPython, "t32.exe");
      writeFileSync(launcher, "windows launcher stub");

      await afterPack({
        appOutDir: temp,
        packager: { appInfo: { productFilename: "vMLX" } },
      });

      expect(afterPack.isBundledPipDistlibWindowsLauncher(launcher)).toBe(true);
      expect(() => readFileSync(launcher)).toThrow();
    } finally {
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("keeps the afterPack hook scoped to pip distlib exe launchers", () => {
    const source = read("scripts/electron-builder-after-pack.cjs");

    expect(source).toContain("site-packages/pip/_vendor/distlib");
    expect(source).toContain("removeBundledWindowsLaunchers");
    expect(source).toContain("removedWindowsLaunchers");
  });

  it("refuses symlinked or cross-checkout release dependencies", () => {
    const source = read("scripts/build-release-dmgs.sh");

    expect(source).toContain("release node_modules must not be a symlink");
    expect(source).toContain("NODE_MODULES_REAL=");
    expect(source).toContain(
      "release node_modules resolves outside this checkout",
    );
  });

  it("uses recursive Developer-ID staging before final audit and reseal", () => {
    const source = read("scripts/build-release-dmgs.sh");
    const stage = source.indexOf("run_electron_builder_action --mac --dir");
    const finalSign = source.indexOf(
      'finalize_release_app_signature "$app_path" "$RELEASE_CODESIGN_IDENTITY"',
    );

    expect(stage).toBeGreaterThan(0);
    expect(finalSign).toBeGreaterThan(stage);
    expect(source).toContain(
      '"$DEV_UNSIGNED" == "1" && "$RELEASE_SCOPE" != "codex_ui_only"',
    );
    expect(source).toContain("export CSC_IDENTITY_AUTO_DISCOVERY=false");
    expect(source).toContain("--config.mac.identity=null");
    expect(source).toContain("inside-out Developer-ID signing");
  });

  it("Developer-ID signs and audits Mach-O leaves outside bundled Python", () => {
    const source = read("scripts/build-release-dmgs.sh");

    expect(source).toContain("sign_remaining_app_macho_leaves()");
    expect(source).toContain(
      "Signature=adhoc|flags=.*adhoc|TeamIdentifier=not set",
    );
    expect(source).toContain("verify_release_macho_leaves()");
    expect(source).toContain("Authority=$EXPECTED_CODESIGN_IDENTITY");
    expect(source).toContain("TeamIdentifier=$EXPECTED_APPLE_TEAM_ID");
    expect(source).toContain("^Timestamp=");
    expect(source).toContain("flags=.*runtime");
    expect(source).toContain(
      'sign_remaining_app_macho_leaves "$app_path" "$identity"',
    );
    expect(source).toContain('verify_release_macho_leaves "$app_path"');
    expect(source).toContain("json.load(sys.stdin)");
    expect(source).not.toContain(
      'json.loads(sys.argv[1])["stdout"]',
    );
  });

  it("parses every package entrypoint and supported artifact hook through electron-builder", async () => {
    const pkg = JSON.parse(read("package.json"));
    const { validateConfig } = requireCjs(
      join(repo, "node_modules/app-builder-lib/out/util/config.js"),
    );

    expect(pkg.scripts.package).toBe("npm run dist");
    expect(pkg.scripts.dist).toBe(
      "VMLX_RELEASE_SCOPE=production ./scripts/build-release-dmgs.sh all",
    );
    expect(pkg.build.beforePack).toBe(
      "scripts/electron-builder-before-pack.cjs",
    );
    expect(pkg.build.artifactBuildStarted).toBe(
      "scripts/electron-builder-after-all-artifact-build.cjs",
    );
    expect(pkg.build.afterAllArtifactBuild).toBe(
      "scripts/electron-builder-after-all-artifact-build.cjs",
    );
    await expect(
      validateConfig(pkg.build, { isEnabled: false, add() {} }),
    ).resolves.toBeUndefined();
  });

  it("runs the source-verification hook at artifact start and after all artifacts", async () => {
    const lifecycleModule = requireCjs(
      join(repo, "scripts/electron-builder-after-all-artifact-build.cjs"),
    );
    const calls: Array<{ panelDir: string; context: unknown }> = [];
    const hook = lifecycleModule.createReleaseArtifactLifecycleHook(
      (panelDir: string, context: unknown) => {
        calls.push({ panelDir, context });
      },
    );
    const started = {
      file: "/private/tmp/vMLX-1.6.20-sequoia-arm64.dmg",
      targetPresentableName: "DMG",
    };
    const completed = {
      outDir: "/private/tmp/release",
      artifactPaths: [started.file],
      platformToTargets: new Map(),
      configuration: {},
    };

    await expect(hook(started)).resolves.toEqual([]);
    await expect(hook(completed)).resolves.toEqual([]);
    expect(calls).toEqual([
      { panelDir: repo, context: started },
      { panelDir: repo, context: completed },
    ]);
  });

  it("rejects unauthenticated or spoofable GitHub release origins", () => {
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );

    for (const value of [
      "https://github.com/jjang-ai/vmlx.git",
      "https://github.com/jjang-ai/vmlx.GIT",
      "ssh://git@github.com/jjang-ai/vmlx.git",
      "ssh://git@github.com/jjang-ai/vmlx.GIT",
      "git@github.com:jjang-ai/vmlx.git",
      "git@github.com:jjang-ai/vmlx.GIT",
    ]) {
      expect(beforePack.canonicalGitHubRepo(value)).toBe("jjang-ai/vmlx");
    }
    for (const value of [
      "git://github.com/jjang-ai/vmlx.git",
      "http://github.com/jjang-ai/vmlx.git",
      "github.com:jjang-ai/vmlx.git",
      "ssh://wrong@github.com/jjang-ai/vmlx.git",
      "ssh://git@github.com:2222/jjang-ai/vmlx.git",
      "https://user@github.com/jjang-ai/vmlx.git",
      "https://github.com.evil.invalid/jjang-ai/vmlx.git",
      "https://github.com/jjang-ai/vmlx/extra.git",
      "file:///tmp/jjang-ai/vmlx.git",
    ]) {
      expect(beforePack.canonicalGitHubRepo(value)).toBe("");
    }
  });

  it("binds the hardened release contract to the explicit production version", () => {
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );

    expect(beforePack.configuredR20Version({})).toBe("1.6.20");
    expect(
      beforePack.configuredR20Version({
        VMLX_R20_EXPECTED_VERSION: "1.6.39",
      }),
    ).toBe("1.6.39");
    expect(() =>
      beforePack.configuredR20Version({
        VMLX_R20_EXPECTED_VERSION: "not-a-release",
      }),
    ).toThrow("invalid vMLX production version");
  });

  it("blocks direct electron-builder packaging of 1.6.20 before expensive hooks", async () => {
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-direct-pack-"));
    const panelDir = join(temp, "panel");
    mkdirSync(panelDir);
    writeFileSync(
      join(panelDir, "package.json"),
      JSON.stringify({
        version: "1.6.20",
        build: { mac: { notarize: { teamId: "55KGF2S5AY" } } },
      }),
    );
    const previous = process.env.VMLX_RELEASE_SCOPE;
    delete process.env.VMLX_RELEASE_SCOPE;
    try {
      await expect(
        beforePack({ packager: { projectDir: panelDir } }),
      ).rejects.toThrow(
        "vMLX 1.6.20 packaging requires VMLX_RELEASE_SCOPE=r20_production",
      );
      expect(() => readFileSync(join(panelDir, "verify-ran"))).toThrow();
    } finally {
      if (previous === undefined) {
        delete process.env.VMLX_RELEASE_SCOPE;
      } else {
        process.env.VMLX_RELEASE_SCOPE = previous;
      }
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("blocks direct prepackaged 1.6.20 artifacts at the completion hook", async () => {
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-direct-artifact-"));
    const panelDir = join(temp, "panel");
    const scriptsDir = join(panelDir, "scripts");
    mkdirSync(scriptsDir, { recursive: true });
    copyFileSync(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
      join(scriptsDir, "electron-builder-before-pack.cjs"),
    );
    copyFileSync(
      join(repo, "scripts/release-python-action.cjs"),
      join(scriptsDir, "release-python-action.cjs"),
    );
    copyFileSync(
      join(repo, "scripts/electron-builder-after-all-artifact-build.cjs"),
      join(scriptsDir, "electron-builder-after-all-artifact-build.cjs"),
    );
    writeFileSync(
      join(panelDir, "package.json"),
      JSON.stringify({
        version: "1.6.20",
        build: { mac: { notarize: { teamId: "55KGF2S5AY" } } },
      }),
    );
    const previous = process.env.VMLX_RELEASE_SCOPE;
    delete process.env.VMLX_RELEASE_SCOPE;
    try {
      const afterAll = requireCjs(
        join(scriptsDir, "electron-builder-after-all-artifact-build.cjs"),
      );
      await expect(afterAll({})).rejects.toThrow(
        "vMLX 1.6.20 packaging requires VMLX_RELEASE_SCOPE=r20_production",
      );
    } finally {
      if (previous === undefined) {
        delete process.env.VMLX_RELEASE_SCOPE;
      } else {
        process.env.VMLX_RELEASE_SCOPE = previous;
      }
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("rejects r20_production with a non-1.6.20 package in both builder hooks", async () => {
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-reciprocal-"));
    const panelDir = join(temp, "panel");
    const scriptsDir = join(panelDir, "scripts");
    mkdirSync(scriptsDir, { recursive: true });
    copyFileSync(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
      join(scriptsDir, "electron-builder-before-pack.cjs"),
    );
    copyFileSync(
      join(repo, "scripts/release-python-action.cjs"),
      join(scriptsDir, "release-python-action.cjs"),
    );
    copyFileSync(
      join(repo, "scripts/electron-builder-after-all-artifact-build.cjs"),
      join(scriptsDir, "electron-builder-after-all-artifact-build.cjs"),
    );
    writeFileSync(
      join(panelDir, "package.json"),
      JSON.stringify({ version: "1.6.17" }),
    );

    const envNames = [
      "VMLX_RELEASE_SCOPE",
      "VMLINUX_RELEASE_SCOPE",
      "VMLX_R20_OFFICIAL_PACKAGING",
    ];
    const previous = Object.fromEntries(
      envNames.map((name) => [name, process.env[name]]),
    );
    delete process.env.VMLX_RELEASE_SCOPE;
    delete process.env.VMLINUX_RELEASE_SCOPE;
    delete process.env.VMLX_R20_OFFICIAL_PACKAGING;
    try {
      const beforePack = requireCjs(
        join(scriptsDir, "electron-builder-before-pack.cjs"),
      );
      const afterAll = requireCjs(
        join(scriptsDir, "electron-builder-after-all-artifact-build.cjs"),
      );
      for (const scopeName of ["VMLX_RELEASE_SCOPE", "VMLINUX_RELEASE_SCOPE"]) {
        process.env[scopeName] = "r20_production";
        await expect(
          beforePack({ packager: { projectDir: panelDir } }),
        ).rejects.toThrow(
          "VMLX_RELEASE_SCOPE=r20_production requires package version 1.6.20, found 1.6.17",
        );
        await expect(afterAll({})).rejects.toThrow(
          "VMLX_RELEASE_SCOPE=r20_production requires package version 1.6.20, found 1.6.17",
        );
        delete process.env[scopeName];
      }
    } finally {
      for (const name of envNames) {
        const value = previous[name];
        if (value === undefined) {
          delete process.env[name];
        } else {
          process.env[name] = value;
        }
      }
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("sanitizes PATH before commands and confines production output before removal", () => {
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-output-confinement-"));
    const shadow = join(temp, "shadow-bin");
    const marker = join(temp, "shadow-command-ran");
    const victim = join(temp, "victim");
    const sentinel = join(victim, "keep.txt");
    mkdirSync(shadow);
    mkdirSync(victim);
    writeFileSync(sentinel, "must survive\n");
    const fakeDirname = join(shadow, "dirname");
    writeFileSync(
      fakeDirname,
      `#!/bin/sh\nprintf shadow > "${marker}"\nexec /usr/bin/dirname "$@"\n`,
    );
    chmodSync(fakeDirname, 0o755);
    try {
      const result = spawnSync(
        "/bin/bash",
        [join(repo, "scripts/build-release-dmgs.sh"), "all"],
        {
          encoding: "utf8",
          env: {
            ...process.env,
            PATH: shadow,
            VMLX_RELEASE_SCOPE: "r20_production",
            VMLINUX_RELEASE_SCOPE: "",
            VMLX_RELEASE_OUTPUT_DIR: victim,
            VMLINUX_RELEASE_OUTPUT_DIR: "",
          },
        },
      );
      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain("release output overrides are forbidden");
      expect(readFileSync(sentinel, "utf8")).toBe("must survive\n");
      expect(() => readFileSync(marker)).toThrow();

      const fakeRoot = join(temp, "fake-checkout");
      const fakePanel = join(fakeRoot, "panel");
      const fakeScripts = join(fakePanel, "scripts");
      mkdirSync(fakeScripts, { recursive: true });
      mkdirSync(join(fakePanel, "node_modules"));
      mkdirSync(join(fakeRoot, ".venv", "bin"), { recursive: true });
      copyFileSync(
        join(repo, "scripts/build-release-dmgs.sh"),
        join(fakeScripts, "build-release-dmgs.sh"),
      );
      copyFileSync(
        join(repo, "scripts/release-python-action.cjs"),
        join(fakeScripts, "release-python-action.cjs"),
      );
      const authoritativePython = join(
        repo,
        "..",
        ".venv",
        "bin",
        "python",
      );
      const fakePythonSource = join(
        fakeRoot,
        ".venv",
        "bin",
        "python-real",
      );
      writeFileSync(
        fakePythonSource,
        "#!/bin/sh\n" +
          "unset __PYVENV_LAUNCHER__\n" +
          `exec ${JSON.stringify(authoritativePython)} "$@"\n`,
      );
      chmodSync(fakePythonSource, 0o755);
      symlinkSync(
        fakePythonSource,
        join(fakeRoot, ".venv", "bin", "python"),
      );
      writeFileSync(
        join(fakeRoot, ".venv", "pyvenv.cfg"),
        "home = /usr/bin\n",
      );
      writeFileSync(
        join(fakePanel, "package.json"),
        JSON.stringify({
          version: "1.6.20",
          build: { mac: { notarize: { teamId: "55KGF2S5AY" } } },
        }),
      );
      symlinkSync(victim, join(fakePanel, "release"));
      const symlinkResult = spawnSync(
        "/bin/bash",
        [join(fakeScripts, "build-release-dmgs.sh"), "all"],
        {
          encoding: "utf8",
          env: {
            ...process.env,
            PATH: shadow,
            VMLX_RELEASE_SCOPE: "r20_production",
            VMLINUX_RELEASE_SCOPE: "",
            VMLX_RELEASE_OUTPUT_DIR: "",
            VMLINUX_RELEASE_OUTPUT_DIR: "",
            PYTHON: "",
            PYTHONHOME: "",
            PYTHONPATH: "",
            VIRTUAL_ENV: "",
          },
        },
      );
      expect(symlinkResult.status).not.toBe(0);
      expect(symlinkResult.stderr).toContain(
        "production release output is not a real directory",
      );
      expect(readFileSync(sentinel, "utf8")).toBe("must survive\n");
      expect(() => readFileSync(marker)).toThrow();
    } finally {
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("revalidates a hash-bound checkout-local tool before reuse", () => {
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-mutated-modules-"));
    const copiedAsar = join(
      temp,
      "node_modules",
      "@electron",
      "asar",
      "bin",
      "asar.js",
    );
    mkdirSync(join(copiedAsar, ".."), { recursive: true });
    copyFileSync(join(repo, "node_modules/@electron/asar/bin/asar.js"), copiedAsar);
    chmodSync(copiedAsar, 0o755);
    const tools = r20Toolchain();
    const copiedRealpath = realpathSync(copiedAsar);
    tools.asar = {
      path: copiedAsar,
      realpath: copiedRealpath,
      sha256: createHash("sha256")
        .update(readFileSync(copiedRealpath))
        .digest("hex"),
    };
    const fixedPath = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin";
    const envNames = [
      "VMLX_R20_FIXED_PATH",
      "PATH",
      ...Object.keys(tools).flatMap((name) => [
        `VMLX_R20_TOOL_${name.toUpperCase()}_PATH`,
        `VMLX_R20_TOOL_${name.toUpperCase()}_REALPATH`,
        `VMLX_R20_TOOL_${name.toUpperCase()}_SHA256`,
      ]),
    ];
    const previous = Object.fromEntries(
      envNames.map((name) => [name, process.env[name]]),
    );
    try {
      process.env.PATH = fixedPath;
      process.env.VMLX_R20_FIXED_PATH = fixedPath;
      for (const [name, tool] of Object.entries(tools)) {
        const prefix = `VMLX_R20_TOOL_${name.toUpperCase()}`;
        process.env[`${prefix}_PATH`] = tool.path;
        process.env[`${prefix}_REALPATH`] = tool.realpath;
        process.env[`${prefix}_SHA256`] = tool.sha256;
      }
      expect(
        beforePack.verifyPinnedToolchain({ fixed_path: fixedPath, tools }),
      ).toBe(true);
      writeFileSync(copiedAsar, "\n// mutated after attestation\n", { flag: "a" });
      expect(() =>
        beforePack.verifyPinnedToolchain({ fixed_path: fixedPath, tools }),
      ).toThrow("pinned asar identity changed");
    } finally {
      for (const name of envNames) {
        const value = previous[name];
        if (value === undefined) {
          delete process.env[name];
        } else {
          process.env[name] = value;
        }
      }
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("fails closed when a bound release tool mutates during its own action", () => {
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-action-mutation-"));
    const panelDir = join(temp, "panel");
    const buildDir = join(temp, "build");
    const planPath = join(buildDir, "r20-release-driver-plan.json");
    const fakeAsar = join(temp, "mutating-asar.js");
    const marker = join(temp, "action-ran");
    mkdirSync(panelDir, { recursive: true });
    mkdirSync(buildDir, { recursive: true });
    writeFileSync(
      fakeAsar,
      "const fs = require('node:fs');\n" +
        "fs.writeFileSync(process.argv[2], 'ran\\n');\n" +
        "fs.appendFileSync(__filename, '\\n// mutated during action\\n');\n",
      { mode: 0o755 },
    );
    const tools = r20Toolchain();
    tools.asar = {
      path: fakeAsar,
      realpath: realpathSync(fakeAsar),
      sha256: createHash("sha256").update(readFileSync(fakeAsar)).digest("hex"),
    };
    const fixedPath = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin";
    const plan = { fixed_path: fixedPath, tools };
    const encoded = `${JSON.stringify(plan)}\n`;
    writeFileSync(planPath, encoded, { mode: 0o600 });
    const planHash = createHash("sha256").update(encoded).digest("hex");
    const envNames = [
      "VMLX_R20_RELEASE_PLAN",
      "VMLX_R20_RELEASE_PLAN_SHA256",
      "VMLX_R20_FIXED_PATH",
      "PATH",
      ...Object.keys(tools).flatMap((name) => [
        `VMLX_R20_TOOL_${name.toUpperCase()}_PATH`,
        `VMLX_R20_TOOL_${name.toUpperCase()}_REALPATH`,
        `VMLX_R20_TOOL_${name.toUpperCase()}_SHA256`,
      ]),
    ];
    const previous = Object.fromEntries(
      envNames.map((name) => [name, process.env[name]]),
    );
    try {
      Object.assign(process.env, {
        VMLX_R20_RELEASE_PLAN: planPath,
        VMLX_R20_RELEASE_PLAN_SHA256: planHash,
        VMLX_R20_FIXED_PATH: fixedPath,
        PATH: fixedPath,
      });
      for (const [name, tool] of Object.entries(tools)) {
        const prefix = `VMLX_R20_TOOL_${name.toUpperCase()}`;
        process.env[`${prefix}_PATH`] = tool.path;
        process.env[`${prefix}_REALPATH`] = tool.realpath;
        process.env[`${prefix}_SHA256`] = tool.sha256;
      }
      expect(() =>
        beforePack.runPlanToolAction(
          panelDir,
          planHash,
          "asar",
          [marker],
          { cwd: panelDir },
        ),
      ).toThrow("pinned asar identity changed");
      expect(readFileSync(marker, "utf8")).toBe("ran\n");
    } finally {
      for (const name of envNames) {
        const value = previous[name];
        if (value === undefined) {
          delete process.env[name];
        } else {
          process.env[name] = value;
        }
      }
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("binds artifact hooks to one live all-flavor release-driver plan", () => {
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-plan-"));
    const buildDir = join(temp, "build");
    const planPath = join(buildDir, "r20-release-driver-plan.json");
    const expectedArtifact = join(
      temp,
      "release",
      "vMLX-1.6.20-sequoia-arm64.dmg",
    );
    const expectedBlockmap = `${expectedArtifact}.blockmap`;
    const stagedApp = join(
      temp,
      "release",
      "sequoia-app",
      "mac-arm64",
      "vMLX.app",
    );
    const hookDirectory = join(temp, "hook-completions");
    const hookAttestation = join(hookDirectory, "sequoia.completion.json");
    const bundleAttestation = join(
      hookDirectory,
      "sequoia.bundle-runtime.json",
    );
    mkdirSync(buildDir, { recursive: true });
    mkdirSync(hookDirectory, { recursive: true, mode: 0o700 });
    const manifestSha256 = "a".repeat(64);
    const sourceCommit = "b".repeat(40);
    const sourceTree = "c".repeat(40);
    const fixedPath = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin";
    const tools = r20Toolchain();
    const runtime = writeR20RuntimeFixture(
      beforePack,
      stagedApp,
      "sequoia",
      sourceCommit,
      sourceTree,
      bundleAttestation,
    );
    const plan = {
      schema_version: 3,
      scope: "r20_production",
      version: "1.6.20",
      source_commit: sourceCommit,
      source_tree: sourceTree,
      manifest_sha256: manifestSha256,
      requested_flavor: "all",
      current_flavor: "sequoia",
      phase: "dmg",
      expected_artifact: expectedArtifact,
      staged_app: stagedApp,
      hook_attestation: hookAttestation,
      bundle_runtime: runtime.reference,
      flavor_contract: r20RuntimeContracts.sequoia,
      driver_pid: process.pid,
      nonce: "d".repeat(64),
      fixed_path: fixedPath,
      tools,
    };
    const encoded = `${JSON.stringify(plan)}\n`;
    writeFileSync(planPath, encoded, { mode: 0o600 });
    const planHash = createHash("sha256").update(encoded).digest("hex");
    const envNames = [
      "VMLX_R20_RELEASE_PLAN",
      "VMLX_R20_RELEASE_PLAN_SHA256",
      "VMLX_R20_RELEASE_DRIVER_PID",
      "VMLX_R20_RELEASE_REQUESTED_FLAVOR",
      "VMLX_R20_RELEASE_CURRENT_FLAVOR",
      "VMLX_R20_RELEASE_PHASE",
      "VMLX_R20_RELEASE_EXPECTED_ARTIFACT",
      "VMLX_R20_RELEASE_DRIVER_NONCE",
      "VMLX_R20_HOOK_ATTESTATION_DIR",
      "VMLX_R20_FIXED_PATH",
      "VMLX_EXPECTED_MLX_WHEEL_PLATFORM",
      "PATH",
      ...Object.keys(tools).flatMap((name) => [
        `VMLX_R20_TOOL_${name.toUpperCase()}_PATH`,
        `VMLX_R20_TOOL_${name.toUpperCase()}_REALPATH`,
        `VMLX_R20_TOOL_${name.toUpperCase()}_SHA256`,
      ]),
    ];
    const previous = Object.fromEntries(
      envNames.map((name) => [name, process.env[name]]),
    );
    Object.assign(process.env, {
      VMLX_R20_RELEASE_PLAN: planPath,
      VMLX_R20_RELEASE_PLAN_SHA256: planHash,
      VMLX_R20_RELEASE_DRIVER_PID: String(process.pid),
      VMLX_R20_RELEASE_REQUESTED_FLAVOR: "all",
      VMLX_R20_RELEASE_CURRENT_FLAVOR: "sequoia",
      VMLX_R20_RELEASE_PHASE: "dmg",
      VMLX_R20_RELEASE_EXPECTED_ARTIFACT: expectedArtifact,
      VMLX_R20_RELEASE_DRIVER_NONCE: plan.nonce,
      VMLX_R20_HOOK_ATTESTATION_DIR: hookDirectory,
      VMLX_R20_FIXED_PATH: fixedPath,
      VMLX_EXPECTED_MLX_WHEEL_PLATFORM:
        r20RuntimeContracts.sequoia.mlx_wheel_platform,
      PATH: fixedPath,
    });
    for (const [name, tool] of Object.entries(tools)) {
      const prefix = `VMLX_R20_TOOL_${name.toUpperCase()}`;
      process.env[`${prefix}_PATH`] = tool.path;
      process.env[`${prefix}_REALPATH`] = tool.realpath;
      process.env[`${prefix}_SHA256`] = tool.sha256;
    }
    try {
      expect(
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          {
            targetPresentableName: "DMG",
            file: expectedArtifact,
            arch: null,
          },
        ),
      ).toEqual(plan);

      delete process.env.VMLX_EXPECTED_MLX_WHEEL_PLATFORM;
      expect(() =>
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          {
            targetPresentableName: "DMG",
            file: expectedArtifact,
            arch: null,
          },
        ),
      ).toThrow(
        "packaging requires VMLX_EXPECTED_MLX_WHEEL_PLATFORM=macosx_14_0_arm64",
      );
      process.env.VMLX_EXPECTED_MLX_WHEEL_PLATFORM = "macosx_26_0_arm64";
      expect(() =>
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          {
            targetPresentableName: "DMG",
            file: expectedArtifact,
            arch: null,
          },
        ),
      ).toThrow(
        "packaging requires VMLX_EXPECTED_MLX_WHEEL_PLATFORM=macosx_14_0_arm64",
      );
      process.env.VMLX_EXPECTED_MLX_WHEEL_PLATFORM =
        r20RuntimeContracts.sequoia.mlx_wheel_platform;

      const buildResultContext = {
        outDir: join(temp, "release"),
        artifactPaths: [expectedArtifact, expectedBlockmap],
        platformToTargets: new Map(),
        configuration: {},
      };
      expect(
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          buildResultContext,
        ),
      ).toEqual(plan);

      process.env.VMLX_R20_TOOL_GIT_PATH = join(temp, "shadowed-git");
      expect(() =>
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          buildResultContext,
        ),
      ).toThrow("pinned git identity changed");
      process.env.VMLX_R20_TOOL_GIT_PATH = tools.git.path;

      expect(() =>
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          {},
        ),
      ).toThrow("unrecognized or ambiguous electron-builder hook context");
      for (const invalidContext of [
        {
          targetPresentableName: "DMG",
          file: expectedArtifact,
        },
        {
          ...buildResultContext,
          targetPresentableName: "DMG",
          file: expectedArtifact,
          arch: null,
        },
      ]) {
        expect(() =>
          beforePack.verifyR20ReleasePlan(
            temp,
            manifestSha256,
            sourceCommit,
            sourceTree,
            invalidContext,
          ),
        ).toThrow("unrecognized or ambiguous electron-builder hook context");
      }
      expect(() =>
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          {
            targetPresentableName: "ZIP",
            file: expectedArtifact,
            arch: null,
          },
        ),
      ).toThrow("artifact build is outside the release-driver plan");

      for (const artifactPaths of [
        [expectedArtifact],
        [expectedBlockmap],
        [
          expectedArtifact,
          expectedBlockmap,
          join(temp, "release", "unexpected.dmg"),
        ],
        [expectedArtifact, expectedBlockmap, expectedArtifact],
        [expectedArtifact, join(temp, "release", "unexpected.dmg.blockmap")],
      ]) {
        expect(() =>
          beforePack.verifyR20ReleasePlan(
            temp,
            manifestSha256,
            sourceCommit,
            sourceTree,
            {
              ...buildResultContext,
              artifactPaths,
            },
          ),
        ).toThrow(
          "completed artifacts do not exactly match the release-driver plan",
        );
      }

      process.env.VMLX_R20_RELEASE_REQUESTED_FLAVOR = "sequoia";
      expect(() =>
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          {},
        ),
      ).toThrow("packaging requires VMLX_R20_RELEASE_REQUESTED_FLAVOR=all");
      process.env.VMLX_R20_RELEASE_REQUESTED_FLAVOR = "all";

      expect(() =>
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          {
            ...buildResultContext,
            artifactPaths: [],
          },
        ),
      ).toThrow(
        "completed artifacts do not exactly match the release-driver plan",
      );

      const stageOutput = join(temp, "release", "sequoia-app");
      const stagePlan = {
        ...plan,
        phase: "stage",
        expected_artifact: stageOutput,
        staged_app: join(stageOutput, "mac-arm64", "vMLX.app"),
      };
      const stageEncoded = `${JSON.stringify(stagePlan)}\n`;
      writeFileSync(planPath, stageEncoded, { mode: 0o600 });
      Object.assign(process.env, {
        VMLX_R20_RELEASE_PLAN_SHA256: createHash("sha256")
          .update(stageEncoded)
          .digest("hex"),
        VMLX_R20_RELEASE_PHASE: "stage",
        VMLX_R20_RELEASE_EXPECTED_ARTIFACT: stageOutput,
      });
      expect(
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          {
            packager: { projectDir: temp },
            outDir: stageOutput,
            appOutDir: join(stageOutput, "mac-arm64"),
            electronPlatformName: "darwin",
            arch: 3,
            targets: [],
          },
        ),
      ).toEqual(stagePlan);
      expect(
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          {
            outDir: stageOutput,
            artifactPaths: [],
            platformToTargets: new Map(),
            configuration: {},
          },
        ),
      ).toEqual(stagePlan);

      process.env.VMLX_R20_RELEASE_PLAN_SHA256 = "e".repeat(64);
      expect(() =>
        beforePack.verifyR20ReleasePlan(
          temp,
          manifestSha256,
          sourceCommit,
          sourceTree,
          {},
        ),
      ).toThrow("release-driver plan SHA-256 does not match");
    } finally {
      for (const name of envNames) {
        const value = previous[name];
        if (value === undefined) {
          delete process.env[name];
        } else {
          process.env[name] = value;
        }
      }
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("emits one sealed completion attestation from the validated production hook path", async () => {
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );
    const asar = requireCjs(join(repo, "node_modules/@electron/asar"));
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-hook-completion-"));
    const hookPanelDir = join(temp, "panel");
    const buildDir = join(temp, "build");
    const releaseDir = join(temp, "release");
    const stagedApp = join(
      releaseDir,
      "sequoia-app",
      "mac-arm64",
      "vMLX.app",
    );
    const resources = join(stagedApp, "Contents", "Resources");
    const asarSource = join(temp, "asar-source");
    const appAsar = join(resources, "app.asar");
    const expectedArtifact = join(
      releaseDir,
      "vMLX-1.6.20-sequoia-arm64.dmg",
    );
    const expectedBlockmap = `${expectedArtifact}.blockmap`;
    const hookDirectory = join(temp, "private", "hook-completions");
    const hookAttestation = join(hookDirectory, "sequoia.completion.json");
    const bundleAttestation = join(
      hookDirectory,
      "sequoia.bundle-runtime.json",
    );
    const planPath = join(buildDir, "r20-release-driver-plan.json");
    const fixedPath = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin";
    const tools = r20Toolchain();
    const manifestSha256 = "a".repeat(64);
    const sourceCommit = "b".repeat(40);
    const sourceTree = "c".repeat(40);
    mkdirSync(hookDirectory, { recursive: true, mode: 0o700 });
    const runtime = writeR20RuntimeFixture(
      beforePack,
      stagedApp,
      "sequoia",
      sourceCommit,
      sourceTree,
      bundleAttestation,
    );
    const plan = {
      schema_version: 3,
      scope: "r20_production",
      version: "1.6.20",
      source_commit: sourceCommit,
      source_tree: sourceTree,
      manifest_sha256: manifestSha256,
      requested_flavor: "all",
      current_flavor: "sequoia",
      phase: "dmg",
      expected_artifact: expectedArtifact,
      staged_app: stagedApp,
      hook_attestation: hookAttestation,
      bundle_runtime: runtime.reference,
      flavor_contract: r20RuntimeContracts.sequoia,
      driver_pid: process.pid,
      nonce: "d".repeat(64),
      fixed_path: fixedPath,
      tools,
    };
    const envNames = [
      "VMLX_R20_RELEASE_PLAN",
      "VMLX_R20_RELEASE_PLAN_SHA256",
      "VMLX_R20_RELEASE_DRIVER_PID",
      "VMLX_R20_RELEASE_REQUESTED_FLAVOR",
      "VMLX_R20_RELEASE_CURRENT_FLAVOR",
      "VMLX_R20_RELEASE_PHASE",
      "VMLX_R20_RELEASE_EXPECTED_ARTIFACT",
      "VMLX_R20_RELEASE_DRIVER_NONCE",
      "VMLX_R20_HOOK_ATTESTATION_DIR",
      "VMLX_R20_FIXED_PATH",
      "VMLX_EXPECTED_MLX_WHEEL_PLATFORM",
      "PATH",
      ...Object.keys(tools).flatMap((name) => [
        `VMLX_R20_TOOL_${name.toUpperCase()}_PATH`,
        `VMLX_R20_TOOL_${name.toUpperCase()}_REALPATH`,
        `VMLX_R20_TOOL_${name.toUpperCase()}_SHA256`,
      ]),
    ];
    const previous = Object.fromEntries(
      envNames.map((name) => [name, process.env[name]]),
    );
    try {
      mkdirSync(buildDir, { recursive: true });
      mkdirSync(hookPanelDir, { recursive: true });
      mkdirSync(resources, { recursive: true });
      mkdirSync(join(asarSource, "out"), { recursive: true });
      mkdirSync(hookDirectory, { recursive: true, mode: 0o700 });
      writeFileSync(
        join(asarSource, "package.json"),
        '{"name":"vmlx-hook-fixture","version":"1.6.20"}\n',
      );
      writeFileSync(
        join(asarSource, "out", "main.js"),
        "console.log('sealed hook fixture')\n",
      );
      await asar.createPackage(asarSource, appAsar);
      writeFileSync(expectedArtifact, "fixture-dmg-payload\n");
      writeFileSync(expectedBlockmap, "fixture-blockmap-payload\n");

      const encoded = `${JSON.stringify(plan)}\n`;
      writeFileSync(planPath, encoded, { mode: 0o600 });
      const planHash = createHash("sha256").update(encoded).digest("hex");
      Object.assign(process.env, {
        VMLX_R20_RELEASE_PLAN: planPath,
        VMLX_R20_RELEASE_PLAN_SHA256: planHash,
        VMLX_R20_RELEASE_DRIVER_PID: String(process.pid),
        VMLX_R20_RELEASE_REQUESTED_FLAVOR: "all",
        VMLX_R20_RELEASE_CURRENT_FLAVOR: "sequoia",
        VMLX_R20_RELEASE_PHASE: "dmg",
        VMLX_R20_RELEASE_EXPECTED_ARTIFACT: expectedArtifact,
        VMLX_R20_RELEASE_DRIVER_NONCE: plan.nonce,
        VMLX_R20_HOOK_ATTESTATION_DIR: hookDirectory,
        VMLX_R20_FIXED_PATH: fixedPath,
        VMLX_EXPECTED_MLX_WHEEL_PLATFORM:
          r20RuntimeContracts.sequoia.mlx_wheel_platform,
        PATH: fixedPath,
      });
      for (const [name, tool] of Object.entries(tools)) {
        const prefix = `VMLX_R20_TOOL_${name.toUpperCase()}`;
        process.env[`${prefix}_PATH`] = tool.path;
        process.env[`${prefix}_REALPATH`] = tool.realpath;
        process.env[`${prefix}_SHA256`] = tool.sha256;
      }

      const buildResultContext = {
        outDir: releaseDir,
        artifactPaths: [expectedArtifact, expectedBlockmap],
        platformToTargets: new Map(),
        configuration: {},
      };
      const verifiedPlan = beforePack.verifyR20ReleasePlan(
        temp,
        manifestSha256,
        sourceCommit,
        sourceTree,
        buildResultContext,
      );
      writeFileSync(join(asarSource, ".DS_Store"), "forbidden metadata");
      rmSync(appAsar, { force: true });
      await asar.createPackage(asarSource, appAsar);
      expect(() =>
        beforePack.emitR20CompletionAttestation(
          hookPanelDir,
          verifiedPlan,
          planHash,
        ),
      ).toThrow("app.asar contains forbidden Finder metadata");
      rmSync(join(asarSource, ".DS_Store"));
      rmSync(appAsar, { force: true });
      await asar.createPackage(asarSource, appAsar);
      const result = beforePack.emitR20CompletionAttestation(
        hookPanelDir,
        verifiedPlan,
        planHash,
      );

      expect(result.path).toBe(hookAttestation);
      expect(result.payload.plan.sha256).toBe(planHash);
      expect(result.payload.bundle_runtime).toEqual(runtime.reference);
      expect(result.payload.runtime_contract).toMatchObject(
        r20RuntimeContracts.sequoia,
      );
      expect(result.payload.staged_app.payload).toEqual(
        beforePack.treePayload(stagedApp),
      );
      expect(result.payload.staged_app.payload.root_mode).toBe(
        statSync(stagedApp).mode & 0o7777,
      );
      expect(
        result.payload.extracted_asar.payload.entries["out/main.js"],
      ).toMatchObject({ kind: "file" });
      expect(result.payload.artifacts.dmg.sha256).toBe(
        createHash("sha256").update(readFileSync(expectedArtifact)).digest("hex"),
      );
      expect(result.payload.artifacts.blockmap.sha256).toBe(
        createHash("sha256").update(readFileSync(expectedBlockmap)).digest("hex"),
      );
      expect(statSync(hookAttestation).mode & 0o777).toBe(0o400);
      expect(() =>
        beforePack.emitR20CompletionAttestation(
          hookPanelDir,
          verifiedPlan,
          planHash,
        ),
      ).toThrow("hook attestation output is reused or not private");
    } finally {
      for (const name of envNames) {
        const value = previous[name];
        if (value === undefined) {
          delete process.env[name];
        } else {
          process.env[name] = value;
        }
      }
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("ignores only ambient Finder metadata in extracted ASAR payloads", () => {
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-asar-finder-"));
    try {
      mkdirSync(join(temp, "node_modules"), { recursive: true });
      writeFileSync(join(temp, "node_modules", "module.js"), "module.exports=1\n");
      writeFileSync(join(temp, "node_modules", ".DS_Store"), "ambient");
      writeFileSync(join(temp, "node_modules", ".other-hidden"), "covered");

      const strict = beforePack.treePayload(temp);
      const filtered = beforePack.treePayload(temp, {
        ignoreAmbientFinderMetadata: true,
      });

      expect(strict.entries["node_modules/.DS_Store"]).toBeDefined();
      expect(filtered.entries["node_modules/.DS_Store"]).toBeUndefined();
      expect(filtered.entries["node_modules/.other-hidden"]).toBeDefined();
      expect(filtered.entries["node_modules/module.js"]).toBeDefined();
    } finally {
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("rejects extra, wrong-version, wrong-arch, and symlinked final DMGs", () => {
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-dmg-set-"));
    const sequoia = join(temp, "vMLX-1.6.20-sequoia-arm64.dmg");
    const tahoe = join(temp, "vMLX-1.6.20-tahoe-arm64.dmg");
    const expected = [sequoia, tahoe];
    const failure = "unexpected DMG set";
    try {
      writeFileSync(sequoia, "sequoia");
      writeFileSync(tahoe, "tahoe");
      expect(
        beforePack.verifyExactDmgDirectory(temp, expected, failure),
      ).toEqual(expected);

      for (const unexpectedName of [
        "rogue.dmg",
        "rogue.DMG",
        "vMLX-1.6.17-sequoia-arm64.dmg",
        "vMLX-1.6.20-tahoe-x64.dmg",
      ]) {
        const unexpected = join(temp, unexpectedName);
        writeFileSync(unexpected, "unexpected");
        expect(() =>
          beforePack.verifyExactDmgDirectory(temp, expected, failure),
        ).toThrow(failure);
        rmSync(unexpected);
      }

      const symlink = join(temp, "linked.dmg");
      symlinkSync(sequoia, symlink);
      expect(() =>
        beforePack.verifyExactDmgDirectory(temp, expected, failure),
      ).toThrow(failure);
    } finally {
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("keeps codesign authority distinct from the prefix-free electron-builder selector", () => {
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-signing-selector-"));
    const fullIdentity =
      "Developer ID Application: ShieldStack LLC (55KGF2S5AY)";
    const selector = "ShieldStack LLC (55KGF2S5AY)";
    const envNames = [
      "VMLX_RELEASE_SCOPE",
      "VMLX_R20_EXPECTED_VERSION",
      "VMLX_R20_OFFICIAL_PACKAGING",
      "VMLX_R20_EXPECTED_TEAM_ID",
      "VMLX_R20_EXPECTED_CODESIGN_IDENTITY",
      "VMLX_R20_PREPACKAGE_MANIFEST",
      "VMLX_R20_PREPACKAGE_MANIFEST_SHA256",
      "VMLX_R20_RELEASE_PLAN",
      "VMLX_R20_RELEASE_PLAN_SHA256",
      "CSC_NAME",
    ];
    const previous = Object.fromEntries(
      envNames.map((name) => [name, process.env[name]]),
    );
    for (const name of [
      "VMLX_R20_PREPACKAGE_MANIFEST",
      "VMLX_R20_PREPACKAGE_MANIFEST_SHA256",
      "VMLX_R20_RELEASE_PLAN",
      "VMLX_R20_RELEASE_PLAN_SHA256",
    ]) {
      delete process.env[name];
    }

    try {
      writeFileSync(
        join(temp, "package.json"),
        JSON.stringify({
          version: "1.6.20",
          build: { mac: { notarize: { teamId: "55KGF2S5AY" } } },
        }),
      );
      process.env.VMLX_RELEASE_SCOPE = "r20_production";
      process.env.VMLX_R20_EXPECTED_VERSION = "1.6.20";
      process.env.VMLX_R20_OFFICIAL_PACKAGING = "1";
      process.env.VMLX_R20_EXPECTED_TEAM_ID = "55KGF2S5AY";
      process.env.VMLX_R20_EXPECTED_CODESIGN_IDENTITY = fullIdentity;

      expect(beforePack.R20_CODESIGN_IDENTITY).toBe(fullIdentity);
      expect(beforePack.R20_CSC_NAME).toBe(selector);
      expect(beforePack.R20_CSC_NAME).not.toBe(
        beforePack.R20_CODESIGN_IDENTITY,
      );
      expect(beforePack.R20_CSC_NAME).not.toMatch(
        /^Developer ID Application:/,
      );

      for (const invalidSelector of [
        fullIdentity,
        ` ${selector}`,
        `${selector} `,
        "55KGF2S5AY",
        "",
      ]) {
        process.env.CSC_NAME = invalidSelector;
        expect(() =>
          beforePack.verifyR20PackagingContext(temp, {}),
        ).toThrow(`packaging requires CSC_NAME=${selector}`);
      }

      process.env.CSC_NAME = selector;
      expect(() =>
        beforePack.verifyR20PackagingContext(temp, {}),
      ).toThrow("requires the official prepackage manifest");

      process.env.VMLX_R20_EXPECTED_CODESIGN_IDENTITY = selector;
      expect(() =>
        beforePack.verifyR20PackagingContext(temp, {}),
      ).toThrow(
        `packaging requires VMLX_R20_EXPECTED_CODESIGN_IDENTITY=${fullIdentity}`,
      );
    } finally {
      for (const name of envNames) {
        const value = previous[name];
        if (value === undefined) {
          delete process.env[name];
        } else {
          process.env[name] = value;
        }
      }
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("pins the production release driver, public source, signing team, and dual flavors", () => {
    const pkg = JSON.parse(read("package.json"));
    const source = read("scripts/build-release-dmgs.sh");
    const beforePackSource = read("scripts/electron-builder-before-pack.cjs");

    expect(pkg.build.dmg.writeUpdateInfo).toBe(true);
    expect(source).toContain(
      'AUTHORITATIVE_PYTHON="$ROOT_DIR/.venv/bin/python"',
    );
    expect(source).toContain(
      "release Python imports vmlx_engine outside the release checkout",
    );
    expect(source).toContain("release Python prefix mismatch");
    expect(source).toContain("run_release_python -I -");
    expect(source).not.toContain('"$PYTHON_BIN" -I -');
    expect(source).toContain(
      'VMLX_R20_RELEASE_PYTHON_INIT_SHA256="$(toolchain_sha256 "$init_path")"',
    );
    expect(source).toContain(
      'VMLX_R20_RELEASE_PYTHON_SERVER_SHA256="$(toolchain_sha256 "$server_path")"',
    );
    expect(source).toContain(
      'VMLX_R20_RELEASE_PYTHON_EXECUTABLE_SHA256="$VMLX_R20_RELEASE_PYTHON_SOURCE_SHA256"',
    );
    expect(source).toContain("VMLX_R20_RELEASE_PYTHON_PYVENV_SHA256");
    expect(source).not.toContain("check-public-repo-hygiene.sh");
    expect(source).toContain('assert_r20_source_identity "before npm ci"');
    expect(source).toContain("canonical release repositories");
    expect(source).toContain('export VMLX_R20_EXPECTED_VERSION="$VERSION"');
    expect(source).toContain("parsed.scheme.lower()");
    expect(source).toContain('scheme == "https"');
    expect(source).toContain('scheme == "ssh"');
    expect(source).not.toContain('scheme == "git"');
    expect(source).toContain(
      're.sub(r"\\.git$", "", parsed.path.strip("/"), flags=re.IGNORECASE)',
    );
    expect(source).toContain('EXPECTED_APPLE_TEAM_ID="55KGF2S5AY"');
    expect(source).toContain(
      'EXPECTED_CODESIGN_IDENTITY="Developer ID Application: ShieldStack LLC (55KGF2S5AY)"',
    );
    expect(source).toContain(
      'EXPECTED_CSC_NAME="ShieldStack LLC (55KGF2S5AY)"',
    );
    expect(source).toContain('export CSC_NAME="$EXPECTED_CSC_NAME"');
    expect(source).not.toContain('export CSC_NAME="$EXPECTED_CODESIGN_IDENTITY"');
    expect(beforePackSource).toContain(
      'const R20_CSC_NAME = "ShieldStack LLC (55KGF2S5AY)"',
    );
    expect(beforePackSource).toContain(
      'requireExactEnv("CSC_NAME", R20_CSC_NAME)',
    );
    expect(beforePackSource).toContain(
      "module.exports.R20_CSC_NAME = R20_CSC_NAME",
    );
    expect(source).toContain(
      "production packaging must build both Sequoia and Tahoe via flavor=all",
    );
    expect(source).toContain("vMLX-${VERSION}-sequoia-arm64.dmg");
    expect(source).toContain("vMLX-${VERSION}-tahoe-arm64.dmg");
    expect(source).toContain("verifyExactDmgDirectory");
    expect(source).toContain('export VMLX_RELEASE_SCOPE="$RELEASE_SCOPE"');
    expect(source).not.toContain("npx electron-vite build");
    expect(beforePackSource).toContain("verifyPinnedToolchain(plan)");
    expect(beforePackSource).toContain("emitR20CompletionAttestation");
    expect(beforePackSource).toContain(
      '["--no-install", "electron-vite", "build"]',
    );
    expect(source).toContain(
      'R20_FIXED_PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"',
    );
    expect(source).toContain("write_r20_build_plan");
    expect(source).toContain("VMLX_R20_RELEASE_REQUESTED_FLAVOR");
    expect(source).toContain(
      'export VMLX_EXPECTED_MLX_WHEEL_PLATFORM="$R20_CURRENT_MLX_WHEEL_PLATFORM"',
    );
    expect(source).toContain("run_bound_release_action");
    expect(source).toContain("capture_bound_release_action");
    expect(source).toContain("run-bound-tool-action");
    expect(source).toContain("capture_toolchain_action git");
    expect(source).toContain("capture_toolchain_action file");
    expect(source).toContain("capture_toolchain_action find");
    expect(source).toContain("toolchain_sha256");
    expect(source).not.toMatch(
      /"\$(?:GIT|SHASUM|AWK|FILE|FIND)_REALPATH"/,
    );
    expect(source).not.toContain("run_pinned_node");
    expect(source).not.toContain("run_pinned_npm");
    expect(source).not.toContain("run_pinned_asar");
    expect(source).not.toContain("run_pinned_electron_builder");
    expect(beforePackSource).toContain("runPlanToolAction");
    expect(beforePackSource).not.toContain(
      "const git = plan.tools.git.realpath",
    );
    expect(beforePackSource).not.toContain(
      "npx = packaging.plan.tools.npx.realpath",
    );
    expect(beforePackSource).not.toContain(
      "run(plan.tools.node.realpath",
    );
  });

  it("ignores only the verified release Python action hardlink in git status", () => {
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );
    const temp = mkdtempSync(join(tmpdir(), "vmlx-r20-status-action-"));
    const runnerDir = join(temp, "tests", "cross_matrix");
    const runner = join(runnerDir, "run_packaged_integrity_contract.py");
    const nonce = "a".repeat(32);
    const actionRelative =
      `tests/cross_matrix/.run_packaged_integrity_contract.py.vmlx-r20-${nonce}`;
    const action = join(temp, actionRelative);
    try {
      mkdirSync(runnerDir, { recursive: true });
      writeFileSync(runner, "print('release gate')\n");
      linkSync(runner, action);

      expect(
        beforePack.stripVerifiedReleasePythonActionFromGitStatus(
          temp,
          `?? ${actionRelative}`,
        ),
      ).toBe("");
      expect(
        beforePack.stripVerifiedReleasePythonActionFromGitStatus(
          temp,
          ` M panel/package.json\n?? ${actionRelative}`,
        ),
      ).toBe(" M panel/package.json");

      rmSync(action);
      writeFileSync(action, "not the runner\n");
      expect(() =>
        beforePack.stripVerifiedReleasePythonActionFromGitStatus(
          temp,
          `?? ${actionRelative}`,
        ),
      ).toThrow("is not the verified runner hardlink");
      expect(
        beforePack.stripVerifiedReleasePythonActionFromGitStatus(
          temp,
          "?? unrelated.txt",
        ),
      ).toBe("?? unrelated.txt");
    } finally {
      rmSync(temp, { recursive: true, force: true });
    }
  });

  it("digest-binds manifest tools and safely creates fresh notary result directories", () => {
    const verifySource = read("scripts/verify-release-dmgs.sh");
    const notarySource = read("scripts/notarize-release-dmgs.sh");

    expect(verifySource).toContain("run-bound-tool-action");
    expect(verifySource).toContain("--expected-document-sha256");
    expect(verifySource).not.toContain("resolve-manifest-tool");
    expect(verifySource).not.toContain("manifest_tool_path");
    expect(verifySource).not.toContain("app_builder_bin");
    expect(notarySource).toContain("run_manifest_app_builder");
    expect(notarySource).not.toContain("app_builder_bin");
    expect(notarySource).toContain("artifact_chain create-private-directory");
    expect(notarySource).toContain("artifact_chain capture-private-command");
    expect(notarySource).toContain(
      "set VMLX_NOTARY_KEYCHAIN_PROFILE from the private release environment",
    );
    expect(verifySource).toContain(
      "set VMLX_NOTARY_KEYCHAIN_PROFILE from the private release environment",
    );
    expect(notarySource).not.toContain("vmlx-notary");
    expect(verifySource).not.toContain("vmlx-notary");
    expect(notarySource).not.toContain('mkdir -p "$result_dir"');
    expect(notarySource).not.toContain('chmod 0700 "$result_dir"');
    expect(notarySource).not.toContain('parent="$(dirname "$output")"');
    expect(notarySource).not.toContain('mkdir -p "$parent"');
    expect(notarySource).not.toContain('chmod 0700 "$parent"');
    expect(notarySource).not.toContain('exec 9>"$temporary"');
    expect(notarySource).not.toContain("artifact_chain seal-capture");
    expect(notarySource).toContain(
      'sequoia_id="$(json_file_path "$result_dir/sequoia.submit.json" id)"',
    );
    expect(notarySource).toContain(
      'tahoe_id="$(json_file_path "$result_dir/tahoe.submit.json" id)"',
    );
    expect(notarySource).not.toContain('sequoia_id="$(\n    capture_apple_records');
    expect(notarySource).not.toContain('tahoe_id="$(\n    capture_apple_records');
    expect(verifySource).toContain('background_entries[0].name != "1.tiff"');
    expect(verifySource).toContain('".fseventsd",');
    expect(verifySource).toContain(
      're.fullmatch(r"[0-9a-f]{16}", name)',
    );
    expect(verifySource).toContain("str(uuid.UUID(fsevent_uuid))");
  });

  it("binds release Python independently and rejects mutation, alias swap, and action swap", () => {
    const helper = requireCjs(
      join(repo, "scripts/release-python-action.cjs"),
    );
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );
    const oldEnv = { ...process.env };

    const makeVenv = (name: string, body: string) => {
      const root = mkdtempSync(join(tmpdir(), `vmlx-python-${name}-`));
      const bin = join(root, ".venv", "bin");
      mkdirSync(bin, { recursive: true });
      writeFileSync(join(root, ".venv", "pyvenv.cfg"), "home = /usr/bin\n");
      const source = join(bin, "python-real");
      writeFileSync(source, body);
      chmodSync(source, 0o755);
      const alias = join(bin, "python");
      symlinkSync(source, alias);
      return { root, source, alias };
    };

    try {
      process.env.PATH = helper.FIXED_PATH;

      const mutation = makeVenv(
        "mutation",
        '#!/bin/sh\nprintf "# mutation\\n" >>"$0"\nexit 0\n',
      );
      const mutationBinding = helper.bindReleasePython(mutation.alias);
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN = mutationBinding.planPath;
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN_SHA256 =
        mutationBinding.planSha256;
      expect(() =>
        helper.runPinnedReleasePythonAction([], { cwd: mutation.root }),
      ).toThrow(/release Python (source|action).*changed/);

      const aliasSwap = makeVenv(
        "alias-swap",
        "#!/bin/sh\nexit 0\n",
      );
      writeFileSync(
        aliasSwap.source,
        "#!/bin/sh\n" +
          `/bin/rm -f ${JSON.stringify(aliasSwap.alias)}\n` +
          `/bin/ln -s /bin/sh ${JSON.stringify(aliasSwap.alias)}\n` +
          "exit 0\n",
      );
      chmodSync(aliasSwap.source, 0o755);
      const aliasBinding = helper.bindReleasePython(aliasSwap.alias);
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN = aliasBinding.planPath;
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN_SHA256 =
        aliasBinding.planSha256;
      expect(() =>
        helper.runPinnedReleasePythonAction([], { cwd: aliasSwap.root }),
      ).toThrow(/binding plan or alias changed/);

      const actionSwap = makeVenv(
        "action-swap",
        "#!/bin/sh\n" +
          'original="$0.original"\n' +
          '/bin/mv "$0" "$original"\n' +
          'printf "#!/bin/sh\\nexit 0\\n" >"$0"\n' +
          '/bin/chmod 755 "$0"\n' +
          "exit 0\n",
      );
      const actionBinding = helper.bindReleasePython(actionSwap.alias);
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN = actionBinding.planPath;
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN_SHA256 =
        actionBinding.planSha256;
      expect(() =>
        helper.runPinnedReleasePythonAction([], { cwd: actionSwap.root }),
      ).toThrow(/release Python action.*changed/);

      rmSync(mutation.root, { recursive: true, force: true });
      rmSync(aliasSwap.root, { recursive: true, force: true });
      rmSync(actionSwap.root, { recursive: true, force: true });
    } finally {
      for (const key of Object.keys(process.env)) {
        if (!(key in oldEnv)) delete process.env[key];
      }
      Object.assign(process.env, oldEnv);
    }
  });

  it("sanitizes Python environment and fails closed on hostile PATH or hardlink failure", () => {
    const helper = requireCjs(
      join(repo, "scripts/release-python-action.cjs"),
    );
    const oldEnv = { ...process.env };
    const root = mkdtempSync(join(tmpdir(), "vmlx-python-env-"));
    const bin = join(root, ".venv", "bin");
    mkdirSync(bin, { recursive: true });
    writeFileSync(join(root, ".venv", "pyvenv.cfg"), "home = /usr/bin\n");
    const source = join(bin, "python-real");
    writeFileSync(
      source,
      "#!/bin/sh\n" +
        '[ -z "${PYTHONPATH:-}" ] || exit 17\n' +
        `[ "$PATH" = ${JSON.stringify(helper.FIXED_PATH)} ] || exit 18\n` +
        'printf "sanitized\\n"\n',
    );
    chmodSync(source, 0o755);
    const alias = join(bin, "python");
    symlinkSync(source, alias);

    try {
      process.env.PATH = helper.FIXED_PATH;
      const binding = helper.bindReleasePython(alias);
      expect(dirname(binding.actionPath)).toBe(dirname(realpathSync(source)));
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN = binding.planPath;
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN_SHA256 =
        binding.planSha256;
      process.env.PYTHONPATH = "/private/hostile";
      expect(
        helper.runPinnedReleasePythonAction([], {
          cwd: root,
          capture: true,
        }),
      ).toBe("sanitized\n");

      process.env.PATH = "/private/hostile";
      expect(() =>
        helper.runPinnedReleasePythonAction([], { cwd: root }),
      ).toThrow(/binding plan or alias changed/);
      process.env.PATH = helper.FIXED_PATH;
      helper.cleanupReleasePythonBinding();

      const crossDevice = Object.assign(new Error("cross-device"), {
        code: "EXDEV",
      });
      expect(() =>
        helper.bindReleasePython(alias, {
          link() {
            throw crossDevice;
          },
        }),
      ).toThrow(/cross-device/);
    } finally {
      rmSync(root, { recursive: true, force: true });
      for (const key of Object.keys(process.env)) {
        if (!(key in oldEnv)) delete process.env[key];
      }
      Object.assign(process.env, oldEnv);
    }
  });

  it("preserves the macOS standalone-Python loader and authoritative venv identity", () => {
    if (process.platform !== "darwin") return;
    const helper = requireCjs(
      join(repo, "scripts/release-python-action.cjs"),
    );
    const python = join(repo, "..", ".venv", "bin", "python");
    const expectedPrefix = join(repo, "..", ".venv");
    const oldEnv = { ...process.env };
    let bound = false;
    try {
      process.env.PATH = helper.FIXED_PATH;
      const inherited = helper.bindingFromEnvironment();
      const hasInherited = Boolean(
        inherited.planPath && inherited.expectedSha256,
      );
      let binding;
      if (hasInherited) {
        const plan = JSON.parse(readFileSync(inherited.planPath, "utf8"));
        binding = { actionPath: plan.action.path };
      } else {
        binding = helper.bindReleasePython(python);
        bound = true;
        process.env.VMLX_R20_RELEASE_PYTHON_PLAN = binding.planPath;
        process.env.VMLX_R20_RELEASE_PYTHON_PLAN_SHA256 =
          binding.planSha256;
      }
      expect(dirname(binding.actionPath)).toBe(
        dirname(realpathSync(python)),
      );
      const probe = JSON.parse(
        helper.runPinnedReleasePythonAction(
          [
            "-c",
            "import json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix}))",
          ],
          { cwd: join(repo, ".."), capture: true },
        ),
      );
      expect(probe.executable).toBe(python);
      expect(probe.prefix).toBe(expectedPrefix);
    } finally {
      if (bound) helper.cleanupReleasePythonBinding();
      for (const key of Object.keys(process.env)) {
        if (!(key in oldEnv)) delete process.env[key];
      }
      Object.assign(process.env, oldEnv);
    }
  });

  it("rejects executable and script swaps at the final pre-spawn boundary", () => {
    const helper = requireCjs(
      join(repo, "scripts/release-python-action.cjs"),
    );
    const beforePack = requireCjs(
      join(repo, "scripts/electron-builder-before-pack.cjs"),
    );
    const oldEnv = { ...process.env };
    const temp = mkdtempSync(join(tmpdir(), "vmlx-python-pre-spawn-"));
    const venv = join(temp, ".venv");
    const bin = join(venv, "bin");
    const python = join(bin, "python");
    const marker = join(temp, "executed");
    mkdirSync(bin, { recursive: true });
    writeFileSync(
      python,
      `#!/bin/sh\nprintf executed >${JSON.stringify(marker)}\nexit 0\n`,
    );
    chmodSync(python, 0o755);
    writeFileSync(join(venv, "pyvenv.cfg"), "home = /private/fake\n");

    try {
      process.env.PATH = helper.FIXED_PATH;
      const executableBinding = helper.bindReleasePython(python);
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN =
        executableBinding.planPath;
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN_SHA256 =
        executableBinding.planSha256;
      expect(() =>
        helper.runPinnedReleasePythonAction([], {
          cwd: temp,
          beforeSpawn: ({ plan }: any) => {
            rmSync(plan.action.path);
            writeFileSync(
              plan.action.path,
              `#!/bin/sh\nprintf attacker >${JSON.stringify(marker)}\n`,
            );
            chmodSync(plan.action.path, 0o755);
          },
        }),
      ).toThrow(/release Python action.*changed|hardlink/);
      expect(() => readFileSync(marker)).toThrow();

      rmSync(temp, { recursive: true, force: true });
      mkdirSync(bin, { recursive: true });
      writeFileSync(
        python,
        `#!/bin/sh\nprintf executed >${JSON.stringify(marker)}\nexit 0\n`,
      );
      chmodSync(python, 0o755);
      writeFileSync(join(venv, "pyvenv.cfg"), "home = /private/fake\n");
      const scriptBinding = helper.bindReleasePython(python);
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN = scriptBinding.planPath;
      process.env.VMLX_R20_RELEASE_PYTHON_PLAN_SHA256 =
        scriptBinding.planSha256;
      const script = join(temp, "gate.py");
      writeFileSync(script, "print('gate')\n");
      expect(() =>
        beforePack.runR20ReleasePythonAction([script], {
          cwd: temp,
          beforeSpawn: ({ script: scriptRecord }: any) => {
            rmSync(scriptRecord.action.path);
            writeFileSync(scriptRecord.action.path, "print('attacker')\n");
          },
        }),
      ).toThrow(/release Python script action/);
      expect(() => readFileSync(marker)).toThrow();
      helper.cleanupReleasePythonBinding();
    } finally {
      rmSync(temp, { recursive: true, force: true });
      for (const key of Object.keys(process.env)) {
        if (!(key in oldEnv)) delete process.env[key];
      }
      Object.assign(process.env, oldEnv);
    }
  });

  it("feeds retained prepackage evidence and exact provenance to the production build", () => {
    const driver = readFileSync(
      join(repo, "scripts/build-release-dmgs.sh"),
      "utf8",
    );
    expect(driver).toContain("--require-prepackage-ready");
    expect(driver).toContain("--require-production-provenance");
    expect(driver).toContain('--scope "$RELEASE_SCOPE"');
    expect(driver).toContain('--jang-source "$VMLX_JANG_TOOLS_SOURCE"');
    expect(driver).toContain("obsolete VMLX_R20_RELEASE_ATTESTATION is forbidden");
    expect(driver).not.toContain("--consume-v5-manifest");
  });
});
