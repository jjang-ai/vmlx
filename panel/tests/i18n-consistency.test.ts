/**
 * i18n consistency regression guard.
 *
 * Locks invariants across every locale file:
 *   1. Valid JSON.
 *   2. Identical key tree — every key present in en.json exists in every
 *      other locale (missing keys fall back to en at runtime, but the
 *      contract is that translations are complete).
 *   3. Interpolation placeholders ({n}, {{count}}, {name}, etc.) are
 *      preserved in translations — missing a placeholder would produce a
 *      silently-wrong UI string.
 *   4. No empty / whitespace-only string values.
 *
 * This test must pass before any release. If you add a new key to en.json,
 * add the translation to every locale file or this test fails.
 */

import { describe, it, expect } from 'vitest'
import { existsSync, readFileSync } from 'fs'
import { resolve } from 'path'

const LOCALES = ['en', 'zh', 'ko', 'ja', 'es'] as const
type LocaleKey = (typeof LOCALES)[number]
const EXPECTED_MAX_THINKING_TOKENS_HELP: Record<LocaleKey, string> = {
  en: "Leave blank to preserve the model/runtime native default and the request's full output budget. Only an explicit value in this UI or API max_thinking_tokens sets a separate reasoning cap. Models without thinking-budget support may ignore it.",
  zh: '留空可保留模型/运行时的原生默认行为和请求的完整输出预算。只有在此界面中输入显式值或在 API 中设置 max_thinking_tokens，才会设置单独的推理上限。不支持思考预算的模型可能会忽略它。',
  ko: '비워 두면 모델/런타임의 기본 동작과 요청의 전체 출력 예산이 유지됩니다. 이 UI에 명시적 값을 입력하거나 API에서 max_thinking_tokens를 설정한 경우에만 별도의 추론 한도가 적용됩니다. 사고 예산을 지원하지 않는 모델은 이를 무시할 수 있습니다.',
  ja: '空欄のままにすると、モデル/ランタイムのネイティブ既定値とリクエストの全出力予算が維持されます。この UI で明示的な値を入力するか、API で max_thinking_tokens を指定した場合にのみ、別個の推論上限が設定されます。思考予算に対応していないモデルでは無視される場合があります。',
  es: 'Déjelo vacío para conservar el valor nativo predeterminado del modelo/runtime y el presupuesto de salida completo de la solicitud. Solo un valor explícito en esta interfaz o max_thinking_tokens en la API establece un límite de razonamiento independiente. Los modelos que no admiten un presupuesto de pensamiento pueden ignorarlo.',
}
const EXPECTED_TOP_P_HELP: Record<LocaleKey, string> = {
  en: 'Nucleus sampling. Keeps the smallest set of highest-probability tokens whose cumulative probability reaches this threshold.',
  zh: '核采样。按概率从高到低保留累计概率达到该阈值所需的最小 token 集合。',
  ko: '핵 샘플링. 누적 확률이 이 임계값에 도달할 때까지 확률이 높은 토큰의 최소 집합을 사용합니다.',
  ja: '核サンプリング。累積確率がこの閾値に達するまで、確率の高い順に最小限のトークン集合を使用します。',
  es: 'Muestreo por núcleo. Conserva el conjunto más pequeño de tokens de mayor probabilidad cuya probabilidad acumulada alcanza este umbral.',
}
const LOCALE_DIR = resolve(
  __dirname,
  '..',
  'src',
  'renderer',
  'src',
  'i18n',
  'locales',
)

function loadLocale(locale: LocaleKey): Record<string, any> {
  const path = resolve(LOCALE_DIR, `${locale}.json`)
  const raw = readFileSync(path, 'utf-8')
  return JSON.parse(raw)
}

function flatten(obj: Record<string, any>, prefix = ''): Record<string, string> {
  const out: Record<string, string> = {}
  for (const [k, v] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${k}` : k
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      Object.assign(out, flatten(v, path))
    } else if (typeof v === 'string') {
      out[path] = v
    }
  }
  return out
}

function extractPlaceholders(value: string): Set<string> {
  // Match both {var} and {{var}} interpolation forms. Returns bare names.
  const out = new Set<string>()
  const matches = value.matchAll(/\{\{?(\w+)\}?\}/g)
  for (const m of matches) {
    out.add(m[1])
  }
  return out
}

describe('i18n locale consistency', () => {
  const locales = Object.fromEntries(
    LOCALES.map((l) => [l, loadLocale(l)]),
  ) as Record<LocaleKey, Record<string, any>>
  const flat = Object.fromEntries(
    LOCALES.map((l) => [l, flatten(locales[l])]),
  ) as Record<LocaleKey, Record<string, string>>

  it('every locale parses as valid JSON', () => {
    for (const l of LOCALES) {
      expect(typeof locales[l]).toBe('object')
      expect(locales[l]).not.toBeNull()
    }
  })

  it('every locale has the same key set as en', () => {
    const enKeys = Object.keys(flat.en).sort()
    for (const l of LOCALES) {
      if (l === 'en') continue
      const lKeys = Object.keys(flat[l]).sort()
      const missing = enKeys.filter((k) => !(k in flat[l]))
      const extra = lKeys.filter((k) => !(k in flat.en))
      expect(
        missing,
        `Locale '${l}' is MISSING these keys present in en.json:\n  ${missing.join('\n  ')}`,
      ).toEqual([])
      expect(
        extra,
        `Locale '${l}' has EXTRA keys not present in en.json (add to en first):\n  ${extra.join('\n  ')}`,
      ).toEqual([])
    }
  })

  it('all interpolation placeholders are preserved in every translation', () => {
    for (const key of Object.keys(flat.en)) {
      const enPlaceholders = extractPlaceholders(flat.en[key])
      if (enPlaceholders.size === 0) continue
      for (const l of LOCALES) {
        if (l === 'en') continue
        const translated = flat[l][key]
        if (!translated) continue
        const translatedPlaceholders = extractPlaceholders(translated)
        const missing = [...enPlaceholders].filter(
          (p) => !translatedPlaceholders.has(p),
        )
        expect(
          missing,
          `Locale '${l}' lost placeholder(s) {${missing.join(',')}} on key '${key}'. en: "${flat.en[key]}" | ${l}: "${translated}"`,
        ).toEqual([])
      }
    }
  })

  it('no string value is empty or whitespace-only', () => {
    for (const l of LOCALES) {
      for (const [key, val] of Object.entries(flat[l])) {
        expect(
          val.trim().length,
          `Locale '${l}' has empty / whitespace-only value at key '${key}'`,
        ).toBeGreaterThan(0)
      }
    }
  })

  it('describes Max Thinking Tokens as an explicit cap without an implicit Auto partition', () => {
    for (const l of LOCALES) {
      expect(locales[l].chat.settings.maxThinkingTokensHelp).toBe(
        EXPECTED_MAX_THINKING_TOKENS_HELP[l],
      )
      expect(locales[l].chat.settings.maxThinkingTokensHelp).not.toMatch(
        /Qwen|3[.]5|3[.]6|1,?024|256/,
      )
    }
  })

  it('describes Top P as a highest-probability nucleus, not tokens above the threshold', () => {
    for (const l of LOCALES) {
      expect(locales[l].chat.settings.topPHelp).toBe(EXPECTED_TOP_P_HELP[l])
    }
  })

  it('LOCALES array in i18n/catalog.ts stays in sync with locale files', () => {
    // If you add a new JSON locale file, also add it to the Locale type
    // and LOCALES / LOCALE_NAMES / LOCALE_FLAGS in i18n/catalog.ts.
    const catalogSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'i18n', 'catalog.ts'),
      'utf-8',
    )
    for (const l of LOCALES) {
      expect(
        catalogSrc.includes(`'${l}'`) || catalogSrc.includes(`"${l}"`),
        `Locale '${l}' JSON exists but is not registered in i18n/catalog.ts`,
      ).toBe(true)
    }
  })

  it('exposes one authoritative locale/key catalog to the visible picker and CDP gate', () => {
    const mainI18n = readFileSync(
      resolve(__dirname, '..', 'src', 'main', 'i18n.ts'),
      'utf8',
    )
    const mainIndex = readFileSync(
      resolve(__dirname, '..', 'src', 'main', 'index.ts'),
      'utf8',
    )
    const preload = readFileSync(
      resolve(__dirname, '..', 'src', 'preload', 'index.ts'),
      'utf8',
    )
    const titleBar = readFileSync(
      resolve(
        __dirname,
        '..',
        'src',
        'renderer',
        'src',
        'components',
        'layout',
        'TitleBar.tsx',
      ),
      'utf8',
    )
    const preflight = readFileSync(
      resolve(__dirname, '..', 'scripts', 'scoped-release-preflight-20.py'),
      'utf8',
    )
    expect(Object.keys(flat.en)).toContain('common.stream')
    expect(mainI18n).toContain('export function getCatalogContract()')
    expect(mainIndex).toContain("ipcMain.handle('i18n:get-catalog-contract'")
    expect(preload).toContain("ipcRenderer.invoke('i18n:get-catalog-contract')")
    expect(titleBar).toContain('LOCALES.map((l) =>')
    expect(titleBar).toContain('data-vmlx-locale-picker="true"')
    expect(titleBar).toContain('data-vmlx-locale-option={l}')
    expect(preflight).toContain('getCatalogContract')
    expect(preflight).toContain('main_ipc_canonical_locale_json')
    expect(preflight).toContain(
      'translationKeys.map((key)=>key.toLowerCase())',
    )
    expect(preflight).toContain('canonicalKeySet.has(token.toLowerCase())')
    expect(preflight).toContain('Emulation.clearDeviceMetricsOverride')
    expect(preflight).not.toContain("const specs=[")
  })

  it('language switcher persists to localStorage key vmlx-locale', () => {
    // Verifies the persistence contract both language pickers depend on.
    const providerSrc = readFileSync(
      resolve(
        __dirname,
        '..',
        'src',
        'renderer',
        'src',
        'i18n',
        'I18nProvider.tsx',
      ),
      'utf-8',
    )
    expect(providerSrc).toMatch(/["']vmlx-locale["']/)
    expect(providerSrc).toContain('localStorage.setItem')
    expect(providerSrc).toContain('localStorage.getItem')
  })

  it('keeps the React context stable across HMR and isolates non-component exports', () => {
    const contextSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'i18n', 'context.ts'),
      'utf-8',
    )
    const providerSrc = readFileSync(
      resolve(
        __dirname,
        '..',
        'src',
        'renderer',
        'src',
        'i18n',
        'I18nProvider.tsx',
      ),
      'utf-8',
    )
    const catalogSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'i18n', 'catalog.ts'),
      'utf-8',
    )

    expect(contextSrc).toMatch(
      /Symbol[.]for[(]["']net[.]vmlx[.]renderer[.]i18n-context["'][)]/,
    )
    expect(contextSrc).toContain('registry[I18N_CONTEXT_GLOBAL] = I18nContext')
    expect(providerSrc).not.toContain('LOCALE_FLAGS')
    expect(providerSrc).not.toContain('LOCALE_NAMES')
    expect(catalogSrc).not.toContain('createContext')
  })

  it('session creation and narrow session drawers use the shared locale catalog', () => {
    const createSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'components', 'sessions', 'CreateSession.tsx'),
      'utf-8',
    )
    const drawerSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'components', 'sessions', 'ServerSettingsDrawer.tsx'),
      'utf-8',
    )
    const viewSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'components', 'sessions', 'SessionView.tsx'),
      'utf-8',
    )
    const chatToolbarSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'components', 'layout', 'ChatModeToolbar.tsx'),
      'utf-8',
    )

    for (const key of [
      'sessions.create.title',
      'sessions.create.localModel',
      'sessions.create.remoteEndpoint',
      'sessions.create.filterPlaceholder',
      'sessions.create.launchSession',
    ]) {
      expect(createSrc).toContain(`t('${key}')`)
    }
    for (const key of [
      'sessions.view.serverSettingsTitle',
      'sessions.settings.runningWarning',
      'sessions.settings.saveAndRestart',
      'common.close',
    ]) {
      expect(drawerSrc).toContain(`t('${key}')`)
    }
    for (const key of [
      'sessions.view.chatButton',
      'sessions.view.server',
      'sessions.view.logs',
      'sessions.view.wake',
      'sessions.view.loadingSession',
      'sessions.view.cacheTitle',
      'sessions.view.benchPanelTitle',
      'sessions.view.embeddingsPanelTitle',
      'sessions.view.performancePanelTitle',
    ]) {
      expect(viewSrc).toContain(`t('${key}')`)
    }
    for (const key of [
      'sessions.view.chatLabel',
      'sessions.view.server',
      'sessions.toolbar.startModel',
      'sessions.toolbar.stopModel',
      'layout.chatToolbar.addLocalModel',
      'layout.chatToolbar.connectRemoteEndpoint',
      'layout.chatToolbar.apiUrlPlaceholder',
    ]) {
      expect(chatToolbarSrc).toContain(`t('${key}')`)
    }
  })

  it('chat history, inference mode, message actions, voice, and API keys use the locale catalog', () => {
    const chatHistorySrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'components', 'layout', 'ChatHistory.tsx'),
      'utf-8',
    )
    const inferenceModeSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'components', 'layout', 'InferenceMode.tsx'),
      'utf-8',
    )
    const messageBubbleSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'components', 'chat', 'MessageBubble.tsx'),
      'utf-8',
    )
    const voiceChatSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'components', 'chat', 'VoiceChat.tsx'),
      'utf-8',
    )
    const appSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'App.tsx'),
      'utf-8',
    )

    for (const key of [
      'layout.chatHistory.clearAll',
      'layout.chatHistory.clearAllConfirm',
      'layout.chatHistory.chatCountHint',
    ]) {
      expect(chatHistorySrc).toContain(`t('${key}'`)
    }
    for (const key of [
      'layout.inferenceMode.casual',
      'layout.inferenceMode.expert',
      'layout.inferenceMode.casualTitle',
      'layout.inferenceMode.expertTitle',
    ]) {
      expect(inferenceModeSrc).toContain(`t('${key}')`)
    }
    for (const key of [
      'chat.bubble.editTitle',
      'chat.bubble.copyTitle',
      'chat.bubble.regenerateTitle',
      'chat.bubble.codeCopy',
      'chat.bubble.codeCopied',
      'chat.bubble.waitingForResponse',
      'chat.bubble.noVisibleResponse',
    ]) {
      expect(messageBubbleSrc).toContain(`t('${key}')`)
    }
    for (const key of [
      'chat.voice.stopRecordingTitle',
      'chat.voice.transcribingTitle',
      'chat.voice.startRecordingTitle',
      'chat.voice.sttMissing',
      'chat.voice.micDenied',
      'chat.voice.startFailed',
      'chat.voice.transcribeFailed',
      'chat.tts.stopTitle',
      'chat.tts.readAloudTitle',
      'chat.tts.speak',
    ]) {
      expect(voiceChatSrc).toContain(`t('${key}')`)
    }
    for (const key of [
      'app.about.version',
      'app.about.apiKeysTitle',
      'app.about.braveKey',
      'app.about.hfToken',
      'app.code.title',
      'app.code.description',
      'app.code.comingSoon',
      'chat.quickStart.title',
      'chat.quickStart.tipSettings',
    ]) {
      expect(appSrc).toContain(`t('${key}'`)
    }
  })

  it('voice icon controls expose localized accessible names', () => {
    const voiceChatSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'components', 'chat', 'VoiceChat.tsx'),
      'utf-8',
    )

    expect(voiceChatSrc).toContain('title={buttonLabel}')
    expect(voiceChatSrc).toContain('aria-label={buttonLabel}')
  })

  it('chat settings agentic controls and accessibility labels use the locale catalog', () => {
    const chatSettingsSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'components', 'chat', 'ChatSettings.tsx'),
      'utf-8',
    )

    for (const key of [
      'common.close',
      'chat.settings.wireCompletions',
      'chat.settings.wireResponses',
      'chat.settings.maxToolIterations',
      'chat.settings.enableBuiltinTools',
      'chat.settings.workingDirectory',
      'chat.settings.browse',
      'chat.settings.toolCategories',
      'chat.settings.toolFileIO',
      'chat.settings.toolSearch',
      'chat.settings.toolShell',
      'chat.settings.toolWebSearch',
      'chat.settings.toolUrlFetch',
      'chat.settings.toolGit',
      'chat.settings.toolUtilities',
      'chat.settings.toolResultLimit',
      'chat.settings.hideToolStatus',
      'chat.settings.reset',
      'chat.settings.braveSearch',
    ]) {
      expect(chatSettingsSrc).toContain(`t('${key}'`)
    }
    for (const staleLiteral of [
      'label="Max Tool Iterations"',
      '>Enable Built-in Coding Tools<',
      '>Working Directory<',
      '>Tool Categories<',
      '>Hide Tool Status<',
      '>Reset<',
    ]) {
      expect(chatSettingsSrc).not.toContain(staleLiteral)
    }
  })

  it('removes the obsolete release modal and localizes the active update banner', () => {
    const appSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'App.tsx'),
      'utf-8',
    )
    const updateNoticePath = resolve(
      __dirname,
      '..',
      'src',
      'renderer',
      'src',
      'components',
      'UpdateNotice.tsx',
    )
    const updateBannerSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'components', 'UpdateBanner.tsx'),
      'utf-8',
    )

    expect(existsSync(updateNoticePath)).toBe(false)
    expect(appSrc).not.toContain('UpdateNotice')
    expect(appSrc).toMatch(
      /import\s+\{\s*UpdateBanner\s*\}\s+from\s+['"][^'"]*components\/UpdateBanner['"]/,
    )
    expect(appSrc).toMatch(/<UpdateBanner\s*\/>/)
    expect(updateBannerSrc).toContain('useTranslation')
    expect(updateBannerSrc).toContain('window.api.app.onUpdateAvailable')
    expect(updateBannerSrc).toContain('return unsub')
    expect(updateBannerSrc).toContain(
      "localStorage.getItem('vmlx-dismissed-update')",
    )
    expect(updateBannerSrc).toContain('prev !== data.latestVersion')
    expect(updateBannerSrc).toContain(
      "localStorage.setItem('vmlx-dismissed-update', update.latestVersion)",
    )
    for (const key of [
      'update.banner.versionLabel',
      'update.banner.available',
      'update.banner.download',
      'update.banner.starOnGitHub',
      'update.banner.dismissTitle',
    ]) {
      expect(updateBannerSrc).toContain(`t('${key}'`)
    }
    for (const staleLiteral of [
      "CURRENT_NOTICE_VERSION = '1.5.45'",
      '>Download<',
      '>Star on GitHub<',
      'title="Dismiss"',
    ]) {
      expect(updateBannerSrc).not.toContain(staleLiteral)
    }

    for (const l of LOCALES) {
      const banner = locales[l].update.banner
      expect(banner.versionLabel).toContain('{version}')
      expect(banner.available.trim()).not.toBe('')
      expect(banner.download.trim()).not.toBe('')
      expect(banner.starOnGitHub.trim()).not.toBe('')
      expect(locales[l].update.notice).toBeUndefined()
    }
  })

  it('keeps the bundle-specific missing-template warning current and localized', () => {
    const sessionViewSrc = readFileSync(
      resolve(
        __dirname,
        '..',
        'src',
        'renderer',
        'src',
        'components',
        'sessions',
        'SessionView.tsx',
      ),
      'utf-8',
    )
    const modelsSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'main', 'ipc', 'models.ts'),
      'utf-8',
    )

    expect(sessionViewSrc).toContain(
      'setMissingChatTemplate(match?.hasChatTemplate === false)',
    )
    expect(sessionViewSrc).toContain(
      'const [missingTemplateNoticeDismissed, setMissingTemplateNoticeDismissed] = useState(false)',
    )
    expect(sessionViewSrc).toContain('setMissingTemplateNoticeDismissed(false)')
    expect(sessionViewSrc).toContain(
      'missingChatTemplate && !missingTemplateNoticeDismissed',
    )
    expect(sessionViewSrc).toContain(
      'onClick={() => setMissingTemplateNoticeDismissed(true)}',
    )
    expect(modelsSrc).toContain('export async function hasUsableChatTemplate')
    expect(sessionViewSrc).not.toContain('jang_redownload_dismissed')
    expect(sessionViewSrc).not.toContain('2026-03-19')

    for (const l of LOCALES) {
      const copy = [
        locales[l].sessions.view.missingChatTemplateHeader,
        locales[l].sessions.view.missingChatTemplateBody,
      ].join('\n')
      expect(copy).not.toContain('2026')
      expect(copy).not.toContain('HuggingFace')
      expect(copy.trim()).not.toBe('')
    }
  })

  it('keeps retired Swift migration UI and stale release-notice proof suppression removed', () => {
    const appSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'App.tsx'),
      'utf-8',
    )
    const swiftMigrationBannerPath = resolve(
      __dirname,
      '..',
      'src',
      'renderer',
      'src',
      'components',
      'layout',
      'SwiftMigrationBanner.tsx',
    )
    const proofScriptPaths = [
      'live-cache-restore-proof.mjs',
      'live-gemma4-media-stress-proof.mjs',
      'live-metal-headroom-ui-proof.mjs',
      'live-mm3-stress-proof.mjs',
    ].map((name) => resolve(__dirname, '..', 'scripts', name))

    expect(existsSync(swiftMigrationBannerPath)).toBe(false)
    expect(appSrc).not.toContain('SwiftMigrationBanner')

    for (const scriptPath of proofScriptPaths) {
      const scriptSrc = readFileSync(scriptPath, 'utf-8')
      expect(scriptSrc).not.toContain('notice_dismissed_version')
      expect(scriptSrc).not.toContain('suppressUpdateNotice')
      expect(scriptSrc).not.toContain('noticeDismissed')
    }
  })
})
