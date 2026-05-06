// Strings.swift
// vMLXApp — §349 i18n catalog
//
// SINGLE SOURCE OF TRUTH for translated user-facing strings in the
// vMLX macOS app. Every entry is a `L10nEntry` constructed with all
// four required labels (en/ja/ko/zh). Omitting any label is a
// compile error — that's the whole enforcement mechanism for §349.
//
// Rules for adding a string:
//   1. Group it under the nearest existing `enum` namespace below.
//      Create a new nested enum if no group fits.
//   2. Use a lowerCamelCase identifier describing the role, not the
//      English phrase — e.g. `sendButton`, not `sendText`.
//   3. Fill in all four translations. If a translation is uncertain,
//      ask a native speaker / use a translation memory tool; do NOT
//      copy the English into other locales "as a placeholder".
//   4. At the call site, render with the current locale:
//        Text(L10n.Chat.sendButton.render(appLocale))
//      where `appLocale` comes from `@Environment(\.appLocale)`.
//
// The seed catalog below covers high-traffic UI: tab names, the
// chat composer, common dialog buttons, menu bar entries, and the
// language picker itself. It is intentionally NOT a full sweep of
// every `Text("…")` in the codebase — see
// docs/audit/i18n-coverage.md for the incremental adoption plan.

import Foundation

public enum L10n {

    // MARK: - Tabs / Top-level modes

    public enum Mode {
        public static let chat = L10nEntry(
            en: "Chat",
            ja: "チャット",
            ko: "채팅",
            zh: "聊天"
        )
        public static let server = L10nEntry(
            en: "Server",
            ja: "サーバー",
            ko: "서버",
            zh: "服务器"
        )
        public static let image = L10nEntry(
            en: "Image",
            ja: "画像",
            ko: "이미지",
            zh: "图像"
        )
        public static let terminal = L10nEntry(
            en: "Terminal",
            ja: "ターミナル",
            ko: "터미널",
            zh: "终端"
        )
        public static let api = L10nEntry(
            en: "API",
            ja: "API",
            ko: "API",
            zh: "API"
        )
    }

    // MARK: - Menu bar / commands

    public enum Menu {
        public static let newChat = L10nEntry(
            en: "New Chat",
            ja: "新規チャット",
            ko: "새 채팅",
            zh: "新建聊天"
        )
        public static let reopenLastClosed = L10nEntry(
            en: "Reopen Last Closed Chat",
            ja: "最後に閉じたチャットを再度開く",
            ko: "마지막으로 닫은 채팅 다시 열기",
            zh: "重新打开最后关闭的聊天"
        )
        public static let view = L10nEntry(
            en: "View",
            ja: "表示",
            ko: "보기",
            zh: "视图"
        )
        public static let commandPalette = L10nEntry(
            en: "Command Palette…",
            ja: "コマンドパレット…",
            ko: "명령 팔레트…",
            zh: "命令面板…"
        )
        public static let downloads = L10nEntry(
            en: "Downloads",
            ja: "ダウンロード",
            ko: "다운로드",
            zh: "下载"
        )
        public static let undo = L10nEntry(
            en: "Undo",
            ja: "取り消す",
            ko: "실행 취소",
            zh: "撤销"
        )
        public static let languageAndSettings = L10nEntry(
            en: "Language & Settings…",
            ja: "言語と設定…",
            ko: "언어 및 설정…",
            zh: "语言与设置…"
        )
        public static let openVMLX = L10nEntry(
            en: "Open vMLX",
            ja: "vMLX を開く",
            ko: "vMLX 열기",
            zh: "打开 vMLX"
        )
        public static let quit = L10nEntry(
            en: "Quit",
            ja: "終了",
            ko: "종료",
            zh: "退出"
        )
    }

    // MARK: - Chat composer / session

    public enum Chat {
        public static let send = L10nEntry(
            en: "Send",
            ja: "送信",
            ko: "보내기",
            zh: "发送"
        )
        public static let stop = L10nEntry(
            en: "Stop",
            ja: "停止",
            ko: "중지",
            zh: "停止"
        )
        public static let placeholder = L10nEntry(
            en: "Message vMLX…",
            ja: "vMLX にメッセージを送る…",
            ko: "vMLX에 메시지 보내기…",
            zh: "向 vMLX 发送消息…"
        )
        public static let attachFile = L10nEntry(
            en: "Attach file",
            ja: "ファイルを添付",
            ko: "파일 첨부",
            zh: "附加文件"
        )
        public static let sessions = L10nEntry(
            en: "Sessions",
            ja: "セッション",
            ko: "세션",
            zh: "会话"
        )
        public static let chatSettings = L10nEntry(
            en: "Chat Settings",
            ja: "チャット設定",
            ko: "채팅 설정",
            zh: "聊天设置"
        )
        public static let clearAllChats = L10nEntry(
            en: "Clear all chats",
            ja: "すべてのチャットを消去",
            ko: "모든 채팅 삭제",
            zh: "清除所有聊天"
        )
        /// Uses %d for token count and %d for char count.
        public static let tokensAndCharsFormat = L10nEntry(
            en: "≈%d tok · %d chars",
            ja: "約 %d トークン · %d 文字",
            ko: "약 %d 토큰 · %d 문자",
            zh: "约 %d 词元 · %d 字"
        )
        public static let reasoning = L10nEntry(
            en: "Reasoning",
            ja: "推論",
            ko: "추론",
            zh: "推理"
        )
        public static let evaluatingPrompt = L10nEntry(
            en: "Evaluating prompt…",
            ja: "プロンプトを評価中…",
            ko: "프롬프트 평가 중…",
            zh: "正在评估提示…"
        )
        public static let noModelsDiscovered = L10nEntry(
            en: "No models discovered",
            ja: "モデルが見つかりません",
            ko: "모델을 찾을 수 없습니다",
            zh: "未发现模型"
        )
        public static let enableThinking = L10nEntry(
            en: "Enable Thinking",
            ja: "思考を有効にする",
            ko: "사고 활성화",
            zh: "启用思考"  // FIXME: needs native review
        )
        public static let reasoningEffort = L10nEntry(
            en: "Reasoning Effort",
            ja: "推論強度",
            ko: "추론 강도",
            zh: "推理强度"  // FIXME: needs native review
        )
        public static let toolChoice = L10nEntry(
            en: "Tool Choice",
            ja: "ツール選択",
            ko: "도구 선택",
            zh: "工具选择"
        )
        public static let effectiveValues = L10nEntry(
            en: "Effective values",
            ja: "適用中の値",
            ko: "적용 중인 값",
            zh: "当前生效值"
        )
        // Section labels in ChatSettingsPopover
        public static let sectionSampling = L10nEntry(
            en: "Sampling — this chat (overrides global)",
            ja: "サンプリング — このチャット(グローバル設定を上書き)",
            ko: "샘플링 — 현재 대화 (전역 설정 무시)",
            zh: "采样 — 当前对话(覆盖全局)"
        )
        public static let sectionReasoning = L10nEntry(
            en: "Reasoning",
            ja: "推論",
            ko: "추론",
            zh: "推理"
        )
        public static let sectionSystemPrompt = L10nEntry(
            en: "System Prompt",
            ja: "システムプロンプト",
            ko: "시스템 프롬프트",
            zh: "系统提示"
        )
        public static let sectionStopSequences = L10nEntry(
            en: "Stop Sequences",
            ja: "停止シーケンス",
            ko: "중지 시퀀스",
            zh: "停止序列"
        )
        public static let sectionTools = L10nEntry(
            en: "Tools",
            ja: "ツール",
            ko: "도구",
            zh: "工具"
        )
        public static let sectionBuiltInTools = L10nEntry(
            en: "Built-in Tools",
            ja: "組み込みツール",
            ko: "내장 도구",
            zh: "内置工具"
        )
        public static let sectionWorkingDirectory = L10nEntry(
            en: "Working Directory",
            ja: "作業ディレクトリ",
            ko: "작업 디렉터리",
            zh: "工作目录"
        )
        public static let addStop = L10nEntry(
            en: "Add stop",
            ja: "停止を追加",
            ko: "중지 추가",
            zh: "添加停止"
        )
    }

    // MARK: - Common actions / dialogs

    public enum Common {
        public static let ok = L10nEntry(
            en: "OK",
            ja: "OK",
            ko: "확인",
            zh: "确定"
        )
        public static let cancel = L10nEntry(
            en: "Cancel",
            ja: "キャンセル",
            ko: "취소",
            zh: "取消"
        )
        public static let save = L10nEntry(
            en: "Save",
            ja: "保存",
            ko: "저장",
            zh: "保存"
        )
        public static let delete = L10nEntry(
            en: "Delete",
            ja: "削除",
            ko: "삭제",
            zh: "删除"
        )
        public static let close = L10nEntry(
            en: "Close",
            ja: "閉じる",
            ko: "닫기",
            zh: "关闭"
        )
        public static let copy = L10nEntry(
            en: "Copy",
            ja: "コピー",
            ko: "복사",
            zh: "复制"
        )
        public static let copied = L10nEntry(
            en: "Copied!",
            ja: "コピーしました",
            ko: "복사됨",
            zh: "已复制"
        )
        public static let search = L10nEntry(
            en: "Search",
            ja: "検索",
            ko: "검색",
            zh: "搜索"
        )
        public static let settings = L10nEntry(
            en: "Settings",
            ja: "設定",
            ko: "설정",
            zh: "设置"
        )
        public static let language = L10nEntry(
            en: "Language",
            ja: "言語",
            ko: "언어",
            zh: "语言"
        )
        public static let help = L10nEntry(
            en: "Help",
            ja: "ヘルプ",
            ko: "도움말",
            zh: "帮助"
        )
        public static let rename = L10nEntry(
            en: "Rename…",
            ja: "名前を変更…",
            ko: "이름 바꾸기…",
            zh: "重命名…"
        )
        public static let reset = L10nEntry(
            en: "Reset",
            ja: "リセット",
            ko: "재설정",
            zh: "重置"
        )
        public static let resetAll = L10nEntry(
            en: "Reset all",
            ja: "すべてリセット",
            ko: "모두 재설정",
            zh: "全部重置"
        )
        public static let exportAsMarkdown = L10nEntry(
            en: "Export as Markdown…",
            ja: "Markdownとしてエクスポート…",
            ko: "Markdown으로 내보내기…",
            zh: "导出为 Markdown…"
        )
        public static let deleteChat = L10nEntry(
            en: "Delete chat",
            ja: "チャットを削除",
            ko: "대화 삭제",
            zh: "删除对话"
        )
        public static let deleteAll = L10nEntry(
            en: "Delete all",
            ja: "すべて削除",
            ko: "모두 삭제",
            zh: "全部删除"
        )
        public static let regenerate = L10nEntry(
            en: "Regenerate",
            ja: "再生成",
            ko: "다시 생성",
            zh: "重新生成"
        )
        public static let halt = L10nEntry(
            en: "Halt",
            ja: "中止",
            ko: "중단",
            zh: "中止"
        )
        public static let pause = L10nEntry(
            en: "Pause",
            ja: "一時停止",
            ko: "일시 중지",
            zh: "暂停"
        )
        public static let resume = L10nEntry(
            en: "Resume",
            ja: "再開",
            ko: "다시 시작",
            zh: "继续"
        )
        public static let retry = L10nEntry(
            en: "Retry",
            ja: "再試行",
            ko: "재시도",
            zh: "重试"
        )
        public static let open = L10nEntry(
            en: "Open",
            ja: "開く",
            ko: "열기",
            zh: "打开"
        )
        public static let loading = L10nEntry(
            en: "Loading…",
            ja: "読み込み中…",
            ko: "불러오는 중…",
            zh: "加载中…"
        )
        public static let waiting = L10nEntry(
            en: "Waiting…",
            ja: "待機中…",
            ko: "대기 중…",
            zh: "等待中…"
        )
        public static let inherit = L10nEntry(
            en: "Inherit",
            ja: "継承",
            ko: "상속",
            zh: "继承"
        )
        public static let off = L10nEntry(
            en: "Off",
            ja: "オフ",
            ko: "끄기",
            zh: "关闭"
        )
        public static let on = L10nEntry(
            en: "On",
            ja: "オン",
            ko: "켜기",
            zh: "开启"
        )
        public static let none = L10nEntry(
            en: "None",
            ja: "なし",
            ko: "없음",
            zh: "无"
        )
        public static let auto = L10nEntry(
            en: "Auto",
            ja: "自動",
            ko: "자동",
            zh: "自动"
        )
        public static let low = L10nEntry(
            en: "Low",
            ja: "低",
            ko: "낮음",
            zh: "低"
        )
        public static let medium = L10nEntry(
            en: "Medium",
            ja: "中",
            ko: "보통",
            zh: "中"
        )
        public static let high = L10nEntry(
            en: "High",
            ja: "高",
            ko: "높음",
            zh: "高"
        )
        /// §388 — DSV4 Flash/Pro adds a fourth effort level above `high`.
        /// Labeled literally as "Max" across locales to match the
        /// jang_config.chat.reasoning.reasoning_effort_levels value.
        public static let maxEffort = L10nEntry(
            en: "Max",
            ja: "最大",
            ko: "최대",
            zh: "最大"
        )
        public static let required = L10nEntry(
            en: "Required",
            ja: "必須",
            ko: "필수",
            zh: "必需"
        )
    }

    // MARK: - Server panel

    public enum Server {
        public static let start = L10nEntry(
            en: "Start Server",
            ja: "サーバーを起動",
            ko: "서버 시작",
            zh: "启动服务器"
        )
        public static let stop = L10nEntry(
            en: "Stop Server",
            ja: "サーバーを停止",
            ko: "서버 중지",
            zh: "停止服务器"
        )
        public static let running = L10nEntry(
            en: "Running",
            ja: "稼働中",
            ko: "실행 중",
            zh: "运行中"
        )
        public static let stopped = L10nEntry(
            en: "Stopped",
            ja: "停止中",
            ko: "중지됨",
            zh: "已停止"
        )
        public static let title = L10nEntry(
            en: "Server",
            ja: "サーバー",
            ko: "서버",
            zh: "服务器"
        )
        public static let subtitle = L10nEntry(
            en: "Pick a model and start serving on a local port",
            ja: "モデルを選択してローカルポートで配信を開始",
            ko: "모델을 선택하고 로컬 포트에서 서비스를 시작하세요",
            zh: "选择模型并在本地端口启动服务"
        )
        public static let loadModel = L10nEntry(
            en: "Load model",
            ja: "モデルを読み込む",
            ko: "모델 불러오기",
            zh: "加载模型"
        )
        public static let noModelLoaded = L10nEntry(
            en: "No model loaded",
            ja: "モデルが読み込まれていません",
            ko: "불러온 모델이 없습니다",
            zh: "未加载模型"
        )
        public static let noModelHint = L10nEntry(
            en: "Pick a model from your local cache to start chatting,\nrun a server, or generate images.",
            ja: "ローカルキャッシュからモデルを選択して、チャット、サーバー実行、画像生成を始めましょう。",
            ko: "로컬 캐시에서 모델을 선택해 채팅, 서버 실행, 이미지 생성을 시작하세요.",
            zh: "从本地缓存中选择模型,即可开始对话、运行服务或生成图像。"  // FIXME: needs native review
        )
        public static let modelsSourceHint = L10nEntry(
            en: "Models are detected from `~/.cache/huggingface/hub/`.\nUse the Downloads window to pull new ones.",
            ja: "モデルは `~/.cache/huggingface/hub/` から検出されます。新規取得はダウンロードウィンドウから行ってください。",
            ko: "모델은 `~/.cache/huggingface/hub/`에서 자동 탐지됩니다. 새 모델은 다운로드 창에서 받으세요.",
            zh: "模型会从 `~/.cache/huggingface/hub/` 自动检测。请从下载窗口获取新模型。"  // FIXME: needs native review
        )
        public static let newSession = L10nEntry(
            en: "New session",
            ja: "新しいセッション",
            ko: "새 세션",
            zh: "新建会话"
        )
    }

    // MARK: - Settings / Preferences

    public enum Settings {
        public static let appearance = L10nEntry(
            en: "Appearance",
            ja: "外観",
            ko: "모양",
            zh: "外观"
        )
        public static let uiLanguage = L10nEntry(
            en: "Interface Language",
            ja: "インターフェースの言語",
            ko: "인터페이스 언어",
            zh: "界面语言"
        )
        public static let uiLanguageHelp = L10nEntry(
            en: "Affects the app's user interface only. Server logs, API responses, and engine output remain in English.",
            ja: "アプリのユーザーインターフェースにのみ適用されます。サーバーログ、API レスポンス、エンジン出力は英語のままです。",
            ko: "앱 사용자 인터페이스에만 적용됩니다. 서버 로그, API 응답, 엔진 출력은 영어로 유지됩니다.",
            zh: "仅影响应用的用户界面。服务器日志、API 响应和引擎输出仍为英文。"
        )
    }

    // MARK: - Downloads window

    public enum Downloads {
        public static let title = L10nEntry(
            en: "Downloads",
            ja: "ダウンロード",
            ko: "다운로드",
            zh: "下载"
        )
        public static let clearCompleted = L10nEntry(
            en: "Clear completed",
            ja: "完了済みを消去",
            ko: "완료된 항목 지우기",
            zh: "清除已完成"
        )
        public static let empty = L10nEntry(
            en: "No downloads yet",
            ja: "まだダウンロードはありません",
            ko: "아직 다운로드가 없습니다",
            zh: "暂无下载"
        )
        public static let emptyHint = L10nEntry(
            en: "Download an MLX model from HuggingFace",
            ja: "HuggingFace から MLX モデルをダウンロード",
            ko: "HuggingFace에서 MLX 모델을 다운로드하세요",
            zh: "从 HuggingFace 下载 MLX 模型"
        )
        public static let openSettings = L10nEntry(
            en: "Open Settings →",
            ja: "設定を開く →",
            ko: "설정 열기 →",
            zh: "打开设置 →"
        )
    }

    // MARK: - Tray / menu-bar item

    public enum Tray {
        public static let session = L10nEntry(
            en: "Session",
            ja: "セッション",
            ko: "세션",
            zh: "会话"
        )
        public static let start = L10nEntry(
            en: "Start",
            ja: "開始",
            ko: "시작",
            zh: "启动"
        )
        public static let stop = L10nEntry(
            en: "Stop",
            ja: "停止",
            ko: "중지",
            zh: "停止"
        )
        public static let restart = L10nEntry(
            en: "Restart",
            ja: "再起動",
            ko: "다시 시작",
            zh: "重启"
        )
        public static let softSleep = L10nEntry(
            en: "Soft Sleep",
            ja: "ソフトスリープ",
            ko: "소프트 슬립",
            zh: "软休眠"
        )
        public static let deepSleep = L10nEntry(
            en: "Deep Sleep",
            ja: "ディープスリープ",
            ko: "딥 슬립",
            zh: "深度休眠"
        )
        public static let wake = L10nEntry(
            en: "Wake",
            ja: "ウェイク",
            ko: "깨우기",
            zh: "唤醒"
        )
        public static let pick = L10nEntry(
            en: "Pick…",
            ja: "選択…",
            ko: "선택…",
            zh: "选择…"
        )
        public static let openLogs = L10nEntry(
            en: "Open Logs",
            ja: "ログを開く",
            ko: "로그 열기",
            zh: "打开日志"
        )
        public static let noRecentLogs = L10nEntry(
            en: "No recent logs.",
            ja: "最近のログはありません。",
            ko: "최근 로그가 없습니다.",
            zh: "暂无最近日志。"
        )
    }

    // MARK: - §356/§357 deep sweep — chat, sidebar, server panels, image

    public enum ChatUI {
        public static let newChat = L10nEntry(
            en: "New chat",
            ja: "新規チャット",
            ko: "새 채팅",
            zh: "新建聊天"
        )
        public static let clearAllChats = L10nEntry(
            en: "Clear all chats",
            ja: "すべてのチャットを消去",
            ko: "모든 채팅 지우기",
            zh: "清除所有聊天"
        )
        public static let clearAllChatsConfirm = L10nEntry(
            en: "All %lld chats and their messages will be permanently removed. This can't be undone.",
            ja: "%lld 件のすべてのチャットとそのメッセージが完全に削除されます。元に戻せません。",
            ko: "%lld개의 모든 채팅과 메시지가 영구적으로 제거됩니다. 취소할 수 없습니다.",
            zh: "所有 %lld 个聊天及其消息将被永久删除。此操作无法撤销。"
        )
        public static let deleteMessageConfirm = L10nEntry(
            en: "The message will be permanently removed from this chat. This can't be undone.",
            ja: "このチャットからメッセージが完全に削除されます。元に戻せません。",
            ko: "이 채팅에서 메시지가 영구적으로 제거됩니다. 취소할 수 없습니다.",
            zh: "消息将从此聊天中永久删除。此操作无法撤销。"
        )
        public static let deleteSessionConfirm = L10nEntry(
            en: "\"%@\" and all its messages will be permanently removed. This can't be undone.",
            ja: "「%@」とそのすべてのメッセージが完全に削除されます。元に戻せません。",
            ko: "\"%@\" 및 모든 메시지가 영구적으로 제거됩니다. 취소할 수 없습니다.",
            zh: "\"%@\" 及其所有消息将被永久删除。此操作无法撤销。"
        )
        public static let quickTweaks = L10nEntry(
            en: "Quick tweaks",
            ja: "クイック調整",
            ko: "빠른 조정",
            zh: "快速调整"
        )
        public static let openFullSettings = L10nEntry(
            en: "Open full settings for more",
            ja: "詳細設定を開く",
            ko: "전체 설정 열기",
            zh: "打开完整设置以查看更多"
        )
        public static let shellToolHelp = L10nEntry(
            en: "Shell maps to BashTool via ChatViewModel → Stream. Enable MCP in the API panel for additional tool providers.",
            ja: "Shell は ChatViewModel → Stream 経由で BashTool にマップされます。追加のツールプロバイダーには API パネルで MCP を有効化してください。",
            ko: "Shell은 ChatViewModel → Stream을 통해 BashTool에 매핑됩니다. 추가 도구 제공자는 API 패널에서 MCP를 활성화하십시오.",
            zh: "Shell 通过 ChatViewModel → Stream 映射到 BashTool。在 API 面板中启用 MCP 以获取其他工具提供者。"
        )
        public static let builtinToolsComingSoon = L10nEntry(
            en: "Coming soon: Web Search, Fetch URL, File Tools, Git, Utility Tools, Brave Search — not yet wired through the stream. Use MCP in the API panel for additional tool providers today.",
            ja: "近日公開: Web Search、Fetch URL、File Tools、Git、Utility Tools、Brave Search — まだストリーム経由で接続されていません。現時点で追加のツールプロバイダーには API パネルの MCP をご利用ください。",
            ko: "곧 출시: 웹 검색, URL 가져오기, 파일 도구, Git, 유틸리티 도구, Brave 검색 — 아직 스트림을 통해 연결되지 않았습니다. 현재는 추가 도구 제공자로 API 패널의 MCP를 사용하십시오.",
            zh: "即将推出：网页搜索、获取 URL、文件工具、Git、实用工具、Brave 搜索 — 尚未通过流连接。目前请使用 API 面板中的 MCP 以获取其他工具提供者。"
        )
        public static let noModelsMatch = L10nEntry(
            en: "No models match `%@`",
            ja: "「%@」に一致するモデルはありません",
            ko: "`%@`와 일치하는 모델이 없습니다",
            zh: "没有匹配 `%@` 的模型"
        )
        public static let output = L10nEntry(
            en: "Output",
            ja: "出力",
            ko: "출력",
            zh: "输出"
        )
    }

    public enum ServerUI {
        public static let performance = L10nEntry(
            en: "PERFORMANCE",
            ja: "パフォーマンス",
            ko: "성능",
            zh: "性能"
        )
        public static let cache = L10nEntry(
            en: "CACHE",
            ja: "キャッシュ",
            ko: "캐시",
            zh: "缓存"
        )
        public static let logs = L10nEntry(
            en: "LOGS",
            ja: "ログ",
            ko: "로그",
            zh: "日志"
        )
        public static let benchmark = L10nEntry(
            en: "BENCHMARK",
            ja: "ベンチマーク",
            ko: "벤치마크",
            zh: "基准测试"
        )
        public static let recentLatency = L10nEntry(
            en: "Recent request latency (ms)",
            ja: "最近のリクエストレイテンシ (ms)",
            ko: "최근 요청 지연 시간 (ms)",
            zh: "最近请求延迟 (ms)"
        )
        public static let engineStoppedNoMetrics = L10nEntry(
            en: "Engine stopped — no metrics",
            ja: "エンジン停止中 — メトリクスなし",
            ko: "엔진 중지됨 — 지표 없음",
            zh: "引擎已停止 — 无指标"
        )
        public static let noCacheActivity = L10nEntry(
            en: "no cache activity yet",
            ja: "キャッシュアクティビティなし",
            ko: "아직 캐시 활동 없음",
            zh: "暂无缓存活动"
        )
        public static let decoded5sAvg = L10nEntry(
            en: "decoded (5s avg)",
            ja: "デコード済み (5秒平均)",
            ko: "디코드됨 (5초 평균)",
            zh: "已解码（5秒平均）"
        )
        public static let noRequestsYet = L10nEntry(
            en: "No requests yet",
            ja: "リクエストなし",
            ko: "아직 요청 없음",
            zh: "尚无请求"
        )
        public static let noModelLoadedSidebar = L10nEntry(
            en: "No model loaded — load one in the sidebar to see cache stats.",
            ja: "モデル未ロード — サイドバーでロードするとキャッシュ統計が表示されます。",
            ko: "로드된 모델 없음 — 사이드바에서 하나를 로드하면 캐시 통계가 표시됩니다.",
            zh: "未加载模型 — 在侧边栏加载一个以查看缓存统计信息。"
        )
        public static let modelArchitecture = L10nEntry(
            en: "Model architecture",
            ja: "モデルアーキテクチャ",
            ko: "모델 아키텍처",
            zh: "模型架构"
        )
        public static let loadModelForCacheBreakdown = L10nEntry(
            en: "Load a model to see per-layer cache breakdown.",
            ja: "モデルをロードすると、レイヤーごとのキャッシュ内訳が表示されます。",
            ko: "모델을 로드하면 레이어별 캐시 분석이 표시됩니다.",
            zh: "加载模型以查看每层缓存细分。"
        )
        public static let hybridInactive = L10nEntry(
            en: "Hybrid mode inactive — SSM companion not used for this model.",
            ja: "ハイブリッドモード無効 — このモデルでは SSM コンパニオンは使用されません。",
            ko: "하이브리드 모드 비활성 — 이 모델에는 SSM 컴패니언이 사용되지 않습니다.",
            zh: "混合模式未启用 — 此模型未使用 SSM 配套。"
        )
        public static let dropCacheHelp = L10nEntry(
            en: "This drops every prefix / paged / SSM / disk cache entry for the loaded model. The next request will re-prefill from scratch.",
            ja: "ロード済みモデルのすべてのプレフィックス/ページ型/SSM/ディスクキャッシュエントリを破棄します。次のリクエストは最初から再プレフィルされます。",
            ko: "로드된 모델의 모든 프리픽스 / 페이지 / SSM / 디스크 캐시 항목을 삭제합니다. 다음 요청은 처음부터 다시 프리필됩니다.",
            zh: "这将删除已加载模型的每个前缀 / 分页 / SSM / 磁盘缓存条目。下一个请求将从头重新预填充。"
        )
        public static let disabled = L10nEntry(
            en: "Disabled",
            ja: "無効",
            ko: "비활성화",
            zh: "已禁用"
        )
        public static let engineError = L10nEntry(
            en: "Engine error",
            ja: "エンジンエラー",
            ko: "엔진 오류",
            zh: "引擎错误"
        )
        public static let remote = L10nEntry(
            en: "Remote",
            ja: "リモート",
            ko: "원격",
            zh: "远程"
        )
        public static let starting = L10nEntry(
            en: "Starting…",
            ja: "起動中…",
            ko: "시작 중…",
            zh: "启动中…"
        )
        public static let lastRunFormat = L10nEntry(
            en: "Last run — %@",
            ja: "最終実行 — %@",
            ko: "마지막 실행 — %@",
            zh: "上次运行 — %@"
        )
        public static let historyLast20 = L10nEntry(
            en: "HISTORY (last 20)",
            ja: "履歴 (最新 20 件)",
            ko: "기록 (최근 20개)",
            zh: "历史记录（最近 20 次）"
        )
        public static let noBenchmarkRuns = L10nEntry(
            en: "No benchmark runs yet.",
            ja: "ベンチマーク実行なし。",
            ko: "아직 벤치마크 실행 없음.",
            zh: "尚无基准测试运行。"
        )
        public static let pidFormat = L10nEntry(
            en: "PID %lld",
            ja: "PID %lld",
            ko: "PID %lld",
            zh: "PID %lld"
        )
    }

    public enum ModelDirs {
        public static let modelDirectories = L10nEntry(
            en: "Model directories",
            ja: "モデルディレクトリ",
            ko: "모델 디렉토리",
            zh: "模型目录"
        )
        public static let folderScanHelp = L10nEntry(
            en: "Folders the library scans for downloaded models.",
            ja: "ダウンロード済みモデルをライブラリがスキャンするフォルダ。",
            ko: "라이브러리가 다운로드된 모델을 스캔하는 폴더.",
            zh: "库扫描已下载模型的文件夹。"
        )
        public static let rescan = L10nEntry(
            en: "Rescan",
            ja: "再スキャン",
            ko: "다시 스캔",
            zh: "重新扫描"
        )
        public static let customDirectories = L10nEntry(
            en: "CUSTOM DIRECTORIES",
            ja: "カスタムディレクトリ",
            ko: "사용자 지정 디렉토리",
            zh: "自定义目录"
        )
        public static let noCustomDirs = L10nEntry(
            en: "No custom directories. Add one to scan an external drive or a workspace folder full of safetensors.",
            ja: "カスタムディレクトリなし。外部ドライブまたは safetensors を含むワークスペースフォルダをスキャンするには追加してください。",
            ko: "사용자 지정 디렉토리가 없습니다. 외부 드라이브 또는 safetensors가 있는 작업 폴더를 스캔하려면 추가하십시오.",
            zh: "没有自定义目录。添加一个以扫描外部驱动器或包含 safetensors 的工作区文件夹。"
        )
        public static let addDirectory = L10nEntry(
            en: "Add directory…",
            ja: "ディレクトリを追加…",
            ko: "디렉토리 추가…",
            zh: "添加目录…"
        )
        public static let lastScanFormat = L10nEntry(
            en: "Last scan: %lld models",
            ja: "最終スキャン: %lld 個のモデル",
            ko: "마지막 스캔: %lld개 모델",
            zh: "上次扫描：%lld 个模型"
        )
        public static let downloadByHF = L10nEntry(
            en: "Download by HuggingFace repo",
            ja: "HuggingFace リポジトリからダウンロード",
            ko: "HuggingFace 리포지토리로 다운로드",
            zh: "通过 HuggingFace 仓库下载"
        )
        public static let hfFormatHint = L10nEntry(
            en: "Enter `{org}/{repo}` format. Gated repos require a HuggingFace token in the API tab.",
            ja: "`{org}/{repo}` 形式で入力してください。ゲート付きリポジトリには API タブで HuggingFace トークンが必要です。",
            ko: "`{org}/{repo}` 형식으로 입력하십시오. 게이트된 리포지토리는 API 탭에서 HuggingFace 토큰이 필요합니다.",
            zh: "输入 `{org}/{repo}` 格式。受限仓库需要在 API 选项卡中配置 HuggingFace 令牌。"
        )
        public static let hfCacheDefault = L10nEntry(
            en: "HuggingFace cache (default — cannot be removed)",
            ja: "HuggingFace キャッシュ (デフォルト — 削除不可)",
            ko: "HuggingFace 캐시 (기본값 — 제거 불가)",
            zh: "HuggingFace 缓存（默认 — 无法删除）"
        )
        public static let missingOnDisk = L10nEntry(
            en: "Missing on disk — folder may have been deleted or the drive unmounted",
            ja: "ディスク上に存在しません — フォルダが削除されたかドライブがマウント解除された可能性があります",
            ko: "디스크에 없음 — 폴더가 삭제되었거나 드라이브가 마운트 해제되었을 수 있습니다",
            zh: "磁盘上不存在 — 文件夹可能已被删除或驱动器未挂载"
        )
        public static let stopScanningHelp = L10nEntry(
            en: "vMLX will stop scanning this folder for models. The folder and any files inside it stay on disk untouched.",
            ja: "vMLX はこのフォルダのモデルスキャンを停止します。フォルダとその中のファイルはディスク上にそのまま残ります。",
            ko: "vMLX는 이 폴더의 모델 스캔을 중지합니다. 폴더와 내부 파일은 디스크에 그대로 남습니다.",
            zh: "vMLX 将停止扫描此文件夹中的模型。文件夹及其内部的所有文件保留在磁盘上不变。"
        )
    }

    public enum ImageUI {
        public static let imageSettings = L10nEntry(
            en: "Image settings",
            ja: "画像設定",
            ko: "이미지 설정",
            zh: "图像设置"
        )
        public static let seed = L10nEntry(
            en: "Seed",
            ja: "シード",
            ko: "시드",
            zh: "种子"
        )
        public static let seedPlaceholder = L10nEntry(
            en: "-1 for random",
            ja: "-1 でランダム",
            ko: "-1은 임의",
            zh: "-1 为随机"
        )
        public static let random = L10nEntry(
            en: "Random",
            ja: "ランダム",
            ko: "임의",
            zh: "随机"
        )
        public static let scheduler = L10nEntry(
            en: "Scheduler",
            ja: "スケジューラー",
            ko: "스케줄러",
            zh: "调度器"
        )
        public static let defaultLabel = L10nEntry(
            en: "Default",
            ja: "デフォルト",
            ko: "기본값",
            zh: "默认"
        )
        public static let strengthHint = L10nEntry(
            en: "Strength lives on the prompt bar.",
            ja: "強度はプロンプトバーにあります。",
            ko: "강도는 프롬프트 바에 있습니다.",
            zh: "强度位于提示栏。"
        )
        public static let saveAsDefault = L10nEntry(
            en: "Save as default",
            ja: "デフォルトとして保存",
            ko: "기본값으로 저장",
            zh: "另存为默认"
        )
        public static let generating = L10nEntry(
            en: "Generating…",
            ja: "生成中…",
            ko: "생성 중…",
            zh: "生成中…"
        )
        public static let stepFormat = L10nEntry(
            en: "Step %lld / %lld",
            ja: "ステップ %lld / %lld",
            ko: "스텝 %lld / %lld",
            zh: "步骤 %lld / %lld"
        )
        public static let elapsedEtaFormat = L10nEntry(
            en: "%llds elapsed · ETA %@",
            ja: "%llds 経過 · 残り %@",
            ko: "%llds 경과 · ETA %@",
            zh: "已用 %llds · 剩余 %@"
        )
        public static let change = L10nEntry(
            en: "Change",
            ja: "変更",
            ko: "변경",
            zh: "更改"
        )
        public static let sourceImage = L10nEntry(
            en: "Source image",
            ja: "ソース画像",
            ko: "원본 이미지",
            zh: "源图像"
        )
        public static let strengthFormat = L10nEntry(
            en: "Strength %@",
            ja: "強度 %@",
            ko: "강도 %@",
            zh: "强度 %@"
        )
        public static let history = L10nEntry(
            en: "HISTORY",
            ja: "履歴",
            ko: "기록",
            zh: "历史记录"
        )
        public static let noHistoryYet = L10nEntry(
            en: "No history yet",
            ja: "履歴なし",
            ko: "아직 기록 없음",
            zh: "尚无历史记录"
        )
        public static let paintMask = L10nEntry(
            en: "Paint mask",
            ja: "マスクを描画",
            ko: "마스크 그리기",
            zh: "绘制蒙版"
        )
        public static let maskLegend = L10nEntry(
            en: "White = edit, transparent = keep",
            ja: "白 = 編集、透明 = 保持",
            ko: "흰색 = 편집, 투명 = 유지",
            zh: "白色 = 编辑，透明 = 保留"
        )
        public static let brushFormat = L10nEntry(
            en: "Brush %lld",
            ja: "ブラシ %lld",
            ko: "브러시 %lld",
            zh: "画笔 %lld"
        )
        public static let opacityFormat = L10nEntry(
            en: "Opacity %@",
            ja: "不透明度 %@",
            ko: "불투명도 %@",
            zh: "不透明度 %@"
        )
    }

    public enum Advanced {
        public static let header = L10nEntry(
            en: "ADVANCED — CORS, RATE LIMIT, TLS",
            ja: "詳細設定 — CORS、レート制限、TLS",
            ko: "고급 — CORS, 속도 제한, TLS",
            zh: "高级 — CORS、速率限制、TLS"
        )
        public static let restartRequired = L10nEntry(
            en: "Restart required to apply",
            ja: "適用するには再起動が必要",
            ko: "적용하려면 재시작 필요",
            zh: "需要重启以应用"
        )
        public static let corsOrigins = L10nEntry(
            en: "CORS allowed origins",
            ja: "CORS 許可されたオリジン",
            ko: "CORS 허용 오리진",
            zh: "CORS 允许的来源"
        )
        public static let corsHint = L10nEntry(
            en: "Comma-separated. * means all. Single exact origin maps to Access-Control-Allow-Origin: <origin>. Multiple non-wildcard entries enable origin-echo mode.",
            ja: "カンマ区切り。* はすべて。単一の完全なオリジンは Access-Control-Allow-Origin: <origin> にマップされます。複数のワイルドカードなしのエントリはオリジンエコーモードを有効にします。",
            ko: "쉼표로 구분. *는 모두를 의미. 단일 정확한 오리진은 Access-Control-Allow-Origin: <origin>에 매핑됩니다. 여러 와일드카드가 아닌 항목은 오리진 에코 모드를 활성화합니다.",
            zh: "以逗号分隔。* 表示全部。单个精确来源映射到 Access-Control-Allow-Origin: <origin>。多个非通配符条目启用来源回显模式。"
        )
        public static let rateLimit = L10nEntry(
            en: "Rate limit (requests / minute / IP)",
            ja: "レート制限 (リクエスト/分/IP)",
            ko: "속도 제한 (요청/분/IP)",
            zh: "速率限制 (请求/分钟/IP)"
        )
        public static let tlsHeader = L10nEntry(
            en: "TLS (HTTPS) — set BOTH to enable, leave blank for HTTP",
            ja: "TLS (HTTPS) — 両方設定で有効化、空欄で HTTP",
            ko: "TLS (HTTPS) — 둘 다 설정하여 활성화, HTTP는 비워 두기",
            zh: "TLS (HTTPS) — 全部设置以启用，留空使用 HTTP"
        )
        public static let tlsBothRequired = L10nEntry(
            en: "Both key and cert must be set to enable TLS",
            ja: "TLS を有効にするには鍵と証明書の両方が必要です",
            ko: "TLS를 활성화하려면 키와 인증서 모두 설정해야 합니다",
            zh: "启用 TLS 需要同时设置密钥和证书"
        )
        public static let tlsFileMissing = L10nEntry(
            en: "One of the TLS files is missing or unreadable",
            ja: "TLS ファイルのいずれかが見つからないか読み取れません",
            ko: "TLS 파일 중 하나가 누락되었거나 읽을 수 없습니다",
            zh: "其中一个 TLS 文件缺失或无法读取"
        )
        public static let tlsWillUse = L10nEntry(
            en: "TLS will use https:// on next restart",
            ja: "TLS は次回再起動時に https:// を使用します",
            ko: "TLS는 다음 재시작 시 https://를 사용합니다",
            zh: "TLS 将在下次重启时使用 https://"
        )
    }

    public enum RequestLog {
        public static let liveHeader = L10nEntry(
            en: "Live request log",
            ja: "ライブリクエストログ",
            ko: "실시간 요청 로그",
            zh: "实时请求日志"
        )
        public static let inspector = L10nEntry(
            en: "Request inspector",
            ja: "リクエストインスペクター",
            ko: "요청 검사기",
            zh: "请求检查器"
        )
        public static let rawLine = L10nEntry(
            en: "Raw log line",
            ja: "生のログ行",
            ko: "원시 로그 행",
            zh: "原始日志行"
        )
        public static let hintFormat = L10nEntry(
            en: "Shows the last %lld server-category log lines. Server tab's LogsPanel has full filters.",
            ja: "最新の %lld 件のサーバーカテゴリログ行を表示します。サーバータブの LogsPanel にはすべてのフィルターがあります。",
            ko: "최근 %lld개의 서버 카테고리 로그 행을 표시합니다. 서버 탭의 LogsPanel에 전체 필터가 있습니다.",
            zh: "显示最近 %lld 条服务器类别日志行。服务器选项卡的 LogsPanel 具有完整过滤器。"
        )
    }

    public enum Downloads2 {
        public static let findModels = L10nEntry(
            en: "Find MLX-compatible models",
            ja: "MLX 互換モデルを検索",
            ko: "MLX 호환 모델 찾기",
            zh: "查找 MLX 兼容模型"
        )
        public static let findModelsHint = L10nEntry(
            en: "Search the HuggingFace Hub. Popular filters: MLX, VLM, image-gen.",
            ja: "HuggingFace Hub を検索します。人気のフィルター: MLX、VLM、image-gen。",
            ko: "HuggingFace Hub를 검색합니다. 인기 필터: MLX, VLM, image-gen.",
            zh: "搜索 HuggingFace Hub。热门过滤器：MLX、VLM、image-gen。"
        )
        public static let searchPlaceholder = L10nEntry(
            en: "Search HuggingFace models — e.g. qwen, gemma, flux",
            ja: "HuggingFace モデルを検索 — 例: qwen、gemma、flux",
            ko: "HuggingFace 모델 검색 — 예: qwen, gemma, flux",
            zh: "搜索 HuggingFace 模型 — 例如 qwen、gemma、flux"
        )
        public static let gated = L10nEntry(
            en: "Gated",
            ja: "ゲート付き",
            ko: "게이트",
            zh: "受限"
        )
        public static let dismiss = L10nEntry(
            en: "Dismiss",
            ja: "閉じる",
            ko: "닫기",
            zh: "忽略"
        )
    }

    public enum DownloadsUI {
        public static let imageTabHint = L10nEntry(
            en: "From the **Image** tab, the model picker has a download button next to each Flux / Z-Image model. Click it and the download appears here automatically.",
            ja: "**画像** タブでは、モデルピッカーの各 Flux / Z-Image モデルの横にダウンロードボタンがあります。クリックするとダウンロードがここに自動的に表示されます。",
            ko: "**이미지** 탭에서 모델 피커의 각 Flux / Z-Image 모델 옆에 다운로드 버튼이 있습니다. 클릭하면 여기에 다운로드가 자동으로 표시됩니다.",
            zh: "在**图像**标签页中，模型选择器的每个 Flux / Z-Image 模型旁都有下载按钮。点击后下载会自动显示在此处。"
        )
        public static let fromCLI = L10nEntry(
            en: "From the CLI:",
            ja: "CLI から:",
            ko: "CLI에서:",
            zh: "从命令行:"
        )
        public static let gatedLicenseHint = L10nEntry(
            en: "Some models require accepting a license on HuggingFace. To download them:",
            ja: "一部のモデルは HuggingFace でライセンスの承諾が必要です。ダウンロードするには:",
            ko: "일부 모델은 HuggingFace에서 라이선스 동의가 필요합니다. 다운로드하려면:",
            zh: "某些模型需要在 HuggingFace 上接受许可证。下载方法:"
        )
        public static let cacheHint = L10nEntry(
            en: "Files are stored in the standard HuggingFace cache. The Server tab's Model Library auto-detects them on the next scan.",
            ja: "ファイルは標準の HuggingFace キャッシュに保存されます。サーバータブのモデルライブラリは次回スキャン時に自動的に検出します。",
            ko: "파일은 표준 HuggingFace 캐시에 저장됩니다. 서버 탭의 모델 라이브러리가 다음 스캔 시 자동으로 감지합니다.",
            zh: "文件存储在标准 HuggingFace 缓存中。服务器选项卡的模型库将在下次扫描时自动检测它们。"
        )
        public static let pauseResumeHint = L10nEntry(
            en: "Pause / cancel / retry are per-row in this window. Resumes use HTTP Range requests so paused downloads pick up from the exact byte they stopped — no re-downloading.",
            ja: "このウィンドウでは一時停止 / キャンセル / 再試行が行ごとに利用できます。再開には HTTP Range リクエストを使用するため、一時停止したダウンロードは停止したバイトから正確に再開されます — 再ダウンロードは不要です。",
            ko: "이 창에서 각 행별로 일시 중지 / 취소 / 재시도가 가능합니다. 재개는 HTTP Range 요청을 사용하여 일시 중지된 다운로드를 정확히 중지된 바이트부터 다시 시작합니다 — 재다운로드 없음.",
            zh: "此窗口中每行都可以暂停 / 取消 / 重试。恢复使用 HTTP Range 请求，暂停的下载从停止的确切字节继续 — 无需重新下载。"
        )
        public static let etaFormat = L10nEntry(
            en: "ETA %@",
            ja: "残り %@",
            ko: "예상 %@",
            zh: "剩余 %@"
        )
        public static let gatedPrompt = L10nEntry(
            en: "Gated / private repo. Paste your HuggingFace token to unlock:",
            ja: "ゲート付き / プライベートリポジトリ。HuggingFace トークンを貼り付けてロック解除:",
            ko: "게이트 / 비공개 리포지토리. HuggingFace 토큰을 붙여넣어 잠금 해제:",
            zh: "受限 / 私有仓库。粘贴您的 HuggingFace 令牌以解锁:"
        )
    }

    public enum Setup {
        public static let welcome = L10nEntry(
            en: "Welcome to vMLX",
            ja: "vMLX へようこそ",
            ko: "vMLX에 오신 것을 환영합니다",
            zh: "欢迎使用 vMLX"
        )
        public static let stepOfFormat = L10nEntry(
            en: "Step %lld of 3",
            ja: "ステップ %lld / 3",
            ko: "3단계 중 %lld단계",
            zh: "步骤 %lld / 3"
        )
        public static let runSOTA = L10nEntry(
            en: "Run state-of-the-art LLMs on Apple Silicon",
            ja: "Apple Silicon で最先端の LLM を実行",
            ko: "Apple Silicon에서 최첨단 LLM 실행",
            zh: "在 Apple Silicon 上运行最先进的 LLM"
        )
        public static let runSOTABlurb = L10nEntry(
            en: "vMLX serves chat, embeddings, images, and tool calls over OpenAI / Anthropic / Ollama APIs — 100% on-device.",
            ja: "vMLX は OpenAI / Anthropic / Ollama 互換APIで、チャット・埋め込み・画像・ツール呼び出しを提供 — 完全オンデバイス。",
            ko: "vMLX는 OpenAI / Anthropic / Ollama API로 채팅, 임베딩, 이미지, 도구 호출을 제공 — 100% 온디바이스.",
            zh: "vMLX 通过兼容 OpenAI / Anthropic / Ollama 的 API 提供聊天、嵌入、图像和工具调用 — 100% 在本地运行。"
        )
        public static let scanningCache = L10nEntry(
            en: "Scanning Hugging Face cache…",
            ja: "Hugging Face キャッシュをスキャン中…",
            ko: "Hugging Face 캐시 스캔 중…",
            zh: "正在扫描 Hugging Face 缓存…"
        )
        public static let pickModel = L10nEntry(
            en: "Pick a model to use for chat",
            ja: "チャットに使用するモデルを選択",
            ko: "채팅에 사용할 모델 선택",
            zh: "选择用于聊天的模型"
        )
        public static let allSet = L10nEntry(
            en: "You're all set",
            ja: "準備完了",
            ko: "모든 준비가 완료되었습니다",
            zh: "一切就绪"
        )
        public static let allSetBody = L10nEntry(
            en: "Open the Server tab to start the engine, then head to Chat. Keys for remote clients live under the API tab.",
            ja: "サーバータブを開いてエンジンを起動し、チャットに進んでください。リモートクライアント用のキーは API タブにあります。",
            ko: "서버 탭을 열어 엔진을 시작한 다음 채팅으로 이동하십시오. 원격 클라이언트용 키는 API 탭에 있습니다.",
            zh: "打开服务器选项卡启动引擎，然后转到聊天。远程客户端的密钥在 API 选项卡下。"
        )
        public static let tokenSaved = L10nEntry(
            en: "Hugging Face token stored — gated repos (Llama, Gemma, Mistral) are accessible.",
            ja: "Hugging Face トークンが保存されました — ゲート付きリポジトリ (Llama、Gemma、Mistral) にアクセスできます。",
            ko: "Hugging Face 토큰이 저장되었습니다 — 게이트된 리포지토리 (Llama, Gemma, Mistral)에 액세스할 수 있습니다.",
            zh: "Hugging Face 令牌已存储 — 可以访问受限仓库（Llama、Gemma、Mistral）。"
        )
        public static let tokenMissing = L10nEntry(
            en: "No Hugging Face token yet — the starter model above is public, but gated repos (Llama, Gemma, Mistral) need one.",
            ja: "まだ Hugging Face トークンがありません — 上記のスターターモデルは公開されていますが、ゲート付きリポジトリ (Llama、Gemma、Mistral) にはトークンが必要です。",
            ko: "아직 Hugging Face 토큰이 없습니다 — 위의 스타터 모델은 공개되지만 게이트된 리포지토리 (Llama, Gemma, Mistral)에는 토큰이 필요합니다.",
            zh: "还没有 Hugging Face 令牌 — 上面的入门模型是公开的，但受限仓库（Llama、Gemma、Mistral）需要令牌。"
        )
        public static let addTokenCTA = L10nEntry(
            en: "Add a token →",
            ja: "トークンを追加 →",
            ko: "토큰 추가 →",
            zh: "添加令牌 →"
        )
        public static let back = L10nEntry(
            en: "Back",
            ja: "戻る",
            ko: "뒤로",
            zh: "返回"
        )
        public static let next = L10nEntry(
            en: "Next",
            ja: "次へ",
            ko: "다음",
            zh: "下一步"
        )
        public static let finish = L10nEntry(
            en: "Finish",
            ja: "完了",
            ko: "완료",
            zh: "完成"
        )
    }

    public enum TerminalUI {
        public static let understandEnable = L10nEntry(
            en: "I understand — enable Terminal mode",
            ja: "理解しました — ターミナルモードを有効化",
            ko: "이해했습니다 — 터미널 모드 활성화",
            zh: "我明白了 — 启用终端模式"
        )
    }

    public enum Misc {
        public static let reset = L10nEntry(
            en: "Reset",
            ja: "リセット",
            ko: "재설정",
            zh: "重置"
        )
        public static let resetToDefaults = L10nEntry(
            en: "Reset to defaults",
            ja: "デフォルトにリセット",
            ko: "기본값으로 재설정",
            zh: "重置为默认"
        )
        public static let create = L10nEntry(
            en: "Create",
            ja: "作成",
            ko: "생성",
            zh: "创建"
        )
        public static let wake = L10nEntry(
            en: "Wake",
            ja: "ウェイク",
            ko: "깨우기",
            zh: "唤醒"
        )
        public static let reconnect = L10nEntry(
            en: "Reconnect",
            ja: "再接続",
            ko: "재연결",
            zh: "重新连接"
        )
        public static let openLogs = L10nEntry(
            en: "Open logs",
            ja: "ログを開く",
            ko: "로그 열기",
            zh: "打开日志"
        )
        public static let start = L10nEntry(
            en: "Start",
            ja: "開始",
            ko: "시작",
            zh: "启动"
        )
        public static let stopBtn = L10nEntry(
            en: "Stop",
            ja: "停止",
            ko: "중지",
            zh: "停止"
        )
        public static let clearCaches = L10nEntry(
            en: "Clear caches",
            ja: "キャッシュをクリア",
            ko: "캐시 지우기",
            zh: "清除缓存"
        )
        public static let searchBtn = L10nEntry(
            en: "Search",
            ja: "検索",
            ko: "검색",
            zh: "搜索"
        )
        public static let download = L10nEntry(
            en: "Download",
            ja: "ダウンロード",
            ko: "다운로드",
            zh: "下载"
        )
        public static let save = L10nEntry(
            en: "Save",
            ja: "保存",
            ko: "저장",
            zh: "保存"
        )
        public static let undo = L10nEntry(
            en: "Undo",
            ja: "元に戻す",
            ko: "실행 취소",
            zh: "撤销"
        )
        public static let clear = L10nEntry(
            en: "Clear",
            ja: "クリア",
            ko: "지우기",
            zh: "清除"
        )
        public static let recallPrompt = L10nEntry(
            en: "Recall prompt",
            ja: "プロンプトを呼び出し",
            ko: "프롬프트 불러오기",
            zh: "调用提示"
        )
        public static let copied = L10nEntry(
            en: "Copied!",
            ja: "コピーしました！",
            ko: "복사됨!",
            zh: "已复制！"
        )
        public static let noMatches = L10nEntry(
            en: "No matches",
            ja: "一致なし",
            ko: "일치하지 않음",
            zh: "无匹配项"
        )
        public static let typeCommand = L10nEntry(
            en: "Type a command…",
            ja: "コマンドを入力…",
            ko: "명령 입력…",
            zh: "输入命令…"
        )
        public static let chatName = L10nEntry(
            en: "Chat name",
            ja: "チャット名",
            ko: "채팅 이름",
            zh: "聊天名称"
        )
        public static let filterModels = L10nEntry(
            en: "Filter models…",
            ja: "モデルをフィルター…",
            ko: "모델 필터링…",
            zh: "过滤模型…"
        )
        public static let selectForChat = L10nEntry(
            en: "Select for this chat",
            ja: "このチャットに選択",
            ko: "이 채팅에 선택",
            zh: "为此聊天选择"
        )
        public static let stopUnload = L10nEntry(
            en: "Stop / unload from RAM",
            ja: "停止 / RAM からアンロード",
            ko: "중지 / RAM에서 언로드",
            zh: "停止 / 从 RAM 卸载"
        )
        public static let startLoad = L10nEntry(
            en: "Start / load into RAM",
            ja: "開始 / RAM にロード",
            ko: "시작 / RAM에 로드",
            zh: "启动 / 加载到 RAM"
        )
        public static let showInServer = L10nEntry(
            en: "Show in Server tab",
            ja: "サーバータブに表示",
            ko: "서버 탭에 표시",
            zh: "在服务器选项卡中显示"
        )
        public static let manageInServer = L10nEntry(
            en: "Manage in Server tab",
            ja: "サーバータブで管理",
            ko: "서버 탭에서 관리",
            zh: "在服务器选项卡中管理"
        )
        public static let noImagesYet = L10nEntry(
            en: "No images yet",
            ja: "画像がまだありません",
            ko: "아직 이미지가 없습니다",
            zh: "尚无图像"
        )
        public static let noImagesHint = L10nEntry(
            en: "Pick a model, type a prompt, and hit Generate.",
            ja: "モデルを選択し、プロンプトを入力して生成を押してください。",
            ko: "모델을 선택하고 프롬프트를 입력한 다음 생성을 누르십시오.",
            zh: "选择模型，输入提示，然后点击生成。"
        )
        public static let removeFromScanList = L10nEntry(
            en: "Remove from scan list",
            ja: "スキャンリストから削除",
            ko: "스캔 목록에서 제거",
            zh: "从扫描列表中移除"
        )
        public static let name = L10nEntry(
            en: "Name",
            ja: "名前",
            ko: "이름",
            zh: "名称"
        )
        public static let command = L10nEntry(
            en: "Command",
            ja: "コマンド",
            ko: "명령",
            zh: "命令"
        )
        public static let argsHint = L10nEntry(
            en: "Args (one per line)",
            ja: "引数 (1 行につき 1 つ)",
            ko: "인수 (한 줄에 하나)",
            zh: "参数（每行一个）"
        )
        public static let urlLabel = L10nEntry(
            en: "URL",
            ja: "URL",
            ko: "URL",
            zh: "URL"
        )
        public static let envKV = L10nEntry(
            en: "KEY=VALUE per line",
            ja: "KEY=VALUE を 1 行につき",
            ko: "KEY=VALUE 줄당",
            zh: "每行 KEY=VALUE"
        )
        public static let enabledToggle = L10nEntry(
            en: "Enabled",
            ja: "有効",
            ko: "활성화",
            zh: "已启用"
        )
        public static let timeout = L10nEntry(
            en: "Timeout",
            ja: "タイムアウト",
            ko: "타임아웃",
            zh: "超时"
        )
        public static let skipSecurity = L10nEntry(
            en: "Skip security validation (dev only)",
            ja: "セキュリティ検証をスキップ (開発専用)",
            ko: "보안 검증 건너뛰기 (개발 전용)",
            zh: "跳过安全验证（仅开发）"
        )
        public static let pidFormat2 = L10nEntry(
            en: "PID %lld",
            ja: "PID %lld",
            ko: "PID %lld",
            zh: "PID %lld"
        )
        public static let apiRoutes = L10nEntry(
            en: "API ROUTES",
            ja: "API ルート",
            ko: "API 경로",
            zh: "API 路由"
        )
        public static let noRoutesMatch = L10nEntry(
            en: "No routes match your filter.",
            ja: "フィルターに一致するルートはありません。",
            ko: "필터와 일치하는 경로가 없습니다.",
            zh: "没有匹配您过滤器的路由。"
        )
        public static let sampleBody = L10nEntry(
            en: "Sample body",
            ja: "サンプルボディ",
            ko: "샘플 본문",
            zh: "示例主体"
        )
        public static let fullCurl = L10nEntry(
            en: "Full curl",
            ja: "完全な curl",
            ko: "전체 curl",
            zh: "完整 curl"
        )
        public static let routeFilterPlaceholder = L10nEntry(
            en: "Filter by path, family, or brief...",
            ja: "パス、ファミリー、または概要でフィルター...",
            ko: "경로, 패밀리 또는 간략 설명으로 필터...",
            zh: "按路径、系列或简介过滤..."
        )
    }

    public enum MCPUI {
        public static let mcpServers = L10nEntry(
            en: "MCP servers",
            ja: "MCP サーバー",
            ko: "MCP 서버",
            zh: "MCP 服务器"
        )
        public static let availableTools = L10nEntry(
            en: "AVAILABLE TOOLS",
            ja: "利用可能なツール",
            ko: "사용 가능한 도구",
            zh: "可用工具"
        )
        public static let noMCPConfigured = L10nEntry(
            en: "No MCP servers configured",
            ja: "MCP サーバーが設定されていません",
            ko: "구성된 MCP 서버가 없습니다",
            zh: "未配置 MCP 服务器"
        )
        public static let noMCPHint = L10nEntry(
            en: "Point at an mcp.json above and hit Reload. See the MCP spec for config format.",
            ja: "上記で mcp.json を指定して [再読み込み] を押してください。設定形式は MCP 仕様を参照してください。",
            ko: "위에서 mcp.json을 지정하고 다시 로드를 누르십시오. 구성 형식은 MCP 사양을 참조하십시오.",
            zh: "在上面指定 mcp.json 并点击重新加载。配置格式请参阅 MCP 规范。"
        )
        public static let reload = L10nEntry(
            en: "Reload",
            ja: "再読み込み",
            ko: "다시 로드",
            zh: "重新加载"
        )
        public static let browse = L10nEntry(
            en: "Browse…",
            ja: "参照…",
            ko: "찾아보기…",
            zh: "浏览…"
        )
        public static let pasteJSON = L10nEntry(
            en: "Paste JSON",
            ja: "JSON を貼り付け",
            ko: "JSON 붙여넣기",
            zh: "粘贴 JSON"
        )
        public static let mcpPathPlaceholder = L10nEntry(
            en: "Path to mcp.json — leave empty to use default",
            ja: "mcp.json のパス — デフォルトを使用するには空のままにしてください",
            ko: "mcp.json 경로 — 기본값을 사용하려면 비워 두십시오",
            zh: "mcp.json 的路径 — 留空以使用默认值"
        )
        public static let remove = L10nEntry(
            en: "Remove",
            ja: "削除",
            ko: "제거",
            zh: "移除"
        )
        public static let removeConfirmFormat = L10nEntry(
            en: "\"%@\" will be removed from mcp.json and stopped if running. This action is not undoable from the app.",
            ja: "「%@」は mcp.json から削除され、実行中の場合は停止されます。この操作はアプリから元に戻せません。",
            ko: "\"%@\"는 mcp.json에서 제거되며 실행 중인 경우 중지됩니다. 이 작업은 앱에서 취소할 수 없습니다.",
            zh: "\"%@\" 将从 mcp.json 中移除，如果正在运行将被停止。此操作无法从应用撤销。"
        )
    }

    public enum HFUI {
        public static let token = L10nEntry(
            en: "HuggingFace token",
            ja: "HuggingFace トークン",
            ko: "HuggingFace 토큰",
            zh: "HuggingFace 令牌"
        )
        public static let tokenHelp = L10nEntry(
            en: "Stored in Keychain. Used to download gated models (Gemma, Llama, etc.).",
            ja: "キーチェーンに保存されます。ゲート付きモデル (Gemma、Llama など) のダウンロードに使用されます。",
            ko: "키체인에 저장됩니다. 게이트된 모델 (Gemma, Llama 등) 다운로드에 사용됩니다.",
            zh: "存储在钥匙串中。用于下载受限模型（Gemma、Llama 等）。"
        )
        public static let saveToken = L10nEntry(
            en: "Save",
            ja: "保存",
            ko: "저장",
            zh: "保存"
        )
        public static let clearToken = L10nEntry(
            en: "Clear",
            ja: "クリア",
            ko: "지우기",
            zh: "清除"
        )
        public static let tokenPlaceholder = L10nEntry(
            en: "<huggingface-token>",
            ja: "<huggingface-token>",
            ko: "<huggingface-token>",
            zh: "<huggingface-token>"
        )
        public static let tokenSaved = L10nEntry(
            en: "Token saved to Keychain",
            ja: "トークンがキーチェーンに保存されました",
            ko: "토큰이 키체인에 저장되었습니다",
            zh: "令牌已保存到钥匙串"
        )
    }

    public enum APIUI {
        public static let endpoint = L10nEntry(
            en: "ENDPOINT",
            ja: "エンドポイント",
            ko: "엔드포인트",
            zh: "端点"
        )
        public static let runningSessions = L10nEntry(
            en: "RUNNING SESSIONS",
            ja: "実行中のセッション",
            ko: "실행 중인 세션",
            zh: "运行中的会话"
        )
        public static let noRunningSessions = L10nEntry(
            en: "No running sessions. Start a model from the Server tab.",
            ja: "実行中のセッションはありません。サーバータブからモデルを起動してください。",
            ko: "실행 중인 세션이 없습니다. 서버 탭에서 모델을 시작하십시오.",
            zh: "没有运行中的会话。请从服务器选项卡启动模型。"
        )
        public static let apiKeys = L10nEntry(
            en: "API KEYS",
            ja: "API キー",
            ko: "API 키",
            zh: "API 密钥"
        )
        public static let adminToken = L10nEntry(
            en: "ADMIN TOKEN",
            ja: "管理者トークン",
            ko: "관리자 토큰",
            zh: "管理令牌"
        )
        public static let adminTokenHelp = L10nEntry(
            en: "— gates /admin/* and /v1/cache/*",
            ja: "— /admin/* と /v1/cache/* を保護",
            ko: "— /admin/* 및 /v1/cache/* 보호",
            zh: "— 保护 /admin/* 和 /v1/cache/*"
        )
        public static let adminTokenPlaceholder = L10nEntry(
            en: "Leave blank to keep admin routes open",
            ja: "空白のままにすると管理ルートが開放されます",
            ko: "비워 두면 관리자 경로가 열린 상태로 유지됩니다",
            zh: "留空以保持管理路由开放"
        )
        public static let randomize = L10nEntry(
            en: "Randomize",
            ja: "ランダム化",
            ko: "임의 생성",
            zh: "随机生成"
        )
        public static let keyLabel = L10nEntry(
            en: "Key label",
            ja: "キーラベル",
            ko: "키 라벨",
            zh: "密钥标签"
        )
        public static let generate = L10nEntry(
            en: "Generate",
            ja: "生成",
            ko: "생성",
            zh: "生成"
        )
        public static let revoke = L10nEntry(
            en: "Revoke",
            ja: "取り消し",
            ko: "취소",
            zh: "吊销"
        )
        public static let revokeConfirm = L10nEntry(
            en: "Any client using this key will immediately lose access. This can't be undone — a new key will have a different value.",
            ja: "このキーを使用しているクライアントは即座にアクセスできなくなります。元に戻せません — 新しいキーは別の値になります。",
            ko: "이 키를 사용하는 모든 클라이언트는 즉시 액세스 권한을 잃게 됩니다. 취소할 수 없습니다 — 새 키는 다른 값을 가집니다.",
            zh: "任何使用此密钥的客户端将立即失去访问权限。此操作无法撤销 — 新密钥将具有不同的值。"
        )
        public static let requireBearer = L10nEntry(
            en: "Require Bearer auth",
            ja: "Bearer 認証を要求",
            ko: "Bearer 인증 필요",
            zh: "需要 Bearer 认证"
        )
        public static let phoneLAN = L10nEntry(
            en: "PHONE / LAN",
            ja: "スマホ / LAN",
            ko: "전화 / LAN",
            zh: "手机 / 局域网"
        )
        public static let scanFromDevice = L10nEntry(
            en: "Scan from another device on your network",
            ja: "ネットワーク上の別のデバイスからスキャン",
            ko: "네트워크의 다른 장치에서 스캔",
            zh: "从网络上的另一台设备扫描"
        )
        public static let portRangeError = L10nEntry(
            en: "Port must be between 1024 and 65535",
            ja: "ポートは 1024 から 65535 の範囲で指定してください",
            ko: "포트는 1024와 65535 사이여야 합니다",
            zh: "端口必须在 1024 到 65535 之间"
        )
        public static let portInUseFormat = L10nEntry(
            en: "Port %lld is already in use by another session",
            ja: "ポート %lld は別のセッションで既に使用されています",
            ko: "포트 %lld은(는) 이미 다른 세션에서 사용 중입니다",
            zh: "端口 %lld 已被另一个会话占用"
        )
        public static let codeSnippets = L10nEntry(
            en: "CODE SNIPPETS",
            ja: "コードスニペット",
            ko: "코드 스니펫",
            zh: "代码片段"
        )
        public static let lanToggle = L10nEntry(
            en: "LAN (0.0.0.0)",
            ja: "LAN (0.0.0.0)",
            ko: "LAN (0.0.0.0)",
            zh: "LAN (0.0.0.0)"
        )
    }

    public enum TrayUI {
        public static let host = L10nEntry(
            en: "Host",
            ja: "ホスト",
            ko: "호스트",
            zh: "主机"
        )
        public static let port = L10nEntry(
            en: "Port",
            ja: "ポート",
            ko: "포트",
            zh: "端口"
        )
        public static let url = L10nEntry(
            en: "URL",
            ja: "URL",
            ko: "URL",
            zh: "URL"
        )
        public static let copy = L10nEntry(
            en: "Copy",
            ja: "コピー",
            ko: "복사",
            zh: "复制"
        )
        public static let gatewayPort = L10nEntry(
            en: "Gateway port",
            ja: "ゲートウェイポート",
            ko: "게이트웨이 포트",
            zh: "网关端口"
        )
        public static let rateLimit = L10nEntry(
            en: "Rate limit (req/min/IP)",
            ja: "レート制限 (req/min/IP)",
            ko: "속도 제한 (req/min/IP)",
            zh: "速率限制 (req/min/IP)"
        )
        public static let rateLimitHelp = L10nEntry(
            en: "0 = unlimited. Applies to every new HTTP listener. Active listeners pick up on next (re)start.",
            ja: "0 = 無制限。新しい HTTP リスナーすべてに適用されます。アクティブなリスナーは次の（再）起動時に反映されます。",
            ko: "0 = 무제한. 모든 새 HTTP 리스너에 적용됩니다. 활성 리스너는 다음 (재)시작 시 반영됩니다.",
            zh: "0 = 无限制。适用于每个新的 HTTP 监听器。活动监听器在下次（重新）启动时应用。"
        )
        public static let maxTokens = L10nEntry(
            en: "Max tokens",
            ja: "最大トークン数",
            ko: "최대 토큰",
            zh: "最大令牌数"
        )
        public static let topK = L10nEntry(
            en: "Top-K",
            ja: "Top-K",
            ko: "Top-K",
            zh: "Top-K"
        )
        public static let kvCache = L10nEntry(
            en: "KV cache",
            ja: "KV キャッシュ",
            ko: "KV 캐시",
            zh: "KV 缓存"
        )
        public static let tqBits = L10nEntry(
            en: "TQ bits",
            ja: "TQ ビット",
            ko: "TQ 비트",
            zh: "TQ 位数"
        )
        public static let maxCacheBlocks = L10nEntry(
            en: "Max cache blocks",
            ja: "最大キャッシュブロック数",
            ko: "최대 캐시 블록",
            zh: "最大缓存块数"
        )
        public static let prefillStepSize = L10nEntry(
            en: "Prefill step size",
            ja: "プレフィルステップサイズ",
            ko: "프리필 스텝 크기",
            zh: "预填充步长"
        )
        public static let unlimited = L10nEntry(
            en: "unlimited",
            ja: "無制限",
            ko: "무제한",
            zh: "无限制"
        )
    }

    public enum Terminal {
        public static let terminal = L10nEntry(
            en: "Terminal",
            ja: "ターミナル",
            ko: "터미널",
            zh: "终端"
        )
        public static let terminalModeHelp = L10nEntry(
            en: "Terminal mode gives the model full shell access",
            ja: "ターミナルモードはモデルにフルシェルアクセスを許可します",
            ko: "터미널 모드는 모델에 완전한 셸 액세스를 제공합니다",
            zh: "终端模式为模型提供完整的 Shell 访问权限"
        )
    }

    // MARK: - Server / SessionConfigForm (§356 — extend i18n coverage)

    public enum SessionConfig {
        public static let sectionSessionConfig = L10nEntry(
            en: "SESSION CONFIG",
            ja: "セッション設定",
            ko: "세션 설정",
            zh: "会话配置"
        )
        public static let sectionModel = L10nEntry(
            en: "MODEL",
            ja: "モデル",
            ko: "모델",
            zh: "模型"
        )
        public static let sectionEngine = L10nEntry(
            en: "ENGINE",
            ja: "エンジン",
            ko: "엔진",
            zh: "引擎"
        )
        public static let sectionCache = L10nEntry(
            en: "CACHE",
            ja: "キャッシュ",
            ko: "캐시",
            zh: "缓存"
        )
        public static let sectionNetwork = L10nEntry(
            en: "NETWORK",
            ja: "ネットワーク",
            ko: "네트워크",
            zh: "网络"
        )
        public static let sectionRemote = L10nEntry(
            en: "REMOTE ENDPOINT",
            ja: "リモートエンドポイント",
            ko: "원격 엔드포인트",
            zh: "远程端点"
        )
        public static let sectionAdvanced = L10nEntry(
            en: "ADVANCED",
            ja: "詳細設定",
            ko: "고급",
            zh: "高级"
        )
        public static let restartRequired = L10nEntry(
            en: "Restart required — changes in this section apply only after the session is stopped and started again.",
            ja: "再起動が必要 — このセクションの変更はセッションを停止して再起動した後にのみ適用されます。",
            ko: "재시작 필요 — 이 섹션의 변경 사항은 세션을 중지하고 다시 시작한 후에만 적용됩니다.",
            zh: "需要重启 — 此部分的更改仅在会话停止并重新启动后生效。"
        )
        public static let noModelsFound = L10nEntry(
            en: "No models found",
            ja: "モデルが見つかりません",
            ko: "모델을 찾을 수 없음",
            zh: "未找到模型"
        )
        public static let browse = L10nEntry(
            en: "Browse…",
            ja: "参照…",
            ko: "찾아보기…",
            zh: "浏览…"
        )
        public static let addCustomDir = L10nEntry(
            en: "Add custom dir…",
            ja: "カスタムフォルダを追加…",
            ko: "사용자 지정 디렉토리 추가…",
            zh: "添加自定义目录…"
        )
        public static let enableLANAccess = L10nEntry(
            en: "Enable LAN access",
            ja: "LAN アクセスを有効化",
            ko: "LAN 액세스 활성화",
            zh: "启用局域网访问"
        )
        public static let lanWarning = L10nEntry(
            en: "Every device on your Wi-Fi will be able to reach this session's HTTP server. Make sure you trust the network before enabling — nothing gates access beyond the optional API-key bearer token. You can turn this off again at any time.",
            ja: "Wi-Fi 上のすべてのデバイスがこのセッションの HTTP サーバーにアクセスできるようになります。有効化する前にネットワークを信頼していることを確認してください — オプションの API キーベアラートークン以外にアクセスを制限するものはありません。いつでもオフに戻せます。",
            ko: "Wi-Fi의 모든 장치가 이 세션의 HTTP 서버에 연결할 수 있게 됩니다. 활성화하기 전에 네트워크를 신뢰하는지 확인하십시오 — 선택적 API 키 베어러 토큰 외에 액세스를 차단하는 것은 없습니다. 언제든지 다시 끌 수 있습니다.",
            zh: "Wi-Fi 上的每个设备都将能够访问此会话的 HTTP 服务器。启用前请确保您信任该网络 — 除了可选的 API 密钥承载令牌外，没有任何限制。您可以随时将其关闭。"
        )
        public static let httpListenerHelp = L10nEntry(
            en: "This session's dedicated HTTP listener. Per-session bind is independent from the global gateway — flip the Tray's Gateway toggle if you want one URL across all sessions instead.",
            ja: "このセッション専用の HTTP リスナー。セッションごとのバインドはグローバルゲートウェイとは独立しています — すべてのセッションで 1 つの URL を使用したい場合はトレイのゲートウェイトグルを切り替えてください。",
            ko: "이 세션 전용 HTTP 리스너입니다. 세션별 바인딩은 전역 게이트웨이와 독립적입니다 — 모든 세션에 대해 하나의 URL을 원한다면 트레이의 게이트웨이 토글을 전환하십시오.",
            zh: "此会话的专用 HTTP 监听器。每个会话的绑定独立于全局网关 — 如果您希望所有会话使用一个 URL，请切换托盘的网关开关。"
        )
        public static let cacheKindHelp = L10nEntry(
            en: "TurboQuant adapts per-layer bits from the KV distribution; Q4/Q8 are fixed-width fallbacks.",
            ja: "TurboQuant は KV 分布からレイヤーごとのビット数を適応させます。Q4/Q8 は固定幅のフォールバックです。",
            ko: "TurboQuant은 KV 분포에서 레이어별 비트 수를 적응시킵니다. Q4/Q8은 고정 너비 대체입니다.",
            zh: "TurboQuant 根据 KV 分布自适应每层位数；Q4/Q8 是固定宽度的回退。"
        )
        public static let remoteEndpointHelp = L10nEntry(
            en: "This session will skip local model loading. All chat / terminal traffic for chats bound to it goes over HTTP to the remote endpoint above. The local engine surface (sleep, cache, gateway) stays inactive while remote mode is on.",
            ja: "このセッションはローカルモデルのロードをスキップします。バインドされたチャット/ターミナルトラフィックはすべて上記のリモートエンドポイントに HTTP 経由で送信されます。リモートモードがオンの間、ローカルエンジン機能 (スリープ、キャッシュ、ゲートウェイ) は非アクティブのままです。",
            ko: "이 세션은 로컬 모델 로딩을 건너뜁니다. 바인딩된 채팅 / 터미널 트래픽은 모두 위의 원격 엔드포인트로 HTTP를 통해 전송됩니다. 원격 모드가 켜져 있는 동안 로컬 엔진 기능 (슬립, 캐시, 게이트웨이)은 비활성 상태로 유지됩니다.",
            zh: "此会话将跳过本地模型加载。绑定到它的所有聊天/终端流量将通过 HTTP 发送到上面的远程端点。启用远程模式时，本地引擎功能（休眠、缓存、网关）保持非活动状态。"
        )
        public static let useRemoteEndpoint = L10nEntry(
            en: "Use remote endpoint instead of local engine",
            ja: "ローカルエンジンの代わりにリモートエンドポイントを使用",
            ko: "로컬 엔진 대신 원격 엔드포인트 사용",
            zh: "使用远程端点而非本地引擎"
        )
        public static let endpointURL = L10nEntry(
            en: "Endpoint URL",
            ja: "エンドポイント URL",
            ko: "엔드포인트 URL",
            zh: "端点 URL"
        )
        public static let proto = L10nEntry(
            en: "Protocol",
            ja: "プロトコル",
            ko: "프로토콜",
            zh: "协议"
        )
        public static let modelName = L10nEntry(
            en: "Model name",
            ja: "モデル名",
            ko: "모델 이름",
            zh: "模型名称"
        )
        public static let apiKey = L10nEntry(
            en: "API key",
            ja: "API キー",
            ko: "API 키",
            zh: "API 密钥"
        )
        // Iter 143 — Smelt removed (Eric directive 2026-05-04). Cold-
        // expert handling now lives in JangPress.
        public static let distributedHelp = L10nEntry(
            en: "Distributed compute across multiple Macs is planned for v1.1 (feat/distributed-rdma branch). Toggle persists to settings but has no runtime effect yet.",
            ja: "複数の Mac にまたがる分散コンピューティングは v1.1 で予定されています (feat/distributed-rdma ブランチ)。トグルは設定に保存されますが、まだ実行時の効果はありません。",
            ko: "여러 Mac에 걸친 분산 컴퓨팅은 v1.1에 계획되어 있습니다 (feat/distributed-rdma 브랜치). 토글은 설정에 저장되지만 아직 런타임 효과는 없습니다.",
            zh: "跨多台 Mac 的分布式计算计划在 v1.1 中推出（feat/distributed-rdma 分支）。切换会保存到设置但尚未产生运行时效果。"
        )
        public static let turboquantBitsFormat = L10nEntry(
            en: "TurboQuant bits: %lld",
            ja: "TurboQuant ビット: %lld",
            ko: "TurboQuant 비트: %lld",
            zh: "TurboQuant 位数: %lld"
        )
        // Iter 143 — JangPress UI scope hint. Eric directive: make the
        // section header explicit about Global vs per-session scope so
        // users moving from the Python panel don't expect Smelt-style
        // per-session behavior.
        public static let jangPressSectionHeader = L10nEntry(
            en: "JangPress (Global — applies to next model load)",
            ja: "JangPress（グローバル — 次回のモデル読み込み時に適用）",
            ko: "JangPress (전역 — 다음 모델 로드 시 적용)",
            zh: "JangPress（全局 — 在下次模型加载时应用）"
        )
        public static let jangPressScopeCaption = L10nEntry(
            en: "Load-time global setting. Changes apply on the next model load — restart the engine to take effect. The mmap backend can drop auxiliary file-backed pages under pressure; it does not replace or compress MLX's canonical model-weight copy. Use cache stats to verify what is actually active.",
            ja: "ロード時のグローバル設定です。変更は次回のモデル読み込み時に適用されます — 反映するにはエンジンを再起動してください。mmap バックエンドは圧迫時に補助的なファイルバッキングページを破棄できます。MLX の正規モデル重みコピーは置き換えや圧縮されません。実際に有効なものはキャッシュ統計で確認してください。",
            ko: "로드 시 전역 설정입니다. 변경 사항은 다음 모델 로드 시 적용됩니다 — 적용하려면 엔진을 재시작하세요. mmap 백엔드는 압박 상황에서 보조 파일 기반 페이지를 폐기할 수 있습니다. MLX의 정식 모델 가중치 사본은 교체되거나 압축되지 않습니다. 실제로 활성화된 항목은 캐시 통계로 확인하세요.",
            zh: "加载时全局设置。更改将在下次模型加载时应用 — 请重启引擎以生效。mmap 后端可在压力下丢弃辅助文件支持的页面；不替换或压缩 MLX 的标准模型权重副本。请使用缓存统计来验证实际生效的内容。"
        )
        public static let diskCacheDirFormat = L10nEntry(
            en: "Disk cache dir: %@",
            ja: "ディスクキャッシュディレクトリ: %@",
            ko: "디스크 캐시 디렉토리: %@",
            zh: "磁盘缓存目录: %@"
        )
        public static let diskCacheMaxFormat = L10nEntry(
            en: "Disk cache max: %lld GB",
            ja: "ディスクキャッシュ最大: %lld GB",
            ko: "디스크 캐시 최대: %lld GB",
            zh: "磁盘缓存上限: %lld GB"
        )
        public static let defaultLabel = L10nEntry(
            en: "default",
            ja: "デフォルト",
            ko: "기본값",
            zh: "默认"
        )
    }

    // MARK: - Reusable picker/option labels

    public enum Option {
        public static let none = L10nEntry(
            en: "None",
            ja: "なし",
            ko: "없음",
            zh: "无"
        )
        public static let off = L10nEntry(
            en: "Off",
            ja: "オフ",
            ko: "끄기",
            zh: "关闭"
        )
        public static let auto = L10nEntry(
            en: "Auto",
            ja: "自動",
            ko: "자동",
            zh: "自动"
        )
    }

    /// §372 — Cursor-hover tooltips (SwiftUI `.help("…")`) across every
    /// surface of the app. Each entry is a short noun/verb phrase so
    /// translators have enough context to keep labels natural in
    /// non-English locales. Added for N6 (Help/info tooltips translated).
    public enum Tooltip {
        // Chat surface
        public static let attachImages = L10nEntry(
            en: "Attach images (drag, paste, or pick from disk)",
            ja: "画像を添付(ドラッグ、貼り付け、またはディスクから選択)",
            ko: "이미지 첨부(드래그, 붙여넣기 또는 디스크에서 선택)",
            zh: "附加图像(拖放、粘贴或从磁盘选取)"
        )
        public static let clickToZoom = L10nEntry(
            en: "Click to zoom",
            ja: "クリックで拡大",
            ko: "클릭하여 확대",
            zh: "点击缩放"
        )
        public static let deleteMessage = L10nEntry(
            en: "Delete message",
            ja: "メッセージを削除",
            ko: "메시지 삭제",
            zh: "删除消息"
        )
        public static let scrollToBottom = L10nEntry(
            en: "Scroll to bottom",
            ja: "一番下までスクロール",
            ko: "맨 아래로 스크롤",
            zh: "滚动到底部"
        )
        public static let scrollToNewestLog = L10nEntry(
            en: "Scroll to newest log line",
            ja: "最新のログ行へスクロール",
            ko: "최신 로그로 스크롤",
            zh: "滚动到最新日志"
        )
        public static let renameChat = L10nEntry(
            en: "Rename chat",
            ja: "チャット名を変更",
            ko: "채팅 이름 변경",
            zh: "重命名对话"
        )
        public static let deleteChat = L10nEntry(
            en: "Delete chat",
            ja: "チャットを削除",
            ko: "채팅 삭제",
            zh: "删除对话"
        )
        public static let copyReasoningOnly = L10nEntry(
            en: "Copy reasoning only",
            ja: "推論のみコピー",
            ko: "추론만 복사",
            zh: "仅复制推理内容"
        )
        public static let modelPicker = L10nEntry(
            en: "Pick a model · colored dot = load state. Right side ▶ / ■ toggles the model in/out of RAM without leaving the Chat tab.",
            ja: "モデルを選択 · 色付きドットは読み込み状態。右側の ▶ / ■ でチャットタブを離れずにモデルをRAMに出し入れできます。",
            ko: "모델 선택 · 색 점은 로드 상태. 오른쪽 ▶ / ■ 로 채팅 탭을 떠나지 않고 모델을 RAM에 로드/언로드합니다.",
            zh: "选择模型 · 彩色圆点表示加载状态。右侧 ▶ / ■ 可在不离开聊天标签的情况下加载/卸载模型。"
        )

        // Server surface
        public static let sessionDashboardLoad = L10nEntry(
            en: "Pick a model from disk and load it into a new server session",
            ja: "ディスクからモデルを選択し、新しいサーバーセッションに読み込みます",
            ko: "디스크에서 모델을 선택하여 새 서버 세션에 로드",
            zh: "从磁盘选取模型并加载到新的服务器会话中"
        )
        public static let resetPeakToCurrent = L10nEntry(
            en: "Reset peak to current",
            ja: "ピークを現在値にリセット",
            ko: "현재 값으로 피크 재설정",
            zh: "将峰值重置为当前值"
        )
        public static let rescanDirs = L10nEntry(
            en: "Force a fresh disk walk of every model directory",
            ja: "すべてのモデルディレクトリを強制的に再スキャン",
            ko: "모든 모델 디렉토리를 새로 디스크 스캔",
            zh: "强制重新扫描所有模型目录"
        )
        public static let pickFolder = L10nEntry(
            en: "Pick a folder containing model directories to scan",
            ja: "スキャンするモデルディレクトリを含むフォルダを選択",
            ko: "스캔할 모델 디렉토리가 포함된 폴더 선택",
            zh: "选取包含模型目录的文件夹进行扫描"
        )
        public static let hfDownload = L10nEntry(
            en: "Queue a HuggingFace repo download. Progress opens in the Downloads window.",
            ja: "HuggingFaceリポジトリのダウンロードをキューに追加。進捗はダウンロードウィンドウに表示されます。",
            ko: "HuggingFace 저장소 다운로드를 큐에 추가. 진행 상황은 다운로드 창에 표시됩니다.",
            zh: "将 HuggingFace 仓库加入下载队列,进度在下载窗口中显示。"
        )
        public static let revealInFinder = L10nEntry(
            en: "Reveal in Finder",
            ja: "Finderで表示",
            ko: "Finder에서 보기",
            zh: "在 Finder 中显示"
        )
        public static let removeDir = L10nEntry(
            en: "Remove this directory from the scan list (does not delete files)",
            ja: "このディレクトリをスキャン一覧から削除(ファイルは削除されません)",
            ko: "스캔 목록에서 이 디렉토리 제거(파일은 삭제되지 않음)",
            zh: "从扫描列表中移除此目录(不会删除文件)"
        )
        public static let cacheWarm5 = L10nEntry(
            en: "Prefills the prefix cache with your 5 most recent user prompts so the next send has a warm cache hit.",
            ja: "最近のユーザープロンプト5件でプレフィックスキャッシュを事前に満たし、次の送信でウォームヒットさせます。",
            ko: "최근 사용자 프롬프트 5건으로 접두 캐시를 미리 채워 다음 전송 시 웜 히트를 얻습니다.",
            zh: "用最近 5 条用户提示词预热前缀缓存,让下一次发送命中热缓存。"
        )
        public static let sectionRestart = L10nEntry(
            en: "Changes to fields under this section require a session restart to apply.",
            ja: "このセクション内のフィールドを変更するには、セッションの再起動が必要です。",
            ko: "이 섹션의 필드 변경은 세션 재시작이 필요합니다.",
            zh: "此部分下字段的更改需要重启会话才能生效。"
        )

        // Image / API / MCP / Tray
        public static let imageSettings = L10nEntry(
            en: "Image settings",
            ja: "画像設定",
            ko: "이미지 설정",
            zh: "图像设置"
        )
        public static let removeStoredToken = L10nEntry(
            en: "Remove stored token",
            ja: "保存されたトークンを削除",
            ko: "저장된 토큰 제거",
            zh: "移除已存储的令牌"
        )
        public static let mcpAddNew = L10nEntry(
            en: "Add a new MCP server to mcp.json",
            ja: "新しいMCPサーバーをmcp.jsonに追加",
            ko: "mcp.json에 새 MCP 서버 추가",
            zh: "向 mcp.json 添加新的 MCP 服务器"
        )
        public static let mcpImportClipboard = L10nEntry(
            en: "Paste a Claude-Desktop-style mcp.json block from the clipboard and import every server.",
            ja: "クリップボードからClaude Desktop形式のmcp.jsonブロックを貼り付け、全サーバーをインポート。",
            ko: "클립보드에서 Claude Desktop 형식 mcp.json 블록을 붙여넣어 모든 서버를 가져옵니다.",
            zh: "从剪贴板粘贴 Claude Desktop 格式的 mcp.json 块,导入全部服务器。"
        )
        public static let mcpEdit = L10nEntry(
            en: "Edit configuration",
            ja: "設定を編集",
            ko: "설정 편집",
            zh: "编辑配置"
        )
        public static let mcpRemove = L10nEntry(
            en: "Remove from mcp.json",
            ja: "mcp.jsonから削除",
            ko: "mcp.json에서 제거",
            zh: "从 mcp.json 中移除"
        )
        public static let mcpCmdPath = L10nEntry(
            en: "Absolute path or PATH-resolvable executable",
            ja: "絶対パスまたはPATHで解決可能な実行ファイル",
            ko: "절대 경로 또는 PATH로 해석 가능한 실행 파일",
            zh: "绝对路径或可通过 PATH 解析的可执行文件"
        )
        public static let mcpSseUrl = L10nEntry(
            en: "Full https:// URL of the MCP SSE endpoint",
            ja: "MCP SSEエンドポイントの完全な https:// URL",
            ko: "MCP SSE 엔드포인트의 완전한 https:// URL",
            zh: "MCP SSE 端点的完整 https:// URL"
        )
        public static let revokeApiKey = L10nEntry(
            en: "Revoke API key",
            ja: "APIキーを取り消す",
            ko: "API 키 해지",
            zh: "吊销 API 密钥"
        )
        public static let inspectRequest = L10nEntry(
            en: "Click to inspect this request",
            ja: "クリックでこのリクエストを検査",
            ko: "클릭하여 이 요청 검사",
            zh: "点击检查此请求"
        )

        // Tray gateway port-bump / failure tooltips. These use %@ so the
        // caller can interpolate port numbers / error messages via
        // L10nEntry.format(locale:_:).
        public static let gatewayPortBumped = L10nEntry(
            en: "Requested port %@ was taken. Bound to %@ instead. Change the port above to reclaim %@ after freeing it.",
            ja: "要求ポート%@は使用中でした。代わりに%@にバインドしました。%@を解放してから上記で再設定してください。",
            ko: "요청한 포트 %@은 사용 중입니다. 대신 %@에 바인딩했습니다. %@를 해제한 후 위에서 재지정하세요.",
            zh: "请求端口 %@ 已被占用,已绑定到 %@。释放后可在上方重新指定 %@。"
        )
        public static let gatewayBindFailed = L10nEntry(
            en: "Gateway failed to bind on port %@. %@",
            ja: "ゲートウェイはポート%@にバインドできませんでした。%@",
            ko: "게이트웨이가 포트 %@ 바인딩에 실패했습니다. %@",
            zh: "网关无法绑定端口 %@。%@"
        )
        public static let unreadErrorLogs = L10nEntry(
            en: "%@ unread error log entries — open Logs to clear",
            ja: "%@件の未読エラーログ — Logsを開いてクリア",
            ko: "읽지 않은 오류 로그 %@건 — Logs에서 확인하여 삭제",
            zh: "%@ 条未读错误日志 — 打开 Logs 清除"
        )
    }

    /// §377 — tray disclosure group headers + cache/MoE/logging toggle
    /// labels. TrayUI already holds the picker / lifecycle labels; this
    /// extends coverage so every user-visible string in TrayItem uses
    /// L10n.
    public enum TrayPanel {
        public static let serverBinding = L10nEntry(
            en: "Server binding",
            ja: "サーバーバインド",
            ko: "서버 바인딩",
            zh: "服务器绑定"
        )
        public static let samplingGlobals = L10nEntry(
            en: "Sampling (global defaults)",
            ja: "サンプリング(グローバル既定)",
            ko: "샘플링(전역 기본값)",
            zh: "采样(全局默认)"
        )
        public static let cache = L10nEntry(
            en: "Cache",
            ja: "キャッシュ",
            ko: "캐시",
            zh: "缓存"
        )
        public static let flashMoE = L10nEntry(
            en: "Flash MoE",
            ja: "Flash MoE",
            ko: "Flash MoE",
            zh: "Flash MoE"
        )
        public static let adapter = L10nEntry(
            en: "Adapter",
            ja: "アダプター",
            ko: "어댑터",
            zh: "适配器"
        )
        public static let logging = L10nEntry(
            en: "Logging",
            ja: "ログ",
            ko: "로깅",
            zh: "日志"
        )
        public static let recentLogs = L10nEntry(
            en: "Recent logs",
            ja: "最近のログ",
            ko: "최근 로그",
            zh: "最近日志"
        )

        // Cache section labels
        public static let l2DiskCache = L10nEntry(
            en: "L2 disk cache",
            ja: "L2ディスクキャッシュ",
            ko: "L2 디스크 캐시",
            zh: "L2 磁盘缓存"
        )
        public static let diskBudgetGB = L10nEntry(
            en: "Disk budget (GB)",
            ja: "ディスク上限 (GB)",
            ko: "디스크 한도 (GB)",
            zh: "磁盘预算 (GB)"
        )
        public static let prefixCache = L10nEntry(
            en: "Prefix cache",
            ja: "プレフィックスキャッシュ",
            ko: "접두 캐시",
            zh: "前缀缓存"
        )
        public static let memoryAwarePrefixCache = L10nEntry(
            en: "Memory-aware prefix cache",
            ja: "メモリ対応プレフィックスキャッシュ",
            ko: "메모리 인식 접두 캐시",
            zh: "内存感知前缀缓存"
        )
        public static let ssmReDerive = L10nEntry(
            en: "SSM re-derive (hybrid+thinking)",
            ja: "SSM再導出(ハイブリッド+思考)",
            ko: "SSM 재유도(하이브리드+사고)",
            zh: "SSM 重新推导(混合+思考)"
        )
        public static let cacheMemoryPercent = L10nEntry(
            en: "Cache memory %",
            ja: "キャッシュメモリ %",
            ko: "캐시 메모리 %",
            zh: "缓存内存 %"
        )

        // Flash MoE section
        public static let enableFlashMoE = L10nEntry(
            en: "Enable Flash MoE (stream experts from SSD)",
            ja: "Flash MoEを有効化 (SSDから専門家をストリーム)",
            ko: "Flash MoE 활성화 (SSD에서 전문가 스트리밍)",
            zh: "启用 Flash MoE(从 SSD 流式加载专家)"
        )
        public static let slotBankSize = L10nEntry(
            en: "Slot bank size",
            ja: "スロットバンクサイズ",
            ko: "슬롯 뱅크 크기",
            zh: "槽位组大小"
        )

        // Adapter section
        public static let adapterBlurb = L10nEntry(
            en: "Active LoRA adapter is managed via Server → Adapter panel.",
            ja: "アクティブなLoRAアダプターは Server → Adapter パネルで管理します。",
            ko: "활성 LoRA 어댑터는 Server → Adapter 패널에서 관리합니다.",
            zh: "活动 LoRA 适配器通过 Server → Adapter 面板管理。"
        )
        public static let openAdapterPanel = L10nEntry(
            en: "Open Adapter panel",
            ja: "Adapter パネルを開く",
            ko: "Adapter 패널 열기",
            zh: "打开 Adapter 面板"
        )

        // Logging section
        public static let defaultLevel = L10nEntry(
            en: "Default level",
            ja: "既定レベル",
            ko: "기본 수준",
            zh: "默认级别"
        )
    }

    // MARK: - NSOpenPanel macOS 26 XPC fallback (iter 134)
    //
    // Iter 128-129 (vmlx#121, #133) added a manual-path entry alert
    // shown when NSOpenPanel's XPC connection fails on macOS 26 ad-
    // hoc-signed builds. The strings below are user-facing — they
    // appear in the alert title / message and in banner text. Iter
    // 134 routes them through L10n so non-English users see localized
    // copy.
    public enum PickerFallback {
        public static let modelDirTitle = L10nEntry(
            en: "Type a model directory path",
            ja: "モデルディレクトリのパスを入力",
            ko: "모델 디렉터리 경로 입력",
            zh: "输入模型目录路径"
        )
        public static let modelDirMessage = L10nEntry(
            en: "macOS blocked the file picker (XPC error common on ad-hoc-signed builds, vmlx#121). Enter the directory path manually below — it will be scanned just like a picker selection.",
            ja: "macOS がファイルピッカーをブロックしました（ad-hoc 署名ビルドで発生する XPC エラー、vmlx#121）。下にディレクトリのパスを直接入力してください — ピッカーで選択した場合と同じようにスキャンされます。",
            ko: "macOS가 파일 선택기를 차단했습니다 (ad-hoc 서명 빌드에서 흔한 XPC 오류, vmlx#121). 아래에 디렉터리 경로를 직접 입력하세요 — 선택기로 고른 것과 동일하게 스캔됩니다.",
            zh: "macOS 已阻止文件选择器 (ad-hoc 签名构建上常见的 XPC 错误，vmlx#121)。请在下方手动输入目录路径 — 它将像选择器选择一样被扫描。"
        )
        public static let cwdTitle = L10nEntry(
            en: "Type working-directory path",
            ja: "作業ディレクトリのパスを入力",
            ko: "작업 디렉터리 경로 입력",
            zh: "输入工作目录路径"
        )
        public static let cwdMessage = L10nEntry(
            en: "macOS blocked the file picker (XPC error common on ad-hoc builds, vmlx#121). Type the directory the bash tool should run in.",
            ja: "macOS がファイルピッカーをブロックしました（ad-hoc ビルドで発生する XPC エラー、vmlx#121）。bash ツールを実行するディレクトリを入力してください。",
            ko: "macOS가 파일 선택기를 차단했습니다 (ad-hoc 빌드에서 흔한 XPC 오류, vmlx#121). bash 도구가 실행될 디렉터리를 입력하세요.",
            zh: "macOS 已阻止文件选择器 (ad-hoc 构建上常见的 XPC 错误，vmlx#121)。请输入 bash 工具运行所在目录。"
        )
        public static let mcpJsonTitle = L10nEntry(
            en: "Type mcp.json path",
            ja: "mcp.json のパスを入力",
            ko: "mcp.json 경로 입력",
            zh: "输入 mcp.json 路径"
        )
        public static let mcpJsonMessage = L10nEntry(
            en: "macOS blocked the file picker (XPC error common on ad-hoc builds, vmlx#121). Type the path to your mcp.json config.",
            ja: "macOS がファイルピッカーをブロックしました（ad-hoc ビルドで発生する XPC エラー、vmlx#121）。mcp.json 設定ファイルのパスを入力してください。",
            ko: "macOS가 파일 선택기를 차단했습니다 (ad-hoc 빌드에서 흔한 XPC 오류, vmlx#121). mcp.json 설정 파일의 경로를 입력하세요.",
            zh: "macOS 已阻止文件选择器 (ad-hoc 构建上常见的 XPC 错误，vmlx#121)。请输入 mcp.json 配置文件的路径。"
        )
        public static let tlsFileTitle = L10nEntry(
            en: "Type TLS file path",
            ja: "TLS ファイルのパスを入力",
            ko: "TLS 파일 경로 입력",
            zh: "输入 TLS 文件路径"
        )
        public static let tlsFileMessage = L10nEntry(
            en: "macOS blocked the file picker (XPC error common on ad-hoc builds, vmlx#121). Type the absolute path to your TLS certificate or key file.",
            ja: "macOS がファイルピッカーをブロックしました（ad-hoc ビルドで発生する XPC エラー、vmlx#121）。TLS 証明書またはキーファイルの絶対パスを入力してください。",
            ko: "macOS가 파일 선택기를 차단했습니다 (ad-hoc 빌드에서 흔한 XPC 오류, vmlx#121). TLS 인증서 또는 키 파일의 절대 경로를 입력하세요.",
            zh: "macOS 已阻止文件选择器 (ad-hoc 构建上常见的 XPC 错误，vmlx#121)。请输入 TLS 证书或密钥文件的绝对路径。"
        )
        public static let chatWorkingDirTitle = L10nEntry(
            en: "Type working directory path",
            ja: "作業ディレクトリのパスを入力",
            ko: "작업 디렉터리 경로 입력",
            zh: "输入工作目录路径"
        )
        public static let chatWorkingDirMessage = L10nEntry(
            en: "macOS blocked the file picker (XPC error, vmlx#121). Type the directory the bash tool should run in.",
            ja: "macOS がファイルピッカーをブロックしました（XPC エラー、vmlx#121）。bash ツールを実行するディレクトリを入力してください。",
            ko: "macOS가 파일 선택기를 차단했습니다 (XPC 오류, vmlx#121). bash 도구가 실행될 디렉터리를 입력하세요.",
            zh: "macOS 已阻止文件选择器 (XPC 错误，vmlx#121)。请输入 bash 工具运行所在目录。"
        )
        /// Banner shown after a successful manual-path fallback. Format
        /// arg is the directory's last-component name.
        public static let manualPathBanner = L10nEntry(
            en: "Used manual path entry (file picker blocked by macOS 26 XPC). Adding %@…",
            ja: "手動パス入力を使用しました（macOS 26 XPC によりファイルピッカーがブロック）。%@ を追加しています…",
            ko: "수동 경로 입력을 사용했습니다 (macOS 26 XPC가 파일 선택기를 차단). %@을(를) 추가 중…",
            zh: "已使用手动路径输入 (macOS 26 XPC 阻止了文件选择器)。正在添加 %@…"
        )
        /// Banner shown when manual fallback also fails (empty path,
        /// non-existent path, etc.). Format arg is the failure reason.
        public static let manualPathFailure = L10nEntry(
            en: "Could not add directory: %@",
            ja: "ディレクトリを追加できませんでした：%@",
            ko: "디렉터리를 추가할 수 없습니다: %@",
            zh: "无法添加目录：%@"
        )
    }
}
