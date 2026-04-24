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
        public static let smeltHelp = L10nEntry(
            en: "Smelt (partial expert loading) is Python-only today. Toggle persists but the Swift engine loads full experts and emits a `smelt mode is enabled but not wired` warning per request. DFlash below is the Swift equivalent for speculative decode.",
            ja: "Smelt (部分的エキスパートロード) は現在 Python 専用です。トグルは保存されますが、Swift エンジンはフルエキスパートをロードし、リクエストごとに `smelt mode is enabled but not wired` 警告を出します。下の DFlash は投機的デコードの Swift 版です。",
            ko: "Smelt (부분 전문가 로딩)은 현재 Python 전용입니다. 토글은 유지되지만 Swift 엔진은 전체 전문가를 로드하고 요청당 `smelt mode is enabled but not wired` 경고를 표시합니다. 아래 DFlash는 추측 디코딩의 Swift 등가물입니다.",
            zh: "Smelt（部分专家加载）目前仅支持 Python。切换会保存但 Swift 引擎会加载完整专家，每次请求发出 `smelt mode is enabled but not wired` 警告。下方的 DFlash 是推测解码的 Swift 等效项。"
        )
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
}
