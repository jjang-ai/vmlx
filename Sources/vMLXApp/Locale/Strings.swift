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
        public static let reset = L10nEntry(
            en: "Reset",
            ja: "リセット",
            ko: "재설정",
            zh: "重置"
        )
        public static let resetAll = L10nEntry(
            en: "Reset all",
            ja: "すべてリセット",
            ko: "전체 재설정",
            zh: "全部重置"
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
