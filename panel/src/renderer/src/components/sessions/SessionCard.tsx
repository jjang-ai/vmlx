import { useState, useEffect } from "react";
import { AlertTriangle, FolderSearch, Settings, ScrollText, Moon, Sun, Trash2 } from "lucide-react";
import { useSessionsContext } from "../../contexts/SessionsContext";
import { useTranslation } from "../../i18n";
import { compactQuantizationBadgeLabel } from "../../lib/quantizationBadge";
import { formatModelBytes, formatResidentLoad } from './loadProgressFormat'

interface Session {
  id: string;
  modelPath: string;
  modelName?: string;
  host: string;
  port: number;
  pid?: number;
  status: "running" | "stopped" | "error" | "loading" | "standby";
  standbyDepth?: "soft" | "deep" | null;
  config: string;
  createdAt: number;
  updatedAt: number;
  lastStartedAt?: number;
  lastStoppedAt?: number;
  type?: "local" | "remote";
  remoteUrl?: string;
  remoteModel?: string;
  modelPathMissing: boolean;
  usableTwinId?: string;
}

interface SessionCardProps {
  session: Session;
  onOpen: (sessionId: string) => void;
  onConfigure: (sessionId: string) => void;
  onStart: (sessionId: string) => void;
  onStop: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
  onRepoint: (sessionId: string) => void;
  onSleep?: (sessionId: string) => void;
  onWake?: (sessionId: string) => void;
}

const statusColors: Record<string, string> = {
  running: "bg-green-500",
  stopped: "bg-muted-foreground",
  error: "bg-destructive",
  loading: "bg-yellow-500 animate-pulse",
  standby: "bg-blue-400",
};

function formatElapsed(secs: number): string {
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}m ${s}s`;
}



export function SessionCard({
  session,
  onOpen,
  onConfigure,
  onStart,
  onStop,
  onDelete,
  onRepoint,
  onSleep,
  onWake,
}: SessionCardProps) {
  const { t } = useTranslation();
  const statusLabels: Record<string, string> = {
    running: t('status.running'),
    stopped: t('status.stopped'),
    error: t('status.error'),
    loading: t('status.loading'),
    standby: t('status.sleeping'),
  };
  const isRemote = session.type === "remote";
  const isImage = (() => {
    try {
      return JSON.parse(session.config || "{}").modelType === "image";
    } catch {
      return false;
    }
  })();
  const shortName =
    session.modelName ||
    session.modelPath.split("/").pop() ||
    session.modelPath;
  const [jangLabel, setJangLabel] = useState<string | undefined>(undefined);
  const [loadingElapsed, setLoadingElapsed] = useState(0);
  const { loadProgress } = useSessionsContext();
  const progress = loadProgress.get(session.id);
  const residentLoad = formatResidentLoad(progress);

  // Elapsed time counter when model is loading
  useEffect(() => {
    if (session.status !== "loading") {
      setLoadingElapsed(0);
      return;
    }
    const interval = setInterval(() => {
      setLoadingElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [session.status]);

  useEffect(() => {
    if (isRemote) return;
    // Show a cheap path-derived fallback immediately, then replace it with the
    // authoritative bundle-derived label from config.json/jang_config.json. The
    // fallback keeps missing/stale paths legible but must never collapse a real
    // JANGTQ/MXTQ bundle to generic JANG once detection completes.
    // Inspect only the bundle basename. Provider directories such as
    // `jangq-ai/` are not evidence that an MXFP child bundle uses JANG.
    const bundleName = session.modelPath.split("/").filter(Boolean).pop() || session.modelPath;
    const name = bundleName.toLowerCase();
    let fallbackLabel: string | undefined;
    if (
      name.includes("jang") ||
      name.includes("mxq") ||
      name.includes("mlxq")
    ) {
      // Extract bits from patterns: JANG_4K, JANG_2S, JANG_2L, JANG-3.99-bit
      const match = name.match(/jang[_-](\d+\.?\d*)/i);
      fallbackLabel = name.includes("jangtq")
        ? "JANGTQ"
        : match
          ? `JANG ${match[1]}-bit`
          : "JANG";
    }
    setJangLabel(fallbackLabel);

    let cancelled = false;
    window.api.models.detectConfig(session.modelPath)
      .then((detected) => {
        // Replace the basename fallback only when bundle metadata supplies a
        // more precise label. Some older JANG bundles have no complete sidecar,
        // so absence of detector metadata is not evidence of base MLX weights.
        if (!cancelled && detected?.quantizationLabel) {
          setJangLabel(detected.quantizationLabel);
        }
      })
      .catch(() => { /* retain path-derived fallback */ });
    return () => { cancelled = true; };
  }, [session.modelPath, isRemote]);

  return (
    <div className="bg-card border border-border rounded-lg p-4 hover:border-primary/50 transition-colors">
      {/* Header */}
      <div className="flex min-w-0 items-start justify-between gap-2 mb-3">
        <div className="flex-1 min-w-0 overflow-hidden">
          <div className="flex min-w-0 items-center gap-1.5">
            {isRemote && (
              <span className="text-xs bg-primary/20 text-primary px-1.5 py-0.5 rounded flex-shrink-0">
                {t('sessions.card.remoteBadge')}
              </span>
            )}
            {isImage && (
              <span className="text-xs bg-violet-500/15 text-violet-400 px-1.5 py-0.5 rounded flex-shrink-0">
                {t('sessions.card.imageBadge')}
              </span>
            )}
            <h3
              className="min-w-0 font-semibold text-sm truncate"
              title={session.modelPath}
            >
              {shortName}
            </h3>
            {jangLabel && (
              <span
                className="min-w-0 max-w-[9rem] shrink truncate text-[10px] px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-400 font-medium"
                title={jangLabel}
                aria-label={`Quantization: ${jangLabel}`}
              >
                {compactQuantizationBadgeLabel(jangLabel)}
              </span>
            )}
            {session.modelPathMissing && (
              <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400 font-medium flex-shrink-0">
                <AlertTriangle className="h-3 w-3" />
                {t('sessions.card.modelPathMissing')}
              </span>
            )}
          </div>
          <p
            className="text-xs text-muted-foreground truncate mt-0.5"
            title={isRemote ? session.remoteUrl : session.modelPath}
          >
            {isRemote ? session.remoteUrl : session.modelPath}
          </p>
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {session.status === "standby" ? (
            <Moon
              className={`h-3 w-3 ${session.standbyDepth === "deep" ? "text-indigo-400" : "text-blue-400"}`}
            />
          ) : (
            <span
              className={`w-2 h-2 rounded-full ${statusColors[session.status]}`}
            />
          )}
          <span className="text-xs text-muted-foreground">
            {session.status === "loading"
              ? t('sessions.card.loadingWithElapsed', { elapsed: formatElapsed(loadingElapsed) })
              : session.status === "standby"
                ? session.standbyDepth === "deep"
                  ? t('status.deepSleep')
                  : t('status.lightSleep')
                : statusLabels[session.status]}
          </span>
        </div>
      </div>

      {session.modelPathMissing && (
        <div className="mb-3 rounded border border-amber-500/25 bg-amber-500/10 px-2.5 py-2 text-xs text-amber-300">
          <p>{t('sessions.card.modelPathMissingDetail')}</p>
          {session.usableTwinId && (
            <p className="mt-1 text-amber-300/80">{t('sessions.card.usableTwin')}</p>
          )}
        </div>
      )}

      {/* Loading progress bar. Also rendered while the session is already
          running but the weights are still settling into RAM (the main
          process keeps emitting resident progress until the model is
          actually resident and then sends the terminal 100%). */}
      {(session.status === "loading" ||
        (session.status === "running" && progress && progress.progress < 100)) && (
        <div className="mb-3">
          <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
            <div
              className={`h-full bg-yellow-500 rounded-full transition-all duration-500 ease-out ${progress?.indeterminate !== false ? 'animate-pulse' : ''}`}
              style={{ width: progress?.indeterminate === false ? `${progress.progress}%` : '100%' }}
            />
          </div>
          {progress && (
            <div className="mt-1 space-y-0.5">
              <p className="text-[10px] text-muted-foreground">
                {/* The main process has no locale catalog, so it ships an i18n key
                    beside the English text; resolve the key and fall back to the
                    literal so a locale missing the entry still reads correctly. */}
                {progress.labelKey
                  ? t(progress.labelKey, {
                      defaultValue: progress.label,
                      ...(progress.labelParams || {}),
                    })
                  : progress.label}{' '}
                {progress.indeterminate === false ? `(${progress.progress}%)` : ''}
              </p>
              {formatModelBytes(progress.modelBytes) && (
                <p className="text-[10px] text-muted-foreground/80">
                  {t('sessions.card.modelFiles')} {formatModelBytes(progress.modelBytes)}
                  {progress.lazyResident ? t('sessions.card.lazyResidentNote') : ''}
                </p>
              )}
              {residentLoad && (
                <p className="text-[10px] text-muted-foreground/80">
                  {t('sessions.card.residentRam')} {residentLoad}
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Info */}
      <div className="flex gap-4 text-xs text-muted-foreground mb-3">
        {isRemote ? (
          <span>
            {session.remoteUrl ? new URL(session.remoteUrl).host : session.host}
          </span>
        ) : (
          <span>
            {session.host}:{session.port}
          </span>
        )}
        {!isRemote && session.pid && <span>{t('sessions.card.pidLabel', { pid: session.pid })}</span>}
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        {session.modelPathMissing ? (
          <>
            {(session.status === "running" || session.status === "loading" || session.status === "standby") && (
              <button
                onClick={() => onStop(session.id)}
              data-vmlx-control="session-card-stop"
              data-vmlx-session-id={session.id}
                className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-destructive hover:text-destructive-foreground hover:border-destructive"
              >
                {t('sessions.card.stop')}
              </button>
            )}
            <button
              onClick={() => onRepoint(session.id)}
              data-vmlx-control="session-card-repoint"
              data-vmlx-session-id={session.id}
              className="flex-1 px-3 py-1.5 text-sm rounded border border-amber-500/40 text-amber-300 hover:bg-amber-500/10 flex items-center justify-center gap-1.5"
            >
              <FolderSearch className="h-3.5 w-3.5" />
              {t('sessions.card.repointModelPath')}
            </button>
            <button
              onClick={() => onDelete(session.id)}
              data-vmlx-control="session-card-delete"
              data-vmlx-session-id={session.id}
              className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-destructive hover:text-destructive-foreground hover:border-destructive flex items-center gap-1.5"
            >
              <Trash2 className="h-3.5 w-3.5" />
              {t('sessions.card.removeSession')}
            </button>
          </>
        ) : (
          <>
        {session.status === "running" && (
          <button
            onClick={() => onOpen(session.id)}
              data-vmlx-control="session-card-open"
              data-vmlx-session-id={session.id}
            className="flex-1 px-3 py-1.5 bg-primary text-primary-foreground text-sm rounded hover:bg-primary/90"
          >
            {t('sessions.card.open')}
          </button>
        )}

        {session.status === "standby" && onWake && (
          <button
            onClick={() => onWake(session.id)}
              data-vmlx-control="session-card-wake"
              data-vmlx-session-id={session.id}
            className="flex-1 px-3 py-1.5 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 flex items-center justify-center gap-1.5"
            title={
              session.standbyDepth === "deep"
                ? t('sessions.card.wakeTitleDeep')
                : t('sessions.card.wakeTitleLight')
            }
          >
            <Sun className="h-3.5 w-3.5" />
            {t('sessions.card.wake')}
          </button>
        )}

        {session.status === "loading" && (
          <>
            <button
              onClick={() => onOpen(session.id)}
              data-vmlx-control="session-card-open"
              data-vmlx-session-id={session.id}
              className="flex-1 px-3 py-1.5 bg-warning/20 text-warning text-sm rounded hover:bg-warning/30 flex items-center justify-center gap-1.5"
            >
              <ScrollText className="h-3.5 w-3.5" />
              {t('sessions.card.logs')}
            </button>
          </>
        )}

        {session.status === "stopped" || session.status === "error" ? (
          <button
            onClick={() => onStart(session.id)}
              data-vmlx-control="session-card-start"
              data-vmlx-session-id={session.id}
            className="flex-1 px-3 py-1.5 bg-success text-success-foreground text-sm rounded hover:bg-success/90"
          >
            {isRemote ? t('sessions.card.connect') : t('sessions.card.start')}
          </button>
        ) : null}

        {!isRemote && (
          <button
            onClick={() => onConfigure(session.id)}
              data-vmlx-control="session-card-configure"
              data-vmlx-session-id={session.id}
            className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-accent"
            title={t('sessions.card.configureTitle')}
          >
            <Settings className="h-4 w-4" />
          </button>
        )}

        {session.status === "running" && !isRemote && onSleep && (
          <button
            onClick={() => onSleep(session.id)}
              data-vmlx-control="session-card-sleep"
              data-vmlx-session-id={session.id}
            className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-blue-500/10 hover:text-blue-400 hover:border-blue-500/30"
            title={t('sessions.card.sleepTitle')}
          >
            <Moon className="h-4 w-4" />
          </button>
        )}

        {(session.status === "running" || session.status === "loading") && (
          <button
            onClick={() => onStop(session.id)}
              data-vmlx-control="session-card-stop"
              data-vmlx-session-id={session.id}
            className="px-3 py-1.5 bg-destructive text-destructive-foreground text-sm rounded hover:bg-destructive/90"
          >
            {isRemote ? t('sessions.card.disconnect') : t('sessions.card.stop')}
          </button>
        )}

        {session.status === "standby" && (
          <button
            onClick={() => onStop(session.id)}
              data-vmlx-control="session-card-stop"
              data-vmlx-session-id={session.id}
            className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-destructive hover:text-destructive-foreground hover:border-destructive"
            title={t('sessions.card.stopStandbyTitle')}
          >
            {t('sessions.card.stop')}
          </button>
        )}

        {session.status === "error" && (
          <button
            onClick={() => onOpen(session.id)}
              data-vmlx-control="session-card-open"
              data-vmlx-session-id={session.id}
            className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-accent"
            title={t('sessions.card.viewCrashLogsTitle')}
          >
            <ScrollText className="h-4 w-4" />
          </button>
        )}

        {(session.status === "stopped" || session.status === "error") && (
          <button
            onClick={() => onDelete(session.id)}
              data-vmlx-control="session-card-delete"
              data-vmlx-session-id={session.id}
            className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-destructive hover:text-destructive-foreground hover:border-destructive"
          >
            {t('sessions.card.delete')}
          </button>
        )}
          </>
        )}
      </div>
    </div>
  );
}
