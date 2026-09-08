import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import {
  PanelLeftClose,
  PanelLeft,
  Info,
} from "lucide-react";
import { ThemeToggle } from "../ui/theme-toggle";
import { useAppState } from "../../contexts/AppStateContext";
import {
  useTranslation,
  LOCALES,
  LOCALE_NAMES,
  LOCALE_FLAGS,
  type Locale,
} from "../../i18n";

export function TitleBar() {
  const { state, dispatch } = useAppState();
  const { t, locale, setLocale } = useTranslation();

  return (
    <div
      className="flex items-center h-[38px] min-w-0 overflow-hidden bg-background border-b border-border flex-shrink-0"
      style={{ WebkitAppRegion: "drag" } as React.CSSProperties}
    >
      {/* macOS traffic light spacer + sidebar toggle */}
      <div
        className="flex w-[102px] shrink-0 items-center gap-1 pl-[72px] pr-2"
        style={{ WebkitAppRegion: "no-drag" } as React.CSSProperties}
      >
        {(
          <button
            onClick={() => dispatch({ type: "TOGGLE_SIDEBAR" })}
            className="p-1 translate-y-[3px] text-muted-foreground hover:text-foreground rounded hover:bg-accent transition-colors focus:outline-none focus-visible:outline-none"
            title={
              state.sidebarCollapsed
                ? t("app.sidebar.show")
                : t("app.sidebar.hide")
            }
          >
            {state.sidebarCollapsed ? (
              <PanelLeft className="h-3.5 w-3.5" />
            ) : (
              <PanelLeftClose className="h-3.5 w-3.5" />
            )}
          </button>
        )}
      </div>

      <div className="flex-1 min-w-0 text-center text-[11px] tracking-[0.14em] text-muted-foreground truncate">
        vmlx / Console
      </div>

      {/* Right: language picker + about + theme toggle */}
      <div
        className="flex items-center gap-1 px-3"
        style={{ WebkitAppRegion: "no-drag" } as React.CSSProperties}
      >
        <ThemeToggle />
        <LanguagePicker locale={locale} setLocale={setLocale} />
        <button
          onClick={() => {
            dispatch({ type: "SET_MODE", mode: "server" });
            dispatch({ type: "SET_SERVER_PANEL", panel: "about" });
          }}
          className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-accent transition-colors"
          title={t("app.about.settings")}
        >
          <Info className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}

function LanguagePicker({
  locale,
  setLocale,
}: {
  locale: Locale;
  setLocale: (l: Locale) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuPos, setMenuPos] = useState<{ top: number; right: number } | null>(
    null,
  );

  // The titlebar bar is h-10 with overflow-hidden, and the main content is a
  // later sibling with its own stacking context — an in-flow absolute dropdown
  // is BOTH clipped by the bar and painted over by main content (a covered,
  // unclickable menu, not a dev-build artifact). Render it in a body portal
  // positioned under the button so it escapes both.
  useEffect(() => {
    if (!open) return undefined;
    const place = () => {
      const r = ref.current?.getBoundingClientRect();
      if (r) setMenuPos({ top: r.bottom + 4, right: window.innerWidth - r.right });
    };
    place();
    function handleClickOutside(e: MouseEvent) {
      const target = e.target as Node;
      if (
        ref.current &&
        !ref.current.contains(target) &&
        menuRef.current &&
        !menuRef.current.contains(target)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    window.addEventListener("resize", place);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      window.removeEventListener("resize", place);
    };
  }, [open]);

  const { t } = useTranslation();

  return (
    <div className="relative" ref={ref}>
      <button
        data-vmlx-locale-picker="true"
        data-vmlx-supported-locales={LOCALES.join(",")}
        onClick={() => setOpen(!open)}
        className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-accent transition-colors text-sm leading-none"
        title={t("titlebar.languageTitle")}
      >
        {LOCALE_FLAGS[locale]}
      </button>
      {open &&
        menuPos &&
        createPortal(
          <div
            ref={menuRef}
            className="fixed bg-popover border border-border rounded-lg shadow-lg py-1 min-w-[140px]"
            style={{
              top: menuPos.top,
              right: menuPos.right,
              zIndex: 2147483647,
              WebkitAppRegion: "no-drag",
            } as React.CSSProperties}
          >
            {LOCALES.map((l) => (
              <button
                key={l}
                data-vmlx-locale-option={l}
                onClick={() => {
                  setLocale(l);
                  setOpen(false);
                }}
                className={`w-full text-left px-3 py-1.5 text-sm flex items-center gap-2 hover:bg-accent transition-colors ${
                  locale === l ? "bg-primary/10 font-medium" : ""
                }`}
              >
                <span>{LOCALE_FLAGS[l]}</span>
                <span>{LOCALE_NAMES[l]}</span>
              </button>
            ))}
          </div>,
          document.body,
        )}
    </div>
  );
}
