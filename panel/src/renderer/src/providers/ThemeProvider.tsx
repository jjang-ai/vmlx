import { useLayoutEffect, type ReactNode } from 'react'

// Console Amber is the only theme. The dark utility variant remains for
// existing component tokens; OS appearance changes do not select another theme.
export function ThemeProvider({ children }: { children: ReactNode }) {
  useLayoutEffect(() => {
    const root = document.documentElement
    root.classList.remove('light')
    root.classList.add('dark')
    root.dataset.theme = 'console-amber'
    // Migrate only the old appearance preference, never the user profile.
    try { localStorage.setItem('vmlx-theme', 'console-amber') } catch { /* restricted storage */ }
  }, [])
  return <>{children}</>
}
