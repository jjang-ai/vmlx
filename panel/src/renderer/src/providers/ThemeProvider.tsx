import { createContext, useContext, useEffect, useState } from 'react'

type Theme = 'dark' | 'light' | 'system'
type ResolvedTheme = 'dark' | 'light'

interface ThemeContextValue {
  theme: Theme
  resolvedTheme: ResolvedTheme
  setTheme: (theme: Theme) => void
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: 'dark',
  resolvedTheme: 'dark',
  setTheme: () => {},
})

export function useTheme() {
  return useContext(ThemeContext)
}

function getSystemTheme(): ResolvedTheme {
  if (typeof window === 'undefined') return 'dark'
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function resolveTheme(theme: Theme): ResolvedTheme {
  return theme === 'system' ? getSystemTheme() : theme
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => {
    const stored = localStorage.getItem('vmlx-theme') as Theme | null
    return stored || 'dark'
  })

  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>(() => resolveTheme(theme))

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme)
    localStorage.setItem('vmlx-theme', newTheme)
  }

  useEffect(() => {
    const resolved = resolveTheme(theme)
    setResolvedTheme(resolved)

    const root = document.documentElement
    root.classList.remove('dark', 'light')
    root.classList.add(resolved)

    if (theme === 'system') {
      const mql = window.matchMedia('(prefers-color-scheme: dark)')
      const handler = () => {
        const newResolved = getSystemTheme()
        setResolvedTheme(newResolved)
        root.classList.remove('dark', 'light')
        root.classList.add(newResolved)
      }
      mql.addEventListener('change', handler)
      return () => mql.removeEventListener('change', handler)
    }
  }, [theme])

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}
