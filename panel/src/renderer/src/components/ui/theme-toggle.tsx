import { Sun, Moon, Monitor } from 'lucide-react'
import { useTheme } from '../../providers/ThemeProvider'
import { Button } from './button'

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  const cycle = () => {
    const next = theme === 'dark' ? 'light' : theme === 'light' ? 'system' : 'dark'
    setTheme(next)
  }

  return (
    <Button variant="ghost" size="icon" onClick={cycle} className="h-6 w-6" title={`Theme: ${theme}`}>
      {theme === 'dark' && <Moon className="h-3.5 w-3.5" />}
      {theme === 'light' && <Sun className="h-3.5 w-3.5" />}
      {theme === 'system' && <Monitor className="h-3.5 w-3.5" />}
    </Button>
  )
}
