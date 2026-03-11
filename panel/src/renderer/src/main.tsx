import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { ThemeProvider } from './providers/ThemeProvider'
import { AppStateProvider } from './contexts/AppStateContext'
import { SessionsProvider } from './contexts/SessionsContext'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider>
      <SessionsProvider>
        <AppStateProvider>
          <App />
        </AppStateProvider>
      </SessionsProvider>
    </ThemeProvider>
  </React.StrictMode>
)
