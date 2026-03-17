import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { ThemeProvider } from './providers/ThemeProvider'
import { AppStateProvider } from './contexts/AppStateContext'
import { SessionsProvider } from './contexts/SessionsContext'
import { DownloadsView } from './components/DownloadsView'
import './index.css'

// Download window: skip all providers, render only the downloads view
const isDownloadWindow = window.location.hash === '#/downloads'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider>
      {isDownloadWindow ? (
        <DownloadsView />
      ) : (
        <SessionsProvider>
          <AppStateProvider>
            <App />
          </AppStateProvider>
        </SessionsProvider>
      )}
    </ThemeProvider>
  </React.StrictMode>
)
