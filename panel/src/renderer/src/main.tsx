import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { ThemeProvider } from './providers/ThemeProvider'
import { I18nProvider } from './i18n'
import { AppStateProvider } from './contexts/AppStateContext'
import { SessionsProvider } from './contexts/SessionsContext'
import { DownloadsView } from './components/DownloadsView'
import './index.css'

// Download window: skip all providers, render only the downloads view
const isDownloadWindow = new URLSearchParams(window.location.search).get('view') === 'downloads'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider>
      <I18nProvider>
        {isDownloadWindow ? (
          <DownloadsView />
        ) : (
          <SessionsProvider>
            <AppStateProvider>
              <App />
            </AppStateProvider>
          </SessionsProvider>
        )}
      </I18nProvider>
    </ThemeProvider>
  </React.StrictMode>
)
