import React, { useState, useEffect, useCallback, useRef } from "react";
import { api } from "./api";
import ChatWindow from "./components/ChatWindow";
import InputBar from "./components/InputBar";
import SyncBadge from "./components/SyncBadge";

// Track cursor position for background glow effect
function useCursorGlow() {
  useEffect(() => {
    const move = (e) => {
      document.body.style.setProperty("--cx", e.clientX + "px");
      document.body.style.setProperty("--cy", e.clientY + "px");
    };
    window.addEventListener("mousemove", move);
    return () => window.removeEventListener("mousemove", move);
  }, []);
}

const SYNC_INTERVAL = 1800; // seconds (30 minutes)

function useCountdown(total) {
  const [count, setCount] = useState(total);
  useEffect(() => {
    const t = setInterval(() => setCount(c => (c <= 1 ? total : c - 1)), 1000);
    return () => clearInterval(t);
  }, [total]);
  return [count, () => setCount(total)];
}

export default function App() {
  // State
  const [messages, setMessages]     = useState([]);
  const [loading, setLoading]       = useState(false);
  const [mode, setMode]             = useState("rag");
  const [syncStatus, setSyncStatus] = useState("idle"); // idle | syncing | error
  const [files, setFiles]           = useState([]);
  const [online, setOnline]         = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [threadId]                  = useState(() => crypto.randomUUID());
  const [conversationId]            = useState(() => `conv_${Date.now()}`);

  const [countdown, resetCountdown] = useCountdown(SYNC_INTERVAL);

  useCursorGlow();

  // ── Online check ──────────────────────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        await api.health();
        setOnline(true);
      } catch {
        setOnline(false);
      }
    };
    check();
    const t = setInterval(check, 15000);
    return () => clearInterval(t);
  }, []);

  // ── Load files ────────────────────────────────────────────────────────────
  const loadFiles = useCallback(async () => {
    try {
      const data = await api.getFiles();
      setFiles(Array.isArray(data) ? data : []);
    } catch {}
  }, []);

  useEffect(() => { loadFiles(); }, [loadFiles]);

  // ── Auto-sync when countdown hits 0 ──────────────────────────────────────
  useEffect(() => {
    if (countdown === 1) handleSync(true);
  }, [countdown]);

  // ── Sync ──────────────────────────────────────────────────────────────────
  const handleSync = useCallback(async (auto = false) => {
    if (syncStatus === "syncing") return;
    setSyncStatus("syncing");
    try {
      await api.sync();
      await loadFiles();
      setSyncStatus("idle");
      resetCountdown();
    } catch {
      setSyncStatus("error");
      setTimeout(() => setSyncStatus("idle"), 4000);
    }
  }, [syncStatus, loadFiles, resetCountdown]);

  // ── Send message ──────────────────────────────────────────────────────────
  const handleSend = useCallback(async (text) => {
    if (!text.trim() || loading) return;

    const userMsg = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const result = await api.chat({
        message: text,
        mode,
        history,
        thread_id: threadId,
        conversation_id: conversationId,
      });

      const aiMsg = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: result.answer || "No response",
        sources: result.sources || [],
        mode: result.mode || mode,
        timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
      };

      setMessages(prev => [...prev, aiMsg]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `⚠️ Error: ${err.message}`,
        timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
      }]);
    } finally {
      setLoading(false);
    }
  }, [messages, mode, loading, threadId, conversationId]);

  // ── Upload ────────────────────────────────────────────────────────────────
  const handleUpload = useCallback(async (file) => {
    const result = await api.upload(file);
    await loadFiles();
    return result;
  }, [loadFiles]);
  const handleClear = useCallback(async () => {
    setMessages([]);
    try { await api.clearMemory(); } catch {}
  }, []);

  // ── Sidebar content ───────────────────────────────────────────────────────
  const [showAllFiles, setShowAllFiles] = useState(false);
  const recentFiles = showAllFiles ? files : files.slice(0, 8);

  return (
    <div className="app">
      {/* ── Sidebar ───────────────────────────────────────────── */}
      <aside className={`sidebar ${sidebarOpen ? "sidebar--open" : "sidebar--closed"}`}>
        <div className="sidebar-header">
          {sidebarOpen && (
            <div className="sidebar-logo">
              <span className="logo-mark">D</span>
              <span className="logo-text">DriveRAG</span>
            </div>
          )}
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(v => !v)}>
            {sidebarOpen ? "←" : "→"}
          </button>
        </div>

        {sidebarOpen && (
          <>
            <nav className="sidebar-nav">
              <button className="nav-item nav-item--active">
                <span>Chat</span>
              </button>
              <button className="nav-item">
                <span>Documents</span>
                {files.length > 0 && <span className="nav-count">{files.length}</span>}
              </button>
            </nav>

            <div className="sidebar-section">
              <div className="sidebar-section-header">Indexed Files</div>
              <div className="file-list">
                {recentFiles.length === 0 ? (
                  <p className="file-empty">No files indexed yet. Click Sync to load your Drive.</p>
                ) : (
                  recentFiles.map((f, i) => (
                    <div key={i} className="file-item">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{color:"var(--text-3)",flexShrink:0}}>
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                      </svg>
                      <span className="file-name" title={f.name || f.file_name}>
                        {f.name || f.file_name || "Document"}
                      </span>
                    </div>
                  ))
                )}
                {files.length > 8 && (
                  <button
                    className="show-more-btn"
                    onClick={() => setShowAllFiles(v => !v)}
                  >
                    {showAllFiles
                      ? "Show less"
                      : `+ ${files.length - 8} more files`}
                  </button>
                )}
              </div>
            </div>

            <div className="sidebar-section">
              <div className="sidebar-section-header">Session</div>
              <div className="session-info">
                <div className="session-row">
                  <span>Messages</span>
                  <span className="session-val">{messages.length}</span>
                </div>
                <div className="session-row">
                  <span>Mode</span>
                  <span className={`session-val mode-val--${mode}`}>
                    {mode === "rag" ? "RAG" : "Direct"}
                  </span>
                </div>
              </div>
            </div>
          </>
        )}
      </aside>

      {/* ── Main ──────────────────────────────────────────────── */}
      <main className="main">
        {/* Topbar */}
        <header className="topbar">
          <div className="topbar-left">
            <div className={`status-dot ${online ? "status-dot--online" : "status-dot--offline"}`} />
            <span className="status-label">
              {online ? "Connected" : "Offline"}
            </span>
          </div>
          <div className="topbar-right">
            <SyncBadge
              status={syncStatus}
              countdown={countdown}
              fileCount={files.length}
              onSync={() => handleSync(false)}
            />
          </div>
        </header>

        {/* Scroll area */}
        <div className="scroll-area">
          <div className="chat-container">
            <ChatWindow
              messages={messages}
              loading={loading}
              fileCount={files.length}
              onSuggestion={handleSend}
            />
          </div>
        </div>

        {/* Fixed input */}
        <div className="input-area">
          <div className="input-container">
            <InputBar
              onSend={handleSend}
              mode={mode}
              onModeChange={setMode}
              onSync={() => handleSync(false)}
              onClear={handleClear}
              onUpload={handleUpload}
              disabled={loading || !online}
              fileCount={files.length}
            />
          </div>
        </div>
      </main>

      <style>{`
        /* ── Layout ─────────────────────────────────────────── */
        .app {
          display: flex;
          height: 100vh;
          overflow: hidden;
          background: var(--bg);
        }

        /* ── Sidebar ────────────────────────────────────────── */
        .sidebar {
          display: flex;
          flex-direction: column;
          background: var(--surface);
          border-right: 1px solid var(--border);
          transition: width 0.25s cubic-bezier(0.16,1,0.3,1);
          overflow: hidden;
          flex-shrink: 0;
        }
        .sidebar--open  { width: 240px; }
        .sidebar--closed { width: 52px; }

        .sidebar-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px 12px;
          border-bottom: 1px solid var(--border);
          min-height: 56px;
          flex-shrink: 0;
        }
        .sidebar-logo { display: flex; align-items: center; gap: 10px; overflow: hidden; }
        .logo-mark {
          width: 28px;
          height: 28px;
          background: linear-gradient(135deg, var(--accent), var(--accent-2));
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: var(--font-display);
          font-size: 14px;
          font-weight: 700;
          color: white;
          flex-shrink: 0;
          box-shadow: 0 4px 12px var(--accent-glow);
        }
        .logo-text {
          font-family: var(--font-display);
          font-size: 17px;
          font-weight: 700;
          color: var(--text);
          white-space: nowrap;
        }
        .sidebar-toggle {
          width: 28px; height: 28px;
          display: flex; align-items: center; justify-content: center;
          border-radius: 7px;
          border: 1px solid var(--border);
          background: none;
          color: var(--text-3);
          flex-shrink: 0;
          transition: all 0.15s;
        }
        .sidebar-toggle:hover { border-color: var(--accent); color: var(--accent-2); }

        .sidebar-nav {
          display: flex;
          flex-direction: column;
          gap: 2px;
          padding: 12px 8px;
          border-bottom: 1px solid var(--border);
        }
        .nav-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 10px;
          border-radius: 8px;
          border: none;
          background: none;
          font-family: var(--font-body);
          font-size: 13.5px;
          color: var(--text-2);
          cursor: pointer;
          transition: all 0.12s;
          text-align: left;
          width: 100%;
        }
        .nav-item:hover { background: var(--surface-2); color: var(--text); }
        .nav-item--active { background: rgba(124,107,255,0.12); color: var(--accent-2); }
        .nav-item--active:hover { background: rgba(124,107,255,0.18); }
        .nav-count {
          margin-left: auto;
          background: var(--surface-3);
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 1px 7px;
          font-size: 11px;
          color: var(--text-2);
        }

        .sidebar-section {
          padding: 16px 8px;
          border-bottom: 1px solid var(--border);
          overflow: hidden;
        }
        .sidebar-section-header {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.07em;
          color: var(--text-3);
          padding: 0 6px 10px;
        }

        .file-list { display: flex; flex-direction: column; gap: 2px; }
        .file-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 6px 8px;
          border-radius: 6px;
          font-size: 12.5px;
          color: var(--text-2);
          transition: all 0.12s;
          cursor: default;
        }
        .file-item:hover { background: var(--surface-2); color: var(--text); }
        .file-icon { color: var(--text-3); flex-shrink: 0; }
        .file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .file-empty { font-size: 12px; color: var(--text-3); padding: 4px 8px; line-height: 1.5; }
        .file-more  { font-size: 11px; color: var(--text-3); padding: 4px 8px; }
        .show-more-btn {
          width: 100%;
          padding: 6px 8px;
          border-radius: 6px;
          border: 1px dashed var(--border-hi);
          background: none;
          font-family: var(--font-body);
          font-size: 12px;
          color: var(--accent-2);
          cursor: pointer;
          text-align: center;
          margin-top: 4px;
          transition: all 0.15s;
        }
        .show-more-btn:hover { background: var(--accent-glow); border-color: var(--accent); }

        .session-info { display: flex; flex-direction: column; gap: 4px; }
        .session-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 5px 8px;
          border-radius: 6px;
          font-size: 12.5px;
          color: var(--text-2);
        }
        .session-val { font-weight: 600; color: var(--text); }
        .mode-val--rag    { color: var(--accent-2) !important; }
        .mode-val--direct { color: var(--green) !important; }

        /* ── Main ───────────────────────────────────────────── */
        .main {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          min-width: 0;
        }

        /* Topbar */
        .topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 24px;
          border-bottom: 1px solid var(--border);
          background: var(--surface);
          flex-shrink: 0;
          min-height: 56px;
        }
        .topbar-left { display: flex; align-items: center; gap: 8px; }
        .status-dot {
          width: 8px; height: 8px;
          border-radius: 50%;
          flex-shrink: 0;
        }
        .status-dot--online  { background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 2s ease-in-out infinite; }
        .status-dot--offline { background: var(--red); }
        @keyframes pulse {
          0%,100% { opacity:1; }
          50% { opacity:0.5; }
        }
        .status-label { font-size: 12.5px; color: var(--text-2); }
        .topbar-right { display: flex; align-items: center; gap: 10px; }

        /* Scroll area */
        .scroll-area {
          flex: 1;
          overflow-y: auto;
          padding: 0 24px;
        }
        .chat-container {
          max-width: 780px;
          margin: 0 auto;
          min-height: 100%;
          display: flex;
          flex-direction: column;
        }

        /* Fixed input */
        .input-area {
          border-top: 1px solid var(--border);
          background: linear-gradient(to top, var(--bg) 80%, transparent);
          padding: 16px 24px 20px;
          flex-shrink: 0;
        }
        .input-container {
          max-width: 780px;
          margin: 0 auto;
        }
      `}</style>
    </div>
  );
}
