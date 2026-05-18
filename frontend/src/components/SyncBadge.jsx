import React from "react";
import { RefreshCw, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";

const fmt = (d) => {
  const h = String(Math.floor(d / 3600)).padStart(2, "0");
  const m = String(Math.floor((d % 3600) / 60)).padStart(2, "0");
  const s = String(d % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
};

export default function SyncBadge({ status, countdown, fileCount, onSync }) {
  const isRunning = status === "syncing";
  const isError   = status === "error";

  return (
    <div className="sync-badge" data-status={status} onClick={!isRunning ? onSync : undefined}>
      <span className="sync-dot" />

      {isRunning ? (
        <Loader2 size={12} className="sync-icon spin" />
      ) : isError ? (
        <AlertCircle size={12} className="sync-icon" />
      ) : (
        <CheckCircle2 size={12} className="sync-icon" />
      )}

      <span className="sync-text">
        {isRunning
          ? "Syncing…"
          : isError
          ? "Sync failed"
          : `${fileCount} file${fileCount !== 1 ? "s" : ""} — ${fmt(countdown)}`}
      </span>

      {!isRunning && (
        <RefreshCw size={11} className="sync-refresh" />
      )}

      <style>{`
        .sync-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 6px 12px 6px 8px;
          background: var(--surface-2);
          border: 1px solid var(--border-hi);
          border-radius: 999px;
          font-family: var(--font-body);
          font-size: 12px;
          font-weight: 500;
          color: var(--text-2);
          cursor: pointer;
          transition: all 0.2s;
          user-select: none;
          letter-spacing: 0.01em;
        }
        .sync-badge:hover {
          border-color: var(--accent);
          color: var(--text);
          background: var(--surface-3);
        }
        .sync-badge[data-status="syncing"] { cursor: default; }
        .sync-badge[data-status="error"] { border-color: var(--red); color: var(--red); }

        .sync-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--green);
          flex-shrink: 0;
          box-shadow: 0 0 6px var(--green);
        }
        .sync-badge[data-status="syncing"] .sync-dot { background: var(--accent); box-shadow: 0 0 6px var(--accent); animation: pulse 1.2s ease-in-out infinite; }
        .sync-badge[data-status="error"] .sync-dot { background: var(--red); box-shadow: 0 0 6px var(--red); }

        .sync-icon { color: var(--green); flex-shrink: 0; }
        .sync-badge[data-status="error"] .sync-icon { color: var(--red); }

        .sync-text { color: inherit; }
        .sync-refresh { opacity: 0.4; flex-shrink: 0; transition: opacity 0.2s; }
        .sync-badge:hover .sync-refresh { opacity: 0.8; }

        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
      `}</style>
    </div>
  );
}
