import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChevronDown, ChevronUp, FileText, Sparkles, User, Copy, Check } from "lucide-react";

function SourceCard({ src, index }) {
  return (
    <div className="source-card">
      <FileText size={12} />
      <div className="source-info">
        <span className="source-file">{src.file || "Document"}</span>
        {src.page && <span className="source-page">p.{src.page}</span>}
      </div>
    </div>
  );
}

export default function MessageBubble({ message }) {
  const [showSources, setShowSources] = useState(false);
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";
  const hasSources = message.sources && message.sources.length > 0;

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`msg-row ${isUser ? "msg-row--user" : "msg-row--ai"}`}>
      {!isUser && (
        <div className="msg-avatar msg-avatar--ai">
          <Sparkles size={14} />
        </div>
      )}

      <div className="msg-body">
        <div className={`msg-bubble ${isUser ? "msg-bubble--user" : "msg-bubble--ai"}`}>
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ inline, children }) {
                  return inline
                    ? <code className="inline-code">{children}</code>
                    : <pre className="code-block"><code>{children}</code></pre>;
                },
                p({ children }) { return <p className="md-p">{children}</p>; },
                ul({ children }) { return <ul className="md-ul">{children}</ul>; },
                ol({ children }) { return <ol className="md-ol">{children}</ol>; },
                li({ children }) { return <li className="md-li">{children}</li>; },
                strong({ children }) { return <strong className="md-strong">{children}</strong>; },
                h1({ children }) { return <h1 className="md-h">{children}</h1>; },
                h2({ children }) { return <h2 className="md-h md-h2">{children}</h2>; },
                h3({ children }) { return <h3 className="md-h md-h3">{children}</h3>; },
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        <div className="msg-meta">
          <span className="msg-time">{message.timestamp}</span>
          {!isUser && message.mode && (
            <span className={`msg-mode-badge ${message.mode === "rag" ? "badge--rag" : "badge--direct"}`}>
              {message.mode === "rag" ? "RAG" : "Direct"}
            </span>
          )}
          {hasSources && (
            <button className="sources-toggle" onClick={() => setShowSources(v => !v)}>
              {showSources ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              {message.sources.length} source{message.sources.length > 1 ? "s" : ""}
            </button>
          )}
          <button className="sources-toggle copy-btn-hover" onClick={handleCopy} title="Copier le message" style={{ marginLeft: "4px" }}>
            {copied ? <Check size={12} style={{ color: "var(--green)" }} /> : <Copy size={12} />}
          </button>
        </div>

        {hasSources && showSources && (
          <div className="sources-list">
            {message.sources.map((src, i) => (
              <SourceCard key={i} src={src} index={i} />
            ))}
          </div>
        )}
      </div>

      {isUser && (
        <div className="msg-avatar msg-avatar--user">
          <User size={14} />
        </div>
      )}

      <style>{`
        .msg-row {
          display: flex;
          gap: 12px;
          align-items: flex-start;
          animation: slideUp 0.3s cubic-bezier(0.16,1,0.3,1) both;
        }
        .msg-row--user { flex-direction: row-reverse; }
        @keyframes slideUp {
          from { opacity:0; transform: translateY(12px); }
          to   { opacity:1; transform: translateY(0); }
        }

        .msg-avatar {
          width: 32px;
          height: 32px;
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
          margin-top: 2px;
        }
        .msg-avatar--ai {
          background: linear-gradient(135deg, #7c6bff22, #a89bff22);
          border: 1px solid var(--accent);
          color: var(--accent-2);
        }
        .msg-avatar--user {
          background: var(--surface-3);
          border: 1px solid var(--border-hi);
          color: var(--text-2);
        }

        .msg-body { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
        .msg-row--user .msg-body { align-items: flex-end; }

        .msg-bubble {
          max-width: 78%;
          padding: 12px 16px;
          border-radius: 14px;
          word-break: break-word;
          line-height: 1.65;
          font-size: 14.5px;
        }
        .msg-bubble--user {
          background: linear-gradient(135deg, var(--accent), #9b8cff);
          color: white;
          border-radius: 14px 4px 14px 14px;
          box-shadow: 0 4px 24px var(--accent-glow);
        }
        .msg-bubble--ai {
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: 4px 14px 14px 14px;
          color: var(--text);
        }

        .msg-meta {
          display: flex;
          align-items: center;
          gap: 8px;
          flex-wrap: wrap;
        }
        .msg-time { font-size: 11px; color: var(--text-3); }

        .msg-mode-badge {
          font-size: 10px;
          font-weight: 600;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          padding: 2px 7px;
          border-radius: 5px;
        }
        .badge--rag   { background: rgba(124,107,255,0.15); color: var(--accent-2); border: 1px solid rgba(124,107,255,0.3); }
        .badge--direct { background: rgba(62,207,142,0.1);  color: var(--green);    border: 1px solid rgba(62,207,142,0.2); }

        .sources-toggle {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          background: none;
          border: 1px solid var(--border);
          border-radius: 5px;
          padding: 2px 8px;
          font-size: 11px;
          color: var(--text-3);
          transition: all 0.15s;
        }
        .sources-toggle:hover { border-color: var(--accent); color: var(--accent-2); }

        .copy-btn-hover {
          opacity: 0;
          pointer-events: none;
          transition: opacity 0.2s;
        }
        .msg-row:hover .copy-btn-hover {
          opacity: 1;
          pointer-events: auto;
        }

        .sources-list {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          max-width: 78%;
        }
        .source-card {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 4px 10px;
          background: var(--surface-3);
          border: 1px solid var(--border);
          border-radius: 7px;
          font-size: 12px;
          color: var(--text-2);
        }
        .source-card svg { color: var(--text-3); flex-shrink: 0; }
        .source-info { display: flex; align-items: center; gap: 6px; }
        .source-file { font-weight: 500; color: var(--text); max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .source-page { font-size: 11px; color: var(--text-3); }

        /* Markdown styles */
        .md-p { margin-bottom: 10px; }
        .md-p:last-child { margin-bottom: 0; }
        .md-ul, .md-ol { margin: 8px 0 10px 20px; }
        .md-li { margin-bottom: 4px; }
        .md-strong { color: var(--text); font-weight: 600; }
        .md-h { font-family: var(--font-display); margin: 14px 0 8px; color: var(--text); }
        .md-h2 { font-size: 16px; }
        .md-h3 { font-size: 14px; }
        .inline-code {
          background: var(--surface-3);
          border: 1px solid var(--border-hi);
          border-radius: 4px;
          padding: 1px 6px;
          font-size: 13px;
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
          color: var(--accent-2);
        }
        .code-block {
          background: var(--surface-3);
          border: 1px solid var(--border-hi);
          border-radius: 8px;
          padding: 12px 14px;
          margin: 10px 0;
          overflow-x: auto;
          font-size: 13px;
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
          color: var(--text);
          line-height: 1.5;
        }
      `}</style>
    </div>
  );
}
