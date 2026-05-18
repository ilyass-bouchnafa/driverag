import React, { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import { Sparkles, Database } from "lucide-react";

const EMPTY_SUGGESTIONS = [
  { icon: "01", title: "Document summary", text: "Summarize all my indexed documents" },
  { icon: "02", title: "Key insights",     text: "What are the most important points in my files?" },
  { icon: "03", title: "Project status",   text: "Which of my projects needs the most attention?" },
  { icon: "04", title: "Deep dive",        text: "Find any risks or issues mentioned in my documents" },
];

export default function ChatWindow({ messages, loading, fileCount, onSuggestion }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  if (messages.length === 0) {
    return (
      <div className="chat-empty">
        <div className="empty-hero">
          <div className="empty-orb">
            <Sparkles size={28} />
          </div>
          <h1 className="empty-title">DriveRAG</h1>
          <p className="empty-sub">
            Ask anything about your Google Drive documents
          </p>

          {fileCount > 0 && (
            <div className="empty-badge">
              <Database size={12} />
              {fileCount} document{fileCount > 1 ? "s" : ""} ready
            </div>
          )}
        </div>

        <div className="empty-cards">
          {EMPTY_SUGGESTIONS.map((s, i) => (
            <button
              key={i}
              className="empty-card"
              onClick={() => onSuggestion(s.text)}
              style={{ animationDelay: `${i * 0.06}s` }}
            >
              <span className="empty-card-num">{s.icon}</span>
              <div className="empty-card-body">
                <div className="empty-card-title">{s.title}</div>
                <div className="empty-card-text">{s.text}</div>
              </div>
            </button>
          ))}
        </div>

        <style>{`
          .chat-empty {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 48px;
            padding: 40px 20px;
          }
          .empty-hero {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 14px;
            text-align: center;
          }
          .empty-orb {
            width: 72px;
            height: 72px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(124,107,255,0.15), rgba(168,155,255,0.08));
            border: 1px solid rgba(124,107,255,0.35);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-2);
            box-shadow: 0 0 40px rgba(124,107,255,0.12), inset 0 1px 0 rgba(255,255,255,0.06);
            animation: orbPulse 3s ease-in-out infinite;
          }
          @keyframes orbPulse {
            0%,100% { box-shadow: 0 0 40px rgba(124,107,255,0.12), inset 0 1px 0 rgba(255,255,255,0.06); }
            50%      { box-shadow: 0 0 60px rgba(124,107,255,0.22), inset 0 1px 0 rgba(255,255,255,0.06); }
          }
          .empty-title {
            font-family: var(--font-display);
            font-size: 42px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--text) 30%, var(--accent-2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
          }
          .empty-sub {
            font-size: 16px;
            color: var(--text-2);
            max-width: 340px;
            line-height: 1.5;
          }
          .empty-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            background: rgba(62,207,142,0.08);
            border: 1px solid rgba(62,207,142,0.2);
            border-radius: 999px;
            font-size: 12px;
            color: var(--green);
            font-weight: 500;
          }

          /* Cards grid */
          .empty-cards {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            width: 100%;
            max-width: 560px;
          }
          .empty-card {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 16px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 12px;
            text-align: left;
            font-family: var(--font-body);
            cursor: pointer;
            transition: all 0.18s;
            animation: cardIn 0.4s cubic-bezier(0.16,1,0.3,1) both;
          }
          @keyframes cardIn {
            from { opacity:0; transform: translateY(16px); }
            to   { opacity:1; transform: translateY(0); }
          }
          .empty-card:hover {
            border-color: var(--accent);
            background: rgba(124,107,255,0.05);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
          }
          .empty-card-icon { font-size: 22px; flex-shrink: 0; line-height: 1; margin-top: 2px; }
          .empty-card-body { display: flex; flex-direction: column; gap: 3px; }
          .empty-card-title { font-size: 13px; font-weight: 600; color: var(--text); }
          .empty-card-text  { font-size: 12px; color: var(--text-2); line-height: 1.45; }
        `}</style>
      </div>
    );
  }

  return (
    <div className="chat-window">
      {messages.map((msg, i) => (
        <MessageBubble key={msg.id || i} message={msg} />
      ))}

      {loading && (
        <div className="typing-row">
          <div className="typing-avatar">
            <Sparkles size={14} />
          </div>
          <div className="typing-bubble">
            <span className="dot" /><span className="dot" /><span className="dot" />
          </div>
        </div>
      )}

      <div ref={bottomRef} />

      <style>{`
        .chat-window {
          display: flex;
          flex-direction: column;
          gap: 20px;
          padding: 24px 0 8px;
        }
        .typing-row {
          display: flex;
          align-items: center;
          gap: 12px;
          animation: slideUp 0.3s cubic-bezier(0.16,1,0.3,1) both;
        }
        @keyframes slideUp {
          from { opacity:0; transform: translateY(12px); }
          to   { opacity:1; transform: translateY(0); }
        }
        .typing-avatar {
          width: 32px; height: 32px;
          border-radius: 10px;
          background: linear-gradient(135deg, rgba(124,107,255,0.2), rgba(168,155,255,0.1));
          border: 1px solid var(--accent);
          color: var(--accent-2);
          display: flex; align-items: center; justify-content: center;
          flex-shrink: 0;
        }
        .typing-bubble {
          display: flex;
          align-items: center;
          gap: 5px;
          padding: 12px 16px;
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: 4px 14px 14px 14px;
        }
        .dot {
          width: 6px; height: 6px;
          border-radius: 50%;
          background: var(--text-3);
          animation: dotBounce 1.2s ease-in-out infinite;
        }
        .dot:nth-child(2) { animation-delay: 0.15s; }
        .dot:nth-child(3) { animation-delay: 0.3s; }
        @keyframes dotBounce {
          0%,60%,100% { transform: translateY(0); opacity:0.4; }
          30% { transform: translateY(-6px); opacity:1; }
        }
      `}</style>
    </div>
  );
}
