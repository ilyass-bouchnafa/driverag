import React, { useState, useRef, useEffect } from "react";
import { Send, Plus, X, FolderSync, Cpu, Database, Trash2, ChevronDown } from "lucide-react";
import { api } from "../api";

const SUGGESTIONS = [
  "What can you tell me about my documents?",
  "Summarize the key points from my files",
  "Which project should I focus on right now?",
  "What are the main topics covered?",
];

export default function InputBar({ onSend, mode, onModeChange, onSync, onClear, onUpload, disabled, fileCount }) {
  const [value, setValue]               = useState("");
  const [menuOpen, setMenuOpen]         = useState(false);
  const [uploading, setUploading]       = useState(false);
  const [uploadMsg, setUploadMsg]       = useState(null);
  const [recording, setRecording]       = useState(false);
  const [transcribing, setTranscribing] = useState(false);

  const textareaRef  = useRef(null);
  const menuRef      = useRef(null);
  const fileInputRef = useRef(null);
  const mediaRecRef  = useRef(null);
  const chunksRef    = useRef([]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, [value]);

  useEffect(() => {
    const handler = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const submit = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg(null);
    setMenuOpen(false);
    try {
      const result = await onUpload(file);
      setUploadMsg({ type: "ok", text: `"${file.name}" imported — ${result.chunks} chunks indexed` });
    } catch (err) {
      setUploadMsg({ type: "err", text: `Error: ${err.message}` });
    } finally {
      setUploading(false);
      e.target.value = "";
      setTimeout(() => setUploadMsg(null), 5000);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunksRef.current = [];
      const mr = new MediaRecorder(stream);
      mediaRecRef.current = mr;
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        
        // Debug — vérifie la taille
        console.log("Audio blob size:", blob.size, "bytes");
        
        if (blob.size < 100) {
          setUploadMsg({ type: "err", text: "Recording too short or empty" });
          setTimeout(() => setUploadMsg(null), 4000);
          return;
        }
        
        setTranscribing(true);
        try {
          const result = await api.transcribe(blob);
          if (result.text) setValue(prev => (prev + " " + result.text).trim());
        } catch (err) {
          setUploadMsg({ type: "err", text: `Transcription error: ${err.message}` });
          setTimeout(() => setUploadMsg(null), 4000);
        } finally {
          setTranscribing(false);
        }
      };
      mr.start();
      setRecording(true);
    } catch (err) {
      setUploadMsg({ type: "err", text: "Microphone access denied" });
      setTimeout(() => setUploadMsg(null), 4000);
    }
  };

  const stopRecording = () => { mediaRecRef.current?.stop(); setRecording(false); };
  const toggleMic = () => { if (recording) stopRecording(); else startRecording(); };

  return (
    <div className="inputbar-shell">
      <input ref={fileInputRef} type="file" accept=".pdf,.docx,.txt,.md,.pptx" style={{ display: "none" }} onChange={handleUpload} />

      {uploadMsg && <div className={`upload-msg upload-msg--${uploadMsg.type}`}>{uploadMsg.text}</div>}
      {transcribing && <div className="upload-msg upload-msg--ok">Transcribing audio…</div>}

      {/* {!value && (
        <div className="suggestions">
          <span className="suggestions-label">Suggestions</span>
          <div className="chips">
            {SUGGESTIONS.map((s, i) => <button key={i} className="chip" onClick={() => setValue(s)}>{s}</button>)}
          </div>
        </div>
      )} */}

      <div className="inputbar">
        <div className="menu-wrap" ref={menuRef}>
          <button className={`icon-btn ${menuOpen ? "active" : ""}`} onClick={() => setMenuOpen(v => !v)}>
            {menuOpen ? <X size={16} /> : <Plus size={16} />}
          </button>
          {menuOpen && (
            <div className="floating-menu">
              <div className="menu-section-label">Import</div>
              <button className="menu-item" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                  <line x1="12" y1="11" x2="12" y2="17"/>
                  <line x1="9" y1="14" x2="15" y2="14"/>
                </svg>
                {uploading ? "Uploading…" : "Import a file"}
                <span style={{ marginLeft: "auto", fontSize: 10, color: "var(--text-3)" }}>PDF · DOCX · TXT</span>
              </button>
              <div className="menu-divider" />
              <div className="menu-section-label">Mode</div>
              <button className={`menu-item ${mode === "rag" ? "menu-item--active" : ""}`} onClick={() => { onModeChange("rag"); setMenuOpen(false); }}>
                <Database size={14} />RAG — Search documents{mode === "rag" && <span className="menu-check">✓</span>}
              </button>
              <button className={`menu-item ${mode === "direct" ? "menu-item--active" : ""}`} onClick={() => { onModeChange("direct"); setMenuOpen(false); }}>
                <Cpu size={14} />Direct — LLM only{mode === "direct" && <span className="menu-check">✓</span>}
              </button>
              <div className="menu-divider" />
              <button className="menu-item" onClick={() => { onSync(); setMenuOpen(false); }}><FolderSync size={14} />Sync Google Drive</button>
              <button className="menu-item menu-item--danger" onClick={() => { onClear(); setMenuOpen(false); }}><Trash2 size={14} />Clear conversation</button>
            </div>
          )}
        </div>

        <button className={`mode-pill ${mode === "rag" ? "mode-pill--rag" : "mode-pill--direct"}`} onClick={() => onModeChange(mode === "rag" ? "direct" : "rag")}>
          {mode === "rag" ? <Database size={10} /> : <Cpu size={10} />}
          {mode === "rag" ? "RAG" : "Direct"}
          <ChevronDown size={10} />
        </button>

        <textarea
          ref={textareaRef}
          className="input-textarea"
          rows={1}
          placeholder={recording ? "Recording… click mic to stop" : transcribing ? "Transcribing…" : mode === "rag" ? "Ask anything about your documents…" : "Ask the AI directly…"}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          disabled={disabled || transcribing}
        />

        <button className={`send-btn ${value.trim() && !disabled ? "send-btn--ready" : ""}`} onClick={submit} disabled={!value.trim() || disabled}>
          <Send size={15} />
        </button>

        <button
          className={`mic-btn ${recording ? "mic-btn--recording" : ""} ${transcribing ? "mic-btn--transcribing" : ""}`}
          onClick={toggleMic}
          disabled={disabled || transcribing}
          title={recording ? "Stop recording" : "Start voice input"}
        >
          {transcribing ? (
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{animation:"spin 1s linear infinite"}}>
              <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
            </svg>
          ) : (
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="23"/>
              <line x1="8" y1="23" x2="16" y2="23"/>
            </svg>
          )}
        </button>
      </div>

      {fileCount > 0 && <p className="inputbar-hint">{fileCount} document{fileCount > 1 ? "s" : ""} indexed · Enter to send, Shift+Enter for new line</p>}

      <style>{`
        .inputbar-shell { width: 100%; display: flex; flex-direction: column; gap: 10px; }
        .upload-msg { padding: 8px 14px; border-radius: 8px; font-size: 13px; font-weight: 500; }
        .upload-msg--ok  { background: rgba(22,163,74,0.08);  border: 1px solid rgba(22,163,74,0.2);  color: var(--green); }
        .upload-msg--err { background: rgba(220,38,38,0.08);  border: 1px solid rgba(220,38,38,0.2);  color: var(--red); }
        .suggestions { display: flex; flex-direction: column; gap: 8px; }
        .suggestions-label { font-size: 11px; color: var(--text-3); font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; }
        .chips { display: flex; flex-wrap: wrap; gap: 6px; }
        .chip { background: var(--surface-2); border: 1px solid var(--border); border-radius: 20px; padding: 6px 14px; font-size: 13px; color: var(--text-2); font-family: var(--font-body); transition: all 0.15s; cursor: pointer; }
        .chip:hover { border-color: var(--accent); color: var(--accent-2); background: rgba(79,70,229,0.05); }
        .inputbar { display: flex; align-items: flex-end; gap: 8px; background: var(--surface-2); border: 1px solid var(--border-hi); border-radius: 14px; padding: 10px; transition: border-color 0.2s; }
        .inputbar:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
        .menu-wrap { position: relative; flex-shrink: 0; }
        .icon-btn { width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; border-radius: 9px; border: 1px solid var(--border); background: var(--surface-3); color: var(--text-2); transition: all 0.15s; cursor: pointer; }
        .icon-btn:hover, .icon-btn.active { border-color: var(--accent); color: var(--accent-2); background: rgba(79,70,229,0.08); }
        .floating-menu { position: absolute; bottom: calc(100% + 10px); left: 0; width: 230px; background: var(--surface); border: 1px solid var(--border-hi); border-radius: 12px; padding: 6px; box-shadow: 0 8px 32px rgba(0,0,0,0.12); z-index: 100; animation: menuIn 0.18s cubic-bezier(0.16,1,0.3,1) both; }
        @keyframes menuIn { from { opacity:0; transform: translateY(8px) scale(0.97); } to { opacity:1; transform: translateY(0) scale(1); } }
        .menu-section-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-3); padding: 4px 8px 6px; }
        .menu-divider { height: 1px; background: var(--border); margin: 4px 0; }
        .menu-item { display: flex; align-items: center; gap: 10px; width: 100%; padding: 9px 10px; border-radius: 8px; border: none; background: none; font-family: var(--font-body); font-size: 13.5px; color: var(--text-2); text-align: left; transition: all 0.12s; cursor: pointer; }
        .menu-item:hover { background: var(--surface-2); color: var(--text); }
        .menu-item--active { color: var(--accent-2); }
        .menu-item--active:hover { background: rgba(79,70,229,0.08); }
        .menu-item--danger:hover { background: rgba(220,38,38,0.08); color: var(--red); }
        .menu-check { margin-left: auto; font-size: 12px; color: var(--accent-2); }
        .mode-pill { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 7px; border: 1px solid var(--border); background: var(--surface-3); font-size: 11px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; cursor: pointer; flex-shrink: 0; align-self: flex-end; margin-bottom: 2px; transition: all 0.15s; height: 26px; }
        .mode-pill--rag    { color: var(--accent-2); border-color: rgba(79,70,229,0.3); }
        .mode-pill--direct { color: var(--green);    border-color: rgba(22,163,74,0.3); }
        .mode-pill:hover { opacity: 0.8; }
        .input-textarea { flex: 1; background: none; border: none; outline: none; resize: none; font-family: var(--font-body); font-size: 14.5px; color: var(--text); line-height: 1.55; min-height: 24px; max-height: 160px; overflow-y: auto; padding: 4px 0; }
        .input-textarea::placeholder { color: var(--text-3); }
        .input-textarea:disabled { opacity: 0.5; cursor: not-allowed; }
        .send-btn { width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; border-radius: 9px; border: 1px solid var(--border); background: var(--surface-3); color: var(--text-3); flex-shrink: 0; transition: all 0.15s; cursor: pointer; }
        .send-btn--ready { background: var(--accent); border-color: var(--accent); color: white; box-shadow: 0 4px 16px var(--accent-glow); }
        .send-btn--ready:hover { transform: scale(1.05); }
        .send-btn:disabled { cursor: not-allowed; }
        .mic-btn { width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; border-radius: 9px; border: 1px solid var(--border); background: var(--surface-3); color: var(--text-2); flex-shrink: 0; transition: all 0.2s; cursor: pointer; }
        .mic-btn:hover { border-color: var(--accent); color: var(--accent-2); }
        .mic-btn--recording { background: var(--red); border-color: var(--red); color: white; animation: micPulse 1s ease-in-out infinite; }
        .mic-btn--transcribing { opacity: 0.6; cursor: not-allowed; }
        @keyframes micPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.4); } 50% { box-shadow: 0 0 0 8px rgba(220,38,38,0); } }
        .inputbar-hint { font-size: 11px; color: var(--text-3); text-align: center; }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
