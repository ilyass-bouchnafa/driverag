const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const handle = async (res) => {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
};

export const api = {
  health: () => fetch(`${BASE}/health`).then(handle),

  chat: (payload) =>
    fetch(`${BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),

  getFiles: () => fetch(`${BASE}/files`).then(handle),

  transcribe: (audioBlob) => {
    const form = new FormData();
    form.append("audio", audioBlob, "recording.webm");
    return fetch(`${BASE}/transcribe`, { method: "POST", body: form }).then(handle);
  },

  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/upload`, { method: "POST", body: form }).then(handle);
  },

  sync: () => fetch(`${BASE}/sync`, { method: "POST" }).then(handle),

  clearMemory: () => fetch(`${BASE}/clear`, { method: "POST" }).then(handle),
};
