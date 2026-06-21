import { useState, useRef, useEffect } from "react";
import { sendChatMessage } from "../api/chat";
import "./ChatWidget.css";

function formatMsg(msg) {
  const lines = msg.split("\n").filter(Boolean);
  return lines.map((line, i) => <p key={i}>{line}</p>);
}

function bold(str) {
  if (!str) return str;
  const parts = str.split(/(\*{1,2}[^*]+\*{1,2})/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }
    return part;
  });
}

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus();
    }
  }, [open]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    const userMsg = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    try {
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      const result = await sendChatMessage(text, history);
      const reply = result?.reply || "Sorry, I didn't get a response.";
      const source = result?.source || "groq";
      setMessages((prev) => [...prev, { role: "assistant", content: reply, source }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I couldn't reach the server. Make sure the backend is running.", source: "error" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={`chat-widget${open ? " chat-widget--open" : ""}`}>
      {!open && (
        <button className="chat-fab" onClick={() => setOpen(true)} aria-label="Open chat">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </button>
      )}

      {open && (
        <div className="chat-panel">
          <div className="chat-header">
            <span className="chat-header-title">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="chat-header-icon">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
              Plant Assistant
            </span>
            <button className="chat-close" onClick={() => setOpen(false)} aria-label="Close chat">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>

          <div className="chat-body" ref={listRef}>
            {messages.length === 0 && (
              <div className="chat-empty">
                <p>Ask me anything about plant diseases, crop care, or how to use Plant Village AI!</p>
                <p className="chat-hints">Try: "How to treat mango anthracnose?" or "What causes rice blast?"</p>
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`chat-bubble chat-bubble--${m.role}`}>
                {formatMsg(m.content).map((p, j) => (
                  <p key={j}>{bold(p.props.children)}</p>
                ))}
                {m.source && m.source !== "groq" && (
                  <span className="chat-source">{m.source === "knowledge_base" ? "Knowledge base" : m.source}</span>
                )}
              </div>
            ))}
            {loading && (
              <div className="chat-bubble chat-bubble--assistant chat-loading">
                <span className="chat-dot-pulse" />
              </div>
            )}
          </div>

          <div className="chat-footer">
            <input
              ref={inputRef}
              className="chat-input"
              type="text"
              placeholder="Type your question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />
            <button className="chat-send" onClick={handleSend} disabled={!input.trim() || loading} aria-label="Send">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
