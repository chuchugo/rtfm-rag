import { useEffect, useRef, useState } from "react";
import ChatMessage from "./ChatMessage";
import StatsPanel from "./StatsPanel";
import { ChartIcon, ManualIcon, SendIcon } from "./Icons";
import type { ChatResponseBody, ConversationTurn, Document, Message } from "./types";

const SUGGESTIONS = [
  "How do I restart the touchscreen?",
  "What tire pressure should I use?",
  "How do I fold the mirrors?",
];

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/documents")
      .then((r) => r.json())
      .then(setDocuments)
      .catch(() => {});
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  async function sendMessage(text?: string) {
    const question = (text ?? input).trim();
    if (!question || loading) return;
    setInput("");
    // Capture history before the state update — messages still refers to
    // the previous state here (React batches the update), so this is exactly
    // the conversation so far, not including the message being sent now.
    const history: ConversationTurn[] = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, history }),
      });
      const data: ChatResponseBody = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources,
          queryLogId: data.query_log_id,
          refused: data.refused,
          needsClarification: data.needs_clarification,
          retrievalLatencyMs: data.retrieval_latency_ms,
          embedLatencyMs: data.embed_latency_ms,
          searchLatencyMs: data.search_latency_ms,
          generationLatencyMs: data.generation_latency_ms,
          retrievedChunks: data.retrieved_chunks,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Something went wrong reaching the server. Is the backend running?" },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function sendFeedback(index: number, feedback: 1 | -1) {
    const message = messages[index];
    if (!message.queryLogId) return;
    setMessages((prev) => prev.map((m, i) => (i === index ? { ...m, feedback } : m)));
    await fetch(`/api/feedback/${message.queryLogId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feedback }),
    });
  }

  const subtitle =
    documents.length === 0
      ? "No manuals ingested yet"
      : documents.length === 1
        ? documents[0].title.replace(/_/g, " ")
        : `${documents.length} manuals loaded`;

  return (
    <div className="app">
      <header className="header">
        <div className="header-titles">
          <h1 className="header-title">rtfm-rag</h1>
          <p className="header-subtitle">{subtitle}</p>
        </div>
        <button
          className={`icon-button ${showStats ? "active" : ""}`}
          onClick={() => setShowStats((s) => !s)}
          aria-label="Show stats"
          title="Stats"
        >
          <ChartIcon size={17} />
        </button>
      </header>

      {showStats && <StatsPanel onClose={() => setShowStats(false)} />}

      {messages.length === 0 ? (
        <div className="empty-state">
          <ManualIcon size={44} />
          <div>
            <p className="empty-state-title">Ask about your manuals</p>
            <p className="empty-state-hint">
              Answers are grounded in the PDFs you've ingested, with page citations — try one of these:
            </p>
          </div>
          <div className="suggestion-row">
            {SUGGESTIONS.map((s) => (
              <button key={s} className="suggestion-chip" onClick={() => sendMessage(s)}>
                {s}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="chat-scroll" ref={scrollRef}>
          {messages.map((m, i) => (
            <ChatMessage key={i} message={m} onFeedback={(fb) => sendFeedback(i, fb)} />
          ))}
          {loading && (
            <div className="message-row assistant">
              <div className="message-col">
                <div className="bubble typing-indicator">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="input-bar">
        <textarea
          className="input-field"
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
          placeholder="Ask about your manuals…"
        />
        <button className="send-button" onClick={() => sendMessage()} disabled={loading || !input.trim()}>
          <SendIcon />
        </button>
      </div>
    </div>
  );
}
