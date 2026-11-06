import { useState } from "react";
import type { Message } from "./types";
import { ChevronIcon, ThumbsDownIcon, ThumbsUpIcon } from "./Icons";

interface Props {
  message: Message;
  onFeedback: (feedback: 1 | -1) => void;
}

export default function ChatMessage({ message: m, onFeedback }: Props) {
  const [detailsOpen, setDetailsOpen] = useState(false);

  return (
    <div className={`message-row ${m.role}`}>
      <div className="message-col">
        <div className="bubble">{m.content}</div>

        {m.refused && <div className="refused-badge">Not found in manual</div>}

        {m.sources && m.sources.length > 0 && (
          <div className="source-chips">
            {m.sources.map((s, j) => (
              <span className="source-chip" key={j} title={s.filename}>
                {shortName(s.filename)} · p.{s.page}
              </span>
            ))}
          </div>
        )}

        {m.retrievedChunks && m.retrievedChunks.length > 0 && !m.needsClarification && (
          <>
            <button
              className={`details-toggle ${detailsOpen ? "open" : ""}`}
              onClick={() => setDetailsOpen((o) => !o)}
            >
              <ChevronIcon size={11} />
              {detailsOpen ? "Hide details" : "How this answer was built"}
            </button>
            {detailsOpen && (
              <div className="details-panel">
                <p className="timing">
                  embed {m.embedLatencyMs}ms + search {m.searchLatencyMs}ms = retrieval{" "}
                  {m.retrievalLatencyMs}ms · generation {m.generationLatencyMs}ms
                </p>
                <ul>
                  {m.retrievedChunks.map((c) => (
                    <li key={c.rank}>
                      #{c.rank + 1} {shortName(c.filename)} p.{c.page} — {c.preview}…
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        {m.role === "assistant" && m.queryLogId && (
          <div className="feedback-row">
            <button
              className={`feedback-button ${m.feedback === 1 ? "selected" : ""}`}
              onClick={() => onFeedback(1)}
              aria-label="Good answer"
              title="Good answer"
            >
              <ThumbsUpIcon />
            </button>
            <button
              className={`feedback-button ${m.feedback === -1 ? "selected" : ""}`}
              onClick={() => onFeedback(-1)}
              aria-label="Bad answer"
              title="Bad answer"
            >
              <ThumbsDownIcon />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function shortName(filename: string): string {
  return filename.replace(/\.pdf$/i, "").replace(/_/g, " ");
}
