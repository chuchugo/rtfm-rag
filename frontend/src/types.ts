export interface Source {
  filename: string;
  page: number;
}

export interface RetrievedChunk {
  rank: number;
  filename: string;
  page: number;
  preview: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  queryLogId?: string;
  feedback?: 1 | -1;
  refused?: boolean;
  needsClarification?: boolean;
  retrievalLatencyMs?: number;
  embedLatencyMs?: number;
  searchLatencyMs?: number;
  generationLatencyMs?: number;
  retrievedChunks?: RetrievedChunk[];
}

export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponseBody {
  answer: string;
  sources: Source[];
  query_log_id: string;
  refused: boolean;
  needs_clarification: boolean;
  retrieval_latency_ms: number;
  embed_latency_ms: number;
  search_latency_ms: number;
  generation_latency_ms: number;
  retrieved_chunks: RetrievedChunk[];
}

export interface Document {
  id: string;
  filename: string;
  title: string;
}
