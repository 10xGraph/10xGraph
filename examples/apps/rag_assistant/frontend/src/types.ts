export interface Source {
  source: string;
  title: string;
  score: number;
  content: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  sources: Source[];
}

export interface ThreadSummary {
  id: string;
  name: string;
}
