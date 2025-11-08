// src/lib/api.ts
export type Retrieved = {
  id?: string;
  rank?: number;
  final_score?: number;
  text?: string;
  metadata?: {
    source_file?: string;
    title?: string;
    authors?: string[] | string;
    year?: number | string;
  };
};

export type ChatPayload = {
  query: string;
  intent?: "conceptual" | "chronological";
  return_context?: boolean;
};

export type ChatResponse = {
  answer: string;
  retrieved?: Retrieved[];
};

const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8001";

export async function apiChat(payload: ChatPayload): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function apiRetrieve(
  query: string,
  intent: "conceptual" | "chronological" = "conceptual"
): Promise<Retrieved[]> {
  const res = await fetch(`${BASE}/api/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, intent }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
