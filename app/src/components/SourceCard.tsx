// src/components/SourceCard.tsx
import type { Retrieved } from "../lib/api";

export default function SourceCard({ r, idx }: { r: Retrieved; idx: number }) {
  const year = r.metadata?.year ?? "—";
  const src = r.metadata?.title || r.metadata?.source_file || "Unknown Source";
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-3 backdrop-blur-sm hover:bg-white/10 transition">
      <div className="text-xs text-neutral-400 mb-1">[{idx}] {year}</div>
      <div className="text-sm text-neutral-200 font-medium">{src}</div>
    </div>
  );
}
