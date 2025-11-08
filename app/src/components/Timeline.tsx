// src/components/Timeline.tsx
import type { Retrieved } from "../lib/api";

export default function Timeline({ items }: { items: Retrieved[] }) {
  const groups = new Map<string, number>();
  for (const r of items) {
    const y = `${r.metadata?.year ?? "n/a"}`;
    const decade = /^\d{4}$/.test(y) ? `${y.slice(0, 3)}0s` : "n/a";
    groups.set(decade, (groups.get(decade) ?? 0) + 1);
  }
  const sorted = Array.from(groups.entries()).sort();
  return (
    <div className="flex gap-2 flex-wrap">
      {sorted.map(([d, n]) => (
        <div
          key={d}
          className="px-2 py-1 rounded-full bg-indigo-600/20 text-indigo-300 text-xs font-medium"
        >
          {d}: {n}
        </div>
      ))}
    </div>
  );
}
