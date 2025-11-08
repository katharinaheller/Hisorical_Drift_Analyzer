import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { Retrieved } from "./lib/api";
import SourceCard from "./components/SourceCard";
import Timeline from "./components/Timeline";

type Msg = {
  role: "user" | "assistant";
  content: string;
  retrieved?: Retrieved[];
  loading?: boolean;
};

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);

  // Always scroll to bottom after message change
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    requestAnimationFrame(() => {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    });
  }, [messages]);

  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);

    // Add user + placeholder assistant message in one state update
    setMessages((prev) => [
      ...prev,
      { role: "user", content: q },
      { role: "assistant", content: "", loading: true },
    ]);

    try {
      const resp = await fetch("http://localhost:8001/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });

      if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
      const data = await resp.json();

      // Update last assistant message
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        updated[updated.length - 1] = {
          ...last,
          loading: false,
          content: data.answer,
          retrieved: data.retrieved,
        };
        return updated;
      });
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `[Error] ${e.message || e}` },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="h-dvh flex flex-col bg-gradient-to-br from-neutral-950 via-black to-neutral-900 text-neutral-100">
      {/* Header */}
      <header className="backdrop-blur-xl bg-white/5 border-b border-white/10 p-5 flex items-center justify-between">
        <h1 className="text-lg font-semibold tracking-tight text-white drop-shadow-sm">
          Historical Drift Analyzer
        </h1>
        <span className="text-xs text-neutral-400 font-light">Local LLM Pipeline</span>
      </header>

      {/* Main layout */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr,360px] overflow-hidden">
        {/* Chat area */}
        <div className="flex flex-col h-full">
          <div
            ref={scroller}
            className="flex-1 overflow-auto px-6 py-8 space-y-6 scroll-smooth"
          >
            <AnimatePresence>
              {messages.map((m, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                  className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <motion.div
                    layout
                    className={`max-w-[70ch] rounded-3xl px-6 py-4 leading-relaxed shadow-lg transition-all duration-300 ${
                      m.role === "user"
                        ? "bg-gradient-to-r from-indigo-600 to-indigo-500 text-white"
                        : "bg-white/10 backdrop-blur-md border border-white/10 text-neutral-50"
                    }`}
                  >
                    <div className="whitespace-pre-wrap text-[15px] tracking-normal">
                      {m.loading ? (
                        <span className="animate-pulse text-indigo-400">▍</span>
                      ) : (
                        m.content
                      )}
                    </div>

                    {m.retrieved && m.retrieved.length > 0 && (
                      <div className="mt-5 space-y-3">
                        <Timeline items={m.retrieved} />
                        <div className="grid sm:grid-cols-2 gap-2">
                          {m.retrieved.slice(0, 6).map((r, idx) => (
                            <SourceCard key={`${r.id}-${idx}`} r={r} idx={idx + 1} />
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                </motion.div>
              ))}
            </AnimatePresence>

            {messages.length === 0 && (
              <div className="h-full flex items-center justify-center text-neutral-500 text-sm">
                <div className="text-center space-y-2 opacity-60">
                  <p className="font-light text-[15px]">Start exploring semantic evolution.</p>
                  <p className="text-xs">Ask how “Artificial Intelligence” changed since the 1980s.</p>
                </div>
              </div>
            )}
          </div>

          {/* Input area */}
          <div className="p-4 border-t border-white/10 backdrop-blur-md bg-white/5">
            <motion.div layout className="flex gap-3 items-center">
              <input
                className="flex-1 bg-white/10 border border-white/10 rounded-3xl px-5 py-3 
                           text-[15px] text-neutral-100 placeholder:text-neutral-500 outline-none 
                           focus:ring-2 focus:ring-indigo-500/40 transition-all duration-200"
                placeholder="Ask a question about AI evolution..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && send()}
              />
              <motion.button
                whileTap={{ scale: 0.96 }}
                onClick={send}
                disabled={busy}
                className="px-6 py-3 rounded-3xl bg-gradient-to-r from-indigo-600 to-purple-600 
                           hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50 
                           text-sm font-medium text-white shadow-md transition-all duration-200"
              >
                {busy ? "..." : "Send"}
              </motion.button>
            </motion.div>
          </div>
        </div>

        {/* Side panel */}
        <aside className="hidden lg:flex flex-col border-l border-white/10 p-6 backdrop-blur-xl bg-white/5">
          <h3 className="font-semibold mb-3 text-neutral-200">Usage Tips</h3>
          <ul className="text-sm text-neutral-400 space-y-2 leading-relaxed">
            <li>Compare term meanings across decades.</li>
            <li>Each answer appears after full generation.</li>
          </ul>
        </aside>
      </main>
    </div>
  );
}
