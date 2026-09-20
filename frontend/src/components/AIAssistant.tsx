import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  PaperAirplaneIcon,
  SparklesIcon,
  XCircleIcon,
} from "@heroicons/react/24/outline";
import { api } from "../services/api";
import { useApp } from "../context/AppContext";
import type {
  AssistantContextSource,
  AssistantMessage,
} from "../types";

interface UiMessage {
  role: "user" | "assistant";
  content: string;
  sources?: AssistantContextSource[];
  llmLabel?: string;
  warnings?: string[];
}

const SUGGESTIONS = [
  "Why was this route selected?",
  "What is the distance between DEMO-B000 and DEMO-B001?",
  "What is the sea-ice forecast for the next 48 hours?",
  "Which iceberg is closest to the vessel?",
  "What factors affect iceberg drift?",
  "What are the risks near the selected route?",
  "Which route has lower estimated fuel consumption?",
];

const HORIZON_OPTIONS = [24, 48, 72, 120];

export function AIAssistant() {
  const { toast } = useApp();
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [horizon, setHorizon] = useState(24);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [llmStatus, setLlmStatus] = useState<{
    provider: string;
    model: string | null;
    configured: boolean;
  } | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  const history = useMemo<AssistantMessage[]>(
    () =>
      messages
        .slice(-14)
        .map((m) => ({ role: m.role, content: m.content })),
    [messages],
  );

  const send = useCallback(
    async (text: string) => {
      const q = text.trim();
      if (!q || busy) return;
      setMessages((m) => [...m, { role: "user", content: q }]);
      setInput("");
      setBusy(true);
      try {
        const res = await api.assistantChat({
          question: q,
          history,
          horizon_hours: horizon,
        });
        setLlmStatus(res.llm);
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            content: res.answer,
            sources: res.sources,
            warnings: res.warnings,
          },
        ]);
        if (!res.llm.configured) {
          toast("No LLM configured — answering from the backend template", "info");
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            content: `I hit an error while querying the backend: ${msg}. Try again, or check that the API server is running.`,
          },
        ]);
        toast("Assistant could not reach the backend", "error");
      } finally {
        setBusy(false);
      }
    },
    [busy, history, horizon, toast],
  );

  const newChat = useCallback(() => {
    setMessages([]);
    setLlmStatus(null);
    setInput("");
  }, []);

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-navy-900">AI Assistant</h1>
          <p className="text-xs text-navy-400 mt-0.5">
            Answers are grounded in the backend API; sources are shown next to each reply.
          </p>
        </div>
        <button
          onClick={newChat}
          disabled={busy}
          className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 text-navy-600 hover:bg-slate-50 transition-colors disabled:opacity-50"
        >
          Clear chat
        </button>
      </div>

      <div className="flex items-center justify-end gap-2">
        <span className="text-xs text-navy-400">Forecast horizon:</span>
        <div className="flex gap-1">
          {HORIZON_OPTIONS.map((h) => (
            <button
              key={h}
              onClick={() => setHorizon(h)}
              disabled={busy}
              className={`px-2.5 py-1 text-xs rounded-md border transition-colors disabled:opacity-50 ${
                horizon === h
                  ? "border-accent-blue bg-blue-50 text-accent-blue font-semibold"
                  : "border-slate-200 text-navy-500 hover:bg-slate-50"
              }`}
            >
              {h}h
            </button>
          ))}
        </div>
        {llmStatus && (
          <span
            className={`ml-1 px-2 py-1 text-[10px] rounded-md border ${
              llmStatus.configured
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : "border-slate-200 bg-slate-50 text-navy-400"
            }`}
          >
            {llmStatus.configured
              ? `LLM: ${llmStatus.provider}${llmStatus.model ? ` / ${llmStatus.model}` : ""}`
              : "No LLM configured — template mode"}
          </span>
        )}
      </div>

      <div className="flex-1 bg-white rounded-xl border border-slate-200 shadow-card flex flex-col min-h-[480px]">
        <div className="px-4 py-3 border-b border-slate-100 flex items-center gap-2">
          <SparklesIcon className="w-4 h-4 text-accent-blue" />
          <span className="text-sm font-semibold text-navy-800">Navigation Assistant</span>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 min-h-[360px]">
          {messages.length === 0 && (
            <div className="text-center py-8">
              <p className="text-sm text-navy-500 mb-3">
                Ask about routes, sea ice, icebergs, distances or fuel — or try one of these:
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    disabled={busy}
                    className="px-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-full text-navy-600 hover:bg-blue-50 hover:border-blue-200 hover:text-accent-blue transition-colors disabled:opacity-50"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}
            >
              <div
                className={`max-w-[85%] px-3 py-2 rounded-lg text-xs leading-relaxed whitespace-pre-wrap ${
                  m.role === "user"
                    ? "bg-accent-blue text-white"
                    : "bg-slate-50 border border-slate-200 text-navy-800"
                }`}
              >
                {m.content}
              </div>

              {m.role === "assistant" && (() => {
                const warnings = m.warnings?.filter(
                  (warning) => warning !== "Predictions are model estimates; no route is guaranteed safe.",
                ) ?? [];
                const sources = m.sources?.filter(
                  (source) => !["route_details", "datasets_status"].includes(source.name),
                ) ?? [];
                if (warnings.length === 0 && sources.length === 0) return null;
                return (
                <div className="mt-1.5 max-w-[85%] space-y-1">
                  {warnings.map((w, wi) => (
                    <div
                      key={wi}
                      className="flex items-start gap-1.5 text-[10px] text-amber-700"
                    >
                      <XCircleIcon className="w-3 h-3 shrink-0 mt-0.5" />
                      <span>{w}</span>
                    </div>
                  ))}
                  {sources.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {sources.map((s) => (
                        <span
                          key={s.name}
                          title={s.note ?? undefined}
                          className={`px-2 py-0.5 rounded-full text-[10px] border ${
                            s.status === "error"
                              ? "border-red-200 bg-red-50 text-red-600"
                              : s.demo
                                ? "border-amber-200 bg-amber-50 text-amber-700"
                                : "border-emerald-200 bg-emerald-50 text-emerald-700"
                          }`}
                        >
                          {s.name}
                          {s.demo ? " · demo" : " · data"}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                );
              })()}
            </div>
          ))}

          {busy && (
            <div className="flex justify-start">
              <div className="px-3 py-2 rounded-lg bg-slate-50 border border-slate-200 text-xs text-navy-400 flex items-center gap-2">
                <span className="w-3 h-3 rounded-full border-2 border-blue-200 border-t-accent-blue animate-spin" />
                Consulting the backend…
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="px-4 py-3 border-t border-slate-100 flex items-center gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send(input)}
            placeholder="Ask about routes, ice, icebergs, drift…"
            className="flex-1 text-sm border border-slate-200 rounded-lg px-3 py-2 bg-white text-navy-800 focus:outline-none focus:ring-2 focus:ring-blue-200"
          />
          <button
            onClick={() => send(input)}
            disabled={busy}
            className="p-2 bg-accent-blue text-white rounded-lg hover:bg-accent-blue-dark transition-colors disabled:opacity-50"
          >
            <PaperAirplaneIcon className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}