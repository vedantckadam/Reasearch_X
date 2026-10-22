"use client";

import React, { useState, useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import { fetchAPI } from "@/lib/api";
import { generateMessageId, getCurrentTimestamp } from "@/lib/chat-utils";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  grounded?: boolean;
  timestamp: string;
}

interface ChatResponse {
  reply: string;
  model: string;
  grounded_context_used: boolean;
  analysis_id?: string | null;
}

const SUGGESTED_QUESTIONS = [
  "Explain my research gaps",
  "Summarize my uploaded paper",
  "Explain the topic analysis",
  "How can I validate a research gap?",
];

// Safe markdown formatter without dangerouslySetInnerHTML
function FormattedMessage({ text }: { text: string }) {
  const lines = text.split("\n");

  return (
    <div className="space-y-1.5 text-xs sm:text-sm leading-relaxed text-slate-800">
      {lines.map((line, idx) => {
        const trimmed = line.trim();

        if (!trimmed) {
          return <div key={idx} className="h-1.5" />;
        }

        // Heading 3: ### Heading
        if (trimmed.startsWith("### ")) {
          return (
            <h4 key={idx} className="font-bold text-slate-900 mt-2 mb-1 text-sm">
              {trimmed.replace(/^###\s+/, "")}
            </h4>
          );
        }

        // Heading 2: ## Heading
        if (trimmed.startsWith("## ")) {
          return (
            <h3 key={idx} className="font-bold text-slate-900 mt-2 mb-1 text-sm sm:text-base border-b border-slate-200 pb-0.5">
              {trimmed.replace(/^##\s+/, "")}
            </h3>
          );
        }

        // Bullet list item
        if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
          const itemText = trimmed.replace(/^[-*]\s+/, "");
          return (
            <div key={idx} className="flex items-start gap-2 pl-1">
              <span className="text-indigo-600 font-bold shrink-0 mt-0.5">•</span>
              <span>{renderInlineStyles(itemText)}</span>
            </div>
          );
        }

        // Numbered list item: 1. Text
        const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
        if (numMatch) {
          return (
            <div key={idx} className="flex items-start gap-2 pl-1">
              <span className="text-indigo-600 font-semibold shrink-0 text-xs mt-0.5">{numMatch[1]}.</span>
              <span>{renderInlineStyles(numMatch[2])}</span>
            </div>
          );
        }

        // Regular paragraph
        return <p key={idx}>{renderInlineStyles(line)}</p>;
      })}
    </div>
  );
}

// Helper to format **bold** and `code` inline without innerHTML
function renderInlineStyles(text: string): React.ReactNode {
  // Tokenize bold **...** and inline code `...`
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*.*?\*\*|`.*?`)/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }

    const token = match[0];
    if (token.startsWith("**") && token.endsWith("**")) {
      parts.push(
        <strong key={match.index} className="font-semibold text-slate-900">
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith("`") && token.endsWith("`")) {
      parts.push(
        <code key={match.index} className="bg-slate-200 text-indigo-700 px-1 py-0.5 rounded text-xs font-mono">
          {token.slice(1, -1)}
        </code>
      );
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  return parts.length > 0 ? parts : text;
}

export default function FloatingChatbot() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Extract analysisId if user is on /analysis/[id]/* route
  const match = pathname ? pathname.match(/\/analysis\/([^/]+)/) : null;
  const analysisId = match ? match[1] : null;

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isOpen, isLoading]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 150);
    }
  }, [isOpen]);

  const handleSendMessage = async (customText?: string) => {
    const textToSend = (customText !== undefined ? customText : inputValue).trim();
    if (!textToSend || isLoading) return;

    setErrorMessage(null);
    const userMsg: ChatMessage = {
      id: generateMessageId("user"),
      role: "user",
      content: textToSend,
      timestamp: getCurrentTimestamp(),
    };

    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    if (!customText) {
      setInputValue("");
    }
    setIsLoading(true);

    try {
      // Build history payload (last 6 messages)
      const historyPayload = newMessages.slice(-6).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const res = await fetchAPI<ChatResponse>("/api/chat", {
        method: "POST",
        body: JSON.stringify({
          message: textToSend,
          history: historyPayload,
          analysis_id: analysisId || undefined,
        }),
      });

      const assistantMsg: ChatMessage = {
        id: generateMessageId("asst"),
        role: "assistant",
        content: res.reply,
        grounded: res.grounded_context_used,
        timestamp: getCurrentTimestamp(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: unknown) {
      const rawMsg = err instanceof Error ? err.message : String(err);
      let detailMsg = rawMsg;
      try {
        const parsed = JSON.parse(rawMsg);
        if (parsed.detail) {
          detailMsg = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
        }
      } catch {}

      if (rawMsg.includes("503") || detailMsg.includes("not configured") || detailMsg.includes("OPENROUTER_API_KEY")) {
        setErrorMessage(
          "AI Service Unconfigured: Set OPENROUTER_API_KEY in backend/.env to chat with Nemotron."
        );
      } else if (rawMsg.includes("504") || detailMsg.includes("timed out") || detailMsg.includes("timeout")) {
        setErrorMessage(
          "The AI provider timed out. OpenRouter may be busy; please retry your question."
        );
      } else if (detailMsg.includes("authentication failed")) {
        setErrorMessage(
          "AI Provider Error: Authentication failed. Please verify your OPENROUTER_API_KEY."
        );
      } else {
        setErrorMessage(detailMsg || "Failed to communicate with AI Assistant.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setErrorMessage(null);
  };

  return (
    <>
      {/* Floating Action Button (Always in bottom-right) */}
      <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2">
        <button
          onClick={() => setIsOpen(!isOpen)}
          aria-label={isOpen ? "Close AI Assistant" : "Open ResearchGap-X AI Assistant"}
          className={`group flex items-center justify-center w-14 h-14 rounded-full shadow-xl transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-indigo-300 ${
            isOpen
              ? "bg-slate-900 text-white rotate-90 scale-95"
              : "bg-gradient-to-r from-slate-900 via-indigo-900 to-purple-900 text-white hover:scale-105 hover:shadow-2xl hover:from-slate-800 hover:to-indigo-800"
          }`}
        >
          {isOpen ? (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          ) : (
            <div className="relative flex items-center justify-center">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white group-hover:scale-110 transition-transform">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                <path d="M12 7v2" strokeWidth="2.5" />
                <path d="M10 11h4" strokeWidth="2.5" />
              </svg>
              {analysisId && (
                <span className="absolute -top-1 -right-1 flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500 border-2 border-slate-900" />
                </span>
              )}
            </div>
          )}
        </button>
      </div>

      {/* Floating Chat Window */}
      {isOpen && (
        <div
          role="dialog"
          aria-label="ResearchGap-X AI Assistant"
          className="fixed bottom-24 right-4 left-4 sm:left-auto sm:right-6 sm:w-[410px] h-[540px] max-h-[80vh] z-50 flex flex-col bg-white rounded-2xl border border-slate-200/90 shadow-2xl overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-200"
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-purple-950 text-white px-4 py-3 flex items-center justify-between border-b border-indigo-900/50 shrink-0">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center shadow-inner text-white">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                </svg>
              </div>
              <div>
                <h3 className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5">
                  ResearchGap-X AI Assistant
                </h3>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className={`w-2 h-2 rounded-full ${analysisId ? "bg-emerald-400 animate-pulse" : "bg-indigo-400"}`} />
                  <span className="text-[11px] text-slate-300 font-medium">
                    {analysisId ? "Grounded in Active Analysis" : "General Methodology Mode"}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-1 text-slate-300">
              {messages.length > 0 && (
                <button
                  onClick={clearChat}
                  title="Clear conversation"
                  aria-label="Clear conversation"
                  className="p-1.5 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </svg>
                </button>
              )}
              <button
                onClick={() => setIsOpen(false)}
                title="Close Assistant"
                aria-label="Close Assistant"
                className="p-1.5 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              >
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          </div>

          {/* Messages Body */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3.5 bg-slate-50/60">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center text-center h-full px-2 py-4">
                <div className="w-12 h-12 rounded-2xl bg-indigo-100 text-indigo-700 flex items-center justify-center mb-3 shadow-sm">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                  </svg>
                </div>
                <h4 className="text-sm font-bold text-slate-900 mb-1">
                  How can I assist your research?
                </h4>
                <p className="text-xs text-slate-500 mb-4 max-w-xs leading-relaxed">
                  {analysisId
                    ? "I am connected to your active corpus analysis. Ask questions about your seed paper, topic models, or identified research gaps."
                    : "I can explain research gap methodologies, SVD matrix decomposition, or guide you through uploading a seed paper."}
                </p>

                {/* Suggested Questions */}
                <div className="w-full space-y-1.5">
                  <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2">
                    Suggested Questions
                  </p>
                  {SUGGESTED_QUESTIONS.map((question, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSendMessage(question)}
                      className="w-full text-left text-xs bg-white hover:bg-indigo-50/70 border border-slate-200 hover:border-indigo-200 text-slate-700 hover:text-indigo-900 p-2.5 rounded-xl transition-all shadow-2xs flex items-center justify-between group"
                    >
                      <span className="truncate">{question}</span>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-slate-400 group-hover:text-indigo-600 transition-colors shrink-0">
                        <polyline points="9 18 15 12 9 6" />
                      </svg>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}
                >
                  <div
                    className={`max-w-[88%] rounded-2xl px-3.5 py-2.5 shadow-2xs text-xs sm:text-sm ${
                      msg.role === "user"
                        ? "bg-indigo-600 text-white rounded-br-xs"
                        : "bg-white border border-slate-200/80 text-slate-800 rounded-bl-xs"
                    }`}
                  >
                    {msg.role === "user" ? (
                      <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                    ) : (
                      <FormattedMessage text={msg.content} />
                    )}
                  </div>
                  <div className="flex items-center gap-1.5 mt-1 px-1">
                    {msg.grounded && (
                      <span className="inline-flex items-center gap-0.5 text-[10px] font-medium text-emerald-600 bg-emerald-50 px-1.5 py-0.2 rounded">
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                        Grounded
                      </span>
                    )}
                    <span className="text-[10px] text-slate-400">{msg.timestamp}</span>
                  </div>
                </div>
              ))
            )}

            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex items-start gap-2">
                <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-xs px-4 py-3 shadow-2xs flex items-center gap-2">
                  <div className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-2 h-2 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-2 h-2 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                  <span className="text-xs text-slate-500 font-medium">Synthesizing research response...</span>
                </div>
              </div>
            )}

            {/* Inline Error Banner */}
            {errorMessage && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700 flex items-start gap-2">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0 mt-0.5">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <div className="flex-1">
                  <p className="font-semibold">Notice</p>
                  <p className="mt-0.5 leading-relaxed">{errorMessage}</p>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Footer Input */}
          <div className="p-3 bg-white border-t border-slate-100 shrink-0">
            <div className="relative flex items-end gap-2 bg-slate-50 border border-slate-200 rounded-xl p-1.5 focus-within:border-indigo-500 focus-within:ring-2 focus-within:ring-indigo-100 transition-all">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  analysisId
                    ? "Ask about paper summary, topics, or gaps..."
                    : "Ask about research gap methodology..."
                }
                rows={1}
                disabled={isLoading}
                className="flex-1 bg-transparent border-0 resize-none px-2 py-1.5 text-xs sm:text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none max-h-24 min-h-[36px]"
              />
              <button
                onClick={() => handleSendMessage()}
                disabled={!inputValue.trim() || isLoading}
                aria-label="Send message"
                className="w-8 h-8 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed text-white flex items-center justify-center shrink-0 transition-colors shadow-2xs"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
            <p className="text-[10px] text-center text-slate-400 mt-1.5">
              Press Enter to send • Shift+Enter for new line
            </p>
          </div>
        </div>
      )}
    </>
  );
}
