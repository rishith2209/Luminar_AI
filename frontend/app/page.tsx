"use client";

import React, { useState, useEffect, useRef } from "react";
import { Send, Plus, BarChart3, CloudUpload, ArrowRight, BookOpen, Trash2, Sliders, Menu, X } from "lucide-react";
import Link from "next/link";
import { Message, useStore } from "../lib/store";
import { StreamingMessage } from "../components/chat/StreamingMessage";
import { AgentTrace } from "../components/agent-trace/AgentTrace";
import { SourceCard } from "../components/source-panel/SourceCard";
import { EvalPill } from "../components/eval-dashboard/EvalPill";

export default function ChatPage() {
  const {
    sessionId,
    messages,
    activeAgent,
    activeTrace,
    selectedCitation,
    sidebarOpen,
    activeTab,
    addMessage,
    updateLastMessage,
    setActiveAgent,
    setActiveTrace,
    toggleSidebar,
    setActiveTab,
    clearChat
  } = useStore();

  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeAgent]);

  // Connect WebSockets for live agent traces
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "trace") {
          // Append incoming trace event to Zustand activeTrace
          useStore.setState((state) => ({
            activeTrace: [
              ...state.activeTrace,
              {
                agent: data.agent,
                status: data.status,
                message: data.message,
                timestamp: Date.now() / 1000,
                latency_ms: data.latency_ms
              }
            ],
            activeAgent: data.status === "active" || data.status === "thinking" ? data.agent : null
          }));
        }
      } catch (err) {
        console.error("Trace websocket parse error", err);
      }
    };

    return () => {
      ws.close();
    };
  }, [sessionId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    const userQuery = input;
    setInput("");
    setIsStreaming(true);
    setActiveTrace([]); // Clear trace logs for new run
    
    // 1. Add User Message
    addMessage({
      id: Math.random().toString(),
      role: "user",
      content: userQuery
    });

    // 2. Add empty Assistant message to stream tokens into
    const assistantMsgId = Math.random().toString();
    addMessage({
      id: assistantMsgId,
      role: "assistant",
      content: ""
    });

    try {
      // Create request payload
      const response = await fetch("http://localhost:8000/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userQuery,
          session_id: sessionId,
          mode: "thorough"
        })
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          
          // Parse SSE event
          const matchEvent = line.match(/^event: (.*)$/m);
          const matchData = line.match(/^data: (.*)$/m);
          
          if (matchEvent && matchData) {
            const eventType = matchEvent[1];
            const eventData = JSON.parse(matchData[1]);

            if (eventType === "synthesis_progress") {
              updateLastMessage((msg) => ({
                ...msg,
                content: eventData.partial_answer
              }));
            } else if (eventType === "done") {
              updateLastMessage((msg) => ({
                ...msg,
                content: eventData.answer,
                citations: eventData.citations,
                eval_scores: eventData.eval_scores,
                agent_trace: eventData.agent_trace
              }));
              // Sync complete trace to side view
              setActiveTrace(eventData.agent_trace);
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      updateLastMessage((msg) => ({
        ...msg,
        content: "An error occurred while streaming response. Please verify backend state."
      }));
    } finally {
      setIsStreaming(false);
      setActiveAgent(null);
    }
  };

  return (
    <div className="flex h-screen bg-[var(--bg-base)] text-white overflow-hidden font-sans">
      
      {/* LEFT PANEL: Navigation Sidebar */}
      <aside className="w-[260px] bg-[var(--bg-surface)] border-r border-[var(--border)] flex flex-col justify-between p-4">
        <div className="space-y-6">
          <div className="flex items-center space-x-2.5 px-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-md shadow-indigo-600/30">
              R
            </div>
            <span className="font-bold text-sm tracking-wide bg-gradient-to-r from-indigo-200 to-indigo-400 bg-clip-text text-transparent">
              RIP ENTERPRISE
            </span>
          </div>

          <button
            onClick={clearChat}
            className="w-full flex items-center space-x-2 px-3 py-2 rounded-xl bg-indigo-600/10 border border-indigo-500/20 text-indigo-300 font-medium text-xs hover:bg-indigo-600/20 transition-all cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            <span>New Research Chat</span>
          </button>

          <nav className="space-y-1">
            <Link
              href="/"
              className="flex items-center space-x-2 px-3 py-2.5 rounded-lg text-xs font-medium bg-neutral-900 border border-[var(--border)] text-white"
            >
              <BookOpen className="w-4 h-4 text-indigo-400" />
              <span>Research Agent</span>
            </Link>
            <Link
              href="/dashboard"
              className="flex items-center space-x-2 px-3 py-2.5 rounded-lg text-xs font-medium text-[var(--text-secondary)] hover:text-white hover:bg-neutral-900 transition-all"
            >
              <BarChart3 className="w-4 h-4 text-neutral-400" />
              <span>Metrics & Knowledge Graph</span>
            </Link>
            <Link
              href="/ingest"
              className="flex items-center space-x-2 px-3 py-2.5 rounded-lg text-xs font-medium text-[var(--text-secondary)] hover:text-white hover:bg-neutral-900 transition-all"
            >
              <CloudUpload className="w-4 h-4 text-neutral-400" />
              <span>Document Ingest</span>
            </Link>
          </nav>
        </div>

        <div className="space-y-3 pt-4 border-t border-[var(--border)]">
          <div className="flex items-center justify-between text-[10px] text-[var(--text-muted)] font-mono">
            <span>Session ID:</span>
            <span>{sessionId.substring(0, 10)}</span>
          </div>
          <button
            onClick={clearChat}
            className="w-full flex items-center justify-center space-x-1.5 py-1.5 rounded-lg text-[10px] text-red-400 hover:bg-red-500/10 cursor-pointer border border-transparent hover:border-red-500/20 font-medium transition-all"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Conversation</span>
          </button>
        </div>
      </aside>

      {/* CENTER PANEL: Streaming Chat View */}
      <main className="flex-1 flex flex-col min-w-0 bg-[var(--bg-base)]">
        <header className="h-14 border-b border-[var(--border)] px-6 flex items-center justify-between bg-[var(--bg-surface)]/60 backdrop-blur">
          <div className="flex items-center space-x-2">
            <h1 className="font-semibold text-sm">Research Workspace</h1>
            {isStreaming && (
              <div className="w-2 h-2 rounded-full bg-green-500 animate-ping" />
            )}
          </div>
          <button
            onClick={toggleSidebar}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs border border-[var(--border)] text-[var(--text-secondary)] hover:text-white hover:bg-neutral-900 cursor-pointer transition-all"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>{sidebarOpen ? "Hide Details" : "Show Details"}</span>
          </button>
        </header>

        {/* Messages list */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-none">
          {messages.length === 0 ? (
            <div className="max-w-2xl mx-auto mt-20 text-center space-y-4">
              <h2 className="text-2xl font-bold text-white">Enterprise Research Assistant</h2>
              <p className="text-sm text-[var(--text-secondary)]">
                Submit complex questions and queries. The platform orchestrates retrieval models, crawls databases, extracts knowledge entities, and evaluates output faithfulness.
              </p>
              <div className="grid grid-cols-2 gap-3 pt-6 text-left">
                <button
                  onClick={() => setInput("Explain the difference between Model Context Protocol (MCP) and A2A Agent routing")}
                  className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] text-xs hover:border-indigo-500/40 text-left transition-all cursor-pointer"
                >
                  <strong className="block text-white mb-1">Comparative query</strong>
                  Explain difference between MCP and A2A agent routing protocol.
                </button>
                <button
                  onClick={() => setInput("What are the core pillars of Advanced RAG architectures?")}
                  className="p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] text-xs hover:border-indigo-500/40 text-left transition-all cursor-pointer"
                >
                  <strong className="block text-white mb-1">Factual lookup</strong>
                  What are the core pillars of Advanced RAG architectures?
                </button>
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className="max-w-4xl mx-auto space-y-2">
                <StreamingMessage
                  content={msg.content}
                  role={msg.role}
                  isStreaming={isStreaming && !msg.content && msg.role === "assistant"}
                  citations={msg.citations}
                />
                
                {/* Eval details pill row below assistant message */}
                {msg.role === "assistant" && msg.eval_scores && (
                  <div className="flex flex-wrap gap-2 pl-4 pb-4 border-b border-[var(--border)] animate-fade-in">
                    <EvalPill
                      label="Faithfulness"
                      score={msg.eval_scores.faithfulness}
                      description="Measures if the generated answer is strictly grounded in the retrieved context passages, ensuring zero speculation."
                      formula="claims_supported_by_context / total_extracted_claims"
                    />
                    <EvalPill
                      label="Answer Relevancy"
                      score={msg.eval_scores.relevancy}
                      description="Calculates whether the answer directly addresses the query topic using cosine similarity of query variants."
                      formula="mean_cosine_similarity(generated_questions, original_query)"
                    />
                    <EvalPill
                      label="Recall Accuracy"
                      score={msg.eval_scores.recall}
                      description="Evaluates context precision, measuring the proportion of useful vs redundant chunks retrieved."
                      formula="useful_retrieved_chunks / total_retrieved_chunks"
                    />
                    <EvalPill
                      label="Overall Performance"
                      score={msg.eval_scores.overall}
                      description="Harmonized average score across the three evaluation sub-metrics."
                      formula="average(faithfulness, relevancy, recall)"
                    />
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Input box */}
        <footer className="p-4 bg-[var(--bg-surface)] border-t border-[var(--border)]">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex items-center space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask research agent..."
              disabled={isStreaming}
              className="flex-1 bg-neutral-900 border border-[var(--border)] rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-indigo-500 text-white placeholder-neutral-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isStreaming || !input.trim()}
              className="px-4 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer disabled:opacity-40 transition-colors flex items-center space-x-1"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </footer>
      </main>

      {/* RIGHT PANEL: Decision Trace and Source Viewer */}
      {sidebarOpen && (
        <aside className="w-[340px] bg-[var(--bg-surface)] border-l border-[var(--border)] flex flex-col">
          {/* Tabs bar */}
          <div className="flex border-b border-[var(--border)]">
            <button
              onClick={() => setActiveTab("trace")}
              className={`flex-1 py-3 text-xs font-semibold uppercase tracking-wider text-center cursor-pointer transition-colors ${
                activeTab === "trace"
                  ? "text-indigo-400 border-b-2 border-indigo-500 bg-neutral-900/50"
                  : "text-[var(--text-secondary)] hover:text-white"
              }`}
            >
              Agent Trace
            </button>
            <button
              onClick={() => setActiveTab("sources")}
              className={`flex-1 py-3 text-xs font-semibold uppercase tracking-wider text-center cursor-pointer transition-colors ${
                activeTab === "sources"
                  ? "text-indigo-400 border-b-2 border-indigo-500 bg-neutral-900/50"
                  : "text-[var(--text-secondary)] hover:text-white"
              }`}
            >
              Context Sources
            </button>
          </div>

          <div className="flex-1 overflow-hidden">
            {activeTab === "trace" ? (
              <AgentTrace />
            ) : (
              <div className="flex flex-col h-full overflow-hidden">
                <div className="p-4 border-b border-[var(--border)] font-semibold flex items-center justify-between text-sm">
                  <span>Context Sources</span>
                  <span className="text-xs text-[var(--text-muted)]">Citations</span>
                </div>
                
                <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-none">
                  {selectedCitation && (
                    <div className="mb-4 p-3 rounded-xl border border-indigo-500/30 bg-indigo-600/5">
                      <div className="text-[10px] uppercase font-bold text-indigo-400 mb-1">Selected Citation</div>
                      <h4 className="font-semibold text-xs text-white mb-1">{selectedCitation.title}</h4>
                      <p className="text-xs text-[var(--text-secondary)] leading-relaxed italic">{selectedCitation.snippet}</p>
                    </div>
                  )}

                  {messages.length > 0 && messages[messages.length - 1].citations ? (
                    messages[messages.length - 1].citations?.map((cit) => (
                      <SourceCard
                        key={cit.index}
                        citation={cit}
                        score={0.88} // Dynamic proxy score
                      />
                    ))
                  ) : (
                    <div className="text-[var(--text-muted)] text-center mt-10 text-xs">
                      No document chunks loaded for the active response.
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </aside>
      )}

    </div>
  );
}
