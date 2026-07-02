"use client";

import React, { useState } from "react";
import Link from "next/link";
import { BookOpen, BarChart3, CloudUpload, FileUp, Globe, Youtube, CheckCircle, Play, AlertCircle } from "lucide-react";

import { useStore } from "../lib/store";

interface IngestStep {
  step: "idle" | "extracting" | "chunking" | "embedding" | "graph" | "completed" | "error";
  percent: number;
  message: string;
}

export default function IngestPage() {
  const sessionId = useStore((state) => state.sessionId);
  const [url, setUrl] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [title, setTitle] = useState("");
  
  const [ingestState, setIngestState] = useState<IngestStep>({
    step: "idle",
    percent: 0,
    message: "Ready to upload documents."
  });

  const triggerIngestSSE = async (formData: FormData) => {
    setIngestState({ step: "extracting", percent: 10, message: "Submitting request..." });
    
    try {
      const response = await fetch("http://localhost:8000/api/ingest", {
        method: "POST",
        body: formData
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
          
          const matchEvent = line.match(/^event: (.*)$/m);
          const matchData = line.match(/^data: (.*)$/m);
          
          if (matchEvent && matchData) {
            const eventType = matchEvent[1];
            const eventData = JSON.parse(matchData[1]);

            if (eventType === "progress") {
              setIngestState({
                step: eventData.step,
                percent: eventData.percent,
                message: eventData.message
              });
            } else if (eventType === "done") {
              if (eventData.error) {
                setIngestState({
                  step: "error",
                  percent: 100,
                  message: `Ingestion failed: ${eventData.error}`
                });
              } else {
                setIngestState({
                  step: "completed",
                  percent: 100,
                  message: `Done! Document ingested successfully. Created ${eventData.chunks_created} chunks.`
                });
                setUrl("");
                setYoutubeUrl("");
                setTitle("");
              }
            }
          }
        }
      }
    } catch (err: any) {
      setIngestState({
        step: "error",
        percent: 100,
        message: `Network error: ${err.message || err}`
      });
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("source_type", file.name.endsWith(".pdf") ? "pdf" : (file.name.endsWith(".docx") ? "docx" : "txt"));
    if (title) formData.append("title", title);
    
    triggerIngestSSE(formData);
  };

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;

    const formData = new FormData();
    formData.append("url", url);
    formData.append("source_type", "url");
    if (title) formData.append("title", title);

    triggerIngestSSE(formData);
  };

  const handleYoutubeSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!youtubeUrl.trim()) return;

    const formData = new FormData();
    formData.append("url", youtubeUrl);
    formData.append("source_type", "youtube");
    if (title) formData.append("title", title);

    triggerIngestSSE(formData);
  };

  return (
    <div className="flex h-screen bg-[var(--bg-base)] text-white overflow-hidden font-sans">
      {/* Navigation Sidebar */}
      <aside className="w-[260px] bg-[var(--bg-surface)] border-r border-[var(--border)] flex flex-col justify-between p-4">
        <div className="space-y-6">
          <div className="flex items-center space-x-2.5 px-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-md">
              R
            </div>
            <span className="font-bold text-sm tracking-wide bg-gradient-to-r from-indigo-200 to-indigo-400 bg-clip-text text-transparent">
              RIP ENTERPRISE
            </span>
          </div>

          <nav className="space-y-1">
            <Link
              href="/"
              className="flex items-center space-x-2 px-3 py-2.5 rounded-lg text-xs font-medium text-[var(--text-secondary)] hover:text-white hover:bg-neutral-900 transition-all"
            >
              <BookOpen className="w-4 h-4 text-neutral-400" />
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
              className="flex items-center space-x-2 px-3 py-2.5 rounded-lg text-xs font-medium bg-neutral-900 border border-[var(--border)] text-white"
            >
              <CloudUpload className="w-4 h-4 text-indigo-400" />
              <span>Document Ingest</span>
            </Link>
          </nav>
        </div>

        <div className="space-y-3 pt-4 border-t border-[var(--border)]">
          <div className="flex items-center justify-between text-[10px] text-[var(--text-muted)] font-mono">
            <span>Session ID:</span>
            <span>{sessionId.substring(0, 10)}</span>
          </div>
        </div>
      </aside>

      {/* Content panel */}
      <main className="flex-1 flex flex-col min-w-0 bg-[var(--bg-base)] overflow-y-auto p-6 space-y-6">
        <header className="flex items-center justify-between pb-4 border-b border-[var(--border)]">
          <div>
            <h1 className="text-xl font-bold tracking-tight">Data Ingestion Hub</h1>
            <p className="text-xs text-[var(--text-secondary)]">Import document files or scrape live URLs into the vector and graph engines</p>
          </div>
        </header>

        <div className="grid grid-cols-5 gap-6">
          {/* Inputs Section */}
          <div className="col-span-3 space-y-6">
            
            {/* Title field */}
            <div className="p-4 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl space-y-2">
              <label className="block text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
                Document Title (Optional)
              </label>
              <input
                type="text"
                placeholder="Give this import a name (e.g. Q2 Report)"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full bg-neutral-900 border border-[var(--border)] rounded-xl px-4 py-2.5 text-xs focus:outline-none focus:border-indigo-500 text-white"
              />
            </div>

            {/* File Ingest Card */}
            <div className="p-5 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl space-y-4">
              <div className="flex items-center space-x-2 text-sm font-semibold">
                <FileUp className="w-4 h-4 text-indigo-400" />
                <span>Upload Document Files</span>
              </div>
              <div className="border border-dashed border-[var(--border)] rounded-xl p-8 text-center bg-neutral-950 relative hover:border-indigo-500/50 transition-colors">
                <input
                  type="file"
                  accept=".pdf,.docx,.txt,.md"
                  onChange={handleFileUpload}
                  className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                />
                <CloudUpload className="w-10 h-10 text-[var(--text-muted)] mx-auto mb-3" />
                <span className="block text-xs font-semibold text-white">Drag and drop files here</span>
                <span className="block text-[10px] text-[var(--text-muted)] mt-1">Supports PDF, DOCX, TXT, Markdown (Max 15MB)</span>
              </div>
            </div>

            {/* URL Scraper Card */}
            <div className="p-5 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl space-y-4">
              <div className="flex items-center space-x-2 text-sm font-semibold">
                <Globe className="w-4 h-4 text-sky-400" />
                <span>Web URL Crawler</span>
              </div>
              <form onSubmit={handleUrlSubmit} className="flex items-center space-x-2">
                <input
                  type="url"
                  placeholder="https://example.com/research-paper"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  className="flex-1 bg-neutral-900 border border-[var(--border)] rounded-xl px-4 py-2.5 text-xs focus:outline-none focus:border-indigo-500 text-white"
                />
                <button
                  type="submit"
                  className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold cursor-pointer"
                >
                  Crawl URL
                </button>
              </form>
            </div>

            {/* YouTube transcript Card */}
            <div className="p-5 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl space-y-4">
              <div className="flex items-center space-x-2 text-sm font-semibold">
                <Youtube className="w-4 h-4 text-red-400" />
                <span>YouTube Transcript Scraper</span>
              </div>
              <form onSubmit={handleYoutubeSubmit} className="flex items-center space-x-2">
                <input
                  type="url"
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  className="flex-1 bg-neutral-900 border border-[var(--border)] rounded-xl px-4 py-2.5 text-xs focus:outline-none focus:border-indigo-500 text-white"
                />
                <button
                  type="submit"
                  className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold cursor-pointer"
                >
                  Fetch Transcript
                </button>
              </form>
            </div>

          </div>

          {/* Progress / Outputs Status Section */}
          <div className="col-span-2 space-y-6">
            
            {/* Status Visualizer */}
            <div className="p-5 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl h-full flex flex-col justify-between min-h-[300px]">
              <div>
                <h3 className="font-semibold text-sm mb-4">Ingestion Telemetry Progress</h3>
                
                {ingestState.step === "idle" ? (
                  <div className="text-xs text-[var(--text-secondary)] bg-neutral-950 p-4 border border-[var(--border)] rounded-xl">
                    No active ingestion. Choose a file or crawler payload on the left to index.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Progress Bar */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold capitalize text-indigo-400">Step: {ingestState.step}</span>
                        <span className="font-mono text-[var(--text-muted)]">{ingestState.percent}%</span>
                      </div>
                      <div className="w-full h-2 bg-neutral-900 rounded-full overflow-hidden border border-[var(--border)]">
                        <div
                          className="h-full bg-indigo-500 transition-all duration-500 ease-out"
                          style={{ width: `${ingestState.percent}%` }}
                        />
                      </div>
                    </div>

                    {/* Step Nodes Checklist */}
                    <div className="space-y-3 pt-3">
                      <div className="flex items-center space-x-2 text-xs">
                        {ingestState.step === "completed" ? (
                          <CheckCircle className="w-4 h-4 text-green-400" />
                        ) : ["extracting", "chunking", "embedding", "graph"].includes(ingestState.step) ? (
                          <Play className="w-4 h-4 text-indigo-400 animate-pulse" />
                        ) : (
                          <div className="w-4 h-4 rounded-full border border-neutral-700" />
                        )}
                        <span className={ingestState.step === "extracting" ? "text-white" : "text-[var(--text-secondary)]"}>
                          Extracting file content
                        </span>
                      </div>

                      <div className="flex items-center space-x-2 text-xs">
                        {["embedding", "graph", "completed"].includes(ingestState.step) ? (
                          <CheckCircle className="w-4 h-4 text-green-400" />
                        ) : ingestState.step === "chunking" ? (
                          <Play className="w-4 h-4 text-indigo-400 animate-pulse" />
                        ) : (
                          <div className="w-4 h-4 rounded-full border border-neutral-700" />
                        )}
                        <span className={ingestState.step === "chunking" ? "text-white" : "text-[var(--text-secondary)]"}>
                          Semantic cosine similarity chunking
                        </span>
                      </div>

                      <div className="flex items-center space-x-2 text-xs">
                        {["graph", "completed"].includes(ingestState.step) ? (
                          <CheckCircle className="w-4 h-4 text-green-400" />
                        ) : ingestState.step === "embedding" ? (
                          <Play className="w-4 h-4 text-indigo-400 animate-pulse" />
                        ) : (
                          <div className="w-4 h-4 rounded-full border border-neutral-700" />
                        )}
                        <span className={ingestState.step === "embedding" ? "text-white" : "text-[var(--text-secondary)]"}>
                          Generating vector embeddings with text-embedding-004
                        </span>
                      </div>

                      <div className="flex items-center space-x-2 text-xs">
                        {ingestState.step === "completed" ? (
                          <CheckCircle className="w-4 h-4 text-green-400" />
                        ) : ingestState.step === "graph" ? (
                          <Play className="w-4 h-4 text-indigo-400 animate-pulse" />
                        ) : (
                          <div className="w-4 h-4 rounded-full border border-neutral-700" />
                        )}
                        <span className={ingestState.step === "graph" ? "text-white" : "text-[var(--text-secondary)]"}>
                          Entity relationship extraction and graph database load
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Status Message Footer */}
              <div className={`mt-6 p-4 rounded-xl border flex items-center space-x-3 text-xs leading-relaxed ${
                ingestState.step === "completed"
                  ? "bg-green-500/10 border-green-500/20 text-green-400"
                  : ingestState.step === "error"
                  ? "bg-red-500/10 border-red-500/20 text-red-400"
                  : "bg-neutral-900 border-[var(--border)] text-[var(--text-secondary)]"
              }`}>
                {ingestState.step === "completed" ? (
                  <CheckCircle className="w-5 h-5 flex-shrink-0" />
                ) : ingestState.step === "error" ? (
                  <AlertCircle className="w-5 h-5 flex-shrink-0" />
                ) : (
                  <Play className="w-5 h-5 text-indigo-400 animate-pulse flex-shrink-0" />
                )}
                <span>{ingestState.message}</span>
              </div>

            </div>

          </div>
        </div>

      </main>
    </div>
  );
}
