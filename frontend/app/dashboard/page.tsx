"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { BookOpen, BarChart3, CloudUpload, ArrowRight, Activity, Database, HeartPulse, Network } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from "recharts";

import { useStore } from "../lib/store";

// Mock data for Recharts dashboard
const faithfulnessData = [
  { time: "09:00", score: 0.82 },
  { time: "11:00", score: 0.88 },
  { time: "13:00", score: 0.85 },
  { time: "15:00", score: 0.92 },
  { time: "17:00", score: 0.90 },
  { time: "19:00", score: 0.94 },
];

const strategyData = [
  { name: "Hybrid Search", value: 64, color: "#6366f1" },
  { name: "HyDE", value: 28, color: "#a855f7" },
  { name: "Graph RAG", value: 18, color: "#ec4899" },
];

const latencyData = [
  { metric: "P50", latency: 120 },
  { metric: "P95", latency: 450 },
  { metric: "P99", latency: 980 },
];

// Interactive SVG Knowledge Graph Visualizer (Lightweight force-directed simulator in pure React)
const KnowledgeGraphViz: React.FC = () => {
  const [nodes, setNodes] = useState([
    { id: "RAG", group: 1, x: 150, y: 150, r: 24, label: "RAG Platform" },
    { id: "Gemini", group: 2, x: 80, y: 80, r: 18, label: "Gemini 2.0" },
    { id: "Qdrant", group: 2, x: 220, y: 80, r: 18, label: "Qdrant DB" },
    { id: "Neo4j", group: 2, x: 80, y: 220, r: 18, label: "Neo4j Graph" },
    { id: "A2A", group: 3, x: 220, y: 220, r: 18, label: "A2A Agent" },
    { id: "MCP", group: 3, x: 150, y: 40, r: 18, label: "MCP Protocol" },
  ]);

  const links = [
    { source: "RAG", target: "Gemini", type: "uses" },
    { source: "RAG", target: "Qdrant", type: "indexes" },
    { source: "RAG", target: "Neo4j", type: "maps" },
    { source: "RAG", target: "A2A", type: "routes" },
    { source: "RAG", target: "MCP", type: "exposes" },
    { source: "A2A", target: "Gemini", type: "executes" },
  ];

  // Very basic simulation to drift nodes slowly on render
  useEffect(() => {
    let frame = 0;
    const interval = setInterval(() => {
      frame++;
      setNodes((prevNodes) =>
        prevNodes.map((n) => {
          if (n.id === "RAG") return n; // Keep root anchored
          // Simple orbit drift animation
          const angle = (frame * 0.02) + (n.group * Math.PI / 2);
          const radius = n.group === 2 ? 90 : 130;
          return {
            ...n,
            x: 150 + Math.cos(angle) * radius,
            y: 150 + Math.sin(angle) * radius,
          };
        })
      );
    }, 30);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full h-[280px] bg-neutral-950 border border-[var(--border)] rounded-xl relative overflow-hidden flex items-center justify-center">
      <svg className="w-full h-full" viewBox="0 0 300 300">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="15" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#4b5563" />
          </marker>
        </defs>

        {/* Draw Links */}
        {links.map((link, idx) => {
          const sourceNode = nodes.find((n) => n.id === link.source);
          const targetNode = nodes.find((n) => n.id === link.target);
          if (!sourceNode || !targetNode) return null;
          return (
            <line
              key={idx}
              x1={sourceNode.x}
              y1={sourceNode.y}
              x2={targetNode.x}
              y2={targetNode.y}
              stroke="#262626"
              strokeWidth="1.5"
              markerEnd="url(#arrow)"
            />
          );
        })}

        {/* Draw Nodes */}
        {nodes.map((node) => (
          <g key={node.id}>
            <circle
              cx={node.x}
              cy={node.y}
              r={node.r}
              className={`${
                node.id === "RAG"
                  ? "fill-indigo-600 stroke-indigo-400"
                  : "fill-neutral-900 stroke-neutral-700"
              } stroke-2 cursor-pointer transition-all hover:stroke-indigo-400`}
            />
            <text
              x={node.x}
              y={node.y + 4}
              className="text-[9px] fill-neutral-300 select-none font-medium text-center"
              textAnchor="middle"
            >
              {node.id}
            </text>
            <title>{node.label}</title>
          </g>
        ))}
      </svg>
      <div className="absolute bottom-2 left-2 text-[9px] text-[var(--text-muted)] font-mono">
        Live force-simulation schema
      </div>
    </div>
  );
};

export default function DashboardPage() {
  const sessionId = useStore((state) => state.sessionId);
  const [stats, setStats] = useState({
    total_documents: 14,
    total_chunks: 582,
    avg_faithfulness: 0.91,
    query_count: 247,
  });

  useEffect(() => {
    // Fetch real stats from API
    fetch("http://localhost:8000/api/stats")
      .then((res) => res.json())
      .then((data) => {
        if (data.status === "healthy") {
          setStats(data);
        }
      })
      .catch((err) => console.log("Stats fetch fallback", err));
  }, []);

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
              className="flex items-center space-x-2 px-3 py-2.5 rounded-lg text-xs font-medium bg-neutral-900 border border-[var(--border)] text-white"
            >
              <BarChart3 className="w-4 h-4 text-indigo-400" />
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
        </div>
      </aside>

      {/* Main Stats Grid Page */}
      <main className="flex-1 flex flex-col min-w-0 bg-[var(--bg-base)] overflow-y-auto p-6 space-y-6">
        <header className="flex items-center justify-between pb-4 border-b border-[var(--border)]">
          <div>
            <h1 className="text-xl font-bold tracking-tight">System Performance & Insights</h1>
            <p className="text-xs text-[var(--text-secondary)]">Real-time telemetry, RAGAS scores, and Neo4j entities schema</p>
          </div>
        </header>

        {/* 4 Cards Row */}
        <div className="grid grid-cols-4 gap-4">
          <div className="p-4 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl flex items-center space-x-4">
            <div className="p-3 rounded-xl bg-indigo-500/10 text-indigo-400">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider block font-semibold">Total Documents</span>
              <strong className="text-lg text-white font-bold">{stats.total_documents}</strong>
            </div>
          </div>

          <div className="p-4 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl flex items-center space-x-4">
            <div className="p-3 rounded-xl bg-purple-500/10 text-purple-400">
              <Network className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider block font-semibold">Total Chunks</span>
              <strong className="text-lg text-white font-bold">{stats.total_chunks}</strong>
            </div>
          </div>

          <div className="p-4 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl flex items-center space-x-4">
            <div className="p-3 rounded-xl bg-green-500/10 text-green-400">
              <HeartPulse className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider block font-semibold">Avg Faithfulness</span>
              <strong className="text-lg text-white font-bold">{Math.round(stats.avg_faithfulness * 100)}%</strong>
            </div>
          </div>

          <div className="p-4 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl flex items-center space-x-4">
            <div className="p-3 rounded-xl bg-sky-500/10 text-sky-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider block font-semibold">Query Count</span>
              <strong className="text-lg text-white font-bold">{stats.query_count}</strong>
            </div>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-3 gap-6">
          {/* Chart 1: Faithfulness Line */}
          <div className="col-span-2 p-5 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl space-y-4">
            <h3 className="font-semibold text-sm">Faithfulness Score Trend</h3>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={faithfulnessData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
                  <XAxis dataKey="time" stroke="#525252" fontSize={10} />
                  <YAxis stroke="#525252" domain={[0, 1]} fontSize={10} />
                  <Tooltip contentStyle={{ backgroundColor: "#111", borderColor: "#262626" }} />
                  <Line type="monotone" dataKey="score" stroke="#6366f1" strokeWidth={2.5} dot={{ fill: "#6366f1" }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* SVG Force Graph */}
          <div className="p-5 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl space-y-4">
            <h3 className="font-semibold text-sm">Active Graph Entities</h3>
            <KnowledgeGraphViz />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          {/* Chart 2: Strategy usage */}
          <div className="p-5 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl space-y-4">
            <h3 className="font-semibold text-sm">Retrieval Strategy Split</h3>
            <div className="h-[180px] flex items-center justify-around">
              <ResponsiveContainer width="50%" height="100%">
                <PieChart>
                  <Pie data={strategyData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={40} outerRadius={60}>
                    {strategyData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-2 text-xs">
                {strategyData.map((e, i) => (
                  <div key={i} className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: e.color }} />
                    <span className="text-[var(--text-secondary)]">{e.name}: {e.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Chart 3: Latency */}
          <div className="p-5 bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl space-y-4">
            <h3 className="font-semibold text-sm">Query Execution Latency</h3>
            <div className="h-[180px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={latencyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
                  <XAxis dataKey="metric" stroke="#525252" fontSize={10} />
                  <YAxis stroke="#525252" unit="ms" fontSize={10} />
                  <Tooltip contentStyle={{ backgroundColor: "#111", borderColor: "#262626" }} />
                  <Bar dataKey="latency" fill="#818cf8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
