import React, { useState } from "react";
import { Play, CheckCircle2, AlertTriangle, Clock } from "lucide-react";
import { AgentTraceStep, useStore } from "../../lib/store";

export const AgentTrace: React.FC = () => {
  const trace = useStore((state) => state.activeTrace);
  const activeAgent = useStore((state) => state.activeAgent);
  const [selectedNode, setSelectedNode] = useState<AgentTraceStep | null>(null);

  const getLatencyColor = (latency_ms?: number) => {
    if (!latency_ms) return "bg-neutral-800";
    if (latency_ms < 200) return "bg-green-500/20 text-green-400 border-green-500/30";
    if (latency_ms < 1000) return "bg-amber-500/20 text-amber-400 border-amber-500/30";
    return "bg-red-500/20 text-red-400 border-red-500/30";
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "thinking":
      case "active":
        return <Play className="w-4 h-4 text-indigo-400 animate-spin" />;
      case "complete":
        return <CheckCircle2 className="w-4 h-4 text-green-400" />;
      case "failed":
        return <AlertTriangle className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-neutral-400" />;
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden text-sm">
      <div className="p-4 border-b border-[var(--border)] font-semibold flex items-center justify-between">
        <span>Agent Decision Trace</span>
        {activeAgent && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 animate-pulse border border-indigo-500/25">
            Active: {activeAgent}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-none">
        {trace.length === 0 ? (
          <div className="text-[var(--text-muted)] text-center mt-10">
            No trace log. Submit a query to see real-time agent coordination.
          </div>
        ) : (
          trace.map((step, idx) => (
            <div
              key={idx}
              onClick={() => setSelectedNode(step)}
              className={`p-3 rounded-xl border border-[var(--border)] cursor-pointer transition-all hover:bg-[var(--bg-elevated)] ${
                selectedNode === step ? "bg-[var(--bg-elevated)] border-indigo-500/40" : "bg-[var(--bg-surface)]"
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center space-x-2 font-medium">
                  {getStatusIcon(step.status)}
                  <span className="text-[var(--text-primary)]">{step.agent}</span>
                </div>
                {step.latency_ms && (
                  <span className={`text-xs px-2 py-0.5 rounded border ${getLatencyColor(step.latency_ms)}`}>
                    {step.latency_ms}ms
                  </span>
                )}
              </div>
              <p className="text-xs text-[var(--text-secondary)] pl-6 leading-relaxed">
                {step.message}
              </p>
            </div>
          ))
        )}
      </div>

      {selectedNode && (
        <div className="p-4 border-t border-[var(--border)] bg-[var(--bg-surface)]">
          <div className="flex items-center justify-between mb-2">
            <span className="font-bold text-xs text-[var(--text-primary)] uppercase tracking-wider">Node Telemetry</span>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-xs text-[var(--text-muted)] hover:text-white cursor-pointer"
            >
              Close
            </button>
          </div>
          <div className="text-xs space-y-1.5 text-[var(--text-secondary)]">
            <div><strong className="text-[var(--text-muted)]">Agent:</strong> {selectedNode.agent}</div>
            <div><strong className="text-[var(--text-muted)]">Status:</strong> {selectedNode.status}</div>
            <div><strong className="text-[var(--text-muted)]">Timestamp:</strong> {new Date(selectedNode.timestamp * 1000).toLocaleTimeString()}</div>
            <div><strong className="text-[var(--text-muted)]">Telemetry Detail:</strong> {selectedNode.message}</div>
          </div>
        </div>
      )}
    </div>
  );
};
