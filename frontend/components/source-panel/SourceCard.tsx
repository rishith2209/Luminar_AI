import React, { useState } from "react";
import { ExternalLink, Database, Network, Globe } from "lucide-react";
import { Citation, useStore } from "../../lib/store";

interface SourceCardProps {
  citation: Citation;
  score?: number;
  graphSource?: boolean;
  liveWeb?: boolean;
}

export const SourceCard: React.FC<SourceCardProps> = ({
  citation,
  score = 0.85,
  graphSource = false,
  liveWeb = false
}) => {
  const [expanded, setExpanded] = useState(false);
  const selectedCitation = useStore((state) => state.selectedCitation);

  const getScoreColor = (s: number) => {
    if (s > 0.85) return "text-green-400 bg-green-500/10 border-green-500/20";
    if (s > 0.6) return "text-amber-400 bg-amber-500/10 border-amber-500/20";
    return "text-red-400 bg-red-500/10 border-red-500/20";
  };

  const isSelected = selectedCitation?.index === citation.index;

  return (
    <div
      className={`p-4 rounded-xl border transition-all duration-300 ${
        isSelected
          ? "bg-[var(--bg-elevated)] border-indigo-500/50 shadow-indigo-500/5 shadow-md"
          : "bg-[var(--bg-surface)] border-[var(--border)]"
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          {graphSource ? (
            <Network className="w-4 h-4 text-purple-400" />
          ) : liveWeb ? (
            <Globe className="w-4 h-4 text-sky-400" />
          ) : (
            <Database className="w-4 h-4 text-indigo-400" />
          )}
          <span className="font-semibold text-xs text-[var(--text-primary)] max-w-[180px] truncate">
            {citation.title || "Context Chunk"}
          </span>
        </div>
        
        <div className="flex items-center space-x-1.5">
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${getScoreColor(score)}`}>
            Relevance: {Math.round(score * 100)}%
          </span>
        </div>
      </div>

      <p className={`text-xs text-[var(--text-secondary)] leading-relaxed mb-3 ${expanded ? "" : "line-clamp-3"}`}>
        {citation.snippet}
      </p>

      <div className="flex items-center justify-between pt-2 border-t border-[var(--border)] text-xs text-[var(--text-muted)]">
        <div className="flex items-center space-x-3">
          <span>Source [{citation.index}]</span>
          {citation.source_url && (
            <a
              href={citation.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-[var(--accent)] flex items-center space-x-1"
            >
              <span>View url</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          )}
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-indigo-400 hover:text-indigo-300 cursor-pointer font-medium"
        >
          {expanded ? "Collapse" : "View Full Chunk"}
        </button>
      </div>
    </div>
  );
};
export default SourceCard;
