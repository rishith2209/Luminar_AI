import React, { useState } from "react";
import { Info } from "lucide-react";

interface EvalPillProps {
  label: string;
  score: number;
  description: string;
  formula: string;
}

export const EvalPill: React.FC<EvalPillProps> = ({
  label,
  score,
  description,
  formula
}) => {
  const [modalOpen, setModalOpen] = useState(false);

  const getScoreColor = (val: number) => {
    if (val >= 0.85) return "text-green-400 bg-green-500/10 border-green-500/20";
    if (val >= 0.7) return "text-amber-400 bg-amber-500/10 border-amber-500/20";
    return "text-red-400 bg-red-500/10 border-red-500/20";
  };

  return (
    <>
      <button
        onClick={() => setModalOpen(true)}
        className={`px-3 py-1 text-xs font-semibold rounded-full border flex items-center space-x-1.5 cursor-pointer transition-all hover:scale-[1.03] ${getScoreColor(score)}`}
      >
        <span>{label}: {Math.round(score * 100) / 100}</span>
        <Info className="w-3.5 h-3.5 opacity-60" />
      </button>

      {modalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
          <div className="w-full max-w-md p-6 rounded-2xl glass border border-[var(--border)] shadow-2xl bg-[var(--bg-surface)]">
            <div className="flex items-center justify-between mb-4 border-b border-[var(--border)] pb-3">
              <h3 className="font-bold text-lg text-white">{label} Evaluation Metric</h3>
              <button
                onClick={() => setModalOpen(false)}
                className="text-sm text-[var(--text-muted)] hover:text-white cursor-pointer"
              >
                Close
              </button>
            </div>
            
            <div className="space-y-4 text-sm text-[var(--text-secondary)]">
              <div>
                <h4 className="font-semibold text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Concept Description</h4>
                <p className="leading-relaxed">{description}</p>
              </div>

              <div>
                <h4 className="font-semibold text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Scoring Methodology</h4>
                <div className="p-3 bg-neutral-900 border border-[var(--border)] rounded-lg font-mono text-xs text-indigo-300">
                  {formula}
                </div>
              </div>

              <div className="pt-2 flex items-center justify-between">
                <span>Current Score:</span>
                <span className="text-lg font-bold text-white">{Math.round(score * 100)}%</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
export default EvalPill;
