import React from "react";
import { Citation, useStore } from "../../lib/store";

interface StreamingMessageProps {
  content: string;
  isStreaming: boolean;
  citations?: Citation[];
  role: "user" | "assistant";
}

export const StreamingMessage: React.FC<StreamingMessageProps> = ({
  content,
  isStreaming,
  citations = [],
  role
}) => {
  const setSelectedCitation = useStore((state) => state.setSelectedCitation);
  const setActiveTab = useStore((state) => state.setActiveTab);

  // Parse inline citations like [1] or [2] and render them as interactive links
  const renderFormattedContent = (text: string) => {
    if (role === "user") {
      return <p className="whitespace-pre-wrap">{text}</p>;
    }

    // Split text by brackets like [1], [2]
    const regex = /(\[\d+\])/g;
    const parts = text.split(regex);

    return (
      <div className="space-y-3 leading-relaxed text-[var(--text-primary)]">
        <p className="whitespace-pre-wrap">
          {parts.map((part, idx) => {
            if (regex.test(part)) {
              const num = parseInt(part.replace(/[\[\]]/g, ""), 10);
              const cit = citations.find((c) => c.index === num);
              
              return (
                <button
                  key={idx}
                  onClick={() => {
                    if (cit) {
                      setSelectedCitation(cit);
                      setActiveTab("sources");
                    }
                  }}
                  className="mx-0.5 px-1.5 py-0.5 text-xs font-semibold rounded bg-indigo-950 text-indigo-300 border border-indigo-800 hover:bg-indigo-900 cursor-pointer transition-colors"
                >
                  {num}
                </button>
              );
            }
            return part;
          })}
        </p>
      </div>
    );
  };

  return (
    <div
      className={`flex w-full ${
        role === "user" ? "justify-end" : "justify-start"
      } mb-4`}
    >
      <div
        className={`max-w-[75%] px-4 py-3 rounded-2xl border transition-all duration-300 ${
          role === "user"
            ? "bg-indigo-600/10 border-indigo-500/20 text-white rounded-br-none"
            : "bg-[var(--bg-surface)] border-[var(--border)] rounded-bl-none shadow-xl"
        }`}
      >
        <div className="text-xs text-[var(--text-muted)] mb-1 font-medium tracking-wider uppercase">
          {role === "user" ? "You" : "Research Intelligence Engine"}
        </div>
        {renderFormattedContent(content)}
        {isStreaming && (
          <span className="inline-block w-1.5 h-4 ml-1 bg-indigo-400 animate-pulse" />
        )}
      </div>
    </div>
  );
};
