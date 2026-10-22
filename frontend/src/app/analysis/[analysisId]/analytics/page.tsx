"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { fetchAPI } from "@/lib/api";

type Topic = {
  topic_idx: number;
  label: string;
  top_words: ([string, number] | string)[];
  corpus_share: number;
};

type TopicResponse = {
  topics: Topic[];
};

export default function AnalyticsPage() {
  const params = useParams();
  const analysisId = params.analysisId as string;
  const [data, setData] = useState<TopicResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const result = await fetchAPI<TopicResponse>(`/api/analysis/${analysisId}/topics`);
        setData(result);
      } catch (err: unknown) {
        if (err instanceof Error) {
          setError(err.message || "Failed to load topic analytics");
        } else {
          setError("Failed to load topic analytics");
        }
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [analysisId]);

  if (loading) {
    return (
      <div className="p-8 flex justify-center items-center h-full text-slate-500">
        <svg className="animate-spin h-8 w-8 text-blue-500 mr-3" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
        Loading Topic Analytics...
      </div>
    );
  }

  if (error) {
    if (error.includes("Topic analysis not found") || error.includes("404")) {
      return (
        <div className="p-12 flex flex-col items-center justify-center text-center">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="text-slate-300 mb-4"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>
          <h2 className="text-xl font-medium text-slate-900 mb-2">Analysis Incomplete</h2>
          <p className="text-slate-500 max-w-sm mb-4">Topic analysis was not found for this analysis. The pipeline might have failed or is still running.</p>
          <button 
            onClick={() => window.location.reload()} 
            className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 transition-colors"
          >
            Refresh
          </button>
        </div>
      );
    }
    return (
      <div className="p-8">
        <div className="bg-red-50 text-red-700 p-4 rounded-md border border-red-200 flex flex-col gap-2">
          <strong>Error Loading Topics</strong>
          <span>{error}</span>
        </div>
      </div>
    );
  }

  if (!data || !data.topics || data.topics.length === 0) {
    return (
      <div className="p-12 flex flex-col items-center justify-center text-center">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="text-slate-300 mb-4"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>
        <h2 className="text-xl font-medium text-slate-900 mb-2">No Topics Extracted</h2>
        <p className="text-slate-500 max-w-sm">The LDA modeling did not produce any valid topics for this corpus.</p>
      </div>
    );
  }

  return (
    <div className="p-8 lg:p-12 max-w-7xl mx-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Topic Analysis</h1>
        <p className="text-slate-500">Latent Dirichlet Allocation (LDA) models identifying core conceptual domains across the literature.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {data.topics.map(topic => (
          <div key={topic.topic_idx} className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden flex flex-col">
            <div className="p-5 border-b border-slate-100 bg-slate-50 flex justify-between items-start">
              <div>
                <div className="text-xs font-bold text-slate-500 tracking-wider mb-1">TOPIC {topic.topic_idx}</div>
                <h3 className="text-lg font-bold text-slate-900 capitalize">{topic.label}</h3>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-blue-600">{(topic.corpus_share * 100).toFixed(1)}%</div>
                <div className="text-xs font-medium text-slate-500">CORPUS SHARE</div>
              </div>
            </div>
            <div className="p-5 flex-1">
              <h4 className="text-xs font-semibold uppercase text-slate-500 tracking-wider mb-3">Dominant Terms</h4>
              <div className="space-y-3">
                {(!topic.top_words || !Array.isArray(topic.top_words) || topic.top_words.length === 0) ? (
                  <div className="text-sm text-slate-500 italic">No terms available for this topic.</div>
                ) : (
                  topic.top_words.slice(0, 5).map((term, i) => {
                    if (!term) return null;
                    let termName = "Unknown";
                    let weight = 0;
                    if (Array.isArray(term)) {
                      termName = typeof term[0] === 'string' ? term[0] : String(term[0] ?? '');
                      weight = typeof term[1] === 'number' ? term[1] : 0;
                    } else if (typeof term === 'string') {
                      termName = term;
                      weight = 1.0;
                    }
                    if (!termName || termName === "Unknown") return null;

                    const firstTerm = topic.top_words[0];
                    const rawFirstWeight = Array.isArray(firstTerm) && typeof firstTerm[1] === 'number'
                      ? firstTerm[1]
                      : 1;
                    const maxWeight = Math.max(0.0001, rawFirstWeight);
                    const percent = Math.min(100, Math.max(0, (weight / maxWeight) * 100));
                    return (
                      <div key={i}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="font-medium text-slate-700">{termName}</span>
                          <span className="text-slate-400 font-mono text-xs">{weight.toFixed(4)}</span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-sm h-1.5">
                          <div className="bg-blue-400 h-1.5 rounded-sm" style={{ width: `${percent}%` }}></div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
