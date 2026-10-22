"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import { fetchAPI } from "@/lib/api";

type Evidence = {
  evidence_type: string;
  evidence_text?: string;
  year?: number;
  evidence_metadata?: Record<string, unknown>;
};

export type GapCandidate = {
  candidate_id: string;
  topic_labels: string[];
  rarity_score: number | null;
  intersection_score: number | null;
  temporal_score: number | null;
  limitation_score: number | null;
  semantic_support_score: number | null;
  final_gap_score: number;
  evidence_count: number;
  confidence_level: string;
  reasoning_signals: string[];
  evidence_references: Evidence[];
};

export type SourceReference = {
  paper_id: string;
  title: string;
  year: number;
  chunk_id?: string | null;
  evidence_field: string;
  similarity_score?: number | null;
};

export type GroundedExplanation = {
  candidate_id: string;
  candidate_gap_statement: string;
  evidence_summary: string;
  literature_context: string;
  limitation_evidence: string;
  temporal_context: string;
  counterevidence: string;
  evidence_limitations: string;
  suggested_direction: string;
  source_references: SourceReference[];
  grounding_status: string;
};

function GapExplanationSection({
  analysisId,
  candidateId,
}: {
  analysisId: string;
  candidateId: string;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [explanation, setExplanation] = useState<GroundedExplanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasAttemptedFetch, setHasAttemptedFetch] = useState(false);

  const fetchExistingExplanation = useCallback(async () => {
    if (hasAttemptedFetch) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAPI<GroundedExplanation>(
        `/api/analysis/${analysisId}/gaps/${candidateId}/explanation`
      );
      if (data && data.candidate_gap_statement) {
        setExplanation(data);
      }
    } catch {
      // 404 is normal if an explanation has not yet been generated
    } finally {
      setLoading(false);
      setHasAttemptedFetch(true);
    }
  }, [analysisId, candidateId, hasAttemptedFetch]);

  const handleToggle = () => {
    const nextState = !isOpen;
    setIsOpen(nextState);
    if (nextState && !hasAttemptedFetch) {
      fetchExistingExplanation();
    }
  };

  const handleGenerate = async () => {
    setGenerating(true);
    setError(null);
    try {
      const data = await fetchAPI<GroundedExplanation>(
        `/api/analysis/${analysisId}/gaps/${candidateId}/explain`,
        { method: "POST" }
      );
      setExplanation(data);
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
        setError(
          "AI Service Unconfigured: Set OPENROUTER_API_KEY in backend/.env to generate Nemotron explanations."
        );
      } else if (rawMsg.includes("504") || detailMsg.includes("timed out") || detailMsg.includes("timeout")) {
        setError(
          "The AI provider timed out during evidence synthesis. Please retry in a moment."
        );
      } else if (detailMsg.includes("authentication failed")) {
        setError(
          "AI Provider Error: Authentication failed. Please verify your OPENROUTER_API_KEY in backend/.env."
        );
      } else {
        setError(detailMsg || "Failed to generate grounded explanation.");
      }
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="border-t border-slate-100 bg-slate-50/50 p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-100 text-indigo-700">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
            </svg>
          </div>
          <div>
            <h4 className="text-sm font-bold text-slate-900">
              AI Grounded Explanation
            </h4>
            <p className="text-xs text-slate-500">
              Evidence-grounded synthesis using OpenRouter NVIDIA Nemotron
            </p>
          </div>
        </div>

        <button
          onClick={handleToggle}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50"
        >
          {explanation ? "View Explanation" : isOpen ? "Collapse" : "Open Explanation"}
          <svg
            className={`h-4 w-4 transition-transform ${isOpen ? "rotate-180" : ""}`}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {isOpen && (
        <div className="mt-4 pt-4 border-t border-slate-200">
          {loading && (
            <div className="flex items-center gap-3 py-4 text-xs text-slate-500">
              <svg className="h-4 w-4 animate-spin text-indigo-600" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
              </svg>
              Checking for existing synthesized explanation...
            </div>
          )}

          {error && (
            <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-3.5 text-xs text-amber-800">
              <div className="flex items-start gap-2.5">
                <svg className="h-4 w-4 mt-0.5 shrink-0 text-amber-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                </svg>
                <div className="flex-1">
                  <p className="font-semibold text-amber-900">{error}</p>
                  {(error.includes("Unconfigured") || error.includes("OPENROUTER_API_KEY")) && (
                    <div className="mt-2.5 space-y-1 text-[11px] text-amber-800/90 border-t border-amber-200/60 pt-2">
                      <p><strong>To resolve:</strong></p>
                      <p>1. Open <code className="bg-amber-100/80 px-1 py-0.5 rounded font-mono">backend/.env</code> in your editor.</p>
                      <p>2. Set your actual key: <code className="bg-amber-100/80 px-1 py-0.5 rounded font-mono">OPENROUTER_API_KEY=sk-or-v1-...</code></p>
                      <p>3. Click <strong>Generate Grounded Explanation</strong> below (the backend detects file changes on-the-fly).</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {!explanation && !loading && (
            <div className="rounded-lg border border-dashed border-slate-200 bg-white p-5 text-center">
              <p className="text-xs text-slate-600 mb-3">
                No grounded explanation has been synthesized for this candidate gap yet.
              </p>
              <button
                onClick={handleGenerate}
                disabled={generating}
                className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow transition hover:bg-indigo-700 disabled:opacity-50"
              >
                {generating ? (
                  <>
                    <svg className="h-3.5 w-3.5 animate-spin text-white" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                    </svg>
                    Synthesizing with Nemotron (30-60s)...
                  </>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
                    </svg>
                    Generate Grounded Explanation
                  </>
                )}
              </button>
            </div>
          )}

          {explanation && (
            <div className="space-y-4">
              <div className="rounded-lg border border-indigo-100 bg-indigo-50/50 p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-700">
                    Candidate Gap Statement
                  </span>
                  <span className="rounded bg-indigo-100 px-2 py-0.5 text-[10px] font-medium text-indigo-800">
                    Grounded &bull; {explanation.grounding_status}
                  </span>
                </div>
                <p className="text-sm font-semibold text-slate-900 leading-relaxed">
                  {explanation.candidate_gap_statement}
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                {explanation.evidence_summary && (
                  <div className="rounded-lg border border-slate-200 bg-white p-3">
                    <h5 className="font-bold text-slate-800 mb-1">Evidence Summary</h5>
                    <p className="text-slate-600 leading-normal">{explanation.evidence_summary}</p>
                  </div>
                )}
                {explanation.literature_context && (
                  <div className="rounded-lg border border-slate-200 bg-white p-3">
                    <h5 className="font-bold text-slate-800 mb-1">Literature Context</h5>
                    <p className="text-slate-600 leading-normal">{explanation.literature_context}</p>
                  </div>
                )}
                {explanation.limitation_evidence && (
                  <div className="rounded-lg border border-slate-200 bg-white p-3">
                    <h5 className="font-bold text-slate-800 mb-1">Stated Author Limitations</h5>
                    <p className="text-slate-600 leading-normal">{explanation.limitation_evidence}</p>
                  </div>
                )}
                {explanation.temporal_context && (
                  <div className="rounded-lg border border-slate-200 bg-white p-3">
                    <h5 className="font-bold text-slate-800 mb-1">Temporal Trajectory</h5>
                    <p className="text-slate-600 leading-normal">{explanation.temporal_context}</p>
                  </div>
                )}
                {explanation.counterevidence && (
                  <div className="rounded-lg border border-slate-200 bg-white p-3">
                    <h5 className="font-bold text-slate-800 mb-1">Counterevidence / Coverage</h5>
                    <p className="text-slate-600 leading-normal">{explanation.counterevidence}</p>
                  </div>
                )}
                {explanation.evidence_limitations && (
                  <div className="rounded-lg border border-slate-200 bg-white p-3">
                    <h5 className="font-bold text-slate-800 mb-1">Evidence Limitations</h5>
                    <p className="text-slate-600 leading-normal">{explanation.evidence_limitations}</p>
                  </div>
                )}
              </div>

              {explanation.suggested_direction && (
                <div className="rounded-lg border border-blue-200 bg-blue-50/70 p-3 text-xs">
                  <h5 className="font-bold text-blue-900 mb-1">Suggested Direction</h5>
                  <p className="text-blue-950 leading-normal mb-2">{explanation.suggested_direction}</p>
                  <p className="text-[11px] font-medium text-blue-700 italic border-t border-blue-200/60 pt-1.5">
                    Notice: This is a potential research-gap signal, not proof of a research gap.
                  </p>
                </div>
              )}

              {explanation.source_references && explanation.source_references.length > 0 && (
                <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs">
                  <h5 className="font-bold text-slate-800 mb-2">
                    Cited Sources ({explanation.source_references.length})
                  </h5>
                  <div className="space-y-1.5">
                    {explanation.source_references.map((src, i) => (
                      <div key={i} className="flex items-start justify-between gap-2 border-b border-slate-100 pb-1.5 last:border-0 last:pb-0">
                        <span className="font-medium text-slate-800">
                          {src.title || src.paper_id}
                        </span>
                        <span className="shrink-0 text-slate-400">
                          {src.year ? src.year : ""}{src.evidence_field ? ` &bull; ${src.evidence_field}` : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex justify-end pt-2">
                <button
                  onClick={handleGenerate}
                  disabled={generating}
                  className="text-[11px] font-semibold text-indigo-600 hover:text-indigo-800 transition disabled:opacity-50"
                >
                  {generating ? "Re-synthesizing..." : "Regenerate explanation &rarr;"}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function GapsPage() {
  const params = useParams();
  const analysisId = params.analysisId as string;
  const [candidates, setCandidates] = useState<GapCandidate[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const result = await fetchAPI<unknown>(`/api/analysis/${analysisId}/gaps`);
        
        // Defensive validation: handle both direct array and { candidates: [...] } response shape
        let cands: GapCandidate[] = [];
        if (Array.isArray(result)) {
          cands = result as GapCandidate[];
        } else if (result && typeof result === "object" && Array.isArray((result as { candidates?: unknown }).candidates)) {
          cands = (result as { candidates: GapCandidate[] }).candidates;
        } else {
          throw new Error("Invalid API response format: expected an array of candidates");
        }
        
        setCandidates(cands);
      } catch (err: unknown) {
        if (err instanceof Error) {
          setError(err.message || "Failed to load gaps data");
        } else {
          setError("Failed to load gaps data");
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
        Loading Gap Candidates...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
        <div className="bg-red-50 text-red-700 p-4 rounded-md border border-red-200">
          Error: {error}
        </div>
      </div>
    );
  }

  // Handle a valid empty candidate list with a useful empty state
  if (candidates && candidates.length === 0) {
    return (
      <div className="p-12 flex flex-col items-center justify-center text-center">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="text-slate-300 mb-4"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>
        <h2 className="text-xl font-medium text-slate-900 mb-2">No Gaps Identified</h2>
        <p className="text-slate-500 max-w-sm">The engine could not hypothesize any candidate gaps with the current seed data.</p>
      </div>
    );
  }

  // Fallback if candidates is somehow null but no error
  if (!candidates) {
    return null;
  }

  return (
    <div className="p-8 lg:p-12 max-w-6xl mx-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Research Gap Explorer</h1>
        <p className="text-slate-500">Hypothesized research directions mathematically prioritized by structural holes and temporal emergence.</p>
      </header>

      <div className="space-y-6">
        {candidates.map((cand, idx) => (
          <div key={cand.candidate_id || idx} className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="p-6 border-b border-slate-100 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <span className="bg-slate-100 text-slate-600 px-2.5 py-1 rounded text-xs font-bold tracking-wider">RANK #{idx + 1}</span>
                  {cand.topic_labels && cand.topic_labels.length > 0 ? (
                    <h2 className="text-lg font-bold text-slate-900">{cand.topic_labels.join(" × ")}</h2>
                  ) : (
                    <h2 className="text-lg font-bold text-slate-900">Intersection Candidate</h2>
                  )}
                </div>
                <div className="flex flex-wrap gap-2 mt-3">
                  {cand.reasoning_signals && cand.reasoning_signals.map(sig => (
                    <span key={sig} className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs font-medium border border-blue-100">
                      {sig}
                    </span>
                  ))}
                </div>
              </div>
              
              <div className="flex items-center gap-6 bg-slate-50 p-4 rounded-lg border border-slate-100">
                <div className="text-center">
                  <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">Score</p>
                  <p className="text-2xl font-bold text-slate-900">{((cand.final_gap_score || 0) * 100).toFixed(0)}<span className="text-lg text-slate-500">%</span></p>
                </div>
                <div className="w-px h-10 bg-slate-200"></div>
                <div className="text-center">
                  <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">Confidence</p>
                  <p className={`text-sm font-bold ${
                    cand.confidence_level === 'High' ? 'text-green-600' : 
                    cand.confidence_level === 'Medium' ? 'text-yellow-600' : 'text-slate-600'
                  }`}>{cand.confidence_level || 'Low'}</p>
                </div>
              </div>
            </div>

            <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-4 flex items-center gap-2">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500"><path d="M12 2v20"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                  Component Signals
                </h3>
                <div className="space-y-4">
                  {[
                    { label: "Rarity (R)", value: cand.rarity_score },
                    { label: "Intersection Sparsity (I)", value: cand.intersection_score },
                    { label: "Temporal (T)", value: cand.temporal_score },
                    { label: "Limitation Extr. (L)", value: cand.limitation_score },
                    { label: "Semantic Support (S)", value: cand.semantic_support_score },
                  ].map(sig => (
                    <div key={sig.label}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-600">{sig.label}</span>
                        <span className={(sig.value === null || sig.value === undefined) ? "text-slate-400 italic" : "font-medium text-slate-900"}>
                          {(sig.value === null || sig.value === undefined) ? "N/A" : (sig.value * 100).toFixed(0) + "%"}
                        </span>
                      </div>
                      <div className="w-full bg-slate-100 rounded-full h-1.5">
                        <div 
                          className={`h-1.5 rounded-full ${(sig.value === null || sig.value === undefined) ? 'bg-transparent' : 'bg-blue-500'}`} 
                          style={{ width: `${(sig.value === null || sig.value === undefined) ? 0 : sig.value * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-4 flex items-center gap-2">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                  Supporting Evidence ({(cand.evidence_references && cand.evidence_references.length) || cand.evidence_count || 0})
                </h3>
                <div className="space-y-3 max-h-[250px] overflow-y-auto pr-2 custom-scrollbar">
                  {cand.evidence_references && cand.evidence_references.slice(0, 3).map((ev, i) => (
                    <div key={i} className="bg-slate-50 p-3 rounded border border-slate-100 text-sm">
                      <div className="font-semibold text-slate-800 mb-1 capitalize text-xs">{ev.evidence_type ? ev.evidence_type.replace('_', ' ') : 'Evidence'}</div>
                      {ev.evidence_text ? (
                        <p className="text-slate-600 italic">&quot;{ev.evidence_text}&quot;</p>
                      ) : (
                        <p className="text-slate-600 text-xs text-slate-400">Statistical pattern observed in latent matrix.</p>
                      )}
                      {ev.evidence_metadata && typeof ev.evidence_metadata.matched_phrase === "string" && (
                        <div className="mt-2 text-xs text-blue-600 bg-blue-50 inline-block px-2 py-0.5 rounded">
                          Keyword: {ev.evidence_metadata.matched_phrase}
                        </div>
                      )}
                    </div>
                  ))}
                  {cand.evidence_references && cand.evidence_references.length > 3 && (
                    <div className="text-center py-2 text-sm text-blue-600 hover:text-blue-700 font-medium cursor-pointer">
                      + View {cand.evidence_references.length - 3} more evidence items
                    </div>
                  )}
                  {(!cand.evidence_references || cand.evidence_references.length === 0) && (
                    <div className="text-sm text-slate-500 italic">
                      No explicit evidence snippets available.
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* AI Grounded Explanation Panel */}
            <GapExplanationSection
              analysisId={analysisId}
              candidateId={cand.candidate_id}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
