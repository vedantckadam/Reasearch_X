"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { fetchAPI } from "@/lib/api";

type Paper = {
  id: string;
  title: string;
  abstract: string;
  year: number;
  authors: string[];
  url: string | null;
  doi: string | null;
  sources: string[];
  relevance_score: number | null;
};

type LiteratureResponse = {
  papers: Paper[];
};

export default function LiteraturePage() {
  const params = useParams();
  const analysisId = params.analysisId as string;
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const result = await fetchAPI<LiteratureResponse>(`/api/analysis/${analysisId}/papers`);
        setPapers(result.papers || []);
      } catch (err: unknown) {
        if (err instanceof Error) {
          setError(err.message || "Failed to load literature");
        } else {
          setError("Failed to load literature");
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
        Loading Literature Matrix...
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

  return (
    <div className="p-8 lg:p-12 max-w-7xl mx-auto">
      <header className="mb-8 flex items-end justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Literature Explorer</h1>
          <p className="text-slate-500">Retrieved and deduplicated documents supporting this analysis.</p>
        </div>
        <div className="text-sm text-slate-500 bg-white border border-slate-200 px-4 py-2 rounded-lg shadow-sm">
          Total Documents: <span className="font-bold text-slate-900">{papers.length}</span>
        </div>
      </header>

      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-600">
            <thead className="text-xs text-slate-500 uppercase bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="px-6 py-4 font-medium">Document Title & Authors</th>
                <th className="px-6 py-4 font-medium w-24">Year</th>
                <th className="px-6 py-4 font-medium w-48">Sources</th>
                <th className="px-6 py-4 font-medium w-24">Rel.</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {papers.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-6 py-12 text-center text-slate-500">
                    No papers retrieved.
                  </td>
                </tr>
              ) : (
                papers.map((paper) => (
                  <tr key={paper.id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-6 py-4">
                      <div className="font-semibold text-slate-900 mb-1">{paper.title}</div>
                      {paper.authors && paper.authors.length > 0 && (
                        <div className="text-xs text-slate-500 truncate max-w-xl">
                          {paper.authors.join(", ")}
                        </div>
                      )}
                      <div className="flex gap-3 mt-2 text-xs">
                        {paper.doi && (
                          <a href={`https://doi.org/${paper.doi}`} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                            DOI: {paper.doi}
                          </a>
                        )}
                        {paper.url && !paper.doi && (
                          <a href={paper.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                            View Source
                          </a>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 font-medium">{paper.year}</td>
                    <td className="px-6 py-4">
                      <div className="flex flex-wrap gap-1">
                        {paper.sources && paper.sources.map(src => (
                          <span key={src} className="px-2 py-0.5 bg-slate-100 border border-slate-200 rounded text-[10px] uppercase font-bold text-slate-600 tracking-wider">
                            {src}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      {paper.relevance_score ? (paper.relevance_score * 100).toFixed(0) + "%" : "N/A"}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
