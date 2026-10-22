"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { fetchAPI } from "@/lib/api";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.type !== "application/pdf") {
        setError("Please upload a valid PDF file.");
        return;
      }
      setFile(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setStatus("Uploading...");
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      // 1. Upload PDF
      const uploadData = await fetchAPI<{ analysis_id: string; message: string }>("/api/upload", {
        method: "POST",
        body: formData,
      });

      setAnalysisId(uploadData.analysis_id);
      setStatus("Initializing pipeline...");

      // 2. Start Analysis
      await fetchAPI("/api/analysis/start", {
        method: "POST",
        body: JSON.stringify({ analysis_id: uploadData.analysis_id, max_papers: 10 }),
      });
      
      setStatus("Processing");
    } catch (err: unknown) {
      setStatus(null);
      if (err instanceof Error) {
        setError(err.message || "An unexpected error occurred during upload.");
      } else {
        setError("An unexpected error occurred during upload.");
      }
    }
  };

  // Poll for status
  useEffect(() => {
    if (!analysisId || !status || status.startsWith("Upload") || status.startsWith("Init")) return;

    const interval = setInterval(async () => {
      try {
        const data = await fetchAPI<{ status: string; message?: string }>(`/api/analysis/${analysisId}/status`);
        setStatus(data.status);

        if (data.status === "COMPLETED") {
          clearInterval(interval);
          // Redirect to the analysis page (we'll build this route later)
          router.push(`/analysis/${analysisId}/gaps`);
        } else if (data.status === "FAILED") {
          clearInterval(interval);
          setError("Analysis failed: " + (data.message || "Unknown error"));
          setStatus(null);
        }
      } catch (err) {
        console.error("Polling error", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [analysisId, status, router]);

  return (
    <div className="min-h-full flex flex-col p-8 lg:p-12">
      <div className="max-w-4xl mx-auto w-full">
        <header className="mb-10">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">New Analysis Workspace</h1>
          <p className="text-slate-500">Upload a seed paper to discover research gaps, identify emerging trends, and explore latent themes across literature.</p>
        </header>

        {error && (
          <div className="mb-6 p-4 rounded-md bg-red-50 border border-red-200 flex items-start gap-3 text-red-700">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0 mt-0.5"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>
            <div>
              <h3 className="font-medium">Upload Error</h3>
              <p className="text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        {!analysisId ? (
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="p-6 sm:p-10 border-b border-slate-100 flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mb-4">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><path d="M12 18v-6"/><path d="m9 15 3-3 3 3"/></svg>
              </div>
              <h3 className="text-lg font-semibold text-slate-900 mb-1">Select Seed Document</h3>
              <p className="text-sm text-slate-500 max-w-sm mb-6">PDF files only. The system will extract the abstract and build a literature matrix around its core topics.</p>
              
              <input
                type="file"
                accept=".pdf"
                className="hidden"
                ref={fileInputRef}
                onChange={handleFileChange}
              />
              
              <div className="flex flex-col sm:flex-row gap-3">
                <button 
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-2 bg-white border border-slate-300 rounded-md text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors shadow-sm"
                >
                  Browse Files
                </button>
                <button 
                  onClick={handleUpload}
                  disabled={!file || status !== null}
                  className="px-6 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center justify-center gap-2"
                >
                  {status ? (
                    <>
                      <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                      {status}
                    </>
                  ) : "Initialize Pipeline"}
                </button>
              </div>

              {file && (
                <div className="mt-4 flex items-center gap-2 text-sm text-slate-600 bg-slate-50 py-1.5 px-3 rounded-full border border-slate-200">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-600"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                  {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </div>
              )}
            </div>
            <div className="bg-slate-50 p-6 text-sm text-slate-500 border-t border-slate-100">
              <h4 className="font-semibold text-slate-700 mb-2">How it works:</h4>
              <ul className="space-y-2 list-disc pl-5">
                <li>Extracts content via PyMuPDF and isolates standard sections.</li>
                <li>Retrieves top related metadata via OpenAlex and Semantic Scholar APIs.</li>
                <li>Constructs Document-Topic and temporal matrices for Latent SVD.</li>
                <li>Applies the multi-signal Gap Engine to hypothesize research directions.</li>
              </ul>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 sm:p-12 text-center flex flex-col items-center justify-center">
            <div className="relative mb-6">
              <div className="w-20 h-20 border-4 border-slate-100 border-t-blue-600 rounded-full animate-spin"></div>
              <div className="absolute inset-0 flex items-center justify-center text-blue-600 font-semibold text-xs">
                {status === "COMPLETED" ? "100%" : "RUN"}
              </div>
            </div>
            <h2 className="text-xl font-bold text-slate-900 mb-2">Analysis in Progress</h2>
            <p className="text-slate-500 mb-6 max-w-md">
              The pipeline is processing {file?.name}. This involves topic modeling, singular value decomposition, and multi-signal scoring.
            </p>
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 text-blue-700 text-sm font-medium rounded-full">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-blue-500"></span>
              </span>
              Current phase: {status}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
