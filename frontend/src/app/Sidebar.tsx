"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Sidebar() {
  const pathname = usePathname();
  
  // Extract analysisId if we are on an /analysis/[id]/* route
  const match = pathname.match(/\/analysis\/([^\/]+)/);
  const analysisId = match ? match[1] : null;

  return (
    <aside className="w-64 bg-slate-900 text-slate-300 flex flex-col shrink-0">
      <div className="h-16 flex items-center px-6 border-b border-slate-800">
        <h1 className="text-xl font-bold text-white tracking-wide flex items-center gap-2">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500">
            <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
            <polyline points="9 22 9 12 15 12 15 22"/>
          </svg>
          ResearchGap-X
        </h1>
      </div>
      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
        <Link 
          href="/" 
          className={`flex items-center gap-3 px-3 py-2 rounded-md font-medium transition-colors ${
            pathname === "/" ? "bg-blue-600 text-white" : "hover:bg-slate-800 hover:text-white"
          }`}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>
          New Analysis
        </Link>
        
        <div className="pt-4 pb-2">
          <p className="px-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">Analysis Results</p>
        </div>
        
        {analysisId ? (
          <>
            <Link 
              href={`/analysis/${analysisId}/gaps`} 
              className={`flex items-center gap-3 px-3 py-2 rounded-md font-medium transition-colors ${
                pathname.includes("/gaps") ? "bg-slate-800 text-white" : "hover:bg-slate-800 hover:text-white text-slate-400"
              }`}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" x2="22" y1="12" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
              Gap Explorer
            </Link>
            <Link 
              href={`/analysis/${analysisId}/literature`} 
              className={`flex items-center gap-3 px-3 py-2 rounded-md font-medium transition-colors ${
                pathname.includes("/literature") ? "bg-slate-800 text-white" : "hover:bg-slate-800 hover:text-white text-slate-400"
              }`}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
              Literature Explorer
            </Link>
            <Link 
              href={`/analysis/${analysisId}/analytics`} 
              className={`flex items-center gap-3 px-3 py-2 rounded-md font-medium transition-colors ${
                pathname.includes("/analytics") ? "bg-slate-800 text-white" : "hover:bg-slate-800 hover:text-white text-slate-400"
              }`}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><rect width="4" height="7" x="7" y="14" rx="1"/><rect width="4" height="12" x="15" y="9" rx="1"/></svg>
              Topic Analysis
            </Link>
            <a 
              href={`http://localhost:8000/api/analysis/${analysisId}/report/pdf`}
              target="_blank" rel="noreferrer"
              className="flex items-center gap-3 px-3 py-2 rounded-md font-medium text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
              Export Report
            </a>
          </>
        ) : (
          <div className="px-4 py-4 text-sm text-slate-500 italic text-center bg-slate-800/50 rounded-md mx-2">
            Upload a seed paper to view analysis features.
          </div>
        )}
      </nav>
      <div className="p-4 border-t border-slate-800 text-xs text-slate-500">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-green-500"></div>
          Backend Online
        </div>
        Version 0.1.0
      </div>
    </aside>
  );
}
