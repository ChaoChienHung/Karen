import { useState, useEffect } from "react";
import Navbar from "./components/Navbar";
import Home from "./components/Home";
import Sandbox from "./components/Sandbox";
import ModelHub from "./components/ModelHub";
import Leaderboard from "./components/Leaderboard";
import Articles from "./components/Articles";
import { Model } from "./types";
import modelsData from "./data/models.json";
import { AlertCircle, Loader2 } from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState<'home' | 'sandbox' | 'hub' | 'leaderboard' | 'articles'>('home');
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Set global language selection. Default is "en" (English) as requested.
  const [lang, setLang] = useState<'en' | 'zh'>('en');

  // Set global theme selection. Default is "dark" (Night mode).
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch("/api/models");
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data) && data.length > 0) {
            setModels(data);
          } else {
            setModels(modelsData as Model[]);
          }
        } else {
          // Fallback to static imported json data if API isn't ready
          setModels(modelsData as Model[]);
        }
      } catch (err) {
        console.warn("API model load failed, using local models fallback:", err);
        setModels(modelsData as Model[]);
      } finally {
        setLoading(false);
      }
    };
    fetchModels();
  }, []);

  // Theme-dependent top level styling classes
  const appBgClass = theme === 'dark' ? 'bg-[#050505] text-slate-200' : 'bg-[#f1f5f9] text-slate-900';
  const loaderTextClass = theme === 'dark' ? 'text-slate-400' : 'text-slate-700 font-medium';
  const borderClass = theme === 'dark' ? 'border-white/10' : 'border-slate-300';
  const footerBgClass = theme === 'dark' ? 'bg-black/40 border-white/10 text-slate-500' : 'bg-white border-slate-300 text-slate-700 shadow-inner';

  return (
    <div className={`flex flex-col min-h-screen ${appBgClass} font-sans selection:bg-indigo-600 selection:text-white antialiased transition-colors duration-300`}>
      
      {/* Universal Sticky Top Header with Lang & Theme states */}
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        lang={lang} 
        setLang={setLang} 
        theme={theme} 
        setTheme={setTheme} 
      />

      {/* Main Page Content Body with entry animation spacing */}
      <main className="flex-grow max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6 overflow-x-hidden w-full">
        
        {loading ? (
          <div className={`flex flex-col items-center justify-center min-h-[60vh] ${loaderTextClass} font-mono text-sm gap-2`}>
            <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
            <span>{lang === 'en' ? "Initializing sandbox models configuration..." : "載入模型配置中..."}</span>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center min-h-[60vh] max-w-md mx-auto text-center">
            <AlertCircle className="w-10 h-10 text-rose-500 mb-3" />
            <p className="text-sm font-semibold">{lang === 'en' ? "Configuration Error" : "配置加載錯誤"}</p>
            <p className="text-xs text-slate-500 mt-1 font-light">{error}</p>
          </div>
        ) : (
          <div className="animate-fade-in transition-opacity duration-300">
            {activeTab === 'home' && <Home setActiveTab={(tab) => setActiveTab(tab as any)} lang={lang} theme={theme} />}
            {activeTab === 'sandbox' && <Sandbox models={models} lang={lang} theme={theme} />}
            {activeTab === 'hub' && <ModelHub models={models} lang={lang} theme={theme} />}
            {activeTab === 'leaderboard' && <Leaderboard models={models} lang={lang} theme={theme} />}
            {activeTab === 'articles' && <Articles lang={lang} theme={theme} />}
          </div>
        )}

      </main>

      {/* Universal Footer */}
      <footer className={`border-t ${borderClass} ${footerBgClass} backdrop-blur-md py-6 text-center text-[10px] uppercase tracking-widest transition-colors duration-300`}>
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4 font-mono">
          <p>
            {lang === 'en' 
              ? "© 2026 Karen Platform • Multi-LLM Parallel Evaluation Playground • Data-Driven Open Platform" 
              : "© 2026 Karen Platform • 多 LLM 模型輸出比較平台 • Data-Driven Open Platform"}
          </p>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
            <span>API STATUS: OPERATIONAL</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
