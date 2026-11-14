import { Terminal } from "lucide-react";

interface HomeProps {
  setActiveTab: (tab: 'home' | 'sandbox' | 'hub' | 'leaderboard' | 'articles') => void;
  lang: 'en' | 'zh';
  theme: 'dark' | 'light';
}

export default function Home({ setActiveTab, lang, theme }: HomeProps) {
  const isDark = theme === 'dark';

  // Dynamic text based on current language
  const description = lang === 'en' 
    ? "A high-performance parallel comparison playground for real-time multi-LLM outputs evaluation, unified prompt testing, and objective benchmark validation."
    : "專業級多模型平行對比沙盒 ── 支援主流生成式 AI 即時輸出對比、提示詞工程工程調優、本機離線連線及客觀學術指標評測。";

  // Class styling configurations based on active theme
  const containerStyle = isDark ? "text-slate-200" : "text-slate-800";
  const descColor = isDark ? "text-slate-400" : "text-slate-600";
  const bgBlurColor = isDark ? "bg-indigo-500/5" : "bg-indigo-500/10";

  return (
    <div className="relative overflow-hidden py-16 sm:py-28 select-none flex flex-col items-center justify-center text-center">
      
      {/* Dynamic Background Blur */}
      <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] ${bgBlurColor} rounded-full blur-3xl pointer-events-none transition-colors duration-300`} />

      <div className="max-w-4xl mx-auto px-4 relative z-10 space-y-10">
        
        {/* Masterfully Styled "KAREN" Header */}
        <div className="relative group select-none">
          {/* Subtle back-glow */}
          <span className="absolute -inset-1 rounded-lg bg-gradient-to-r from-indigo-500 via-purple-600 to-pink-500 opacity-20 blur-xl group-hover:opacity-45 transition-opacity duration-700 pointer-events-none" />
          
          <h1 className="relative font-sans text-8xl sm:text-9xl md:text-[11rem] font-black tracking-widest uppercase transition-transform duration-500 group-hover:scale-[1.01] select-none">
            <span className={`bg-gradient-to-br bg-clip-text text-transparent font-extrabold tracking-widest ${
              isDark 
                ? "from-white via-indigo-200 to-indigo-400" 
                : "from-indigo-900 via-indigo-850 to-indigo-800"
            }`}>
              K
            </span>
            <span className={`bg-gradient-to-br bg-clip-text text-transparent ${
              isDark 
                ? "from-indigo-400 via-purple-400 to-pink-400" 
                : "from-indigo-700 via-violet-700 to-purple-800"
            }`}>
              AREN
            </span>
          </h1>
        </div>
        
        {/* Subtitle / Description Section styled with elegant serif italic format */}
        <div className="max-w-2xl mx-auto">
          <p className={`text-base sm:text-xl leading-relaxed tracking-wide font-serif italic ${descColor} transition-colors duration-300`}>
            “{description}”
          </p>
        </div>

        {/* Call to Action Section */}
        <div className="flex flex-col items-center gap-6 pt-4">
          
          {/* Primary CTA Button (without terminal icon as requested) with enlarged font size */}
          <button
            onClick={() => setActiveTab('sandbox')}
            className="inline-flex items-center justify-center px-12 py-5 sm:py-5.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm sm:text-lg font-bold shadow-lg shadow-indigo-600/20 transition-all hover:scale-[1.03] active:scale-[0.98] cursor-pointer"
            id="home-open-sandbox-btn"
          >
            <span>{lang === 'en' ? "Open Sandbox Testing" : "開始沙盒對比"}</span>
          </button>

        </div>

      </div>
    </div>
  );
}
