import { useState, useRef, useEffect } from "react";
import { BarChart2, Library, Terminal, Home as HomeIcon, Globe, Sun, Moon, ChevronDown } from "lucide-react";

interface NavbarProps {
  activeTab: 'home' | 'sandbox' | 'hub' | 'leaderboard' | 'articles';
  setActiveTab: (tab: 'home' | 'sandbox' | 'hub' | 'leaderboard' | 'articles') => void;
  lang: 'en' | 'zh';
  setLang: (lang: 'en' | 'zh') => void;
  theme: 'dark' | 'light';
  setTheme: (theme: 'dark' | 'light') => void;
}

export default function Navbar({ activeTab, setActiveTab, lang, setLang, theme, setTheme }: NavbarProps) {
  const [langDropdownOpen, setLangDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const navItems = [
    { id: 'home', label: lang === 'en' ? 'Home' : '首頁', icon: HomeIcon },
    { id: 'sandbox', label: lang === 'en' ? 'Sandbox' : '沙盒測試', icon: Terminal },
    { id: 'hub', label: lang === 'en' ? 'Model Hub' : '模型庫', icon: Library },
    { id: 'leaderboard', label: lang === 'en' ? 'Leaderboard' : '評測報告', icon: BarChart2 },
    { id: 'articles', label: lang === 'en' ? 'Articles' : '專題文章', icon: Library },
  ] as const;

  // Handle click outside to close language dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setLangDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Theme styling definitions
  const isDark = theme === 'dark';
  const headerBg = isDark ? "bg-[#050505]/95 border-white/10" : "bg-white/95 border-slate-300 shadow-sm shadow-slate-900/5";
  const brandText = isDark ? "text-white" : "text-slate-900 font-extrabold";
  const btnBorder = isDark ? "border-white/10 bg-white/5 text-slate-300 hover:text-white hover:bg-white/10" : "border-slate-300 bg-white text-slate-800 hover:text-slate-950 hover:bg-slate-100 shadow-sm";
  const dropdownBg = isDark ? "bg-[#0d0d0d] border-white/10 text-slate-300" : "bg-white border-slate-300 text-slate-800 shadow-xl";
  const dropdownItemHover = isDark ? "hover:bg-white/5 hover:text-white" : "hover:bg-slate-100 hover:text-slate-950";

  return (
    <header className={`sticky top-0 z-50 w-full border-b backdrop-blur-md transition-colors duration-300 ${headerBg}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          
          {/* Brand Logo: Clean styled large 'K', no subtitle playground */}
          <div 
            onClick={() => setActiveTab('home')}
            className="flex items-center gap-2 cursor-pointer group select-none"
            id="brand-logo"
          >
            <div className="flex items-center justify-center transition-transform group-hover:scale-105">
              <span className={`font-sans font-black text-3xl tracking-tighter bg-gradient-to-r bg-clip-text text-transparent ${
                isDark 
                  ? "from-indigo-400 to-purple-400" 
                  : "from-indigo-700 to-violet-800 font-black"
              }`}>
                K
              </span>
            </div>
            <div className="flex flex-col items-start leading-none ml-1">
              <span className={`font-sans text-sm font-bold tracking-tight uppercase ${brandText}`}>
                Karen
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex space-x-0.5 sm:space-x-1.5">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              
              // Active / Inactive states for Light & Dark themes
              const tabStyle = isActive
                ? (isDark 
                    ? "text-indigo-400 bg-indigo-500/5 border-indigo-500/15 shadow-inner" 
                    : "text-indigo-600 bg-indigo-50 border-indigo-200/50 shadow-sm")
                : (isDark 
                    ? "text-slate-400 hover:text-white hover:bg-white/5 border-transparent" 
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100 border-transparent");

              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`relative flex items-center justify-center px-2 py-1.5 sm:px-4 sm:py-2 text-xs sm:text-[13px] md:text-sm font-bold rounded-lg border transition-all duration-300 cursor-pointer ${tabStyle}`}
                  id={`nav-tab-${item.id}`}
                >
                  <span>{item.label}</span>
                  {isActive && (
                    <span className="absolute bottom-0 inset-x-3 h-[2px] bg-indigo-500 rounded-full" />
                  )}
                </button>
              );
            })}
          </nav>

          {/* Right Header Controls (Language and Theme Selection) */}
          <div className="flex items-center gap-2 sm:gap-3">
            
            {/* Theme Toggle (Day / Night mode switcher displaying current status) */}
            <button
              onClick={() => setTheme(isDark ? 'light' : 'dark')}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg border text-xs font-mono transition-all cursor-pointer ${btnBorder}`}
              title={isDark ? (lang === 'en' ? "Switch to Day Mode" : "切換至淺色模式") : (lang === 'en' ? "Switch to Night Mode" : "切換至深色模式")}
              id="theme-toggler"
            >
              {isDark ? (
                <>
                  <Moon className="w-3.5 h-3.5 text-indigo-400 fill-indigo-400/20" />
                  <span className="font-sans font-medium hidden sm:inline">{lang === 'en' ? "Night Mode" : "深色模式"}</span>
                </>
              ) : (
                <>
                  <Sun className="w-3.5 h-3.5 text-amber-500 fill-amber-500/10" />
                  <span className="font-sans font-medium hidden sm:inline">{lang === 'en' ? "Day Mode" : "淺色模式"}</span>
                </>
              )}
            </button>

            {/* Language Selection Dropdown */}
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setLangDropdownOpen(!langDropdownOpen)}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg border text-xs font-mono transition-all cursor-pointer ${btnBorder}`}
                id="language-dropdown-toggle"
              >
                <Globe className="w-3.5 h-3.5 text-indigo-500" />
                <span className="font-sans font-medium">
                  {lang === 'en' ? "English" : "繁體中文"}
                </span>
                <ChevronDown className={`w-3 h-3 transition-transform duration-200 ${langDropdownOpen ? "rotate-180" : ""}`} />
              </button>

              {langDropdownOpen && (
                <div className={`absolute right-0 mt-1.5 w-32 rounded-lg border py-1 z-50 animate-fade-in ${dropdownBg}`} id="language-dropdown-menu">
                  <button
                    onClick={() => {
                      setLang('en');
                      setLangDropdownOpen(false);
                    }}
                    className={`w-full text-left px-3 py-1.5 text-xs font-sans flex items-center justify-between cursor-pointer ${dropdownItemHover}`}
                  >
                    <span>English</span>
                    {lang === 'en' && <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />}
                  </button>
                  <button
                    onClick={() => {
                      setLang('zh');
                      setLangDropdownOpen(false);
                    }}
                    className={`w-full text-left px-3 py-1.5 text-xs font-sans flex items-center justify-between cursor-pointer ${dropdownItemHover}`}
                  >
                    <span>繁體中文</span>
                    {lang === 'zh' && <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />}
                  </button>
                </div>
              )}
            </div>

          </div>

        </div>
      </div>
    </header>
  );
}
