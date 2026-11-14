import { useState } from "react";
import { BookOpen, Search, Filter, Calendar, Clock, ChevronRight, X, Sparkles, ChevronDown } from "lucide-react";

interface Article {
  id: number;
  title: string;
  tag: string;
  date: string;
  readTime: string;
  summary: string;
  content: string;
}

interface ArticlesProps {
  lang: 'en' | 'zh';
  theme: 'dark' | 'light';
}

export default function Articles({ lang, theme }: ArticlesProps) {
  const isDark = theme === 'dark';
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTag, setSelectedTag] = useState<string>("all");
  const [selectedArticleId, setSelectedArticleId] = useState<number | null>(null);
  const [showTags, setShowTags] = useState(false);

  // Expanded articles repository with high-quality insights
  const articles_en: Article[] = [
    {
      id: 1,
      title: "Reasoning LLM Comparative Analysis: DeepSeek R1 vs. Claude 3.5 Sonnet",
      tag: "Logical Reasoning",
      date: "2026-06-12",
      readTime: "6 min read",
      summary: "Evaluates algorithm design, dynamic planning, and boundary conditions. R1's native reinforcement learning excels in math precision by 15%, while Sonnet maintains superior readability and architectural formatting.",
      content: `### Evaluation Context
This test covers 50 high-difficulty algorithm designs (LeetCode Hard) and 30 advanced mathematical analysis equations. We analyzed how models handle edge-cases, dynamic state tracking, and algorithmic optimization.

### Key Discoveries
1. **Thought Log Transparency**: DeepSeek R1 generates an average of 1,200 words in its thinking chain, capturing errors, backtracking, and trial-and-error steps. This increases model explainability.
2. **Extreme Logical Derivation**: On the MATH test set, DeepSeek R1 successfully solved complex ordinary differential equations with extraordinary precision.
3. **Coding Architecture**: Claude 3.5 Sonnet remains the gold standard in modular design, software decoupling, and commenting. Its code is production-ready and highly maintainable.

### Comparative Takeaway
For algorithmic challenges and mathematical derivation where correctness is paramount, DeepSeek R1 represents a monumental leap forward. However, for clean production codebases requiring solid architectural patterns, Claude 3.5 Sonnet remains unmatched.`
    },
    {
      id: 2,
      title: "Traditional Chinese Localization & Nuance: GPT-4o vs. Gemini 3.5 Flash",
      tag: "Multilingual Capability",
      date: "2026-05-30",
      readTime: "5 min read",
      summary: "Evaluates Taiwanese idiomatic translation, marketing copy, and legal agreements. Gemini 3.5 Flash shines in layout retention and formal translation, while GPT-4o leads in creative prose and flow.",
      content: `### Evaluation Context
Evaluates localized prose, business marketing slogans, and legal contractual clauses across 100 benchmark examples representing local Taiwanese idioms and legal syntax.

### Key Discoveries
1. **Idiomatic Adaptation**: GPT-4o excels in converting English idioms into natural, vivid local colloquialisms that fit Taiwanese reading habits effortlessly.
2. **Professional Documentation**: Gemini 3.5 Flash dominates in maintaining precise layout structure, nested lists, and Markdown tables.
3. **Concise Summary**: Gemini produces highly objective, summarized reports devoid of conversational fluff, making it ideal for software API integrations.

### Practical Engineering Advice
Choose GPT-4o when designing customer-facing chat interfaces that require friendly localized banter. Use Gemini 3.5 Flash for automated file ingestion pipelines, layout-heavy summarization, and developer tools where throughput speed and structured JSON formats are key.`
    },
    {
      id: 3,
      title: "The Battle of Cost and Latency: Gemini 3.1 Flash Lite in Production Workflow",
      tag: "Performance & Cost",
      date: "2026-05-15",
      readTime: "4 min read",
      summary: "An engineering deep-dive on running high-frequency classification, lightweight summarization, and routing tasks using the new ultra-low cost Gemini 3.1 Flash Lite.",
      content: `### Evaluation Context
We simulated a production-grade customer support routing agent handling 10,000 incoming multi-modal requests per hour, measuring API response latency, cold start overheads, and token expenditure.

### Key Discoveries
1. **Cost Minimization**: Flash Lite reduces operational API expenditures by up to 60% compared to standard Flash models, making high-volume agent routing viable.
2. **First-Token Latency**: Achieved an average of 140ms time-to-first-token, making it exceptionally suited for real-time edge interactions.
3. **Few-Shot Adherence**: While zero-shot performance is moderate on complex logic, adding just 2 or 3 few-shot examples inside the System Prompt raises routing accuracy to 98.4%.

### Implementation Advice
Do not use Flash Lite for complex code refactoring or multi-step logic. Instead, deploy it as the entry-point classifier (Intent Router) in multi-agent workflows to inspect user intent before passing requests to larger model panels.`
    }
  ];

  const articles_zh: Article[] = [
    {
      id: 1,
      title: "推理模型深度比對：DeepSeek R1 vs. Claude 3.5 Sonnet",
      tag: "邏輯推理",
      date: "2026-06-12",
      readTime: "6 分鐘閱讀",
      summary: "在涉及複雜代碼優化、動態規劃及邊界條件驗證時，原生強化學習 R1 在數學推導精確度上超過 Sonnet 達 15%，然而 Sonnet 在系統化架構規劃上擁有更好的敘述邏輯性與易讀性。",
      content: `### 評測背景
本次評測聚焦於「強邏輯任務」，包含 50 道高難度算法設計（LeetCode Hard）及 30 道高等數學分析題。我們評估了模型在面對極限邊界條件時的自我校對能力。

### 核心發現
1. **思維路徑透明度**：DeepSeek R1 在遇到複雜算法時，會生成平均 1,200 字的思考鏈（Thought Log），完整記錄了回溯、查錯及重試過程。這極大地提高了可解釋性。
2. **極限推導**：在 MATH 測試集上，DeepSeek R1 憑藉強化學習自主演進，成功求解了多道常微分方程，精確率表現極佳。
3. **編程架構**：Claude 3.5 Sonnet 在系統架構、模組解耦和注釋撰寫上依舊保持業界最高標準，代碼可以直接部署，可維護性極高。

### 綜合實測建議
面對極致硬核的演算法優化與科學推導，DeepSeek R1 展現了非凡的演算實力；但若需要產出結構優雅、模組化且易於維護的生產代碼，Claude 3.5 Sonnet 仍然是無可替代的首席工程師。`
    },
    {
      id: 2,
      title: "繁體中文在地化翻譯與語境評估：GPT-4o vs. Gemini 3.5 Flash",
      tag: "多語言能力",
      date: "2026-05-30",
      readTime: "5 分鐘閱讀",
      summary: "評估繁中成語、在地俚語、以及法律契約等特殊翻譯場景。Gemini 3.5 Flash 在排版格式及學術翻譯精準度表現出色，GPT-4o 則在文學口語化、流暢性上略微占優。",
      content: `### 評測背景
測試集包含 100 篇台灣與香港在地俚語、法律合約及行銷文案，全面比對在地化繁中語境理解。

### 核心發現
1. **在地俚語翻譯**：針對「接地氣」的口語，GPT-4o 翻譯非常生動，能將英文 idiom 轉譯為合適的台灣日常俗語。
2. **專業與法律文獻**：Gemini 3.5 Flash 發揮了強大脈絡理解優勢，在保持原句段落格式、排版結構（Markdown 表格）上極其精準。
3. **長文總結**：Gemini 的長文總結沒有多餘修飾，語氣極為客觀且格式標準，是工程界整合 API 應用的首選。

### 實務部署指南
若是開發需要生動有趣、在地化親和力高的客服或寫作助理，首選 GPT-4o；若是開發自動化文件分析系統、資訊摘要或極限傳輸 API，Gemini 3.5 Flash 的極速回傳與結構化 JSON 對齊能力將帶來更佳效益。`
    },
    {
      id: 3,
      title: "極限性價比對決：Gemini 3.1 Flash Lite 實戰部署與超低延遲調優",
      tag: "性能與成本",
      date: "2026-05-15",
      readTime: "4 分鐘閱讀",
      summary: "工程師深度指南：如何利用全新極低成本的 Gemini 3.1 Flash Lite 處理高頻分類、輕量摘要及意圖分流任務，實現高達 60% 費用節省。",
      content: `### 評測背景
我們模擬了高負載的生產級 AI 代理工作流，每小時處理 10,000 次多模態請求，針對 API 調用成本、首字延遲（TTFT）及指令遵循率進行端到端實測。

### 核心發現
1. **極致成本控制**：在大量調用語境下，Flash Lite 比標準版 Flash 降低了約 60% 的 API 支出，解決了長文本推理因成本過高難以落地的大難題。
2. **毫秒級首字響應**：首字回傳延遲平均僅 140 毫秒，在實時問答及語音對話系統中表現出類拔萃。
3. **少量樣本（Few-Shot）加持**：儘管零樣本表現中規中矩，但只要在 Prompt 中加入 2-3 個分類範例，意圖分流準確率便能飆升至 98.4%。

### 實戰應用建議
不要強求 Flash Lite 處理大體量的代碼重構或多步驟推導。相反地，應將其置於多代理（Multi-Agent）系統的最前線，做為輕量分流器（Intent Router）識別用戶意圖，能大幅優化系統反應速度並幫企業省下驚人預算。`
    }
  ];

  const articles = lang === 'en' ? articles_en : articles_zh;

  // Extract unique tags for filtering
  const tags = ["all", ...Array.from(new Set(articles.map(a => a.tag)))];

  // Filtering & Search logical processing
  const filteredArticles = articles.filter(article => {
    const matchesTag = selectedTag === "all" || article.tag === selectedTag;
    const searchLower = searchQuery.toLowerCase();
    const matchesSearch = 
      article.title.toLowerCase().includes(searchLower) ||
      article.summary.toLowerCase().includes(searchLower) ||
      article.tag.toLowerCase().includes(searchLower) ||
      article.content.toLowerCase().includes(searchLower);
    return matchesTag && matchesSearch;
  });

  const selectedArticle = articles.find(a => a.id === selectedArticleId);

  // Dynamic Styles
  const cardBgClass = isDark 
    ? "bg-[#0d0d0d] border-white/5 hover:border-white/10 hover:bg-black/40 text-slate-300" 
    : "bg-white border-slate-300 hover:border-indigo-500/50 hover:bg-slate-50/50 text-slate-900 shadow-sm shadow-slate-900/5 hover:shadow-md";
  const titleColor = isDark ? "text-white" : "text-slate-950 font-bold";
  const descColor = isDark ? "text-slate-400" : "text-slate-700 font-medium";
  const inputBgClass = isDark ? "border-slate-800 bg-slate-900/60 text-slate-200 placeholder-slate-500 focus:border-indigo-500" : "border-slate-300 bg-white text-slate-900 placeholder-slate-500 focus:border-indigo-500 shadow-sm";

  // If an article is selected, render the gorgeous full-page reader
  if (selectedArticle) {
    return (
      <div className="py-6 select-text animate-fade-in">
        <div className="max-w-4xl mx-auto px-1 sm:px-6">
          {/* Elegant Back button */}
          <button
            onClick={() => setSelectedArticleId(null)}
            className={`mb-6 inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold transition-all border cursor-pointer select-none ${
              isDark 
                ? "bg-white/5 border-white/10 text-slate-300 hover:text-white hover:bg-white/10" 
                : "bg-white border-slate-200 text-slate-700 hover:text-indigo-600 hover:border-slate-300 hover:bg-slate-50 shadow-sm"
            }`}
          >
            <span>←</span>
            <span>{lang === 'en' ? "Back to Articles List" : "返回評測專題列表"}</span>
          </button>

          {/* Full Page Reading Container */}
          <article className={`p-6 sm:p-10 rounded-2xl border ${
            isDark 
              ? "bg-[#0d0d0d] border-white/5 text-slate-200 shadow-2xl" 
              : "bg-white border-slate-200 text-slate-900 shadow-lg"
          }`}>
            {/* Metadata and Badge Bar */}
            <div className="flex flex-wrap items-center gap-4 mb-6 text-xs font-mono font-bold">
              <span className="px-2.5 py-1 rounded bg-indigo-500/10 border border-indigo-500/20 text-indigo-600 dark:text-indigo-300 uppercase tracking-wider">
                {selectedArticle.tag}
              </span>
              <span className={`flex items-center gap-1 ${isDark ? "text-slate-400" : "text-slate-700 font-semibold"}`}>
                <Calendar className="w-3.5 h-3.5 text-indigo-500" />
                {lang === 'en' ? "Published: " : "發布日期："} {selectedArticle.date}
              </span>
              <span className={`flex items-center gap-1 ${isDark ? "text-slate-400" : "text-slate-700 font-semibold"}`}>
                <Clock className="w-3.5 h-3.5 text-indigo-500" />
                {selectedArticle.readTime}
              </span>
            </div>

            {/* Main Display Title */}
            <h1 className={`font-display text-2xl sm:text-3xl font-extrabold tracking-tight mb-6 leading-tight ${
              isDark ? "text-white" : "text-slate-950"
            }`}>
              {selectedArticle.title}
            </h1>

            {/* Divider */}
            <div className={`h-[1px] w-full mb-8 border-t border-dashed ${
              isDark ? "border-white/10" : "border-slate-200"
            }`} />

            {/* Beautiful, High-Contrast Body Text */}
            <div className={`space-y-6 text-sm sm:text-base leading-relaxed ${
              isDark ? "text-slate-200 font-light" : "text-slate-900 font-normal"
            }`}>
              {selectedArticle.content.split("\n\n").map((para, pIdx) => {
                if (para.startsWith("### ")) {
                  return (
                    <h3 key={pIdx} className={`text-base sm:text-lg font-bold font-sans mt-8 mb-4 border-l-4 border-indigo-500 pl-3 ${
                      isDark ? "text-white" : "text-slate-950"
                    }`}>
                      {para.substring(4)}
                    </h3>
                  );
                }
                if (para.startsWith("1. ") || para.startsWith("2. ") || para.startsWith("3. ")) {
                  const items = para.split("\n");
                  return (
                    <ol key={pIdx} className="list-decimal list-inside pl-1 space-y-3 my-4">
                      {items.map((it, iIdx) => {
                        const parsed = it.replace(/^\d+\.\s+/, "");
                        const colonIndex = parsed.indexOf(":");
                        if (colonIndex !== -1) {
                          const boldPart = parsed.substring(0, colonIndex);
                          const regularPart = parsed.substring(colonIndex);
                          return (
                            <li key={iIdx} className={`leading-relaxed ${isDark ? "text-slate-300" : "text-slate-900 font-medium"}`}>
                              <strong className={isDark ? "text-white font-semibold" : "text-slate-950 font-bold"}>
                                {boldPart}
                              </strong>
                              {regularPart}
                            </li>
                          );
                        }
                        return (
                          <li key={iIdx} className={`leading-relaxed ${isDark ? "text-slate-300" : "text-slate-900 font-medium"}`}>
                            {parsed}
                          </li>
                        );
                      })}
                    </ol>
                  );
                }
                return (
                  <p key={pIdx} className={`leading-relaxed ${
                    isDark ? "text-slate-300" : "text-slate-900 font-medium"
                  }`}>
                    {para}
                  </p>
                );
              })}
            </div>

            {/* Back action footer */}
            <div className={`mt-12 pt-6 border-t flex flex-col sm:flex-row justify-between items-center gap-4 ${
              isDark ? "border-white/5" : "border-slate-100"
            }`}>
              <span className="text-xs text-slate-500 font-medium font-mono">
                {lang === 'en' ? "K-AREN Pro Sandbox Insights" : "K-AREN Pro 專業級沙盒模型深度報告"}
              </span>
              <button
                onClick={() => setSelectedArticleId(null)}
                className="w-full sm:w-auto px-6 py-2.5 rounded-xl text-xs font-bold transition-all bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/10 cursor-pointer"
              >
                {lang === 'en' ? "Back to Articles list" : "關閉並返回列表"}
              </button>
            </div>
          </article>
        </div>
      </div>
    );
  }

  return (
    <div className="py-6 select-none animate-fade-in">
      <div className="max-w-7xl mx-auto px-1 sm:px-6 lg:px-8">
        
        {/* Header Title Block */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6 mb-8 border-b border-slate-200/40 dark:border-white/5 pb-6">
          <div>
            <h1 className={`text-2xl sm:text-3xl font-display font-bold flex items-center gap-2.5 ${titleColor}`}>
              <BookOpen className="w-6 h-6 text-indigo-500" />
              <span>{lang === 'en' ? "Expert Reviews & Articles" : "深度評測專題與文章"}</span>
            </h1>
            <p className={`text-xs sm:text-sm mt-1.5 ${descColor}`}>
              {lang === 'en' 
                ? "Deep-dive evaluations comparing reasoning speed, code refactoring, and multi-lingual prompt limits."
                : "專注於 LLM 的極限邏輯、在地化繙譯及性價比對決的實測文章與專業報告。"}
            </p>
          </div>
        </div>

        {/* Search & Filter bar */}
        <div className="flex flex-col gap-4 mb-8">
          <div className="flex flex-col md:flex-row gap-4">
            {/* Search Input */}
            <div className="relative flex-grow">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
              <input
                type="text"
                placeholder={lang === 'en' ? "Search articles, insights, or technology..." : "搜尋評測文章、關鍵技術或模型..."}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className={`w-full pl-12 pr-4 py-3.5 rounded-2xl border text-base font-light transition-all focus:outline-none focus:ring-1 focus:ring-indigo-500 ${inputBgClass}`}
              />
            </div>

            {/* Filter Toggle Button */}
            <button
              onClick={() => setShowTags(!showTags)}
              className={`px-5 py-3.5 rounded-2xl border font-bold text-sm tracking-wide transition-all flex items-center justify-center gap-2 cursor-pointer select-none shrink-0 ${
                selectedTag !== "all" || showTags
                  ? "bg-indigo-600 text-white border-indigo-500 shadow-lg shadow-indigo-600/10"
                  : isDark
                    ? "bg-slate-900 border-slate-800 text-slate-200 hover:border-slate-700 hover:bg-slate-850"
                    : "bg-white border-slate-200 text-slate-700 hover:border-slate-300 hover:bg-slate-50 shadow-sm"
              }`}
            >
              <Filter className="w-4 h-4" />
              <span>{lang === 'en' ? "Filter by Tag" : "按分類篩選"}</span>
              {selectedTag !== "all" && (
                <span className="ml-1 px-2 py-0.5 text-[10px] bg-white text-indigo-600 rounded-full font-black uppercase">
                  {selectedTag}
                </span>
              )}
              <ChevronDown className={`w-4 h-4 transition-transform duration-300 ${showTags ? "rotate-180" : ""}`} />
            </button>
          </div>

          {/* Collapsible tags drawer/dropdown */}
          {showTags && (
            <div className={`p-4 sm:p-5 rounded-2xl border transition-all ${
              isDark ? "bg-[#0d0d0d] border-white/5" : "bg-slate-50 border-slate-200/60"
            }`}>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3 pb-2.5 border-b border-dashed border-slate-200 dark:border-white/5">
                <span className={`text-xs font-mono font-bold tracking-wider uppercase ${isDark ? "text-indigo-400" : "text-indigo-600"}`}>
                  {lang === 'en' ? "Select Category tag" : "選擇分類標籤"}
                </span>
                {selectedTag !== "all" && (
                  <button
                    onClick={() => setSelectedTag("all")}
                    className="text-[11px] font-bold text-rose-500 hover:underline flex items-center gap-1 cursor-pointer"
                  >
                    <X className="w-3 h-3" />
                    <span>{lang === 'en' ? "Clear Filter" : "重設篩選"}</span>
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-2.5">
                {tags.map((tag) => (
                  <button
                    key={tag}
                    onClick={() => {
                      setSelectedTag(tag);
                    }}
                    className={`px-4 py-2 rounded-xl text-xs font-bold border transition-all uppercase tracking-wider cursor-pointer ${
                      selectedTag === tag
                        ? "bg-indigo-600 text-white border-indigo-500 shadow-md"
                        : isDark
                          ? "bg-black/40 text-slate-400 border-slate-800 hover:border-slate-700 hover:text-white"
                          : "bg-white text-slate-700 border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                    }`}
                  >
                    {tag === "all" ? (lang === 'en' ? "Show All" : "顯示全部") : tag}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Articles List Grid */}
        {filteredArticles.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" id="articles-hub-grid">
            {filteredArticles.map((article) => (
              <div
                key={article.id}
                onClick={() => setSelectedArticleId(article.id)}
                className={`group p-6 rounded-2xl border transition-all duration-300 cursor-pointer flex flex-col justify-between h-[250px] relative overflow-hidden ${cardBgClass}`}
              >
                {/* Decorative glow */}
                <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/5 rounded-full blur-2xl group-hover:bg-indigo-500/10 transition-all duration-500 pointer-events-none" />

                <div>
                  <div className="flex items-center justify-between text-[10px] font-mono mb-3">
                    <span className="px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20 text-indigo-600 dark:text-indigo-300 font-bold uppercase tracking-wider">
                      {article.tag}
                    </span>
                    <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400 font-semibold">
                      <span className="flex items-center gap-0.5">
                        <Calendar className="w-3.5 h-3.5 text-indigo-500" />
                        {article.date}
                      </span>
                    </div>
                  </div>

                  <h3 className={`font-display text-base font-bold group-hover:text-indigo-500 transition-colors line-clamp-2 ${titleColor}`}>
                    {article.title}
                  </h3>

                  <p className="text-xs leading-relaxed mt-2.5 line-clamp-3 text-slate-600 dark:text-slate-400 font-normal">
                    {article.summary}
                  </p>
                </div>

                <div className="flex items-center justify-between mt-4 border-t border-slate-200/80 dark:border-white/5 pt-3">
                  <span className="text-[10px] font-mono text-slate-600 dark:text-slate-400 flex items-center gap-1 font-bold">
                    <Clock className="w-3.5 h-3.5 text-indigo-500 dark:text-indigo-400" />
                    {article.readTime}
                  </span>
                  
                  <span className="text-xs font-semibold text-indigo-500 group-hover:translate-x-1 transition-transform flex items-center gap-0.5">
                    <span>{lang === 'en' ? "Read Full" : "完整文章"}</span>
                    <span>→</span>
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className={`text-center py-20 border border-dashed rounded-2xl ${isDark ? "border-slate-800" : "border-slate-200"}`}>
            <Sparkles className="w-8 h-8 text-slate-400 mx-auto mb-3 animate-pulse" />
            <h3 className={`text-sm font-medium ${isDark ? "text-slate-300" : "text-slate-700"}`}>
              {lang === 'en' ? "No articles match your query" : "未找到符合篩選條件的評測文章"}
            </h3>
            <p className="text-xs text-slate-500 mt-1 font-light">
              {lang === 'en' ? "Try adjusting your search query or choosing another tag filter." : "請嘗試調整搜尋關鍵字或點選其他分類按鈕。"}
            </p>
          </div>
        )}

      </div>
    </div>
  );
}
