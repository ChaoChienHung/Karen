import { useState } from "react";
import { Trophy, FileText, Check, HelpCircle, ArrowUpDown, ChevronRight, ChevronDown } from "lucide-react";
import { Model } from "../types";

interface LeaderboardProps {
  models: Model[];
  lang: 'en' | 'zh';
  theme?: 'dark' | 'light';
}

export default function Leaderboard({ models, lang, theme }: LeaderboardProps) {
  const [selectedMetric, setSelectedMetric] = useState<'MMLU' | 'HumanEval' | 'MATH' | 'GPQA'>('MMLU');
  const [selectedReport, setSelectedReport] = useState<number | null>(null);
  const [metricDropdownOpen, setMetricDropdownOpen] = useState(false);

  const isDark = theme !== "light";

  // Dynamic Theme Styling Clones
  const cardBgClass = isDark ? "bg-[#0d0d0d] border-white/10 shadow-xl" : "bg-white border-slate-300 shadow-sm shadow-slate-900/5";
  const itemBgClass = isDark ? "bg-black/40 border-white/5" : "bg-slate-100/90 border-slate-200";
  const subItemBgClass = isDark ? "bg-black/20 border-white/5" : "bg-slate-50 border-slate-250";
  const textColorPrimary = isDark ? "text-white" : "text-slate-950 font-bold";
  const textColorSecondary = isDark ? "text-slate-400" : "text-slate-800 font-medium";
  const textColorMuted = isDark ? "text-slate-500" : "text-slate-600 font-medium";
  const borderColClass = isDark ? "border-white/10" : "border-slate-200";

  const metrics_en = [
    { id: 'MMLU', name: 'MMLU', desc: 'Academic Breadth', tooltip: 'Massive Multitask Language Understanding. Evaluates multi-disciplinary (humanities, social sciences, STEM) facts and reasoning.' },
    { id: 'HumanEval', name: 'HumanEval', desc: 'Coding Capability', tooltip: 'Evaluates Python programming logic and automated coding correctness.' },
    { id: 'MATH', name: 'MATH', desc: 'Advanced Math Logic', tooltip: 'Complex mathematical problem-solving across algebra, calculus, and geometry.' },
    { id: 'GPQA', name: 'GPQA', desc: 'Expert level Q&A', tooltip: 'Graduate-Level Google-Proof Q&A. Advanced PhD-level physics, chemistry, and biology reasoning.' },
  ] as const;

  const metrics_zh = [
    { id: 'MMLU', name: 'MMLU', desc: '綜合學科理解', tooltip: 'Massive Multitask Language Understanding. 測試多學科（人文、社科、STEM）綜合知識與學術邏輯。' },
    { id: 'HumanEval', name: 'HumanEval', desc: '編程代碼能力', tooltip: '測試 Python 代碼生成、基礎邏輯編程與代碼正確性。' },
    { id: 'MATH', name: 'MATH', desc: '多步數學推理', tooltip: '複雜數學推導難題。測試多步驟推理與精確求解能力。' },
    { id: 'GPQA', name: 'GPQA', desc: '硬科學專家評測', tooltip: 'Graduate-Level Google-Proof Q&A. 研究生難度物理、化學、生物前沿難題。' },
  ] as const;

  const metrics = lang === 'en' ? metrics_en : metrics_zh;

  // Sort models based on the selected metric score descending
  const sortedModels = [...models].sort((a, b) => b.benchmarks[selectedMetric] - a.benchmarks[selectedMetric]);

  const reports_en = [
    {
      id: 1,
      title: "Reasoning LLM Comparative Analysis: DeepSeek R1 vs. Claude 3.5 Sonnet",
      tag: "Logical Reasoning",
      date: "2026-06-12",
      summary: "Evaluates algorithm design, dynamic planning, and boundary conditions. R1's native reinforcement learning excels in math precision by 15%, while Sonnet maintains superior readability and architectural formatting.",
      content: `### Evaluation Context
This test covers 50 high-difficulty algorithm designs (LeetCode Hard) and 30 advanced mathematical analysis equations.

### Key Discoveries
1. **Thought Log Transparency**: DeepSeek R1 generates an average of 1,200 words in its thinking chain, capturing errors, backtracking, and trial-and-error steps. This increases model explainability.
2. **Extreme Logical Derivation**: On the MATH test set, DeepSeek R1 successfully solved complex ordinary differential equations with extraordinary precision.
3. **Coding Architecture**: Claude 3.5 Sonnet remains the gold standard in modular design, software decoupling, and commenting. Its code is production-ready and highly maintainable.`
    },
    {
      id: 2,
      title: "Traditional Chinese Localization & Nuance: GPT-4o vs. Gemini 3.5 Flash",
      tag: "Multilingual Capability",
      date: "2026-05-30",
      summary: "Evaluates Taiwanese idiomatic translation, marketing copy, and legal agreements. Gemini 3.5 Flash shines in layout retention and formal translation, while GPT-4o leads in creative prose and flow.",
      content: `### Evaluation Context
Evaluates localized prose, business marketing slogans, and legal contractual clauses across 100 benchmark examples.

### Key Discoveries
1. **Idiomatic Adaptation**: GPT-4o excels in converting English idioms into natural, vivid local colloquialisms.
2. **Professional Documentation**: Gemini 3.5 Flash dominates in maintaining precise layout structure, nested lists, and Markdown tables.
3. **Concise Summary**: Gemini produces highly objective, summarized reports devoid of conversational fluff, making it ideal for software API integrations.`
    }
  ];

  const reports_zh = [
    {
      id: 1,
      title: "推理模型深度比對：DeepSeek R1 vs. Claude 3.5 Sonnet",
      tag: "邏輯推理",
      date: "2026-06-12",
      summary: "在涉及複雜代碼優化、動態規劃及邊界條件驗證時，原生強化學習 R1 在數學推導精確度上超過 Sonnet 達 15%，然而 Sonnet 在系統化架構規劃上擁有更好的敘述邏輯性與易讀性。",
      content: `### 評測背景
本次評測聚焦於「強邏輯任務」，包含 50 道高難度算法設計（LeetCode Hard）及 30 道高等數學分析題。

### 核心發現
1. **思維路徑透明度**：DeepSeek R1 在遇到複雜算法時，會生成平均 1,200 字的思考鏈（Thought Log），完整記錄了回溯、查錯及重試過程。這極大地提高了可解釋性。
2. **極限推導**：在 MATH 測試集上，DeepSeek R1 憑藉強化學習自主演進，成功求解了多道常微分方程，精確率表現極佳。
3. **編程架構**：Claude 3.5 Sonnet 在系統架構、模組解耦和注釋撰寫上依舊保持業界最高標準，代碼可以直接部署，可維護性極高。`
    },
    {
      id: 2,
      title: "繁體中文在地化翻譯與語境評估：GPT-4o vs. Gemini 3.5 Flash",
      tag: "多語言能力",
      date: "2026-05-30",
      summary: "評估繁中成語、在地俚語、以及法律契約等特殊翻譯場景。Gemini 3.5 Flash 在排版格式及學術翻譯精準度表現出色，GPT-4o 則在文學口語化、流暢性上略微占優。",
      content: `### 評測背景
測試集包含 100 篇台灣與香港在地俚語、法律合約及行銷文案，全面比對在地化繁中語境理解。

### 核心發現
1. **在地俚語翻譯**：針對「接地氣」的口語，GPT-4o 翻譯非常生動，能將 idiom 轉譯為合適的英文日常俗語。
2. **專業與法律文獻**：Gemini 3.5 Flash 發揮了強大脈絡理解優勢，在保持原句段落格式、排版結構（Markdown 表格）上極其精準。
3. **長文總結**：Gemini 的長文總結沒有多餘修飾，語氣極為客觀且格式標準，是工程界整合 API 應用的首選。`
    }
  ];

  const reports = lang === 'en' ? reports_en : reports_zh;

  return (
    <div className="py-8 select-none">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Header */}
        <div className="mb-8">
          <h1 className={`text-2xl sm:text-3xl font-display font-bold flex items-center gap-2 ${textColorPrimary}`}>
            <Trophy className="w-6 h-6 text-indigo-500 dark:text-indigo-400" />
            <span>{lang === 'en' ? "Benchmarks & Evaluations" : "評測與性能報告"}</span>
          </h1>
          <p className={`text-xs sm:text-sm mt-1 font-light ${textColorSecondary}`}>
            {lang === 'en'
              ? "Quantified academic benchmark rankings paired with rigorous, expert comparative reports."
              : "結合業界公認學術標準 (MMLU, HumanEval) 的客觀量化數據，輔以深度專題對比報告。"}
          </p>
        </div>

        {/* Bento Grid layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
          
          {/* Main Benchmark Progress Chart (Left 2 Columns) */}
          <div className={`lg:col-span-2 p-6 rounded-xl border ${cardBgClass}`}>
            
            {/* Metric select headers refactored to a dropdown */}
            <div className={`flex items-center justify-between gap-4 mb-6 border-b pb-4 ${borderColClass}`}>
              <h3 className={`font-sans text-sm font-bold ${textColorPrimary}`}>
                {lang === 'en' ? "Objective Standard Leaderboard" : "客觀標準排行榜"}
              </h3>
              
              {/* Scalable Custom Dropdown */}
              <div className="relative">
                <button
                  onClick={() => setMetricDropdownOpen(!metricDropdownOpen)}
                  className={`px-3 py-2 rounded-xl text-xs font-bold font-mono border flex items-center gap-2 transition-all cursor-pointer ${
                    isDark
                      ? "bg-black/50 border-white/10 text-indigo-400 hover:text-indigo-300 hover:border-white/20"
                      : "bg-white border-slate-200 text-indigo-600 hover:text-indigo-700 shadow-sm"
                  }`}
                  id="metric-dropdown-trigger"
                >
                  <span>{selectedMetric}</span>
                  <ChevronDown className="w-3.5 h-3.5 opacity-80" />
                </button>

                {metricDropdownOpen && (
                  <>
                    <div className="fixed inset-0 z-30" onClick={() => setMetricDropdownOpen(false)} />
                    <div
                      className={`absolute right-0 mt-1.5 w-60 rounded-xl border p-1 shadow-2xl z-40 transition-all animate-fade-in ${
                        isDark
                          ? "bg-[#0f0f0f] border-slate-800 text-slate-200"
                          : "bg-white border-slate-200 text-slate-900 shadow-indigo-100/40"
                      }`}
                    >
                      {metrics.map((m) => {
                        const isSelected = selectedMetric === m.id;
                        return (
                          <button
                            key={m.id}
                            onClick={() => {
                              setSelectedMetric(m.id);
                              setMetricDropdownOpen(false);
                            }}
                            className={`w-full text-left px-3.5 py-2.5 text-xs rounded-lg flex items-center justify-between transition-colors cursor-pointer ${
                              isSelected
                                ? "bg-indigo-600/10 text-indigo-500 font-bold"
                                : isDark
                                  ? "hover:bg-white/5 text-slate-300"
                                  : "hover:bg-slate-50 text-slate-800 font-semibold"
                            }`}
                          >
                            <div className="flex flex-col">
                              <span className="font-bold">{m.name}</span>
                              <span className={`text-[10px] font-sans mt-0.5 ${isSelected ? "text-indigo-500" : "opacity-60"}`}>{m.desc}</span>
                            </div>
                            {isSelected && <Check className="w-3.5 h-3.5 text-indigo-500" />}
                          </button>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Metric explanation */}
            <div className={`mb-6 p-3.5 rounded-lg border text-xs leading-relaxed font-light flex items-start gap-2 ${itemBgClass}`}>
              <HelpCircle className="w-4 h-4 text-indigo-500 dark:text-indigo-400 flex-shrink-0 mt-0.5" />
              <div className={textColorSecondary}>
                <span className={`font-bold ${textColorPrimary}`}>{selectedMetric} ({metrics.find(m => m.id === selectedMetric)?.desc})</span>:{" "}
                {metrics.find(m => m.id === selectedMetric)?.tooltip}
              </div>
            </div>

            {/* Rankings vertical list */}
            <div className="space-y-4">
              {sortedModels.map((model, index) => {
                const score = model.benchmarks[selectedMetric];
                const rank = index + 1;
                
                const rankBadgeColor = 
                  rank === 1 ? "bg-amber-500/10 text-amber-500 dark:text-amber-400 border-amber-500/20" :
                  rank === 2 ? (isDark ? "bg-slate-300/10 text-slate-300 border-slate-300/20" : "bg-slate-200/50 text-slate-700 border-slate-300") :
                  rank === 3 ? "bg-amber-700/10 text-amber-700 dark:text-amber-600 border-amber-700/20" :
                  (isDark ? "bg-white/5 text-slate-500 border-white/5" : "bg-slate-100 text-slate-600 border-slate-200");

                return (
                  <div key={model.id} className="group relative">
                    <div className="flex items-center justify-between mb-1.5 text-xs">
                      <div className="flex items-center gap-2">
                        <span className={`w-5 h-5 flex items-center justify-center rounded-md border text-[10px] font-mono font-bold ${rankBadgeColor}`}>
                          {rank}
                        </span>
                        <span className={`font-semibold group-hover:text-indigo-500 dark:group-hover:text-indigo-400 transition-colors ${textColorPrimary}`}>
                          {model.name}
                        </span>
                        <span className={`text-[9px] font-mono ${textColorMuted}`}>
                          {model.provider}
                        </span>
                      </div>
                      <span className="font-mono font-bold text-indigo-700 dark:text-indigo-300">{score}%</span>
                    </div>

                    {/* Progress Bar Container */}
                    <div className={`h-2 w-full rounded-full overflow-hidden border ${isDark ? "bg-black border-white/5" : "bg-slate-100 border-slate-200/50"}`}>
                      <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-1000 ease-out"
                        style={{ width: `${score}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Metric Overview */}
          <div className={`p-6 rounded-xl border flex flex-col justify-between ${cardBgClass}`}>
            <div>
              <h3 className={`font-sans text-sm font-bold mb-4 border-b pb-3 ${borderColClass} ${textColorPrimary}`}>
                {lang === 'en' ? "Standard Datasets" : "學術測試集概述"}
              </h3>
              
              <div className="space-y-4">
                {metrics.map((m) => (
                  <div key={m.id} className={`p-3 rounded-lg border ${subItemBgClass}`}>
                    <div className="flex justify-between items-center mb-1">
                      <span className={`text-xs font-bold font-mono ${textColorPrimary}`}>{m.name}</span>
                      <span className="text-[10px] text-indigo-700 dark:text-indigo-400 font-bold">{m.desc}</span>
                    </div>
                    <p className={`text-[11px] leading-normal font-light ${textColorSecondary}`}>
                      {m.tooltip.split(". ")[0]}.
                    </p>
                  </div>
                ))}
              </div>
            </div>
            
            <div className={`mt-6 pt-4 border-t text-[10px] font-light italic ${borderColClass} ${textColorMuted}`}>
              {lang === 'en' 
                ? "* Benchmark score data aggregated from official model white papers and LMSYS open evaluation logs."
                : "* 公開學術測試基準數據均來源於各模型官方技術白皮書及 LMSYS 開源評測平台。"}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
