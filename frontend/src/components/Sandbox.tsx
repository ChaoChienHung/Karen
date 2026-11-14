import { useState, useRef, useEffect } from "react";
import { 
  Play, 
  Plus, 
  Lock,
  Pin,
  PinOff,
  Trash2, 
  Copy, 
  Check, 
  RefreshCw, 
  Sparkles, 
  SlidersHorizontal, 
  LayoutGrid, 
  HelpCircle,
  AlertTriangle,
  Mic,
  Send,
  Languages,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  X
} from "lucide-react";
import { Model, GlobalConfig, OutputPanel } from "../types";
import MarkdownRenderer from "./MarkdownRenderer";

interface SandboxProps {
  models: Model[];
  lang: 'en' | 'zh';
  theme: 'dark' | 'light';
}

// Interactive Prompt Templates for testing - EN Version
const PROMPT_TEMPLATES_EN = [
  {
    label: "Code Refactoring & Optimization",
    system: "You are a senior software architect focused on clean code, memory efficiency, and optimal time complexity.",
    user: "Please refactor this JavaScript function to make it run faster and consume less memory. Analyze the Big-O time and space complexity before and after your refactoring:\n\n```javascript\nfunction findDuplicates(arr) {\n  let duplicates = [];\n  for (let i = 0; i < arr.length; i++) {\n    for (let j = i + 1; j < arr.length; j++) {\n      if (arr[i] === arr[j] && !duplicates.includes(arr[i])) {\n        duplicates.push(arr[i]);\n      }\n    }\n  }\n  return duplicates;\n}\n```"
  },
  {
    label: "Idiomatic Translation & Polish",
    system: "You are a professional literary translator specializing in translating text between English and elegant Traditional Chinese, capturing local idioms, nuances, and tones.",
    user: "Translate the following passage into natural, elegant Traditional Chinese (Taiwanese localized idiom style). Include a short commentary on key localized business terminology:\n\n\"The engineering team decided to bite the bullet and rewrite the legacy system from scratch. They knew it was a long shot, but keeping the old servers running was no longer a viable option. At the end of the day, quality code is what keeps customers happy.\""
  },
  {
    label: "Extreme Logical Reasoning Challenge",
    system: "You are a world-class logician who solves puzzles step-by-step with rigorous logic and absolute clarity.",
    user: "There are three boxes: one with apples, one with oranges, and one with both apples and oranges. Every box is labeled, but ALL labels are incorrect.\nIf you can only pick one fruit from one box (without looking inside), which box should you pick from to correctly identify all three boxes? Walk through your logical reasoning step-by-step."
  }
];

// Interactive Prompt Templates for testing - ZH Version
const PROMPT_TEMPLATES_ZH = [
  {
    label: "代碼重構與優化",
    system: "你是一位資深的系統架構師，專注於代碼品質、執行效率、時間/空間複雜度優化、以及乾淨代碼（Clean Code）規範。",
    user: "請重構以下 JavaScript 函數，優化其執行效率，並分析優化前後的時間複雜度與空間複雜度：\n\n```javascript\nfunction findDuplicates(arr) {\n  let duplicates = [];\n  for (let i = 0; i < arr.length; i++) {\n    for (let j = i + 1; j < arr.length; j++) {\n      if (arr[i] === arr[j] && !duplicates.includes(arr[i])) {\n        duplicates.push(arr[i]);\n      }\n    }\n  }\n  return duplicates;\n}\n```"
  },
  {
    label: "繁簡 idioms 翻譯與在地潤色",
    system: "你是一位專業的翻譯官，精通繁體中文（台灣語境）與英文之間的雙向翻譯，注重在地化俚語、成語與語氣轉換。",
    user: "請將以下段落翻譯成優雅流暢的台灣繁體中文，並附上關鍵字詞的對比說明：\n\n\"The engineering team decided to bite the bullet and rewrite the legacy system from scratch. They knew it was a long shot, but keeping the old servers running was no longer a viable option. At the end of the day, quality code is what keeps customers happy.\""
  },
  {
    label: "極限邏輯思維挑戰",
    system: "你是一位世界級的邏輯學家，思維嚴密、分析透徹、擅長使用歸納法與演繹法進行推理，解答過程條理分明。",
    user: "有三個盒子：一個裝有蘋果，一個裝有橘子，另一個裝有蘋果和橘子。每個盒子外面都貼了標籤，但標籤全部都貼錯了。\n如果你只能從其中一個盒子裡拿出一顆水果（且不能看盒子內部），你應該從哪一個盒子裡拿，才能百分之百正確推斷出所有三個盒子裡各自裝的是什麼水果？請給出詳細的邏輯推導步驟。"
  }
];

export default function Sandbox({ models, lang, theme }: SandboxProps) {
  const isDark = theme === 'dark';

  // Local Config state
  const [globalConfig, setGlobalConfig] = useState<GlobalConfig>({
    systemPrompt: "",
    userPrompt: "",
    temperature: 0.7,
    maxTokens: 1200,
    syncScroll: true
  });

  // Dynamic Panels workspace state - Default load flash and deepseek
  const [panels, setPanels] = useState<OutputPanel[]>([
    { id: "panel-1", modelId: "gemini-3.5-flash", output: "", isStreaming: false, isCompleted: false },
    { id: "panel-2", modelId: "deepseek-r1", output: "", isStreaming: false, isCompleted: false }
  ]);

  // Baseline Model ID (Pinned on the left for benchmark comparison)
  const [baselinePanelId, setBaselinePanelId] = useState<string | null>("panel-1");

  // Sidebar hover state
  const [isHovered, setIsHovered] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  
  // Copy state trackers
  const [copiedPanelId, setCopiedPanelId] = useState<string | null>(null);
  const [copiedAll, setCopiedAll] = useState(false);

  // Custom panel model selector dropdown state
  const [openDropdownId, setOpenDropdownId] = useState<string | null>(null);

  // Carousel scrolling ref & helper for the right comparison stream
  const carouselScrollRef = useRef<HTMLDivElement>(null);
  const scrollCarousel = (direction: 'left' | 'right') => {
    if (carouselScrollRef.current) {
      const scrollAmount = direction === 'left' ? -496 : 496; // width (480px) + gap (16px)
      carouselScrollRef.current.scrollBy({ left: scrollAmount, behavior: 'smooth' });
    }
  };

  // Voice recognition state
  const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<any>(null);

  // Translation & Feedback state
  const [isTranslating, setIsTranslating] = useState(false);
  const [notificationToast, setNotificationToast] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setNotificationToast(msg);
    setTimeout(() => setNotificationToast(null), 3000);
  };

  // Max models calculation based on available pool
  const maxAllowedModels = models.length > 0 ? models.length : 12;
  const isMaxModelsReached = panels.length >= maxAllowedModels;

  // Initialize Speech Recognition if supported
  useEffect(() => {
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = lang === 'en' ? 'en-US' : 'zh-TW';
      
      recognition.onstart = () => {
        setIsListening(true);
      };
      
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setGlobalConfig(prev => ({
          ...prev,
          userPrompt: prev.userPrompt + transcript
        }));
      };
      
      recognition.onerror = (event: any) => {
        console.error("Speech recognition error:", event);
        setIsListening(false);
      };
      
      recognition.onend = () => {
        setIsListening(false);
      };
      
      recognitionRef.current = recognition;
    }
  }, [lang]);

  const toggleListening = () => {
    if (!recognitionRef.current) return;
    if (isListening) {
      recognitionRef.current.stop();
    } else {
      recognitionRef.current.start();
    }
  };

  const handleTranslatePrompt = async () => {
    if (!globalConfig.userPrompt.trim()) return;
    setIsTranslating(true);
    try {
      const targetLang = lang === 'en' ? 'Traditional Chinese (Taiwan)' : 'English';
      const response = await fetch(`/api/compare/stream?modelId=gemini-3.5-flash&prompt=${encodeURIComponent(`Please translate the following text directly into natural, smooth ${targetLang}. Return ONLY the direct translation and absolutely nothing else. Keep terms like code or systems standard. Text to translate:\n\n${globalConfig.userPrompt}`)}&systemPrompt=You are an expert translator. Output ONLY the raw translation with no introductory text or quotes.&temperature=0.1&maxTokens=600`);
      
      if (response.ok) {
        const reader = response.body?.getReader();
        const decoder = new TextDecoder("utf-8");
        if (reader) {
          let done = false;
          let translatedText = "";
          let chunkBuffer = "";
          while (!done) {
            const { value, done: doneReading } = await reader.read();
            done = doneReading;
            chunkBuffer += decoder.decode(value, { stream: !doneReading });
            const lines = chunkBuffer.split("\n");
            chunkBuffer = lines.pop() || "";
            for (const line of lines) {
              const trimmed = line.trim();
              if (trimmed.startsWith("data: ")) {
                const dataStr = trimmed.substring(6).trim();
                if (dataStr === "[DONE]") {
                  done = true;
                  break;
                }
                try {
                  const parsed = JSON.parse(dataStr);
                  if (parsed.text) {
                    translatedText += parsed.text;
                  }
                } catch (e) {}
              }
            }
          }
          if (translatedText.trim()) {
            setGlobalConfig(prev => ({
              ...prev,
              userPrompt: translatedText.trim()
            }));
          }
        }
      }
    } catch (err) {
      console.error("Translation failed:", err);
    } finally {
      setIsTranslating(false);
    }
  };

  // References to keep track of columns for sync scroll
  const columnRefs = useRef<{ [panelId: string]: HTMLDivElement | null }>({});
  const activeScrollPanelId = useRef<string | null>(null);

  // Sync scroll listener logic
  const handleScroll = (panelId: string) => {
    if (!globalConfig.syncScroll) return;
    if (activeScrollPanelId.current !== panelId) return;

    const sourceElement = columnRefs.current[panelId];
    if (!sourceElement) return;

    const scrollTop = sourceElement.scrollTop;

    // Apply exact scrollTop to all other visible columns
    panels.forEach((p) => {
      if (p.id === panelId) return;
      const targetElement = columnRefs.current[p.id];
      if (targetElement) {
        targetElement.scrollTop = scrollTop;
      }
    });
  };

  // Run generation for a specific single panel
  const generatePanelOutput = async (panelId: string, currentPanelsState: OutputPanel[]) => {
    const targetPanelIndex = currentPanelsState.findIndex(p => p.id === panelId);
    if (targetPanelIndex === -1) return;

    const targetPanel = currentPanelsState[targetPanelIndex];

    // Reset panel status to streaming
    setPanels(prev => prev.map(p => p.id === panelId ? {
      ...p,
      output: "",
      isStreaming: true,
      isCompleted: false,
      error: undefined
    } : p));

    // Formulate the SSE URL
    const url = `/api/compare/stream?modelId=${targetPanel.modelId}&prompt=${encodeURIComponent(globalConfig.userPrompt)}&systemPrompt=${encodeURIComponent(globalConfig.systemPrompt)}&temperature=${globalConfig.temperature}&maxTokens=${globalConfig.maxTokens}`;

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP status ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      if (!reader) {
        throw new Error("Failed to initialize reader.");
      }

      let done = false;
      let accumulatedOutput = "";
      let chunkBuffer = "";

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        chunkBuffer += decoder.decode(value, { stream: !doneReading });

        const lines = chunkBuffer.split("\n");
        // Maintain trailing incomplete line in buffer
        chunkBuffer = lines.pop() || "";

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (trimmedLine.startsWith("data: ")) {
            const dataStr = trimmedLine.substring(6).trim();
            if (dataStr === "[DONE]") {
              done = true;
              break;
            }

            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.error) {
                throw new Error(parsed.error);
              }
              if (parsed.text) {
                accumulatedOutput += parsed.text;
                setPanels(prev => prev.map(p => p.id === panelId ? {
                  ...p,
                  output: accumulatedOutput
                } : p));
              }
            } catch (err: any) {
              // Ignore partial parsing errors except direct backend throws
              if (err.message && err.message.includes("is not configured")) {
                throw err;
              }
            }
          }
        }
      }

      // Finalize status
      setPanels(prev => prev.map(p => p.id === panelId ? {
        ...p,
        isStreaming: false,
        isCompleted: true
      } : p));

    } catch (error: any) {
      console.error(`Error on panel ${panelId}:`, error);
      setPanels(prev => prev.map(p => p.id === panelId ? {
        ...p,
        isStreaming: false,
        isCompleted: false,
        error: error.message || "Failed to stream output."
      } : p));
    }
  };

  // Submit all active panels concurrently
  const handleCompareAll = () => {
    if (!globalConfig.userPrompt.trim()) return;

    // Trigger all panels parallelly
    panels.forEach((p) => {
      generatePanelOutput(p.id, panels);
    });
  };

  // Add a new comparison panel dynamically
  const handleAddPanel = (modelId?: string) => {
    if (panels.length >= maxAllowedModels) {
      showToast(lang === 'en' ? `Model limit reached (${panels.length}/${maxAllowedModels}).` : `已達可用模型數量上限 (${panels.length}/${maxAllowedModels})。`);
      return;
    }

    const nextId = `panel-${Date.now()}`;
    const unusedModel = models.find(m => !panels.some(p => p.modelId === m.id));
    const selectedModelId = modelId || unusedModel?.id || models[0]?.id || "gemini-3.5-flash";

    setPanels(prev => [...prev, {
      id: nextId,
      modelId: selectedModelId,
      output: "",
      isStreaming: false,
      isCompleted: false
    }]);

    showToast(lang === 'en' ? "Added new model window" : "已新增模型對比視窗");
  };

  // Remove a comparison panel
  const handleRemovePanel = (id: string) => {
    setPanels(prev => {
      const next = prev.filter(p => p.id !== id);
      if (baselinePanelId === id) {
        setBaselinePanelId(next.length > 0 ? next[0].id : null);
      }
      return next;
    });
  };

  // Toggle or designate baseline model
  const toggleBaseline = (panelId: string) => {
    const targetPanel = panels.find(p => p.id === panelId);
    const modelName = models.find(m => m.id === targetPanel?.modelId)?.name || targetPanel?.modelId || "Model";

    if (baselinePanelId === panelId) {
      setBaselinePanelId(null);
      showToast(lang === 'en' ? `Unpinned ${modelName} baseline` : `已取消 ${modelName} 基準固定`);
    } else {
      setBaselinePanelId(panelId);
      showToast(lang === 'en' ? `Pinned ${modelName} as Baseline on the left` : `已將 ${modelName} 固定為左側基準模型`);
    }
  };

  // Model switch of an individual panel
  const handlePanelModelChange = (id: string, newModelId: string) => {
    setPanels(prev => prev.map(p => p.id === id ? {
      ...p,
      modelId: newModelId,
      output: "",
      isCompleted: false,
      error: undefined
    } : p));
  };

  // Apply template values directly to input fields
  const handleApplyTemplate = (tpl: { system: string; user: string }) => {
    setGlobalConfig(prev => ({
      ...prev,
      systemPrompt: tpl.system,
      userPrompt: tpl.user
    }));
  };

  // Copy helper
  const copyTextToClipboard = (text: string, stateSetter: () => void) => {
    navigator.clipboard.writeText(text).then(() => {
      stateSetter();
    });
  };

  const handleCopyAll = () => {
    let combined = `=== SYSTEM PROMPT ===\n${globalConfig.systemPrompt}\n\n=== USER PROMPT ===\n${globalConfig.userPrompt}\n\n`;
    panels.forEach(p => {
      const modelName = models.find(m => m.id === p.modelId)?.name || p.modelId;
      combined += `\n========================================\n[MODEL OUTPUT: ${modelName}]\n========================================\n${p.output}\n`;
    });
    copyTextToClipboard(combined, () => {
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 2000);
    });
  };

  // Dynamic Theme Styling Clones
  const workspaceTitleClass = isDark ? "text-white" : "text-slate-900 font-extrabold";
  const headerBgClass = isDark ? "bg-[#0d0d0d] border-white/10" : "bg-white border-slate-300 shadow-sm shadow-slate-900/5";
  const panelBgClass = isDark ? "bg-[#0d0d0d] border-white/10" : "bg-white border-slate-300 shadow-sm shadow-slate-900/5";
  const panelContentBgClass = isDark ? "bg-slate-950/20" : "bg-slate-50/70";
  const textareaBgClass = isDark ? "bg-black/50 border-white/10 text-slate-100 placeholder-slate-700 focus:border-indigo-500/50" : "bg-white border-slate-300 text-slate-900 placeholder-slate-400 focus:border-indigo-500 shadow-sm";
  const selectBgClass = isDark ? "bg-[#0d0d0d] text-slate-200" : "bg-white text-slate-800";
  const settingsSidebarBg = isDark ? "bg-[#0d0d0d] border-white/10" : "bg-white border-slate-300 shadow-2xl";
  const settingsInputBg = isDark ? "bg-black/50 border-white/10 text-slate-200" : "bg-slate-100 border-slate-300 text-slate-900";

  // Reusable Single Panel Card Renderer
  const renderPanelCard = (panel: OutputPanel, isBaseline: boolean, customWidthClass?: string) => {
    const associatedModel = models.find(m => m.id === panel.modelId);
    const hasOutput = panel.output !== "";
    const widthClass = customWidthClass || (panels.length === 1 ? "w-full" : "w-full md:w-[480px] shrink-0 snap-center");

    return (
      <div
        key={panel.id}
        className={`${widthClass} flex flex-col overflow-hidden h-full min-h-[280px] transition-all rounded-xl border ${
          isBaseline 
            ? (isDark ? "bg-[#110f0b] border-amber-500/35 ring-1 ring-amber-500/20 shadow-lg shadow-amber-500/5" : "bg-white border-amber-400/80 shadow-md shadow-amber-500/5 ring-1 ring-amber-400/25")
            : panelBgClass
        } ${
          panel.isStreaming 
            ? "border-indigo-500/40 ring-1 ring-indigo-500/20" 
            : ""
        }`}
      >
        {/* Panel Header */}
        <div className={`p-3 border-b border-inherit ${isBaseline ? (isDark ? "bg-amber-500/5" : "bg-amber-50/50") : "bg-inherit"} flex items-center justify-between gap-1 select-none shrink-0`}>
          
          {/* Beautiful Custom Dropdown Selection */}
          <div className="flex-1 min-w-0 relative">
            <div className="flex items-center gap-1.5 flex-wrap">
              <button
                onClick={() => setOpenDropdownId(openDropdownId === panel.id ? null : panel.id)}
                className={`inline-flex items-center text-left text-xs font-bold font-display focus:outline-none transition-all truncate gap-1 cursor-pointer hover:text-indigo-500 ${
                  isDark ? "text-slate-100" : "text-slate-900"
                }`}
                id={`dropdown-btn-${panel.id}`}
              >
                <span className="truncate max-w-[140px] sm:max-w-[200px]">{associatedModel?.name || panel.modelId}</span>
                <ChevronDown className="w-3.5 h-3.5 opacity-60 flex-shrink-0" />
              </button>
              
              {isBaseline && (
                <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[9px] font-bold font-mono uppercase bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30">
                  <Pin className="w-2.5 h-2.5 fill-current rotate-45" />
                  {lang === 'en' ? "Baseline" : "基準模型"}
                </span>
              )}
            </div>
            
            <div className={`text-[9px] font-mono uppercase tracking-wide mt-0.5 ${isDark ? "text-slate-500" : "text-slate-700 font-medium"}`}>
              {associatedModel?.provider || "Cloud LLM"}
            </div>

            {/* Custom styled dropdown overlay menu */}
            {openDropdownId === panel.id && (
              <>
                <div 
                  className="fixed inset-0 z-40" 
                  onClick={() => setOpenDropdownId(null)} 
                />
                <div 
                  className={`absolute left-0 mt-1.5 w-64 rounded-xl border max-h-60 overflow-y-auto shadow-2xl z-50 py-1 sandbox-scrollbar transition-all animate-fade-in ${
                    isDark 
                      ? "bg-[#0f0f0f] border-slate-800 text-slate-200" 
                      : "bg-white border-slate-200 text-slate-900 shadow-indigo-100/40"
                  }`}
                >
                  {models.map((m) => {
                    const isSelected = m.id === panel.modelId;
                    return (
                      <button
                        key={m.id}
                        onClick={() => {
                          handlePanelModelChange(panel.id, m.id);
                          setOpenDropdownId(null);
                        }}
                        className={`w-full text-left px-3.5 py-2 text-xs font-sans flex items-center justify-between transition-colors cursor-pointer ${
                          isSelected
                            ? "bg-indigo-600/10 text-indigo-500 font-semibold"
                            : isDark
                              ? "hover:bg-white/5 text-slate-300"
                              : "hover:bg-slate-50 text-slate-800"
                        }`}
                      >
                        <div className="flex flex-col">
                          <span className="font-semibold">{m.name}</span>
                          <span className={`text-[10px] font-mono ${isSelected ? "text-indigo-400" : "opacity-60"}`}>{m.provider}</span>
                        </div>
                        {isSelected && <Check className="w-3.5 h-3.5 text-indigo-500" />}
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </div>

          {/* Panel status icons / controls */}
          <div className="flex items-center gap-1 flex-shrink-0">
            
            {/* Baseline Model Pin Button */}
            <button
              onClick={() => toggleBaseline(panel.id)}
              title={isBaseline 
                ? (lang === 'en' ? "Baseline Model (Click to Unpin)" : "基準模型 (點擊取消固定)") 
                : (lang === 'en' ? "Pin as Baseline Model (Fixed on Left)" : "設為基準模型 (固定於左側)")
              }
              className={`p-1.5 rounded-lg transition-all cursor-pointer ${
                isBaseline
                  ? "text-amber-500 bg-amber-500/15 hover:bg-amber-500/25"
                  : (isDark ? "text-slate-500 hover:text-amber-400 hover:bg-white/5" : "text-slate-400 hover:text-amber-600 hover:bg-slate-100")
              }`}
            >
              <Pin className={`w-3.5 h-3.5 ${isBaseline ? "fill-amber-500 rotate-45" : ""}`} />
            </button>

            {/* Copy button for this column */}
            {hasOutput && (
              <button
                onClick={() => copyTextToClipboard(panel.output, () => {
                  setCopiedPanelId(panel.id);
                  setTimeout(() => setCopiedPanelId(null), 2000);
                })}
                title={lang === 'en' ? "Copy Output" : "複製輸出"}
                className="p-1.5 rounded-lg text-slate-500 hover:text-indigo-500 hover:bg-slate-100 dark:hover:bg-slate-800/60 transition-colors cursor-pointer"
              >
                {copiedPanelId === panel.id ? (
                  <Check className="w-3.5 h-3.5 text-emerald-500" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            )}

            {/* Regenerate this panel only */}
            <button
              onClick={() => generatePanelOutput(panel.id, panels)}
              disabled={!globalConfig.userPrompt.trim() || panel.isStreaming}
              title={lang === 'en' ? "Regenerate" : "重新生成"}
              className="p-1.5 rounded-lg text-slate-500 hover:text-indigo-500 hover:bg-slate-100 dark:hover:bg-slate-800/60 transition-colors disabled:opacity-30 disabled:hover:bg-transparent cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${panel.isStreaming ? "animate-spin text-indigo-500" : ""}`} />
            </button>

            {/* Remove this panel */}
            <button
              onClick={() => handleRemovePanel(panel.id)}
              title={lang === 'en' ? "Close panel" : "關閉視窗"}
              className="p-1.5 rounded-lg text-slate-500 hover:text-rose-500 hover:bg-slate-100 dark:hover:bg-slate-800/60 transition-colors cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>

        </div>

        {/* Output Scroll Content Block */}
        <div
          ref={(el) => {
            columnRefs.current[panel.id] = el;
          }}
          onScroll={() => handleScroll(panel.id)}
          onMouseEnter={() => {
            activeScrollPanelId.current = panel.id;
          }}
          className={`flex-1 overflow-y-auto p-4 text-xs leading-relaxed space-y-3 sandbox-scrollbar ${
            isBaseline 
              ? (isDark ? "bg-slate-950/40" : "bg-amber-50/10") 
              : panelContentBgClass
          }`}
        >
          {/* Loading status representation */}
          {panel.isStreaming && !hasOutput && (
            <div className="flex items-center gap-2 text-slate-500 font-mono py-2 text-[10px]">
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
              </span>
              <span>{lang === 'en' ? "Connecting to stream endpoint..." : "正在初始化連線..."}</span>
            </div>
          )}

          {/* Error container */}
          {panel.error && (
            <div className="p-3.5 rounded-xl bg-rose-500/5 border border-rose-500/10 text-rose-500 text-xs">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-rose-600 dark:text-rose-400">{lang === 'en' ? "Connection Failed" : "連線失敗"}</p>
                  <p className="font-light text-[11px] mt-1 leading-normal">{panel.error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Display Markdown parsed results */}
          {hasOutput && (
            <MarkdownRenderer content={panel.output} />
          )}

          {/* Blank state */}
          {!hasOutput && !panel.isStreaming && !panel.error && (
            <div className="h-full flex flex-col items-center justify-center text-center py-20 px-4">
              <HelpCircle className="w-7 h-7 text-slate-400 mb-2 animate-pulse" />
              <p className="text-xs text-slate-500 font-light">
                {lang === 'en' ? "Input prompt below and hit send arrow" : "在下方輸入問題並點選發送圖示"}
              </p>
            </div>
          )}
        </div>

        {/* Small visual stream info footer */}
        {hasOutput && (
          <div className="px-3.5 py-2.5 border-t border-inherit bg-inherit text-[9px] font-mono text-slate-500 flex justify-between items-center shrink-0">
            <span className="capitalize">{panel.isStreaming ? (lang === 'en' ? "● STREAMING" : "● 串流中") : (lang === 'en' ? "● COMPLETED" : "● 讀取完畢")}</span>
            <span>{panel.output.length} chars</span>
          </div>
        )}

      </div>
    );
  };

  // Split panels based on baseline designation
  const activeBaselinePanel = panels.find(p => p.id === baselinePanelId);
  const comparisonPanels = panels.filter(p => p.id !== baselinePanelId);

  return (
    <div className="py-1 select-none relative h-[calc(100vh-5rem)] flex flex-col justify-between overflow-hidden">
      
      {/* Toast Feedback Notification */}
      {notificationToast && (
        <div className="fixed top-20 left-1/2 -translate-x-1/2 z-50 px-4 py-2.5 rounded-xl bg-slate-900 dark:bg-indigo-600 text-white text-xs font-semibold shadow-xl shadow-black/20 flex items-center gap-2 border border-slate-700 dark:border-indigo-500 animate-fade-in">
          <Sparkles className="w-4 h-4 text-amber-400 shrink-0" />
          <span>{notificationToast}</span>
        </div>
      )}

      {/* Top Header Workspace Status Bar */}
      <div className="flex items-center justify-between mb-3 flex-wrap gap-4 px-1 shrink-0">
        <div>
          <h1 className={`text-base sm:text-lg font-bold flex items-center gap-2 font-display ${workspaceTitleClass}`}>
            <span>{lang === 'en' ? "Parallel Sandbox" : "多模型平行沙盒"}</span>
          </h1>
        </div>

        {/* Workspace global parameters */}
        <div className="flex items-center gap-3">
          
          {/* Synchronous Scrolling Toggler Switch */}
          <div className="flex items-center gap-2">
            <span className={`text-xs font-semibold ${isDark ? "text-slate-400" : "text-slate-600"}`}>
              {lang === 'en' ? "Sync Scroll" : "同步捲動"}
            </span>
            <button
              onClick={() => setGlobalConfig(prev => ({ ...prev, syncScroll: !prev.syncScroll }))}
              className={`relative inline-flex h-5 w-10 items-center rounded-full transition-colors duration-300 focus:outline-none cursor-pointer ${
                globalConfig.syncScroll ? "bg-indigo-600" : (isDark ? "bg-slate-800" : "bg-slate-200")
              }`}
              id="sync-scroll-switch"
              title={lang === 'en' ? "Toggle Synchronized Scroll" : "切換同步滾動"}
            >
              <span
                className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform duration-300 ${
                  globalConfig.syncScroll ? "translate-x-5.5" : "translate-x-1"
                }`}
              />
            </button>
          </div>

          {/* Add Model Panel directly in header - with Lock icon when limit is reached */}
          <button
            onClick={() => handleAddPanel()}
            disabled={isMaxModelsReached}
            title={
              isMaxModelsReached
                ? (lang === 'en' ? `Maximum model limit reached (${panels.length}/${maxAllowedModels})` : `已達可用模型數量上限 (${panels.length}/${maxAllowedModels})`)
                : (lang === 'en' ? "Add Model" : "新增模型")
            }
            className={`px-2.5 py-1.5 rounded-lg border text-xs font-semibold flex items-center gap-1.5 transition-all ${
              isMaxModelsReached
                ? (isDark 
                    ? "border-white/5 bg-white/5 text-slate-500 cursor-not-allowed opacity-60" 
                    : "border-slate-200 bg-slate-100 text-slate-400 cursor-not-allowed opacity-60")
                : "border-indigo-500/20 bg-indigo-500/5 hover:bg-indigo-500/15 text-indigo-500 dark:text-indigo-400 cursor-pointer shadow-sm"
            }`}
            id="header-add-model-btn"
          >
            {isMaxModelsReached ? (
              <Lock className="w-3.5 h-3.5" />
            ) : (
              <Plus className="w-3.5 h-3.5" />
            )}
            <span>{lang === 'en' ? "Add Model" : "新增模型"}</span>
          </button>

          {/* Clean all outputs */}
          <button
            onClick={() => setPanels(prev => prev.map(p => ({ ...p, output: "", isCompleted: false, error: undefined })))}
            className={`px-2.5 py-1.5 rounded-lg border flex items-center gap-1.5 text-xs font-semibold transition-all cursor-pointer ${
              isDark 
                ? "border-white/10 hover:border-white/20 bg-black/40 text-slate-400 hover:text-slate-200" 
                : "border-slate-200 hover:border-slate-300 bg-white text-slate-600 hover:text-slate-800 shadow-sm"
            }`}
            id="header-clear-output-btn"
          >
            <Trash2 className="w-3.5 h-3.5 text-rose-500" />
            <span>{lang === 'en' ? "Clear" : "清除"}</span>
          </button>

          {/* Settings Drawer Button */}
          <button
            onClick={() => setShowSettings(true)}
            className={`px-2.5 py-1.5 rounded-lg border flex items-center gap-1.5 text-xs font-semibold transition-all cursor-pointer ${
              isDark 
                ? "border-white/10 hover:border-white/20 bg-black/40 text-slate-400 hover:text-slate-200" 
                : "border-slate-200 hover:border-slate-300 bg-white text-slate-600 hover:text-slate-800 shadow-sm"
            }`}
            id="header-settings-toggle-btn"
          >
            <SlidersHorizontal className="w-3.5 h-3.5 text-indigo-500" />
            <span>{lang === 'en' ? "Settings" : "參數設定"}</span>
          </button>
        </div>
      </div>

      {/* Multi-Panel Workspace - Supports Left Pinned Baseline + Right Scrollable Comparison Stream */}
      <div className="relative flex-1 min-h-0 mb-3 pr-0.5 overflow-hidden">
        {panels.length > 0 ? (
          activeBaselinePanel && comparisonPanels.length > 0 ? (
            /* Split layout: Fixed Baseline on Left + Carousel on Right */
            <div className="w-full h-full flex flex-col md:flex-row gap-3 overflow-hidden items-stretch" id="panels-baseline-workspace">
              
              {/* Left Sticky/Fixed Baseline Model Panel */}
              <div className="w-full md:w-[460px] lg:w-[480px] shrink-0 h-[calc(100vh-21rem)] md:h-[calc(100vh-22rem)] min-h-[280px]">
                {renderPanelCard(activeBaselinePanel, true, "w-full h-full")}
              </div>

              {/* Subtle visual divider between baseline and comparison models */}
              <div className="hidden md:flex items-center justify-center shrink-0">
                <div className="h-[90%] w-px bg-slate-200 dark:bg-white/10" />
              </div>

              {/* Right Scrollable Comparison Models Carousel */}
              <div className="flex-1 min-w-0 h-[calc(100vh-21rem)] md:h-[calc(100vh-22rem)] min-h-[280px] relative group/carousel flex items-center">
                
                {/* Floating Left Scroll Button for right carousel */}
                {comparisonPanels.length > 1 && (
                  <button
                    onClick={() => scrollCarousel('left')}
                    className={`absolute left-2 z-30 p-2 rounded-full border shadow-lg backdrop-blur-md transition-all opacity-0 group-hover/carousel:opacity-100 hidden md:flex items-center justify-center cursor-pointer hover:scale-110 active:scale-95 ${
                      isDark 
                        ? "bg-slate-900/95 border-slate-800 text-slate-300 hover:text-white hover:bg-slate-800" 
                        : "bg-white border-slate-200 text-slate-800 hover:text-slate-950 hover:bg-slate-50 shadow-sm"
                    }`}
                    title={lang === 'en' ? "Scroll Left" : "向左滾動"}
                  >
                    <ChevronLeft className="w-5 h-5" />
                  </button>
                )}

                {/* Floating Right Scroll Button for right carousel */}
                {comparisonPanels.length > 1 && (
                  <button
                    onClick={() => scrollCarousel('right')}
                    className={`absolute right-2 z-30 p-2 rounded-full border shadow-lg backdrop-blur-md transition-all opacity-0 group-hover/carousel:opacity-100 hidden md:flex items-center justify-center cursor-pointer hover:scale-110 active:scale-95 ${
                      isDark 
                        ? "bg-slate-900/95 border-slate-800 text-slate-300 hover:text-white hover:bg-slate-800" 
                        : "bg-white border-slate-200 text-slate-800 hover:text-slate-950 hover:bg-slate-50 shadow-sm"
                    }`}
                    title={lang === 'en' ? "Scroll Right" : "向右滾動"}
                  >
                    <ChevronRight className="w-5 h-5" />
                  </button>
                )}

                {/* Scrollable Container */}
                <div 
                  ref={carouselScrollRef}
                  className="w-full h-full flex gap-4 overflow-x-auto scroll-smooth snap-x snap-mandatory sandbox-scrollbar pb-1"
                  id="panels-carousel-container"
                >
                  {comparisonPanels.map((panel) => 
                    renderPanelCard(panel, false, comparisonPanels.length === 1 ? "w-full md:w-[480px] shrink-0 snap-center h-full" : "w-full md:w-[480px] shrink-0 snap-center h-full")
                  )}
                </div>

              </div>

            </div>
          ) : (
            /* Single unified carousel when no baseline or only 1 panel */
            <div className="relative w-full h-full group/carousel flex items-center">
              {panels.length > 1 && (
                <button
                  onClick={() => scrollCarousel('left')}
                  className={`absolute left-2 z-30 p-2 rounded-full border shadow-lg backdrop-blur-md transition-all opacity-0 group-hover/carousel:opacity-100 hidden md:flex items-center justify-center cursor-pointer hover:scale-110 active:scale-95 ${
                    isDark 
                      ? "bg-slate-900/95 border-slate-800 text-slate-300 hover:text-white hover:bg-slate-800" 
                      : "bg-white border-slate-200 text-slate-800 hover:text-slate-950 hover:bg-slate-50 shadow-sm"
                  }`}
                  title={lang === 'en' ? "Scroll Left" : "向左滾動"}
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
              )}

              {panels.length > 1 && (
                <button
                  onClick={() => scrollCarousel('right')}
                  className={`absolute right-2 z-30 p-2 rounded-full border shadow-lg backdrop-blur-md transition-all opacity-0 group-hover/carousel:opacity-100 hidden md:flex items-center justify-center cursor-pointer hover:scale-110 active:scale-95 ${
                    isDark 
                      ? "bg-slate-900/95 border-slate-800 text-slate-300 hover:text-white hover:bg-slate-800" 
                      : "bg-white border-slate-200 text-slate-800 hover:text-slate-950 hover:bg-slate-50 shadow-sm"
                  }`}
                  title={lang === 'en' ? "Scroll Right" : "向右滾動"}
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
              )}

              <div 
                ref={carouselScrollRef}
                className="w-full h-full flex gap-4 overflow-x-auto scroll-smooth snap-x snap-mandatory sandbox-scrollbar pb-1"
                id="panels-carousel-container"
              >
                {panels.map((panel) => 
                  renderPanelCard(panel, panel.id === baselinePanelId, panels.length === 1 ? "w-full h-[calc(100vh-21rem)] md:h-[calc(100vh-22rem)]" : "w-full md:w-[480px] shrink-0 snap-center h-[calc(100vh-21rem)] md:h-[calc(100vh-22rem)]")
                )}
              </div>
            </div>
          )
        ) : (
          <div className={`text-center py-20 border border-dashed rounded-3xl flex flex-col items-center justify-center h-full ${
            isDark ? "border-white/10 bg-black/20" : "border-slate-200 bg-white shadow-sm"
          }`}>
            <LayoutGrid className="w-8 h-8 text-slate-400 mb-3" />
            <h3 className={`font-semibold text-sm ${isDark ? "text-slate-300" : "text-slate-700"}`}>
              {lang === 'en' ? "Workspace is Empty" : "工作區目前沒有模型視窗"}
            </h3>
            <p className="text-xs text-slate-500 mt-1 font-light mb-4">
              {lang === 'en' ? "Click the button below or settings to add model testing window." : "請點擊下方按鈕或右上新增按鈕來建立模型視窗。"}
            </p>
            <button
              onClick={() => handleAddPanel("gemini-3.5-flash")}
              className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all inline-flex items-center gap-1.5 cursor-pointer shadow-md shadow-indigo-600/20"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>{lang === 'en' ? "Add Gemini 3.5 Flash" : "新增 Gemini 3.5 Flash"}</span>
            </button>
          </div>
        )}
      </div>

      {/* Model Output Vote Comparison Buttons (Centered right below the panels) */}
      {panels.length === 2 && panels.some(p => p.output !== "") && (
        <div className="flex flex-wrap items-center justify-center gap-2 mb-2.5 shrink-0 animate-fade-in px-1">
          <button 
            onClick={() => showToast(lang === 'en' ? "Feedback recorded: Left model response is better!" : "評價已送出：左側模型表現較佳！")}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-full border transition-all flex items-center gap-1 cursor-pointer ${
              isDark ? "bg-[#111] border-white/10 hover:bg-white/5 text-slate-300" : "bg-white border-slate-200 hover:bg-slate-50 text-slate-700 shadow-sm"
            }`}
          >
            <span>← {lang === 'en' ? "Left is Better" : "左側較佳"}</span>
          </button>
          <button 
            onClick={() => showToast(lang === 'en' ? "Feedback recorded: Both models performed well!" : "評價已送出：兩者表現皆佳！")}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-full border transition-all flex items-center gap-1 cursor-pointer ${
              isDark ? "bg-[#111] border-white/10 hover:bg-white/5 text-slate-300" : "bg-white border-slate-200 hover:bg-slate-50 text-slate-700 shadow-sm"
            }`}
          >
            <span>👍 {lang === 'en' ? "Both are Good" : "兩者皆好"}</span>
          </button>
          <button 
            onClick={() => showToast(lang === 'en' ? "Feedback recorded: Both models need improvement!" : "評價已送出：兩者皆需改進！")}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-full border transition-all flex items-center gap-1 cursor-pointer ${
              isDark ? "bg-[#111] border-white/10 hover:bg-white/5 text-slate-300" : "bg-white border-slate-200 hover:bg-slate-50 text-slate-700 shadow-sm"
            }`}
          >
            <span>👎 {lang === 'en' ? "Both are Bad" : "兩者皆差"}</span>
          </button>
          <button 
            onClick={() => showToast(lang === 'en' ? "Feedback recorded: Right model response is better!" : "評價已送出：右側模型表現較佳！")}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-full border transition-all flex items-center gap-1 cursor-pointer ${
              isDark ? "bg-[#111] border-white/10 hover:bg-white/5 text-slate-300" : "bg-white border-slate-200 hover:bg-slate-50 text-slate-700 shadow-sm"
            }`}
          >
            <span>{lang === 'en' ? "Right is Better" : "右側較佳"} →</span>
          </button>
        </div>
      )}

      {/* Highly Prominent Bottom Chatbox Card with Accent Border */}
      <div className="shrink-0 mb-2 w-full max-w-4xl mx-auto px-1" id="prominent-bottom-chatbox">
        <div className={`p-3 rounded-2xl border transition-all duration-300 shadow-xl ${
          isDark 
            ? "bg-[#0b0b0b] border-white/10 focus-within:border-orange-500/80 focus-within:ring-1 focus-within:ring-orange-500/40" 
            : "bg-white border-slate-300 focus-within:border-indigo-600 focus-within:ring-1 focus-within:ring-indigo-600/30 shadow-md shadow-slate-900/5"
        }`}>
          
          {/* Textarea for message entry */}
          <textarea
            value={globalConfig.userPrompt}
            onChange={(e) => setGlobalConfig(prev => ({ ...prev, userPrompt: e.target.value }))}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleCompareAll();
              }
            }}
            placeholder={lang === 'en' ? "Ask followup..." : "輸入您的測試問題或指令..."}
            rows={2}
            className="w-full bg-transparent border-0 resize-none outline-none focus:ring-0 p-1.5 text-sm text-slate-800 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-600 font-sans"
            id="prompt-textarea"
          />

          {/* Toolbar footer inside input card */}
          <div className="flex items-center justify-between border-t border-slate-100 dark:border-white/5 pt-2 mt-1.5">
            
            {/* Left tool Actions: Languages Translator Action */}
            <div className="flex items-center gap-2">
              <button
                onClick={handleTranslatePrompt}
                disabled={isTranslating || !globalConfig.userPrompt.trim()}
                title={lang === 'en' ? "Translate Prompt" : "翻譯提示詞"}
                className={`p-1.5 rounded-lg transition-colors cursor-pointer flex items-center gap-1 ${
                  isDark 
                    ? "text-slate-500 hover:text-orange-400 hover:bg-white/5 disabled:opacity-30" 
                    : "text-slate-500 hover:text-orange-600 hover:bg-slate-50 disabled:opacity-30"
                }`}
              >
                <Languages className={`w-3.5 h-3.5 ${isTranslating ? "animate-spin text-orange-500" : ""}`} />
                <span className="text-[10px] font-medium hidden sm:inline">
                  {isTranslating ? (lang === 'en' ? "Translating..." : "翻譯中...") : (lang === 'en' ? "Translate Prompt" : "一鍵翻譯")}
                </span>
              </button>

              {/* Status Indicator / Limit Badge */}
              {isMaxModelsReached ? (
                <span className="text-[10px] text-amber-600 dark:text-amber-400 font-medium flex items-center gap-1 ml-2 bg-amber-500/10 dark:bg-amber-500/15 border border-amber-500/20 px-2 py-0.5 rounded-full font-sans">
                  <Lock className="w-3 h-3" />
                  <span>{lang === 'en' ? `Max models reached (${panels.length}/${maxAllowedModels})` : `已達模型上限 (${panels.length}/${maxAllowedModels})`}</span>
                </span>
              ) : (
                <span className="text-[10px] text-slate-400 dark:text-slate-600 font-mono hidden md:inline ml-2">
                  {panels.length}/{maxAllowedModels} {lang === 'en' ? "configured windows" : "個視窗已配置"}
                </span>
              )}
            </div>

            {/* Right tool Actions: Mic, Template Dropup, Submit */}
            <div className="flex items-center gap-1.5">
              
              {/* Copy entire report (floating icon) */}
              {panels.some(p => p.output !== "") && (
                <button
                  onClick={handleCopyAll}
                  title={lang === 'en' ? "Copy Entire Comparison Report" : "複製完整對比報告"}
                  className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
                    isDark ? "text-slate-500 hover:text-slate-200 hover:bg-white/5" : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
                  }`}
                >
                  {copiedAll ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              )}

              {/* Voice Speech Microphone */}
              {SpeechRecognition && (
                <button
                  onClick={toggleListening}
                  title={isListening ? (lang === 'en' ? "Stop Listening" : "停止錄音") : (lang === 'en' ? "Start Voice Input" : "啟用語音輸入")}
                  className={`p-1.5 rounded-lg transition-all cursor-pointer ${
                    isListening 
                      ? "bg-rose-500/10 text-rose-500 border border-rose-500/20 animate-pulse" 
                      : (isDark ? "text-slate-500 hover:text-slate-200 hover:bg-white/5" : "text-slate-500 hover:text-slate-800 hover:bg-slate-50")
                  }`}
                >
                  <Mic className="w-3.5 h-3.5" />
                </button>
              )}

              {/* Parameter Settings Trigger Icon */}
              <button
                onClick={() => setShowSettings(true)}
                title={lang === 'en' ? "Model Parameters & Settings" : "模型參數與設定"}
                className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
                  isDark ? "text-slate-500 hover:text-slate-200 hover:bg-white/5" : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
                }`}
              >
                <SlidersHorizontal className="w-3.5 h-3.5" />
              </button>

              {/* Templates dropdown trigger icon / Append helper */}
              <div className="relative group">
                <button
                  title={lang === 'en' ? "Quick Prompt Templates" : "快捷測試範本"}
                  className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
                    isDark ? "text-slate-500 hover:text-slate-200 hover:bg-white/5" : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
                  }`}
                >
                  <Plus className="w-3.5 h-3.5" />
                </button>
                
                {/* Quick templates dropup menu on hover */}
                <div className={`absolute bottom-full right-0 mb-2 w-56 rounded-xl border p-2 z-50 shadow-xl hidden group-hover:block animate-fade-in ${
                  isDark ? "bg-[#0f0f0f] border-white/10 text-slate-300" : "bg-white border-slate-200 text-slate-700"
                }`}>
                  <div className="text-[9px] font-mono font-bold text-slate-400 uppercase tracking-wider px-2 py-1 mb-1 border-b border-slate-100 dark:border-white/5">
                    {lang === 'en' ? "Quick Scenarios" : "選擇測試範本"}
                  </div>
                  {(lang === 'en' ? PROMPT_TEMPLATES_EN : PROMPT_TEMPLATES_ZH).map((tpl, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleApplyTemplate(tpl)}
                      className={`w-full text-left px-2 py-1.5 text-xs rounded-lg truncate block cursor-pointer transition-colors ${
                        isDark ? "hover:bg-white/5 hover:text-white" : "hover:bg-slate-50 hover:text-slate-900"
                      }`}
                    >
                      {tpl.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Submit Comparison Arrow Button */}
              <button
                onClick={handleCompareAll}
                disabled={!globalConfig.userPrompt.trim() || panels.length === 0}
                className="p-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-200/50 dark:disabled:bg-white/5 text-white disabled:text-slate-400 dark:disabled:text-slate-600 disabled:cursor-not-allowed transition-all cursor-pointer flex items-center justify-center"
                id="prompt-compare-submit-btn"
              >
                <Send className="w-3.5 h-3.5" />
              </button>

            </div>

          </div>

        </div>
      </div>

      {/* Redesigned Sliding Drawer for Parameter Settings */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex justify-end">
          {/* Backdrop */}
          <div 
            onClick={() => setShowSettings(false)}
            className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm transition-opacity" 
          />
          
          {/* Sliding Content Container */}
          <div className={`relative w-full max-w-sm sm:max-w-md h-full flex flex-col justify-between border-l z-10 shadow-2xl animate-slide-left ${
            isDark ? "bg-[#0d0d0d] border-white/10 text-slate-200" : "bg-white border-slate-200 text-slate-800"
          }`}>
            
            {/* Drawer Header */}
            <div className={`px-5 py-4 border-b flex items-center justify-between ${
              isDark ? "border-white/10 bg-black/40" : "border-slate-100 bg-slate-50"
            }`}>
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="w-4.5 h-4.5 text-indigo-500" />
                <h3 className={`font-display text-sm sm:text-base font-bold ${isDark ? "text-white" : "text-slate-900"}`}>
                  {lang === 'en' ? "Parameters & Settings" : "沙盒參數與進階設定"}
                </h3>
              </div>
              <button
                onClick={() => setShowSettings(false)}
                className={`p-1.5 rounded-lg text-xs transition-colors cursor-pointer ${
                  isDark ? "text-slate-400 hover:text-white hover:bg-white/5" : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
                }`}
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Scrollable Form Content */}
            <div className="p-5 overflow-y-auto space-y-6 flex-1 sandbox-scrollbar">
              
              {/* System Prompt block */}
              <div className="space-y-2">
                <label className={`block text-xs sm:text-sm font-bold ${isDark ? "text-slate-200" : "text-slate-800"}`}>
                  {lang === 'en' ? "Global System Instruction" : "全域系統提示詞 (System Prompt)"}
                </label>
                <p className="text-[11px] text-slate-500 font-light leading-normal">
                  {lang === 'en' 
                    ? "Defines the persona and constraints for all active comparison models globally." 
                    : "為所有沙盒視窗內運行的 AI 模型設定一致的角色定位與回覆規範。"}
                </p>
                <textarea
                  value={globalConfig.systemPrompt}
                  onChange={(e) => setGlobalConfig(prev => ({ ...prev, systemPrompt: e.target.value }))}
                  placeholder={lang === 'en' ? "e.g., You are a strict software developer advisor..." : "例如：你是一位嚴格的程式碼審查專家..."}
                  rows={4}
                  className={`w-full p-3 rounded-xl border text-xs sm:text-sm font-light resize-none focus:outline-none focus:ring-1 focus:ring-indigo-500 ${
                    isDark ? "bg-black/40 border-white/10 text-white placeholder-slate-700" : "bg-slate-50 border-slate-200 text-slate-800 placeholder-slate-400"
                  }`}
                />
              </div>

              {/* Temperature Slider */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <label className={`text-xs sm:text-sm font-bold ${isDark ? "text-slate-200" : "text-slate-800"}`}>
                    {lang === 'en' ? "Temperature" : "隨機度 (Temperature)"}
                  </label>
                  <span className="text-xs font-mono font-bold text-indigo-500 bg-indigo-500/10 px-2 py-0.5 rounded">
                    {globalConfig.temperature}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 font-light leading-normal">
                  {lang === 'en' 
                    ? "Higher temperature values lead to more creative and diverse responses." 
                    : "數值越高回覆越具創意與隨機性，越低則越精準客觀。"}
                </p>
                <input
                  type="range"
                  min="0.0"
                  max="1.5"
                  step="0.1"
                  value={globalConfig.temperature}
                  onChange={(e) => setGlobalConfig(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                  className={`w-full h-1.5 rounded-lg cursor-pointer outline-none transition-all ${
                    isDark ? "bg-slate-800" : "bg-indigo-100/80"
                  }`}
                />
              </div>

              {/* Max Tokens Slider */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <label className={`text-xs sm:text-sm font-bold ${isDark ? "text-slate-200" : "text-slate-800"}`}>
                    {lang === 'en' ? "Max Output Length" : "單次最大生成長度 (Max Tokens)"}
                  </label>
                  <span className="text-xs font-mono font-bold text-indigo-500 bg-indigo-500/10 px-2 py-0.5 rounded">
                    {globalConfig.maxTokens}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 font-light leading-normal">
                  {lang === 'en' 
                    ? "Caps the response token length to avoid budget waste." 
                    : "限制單一模型輸出的最大長度，防止不必要的超長回覆浪費額度。"}
                </p>
                <input
                  type="range"
                  min="200"
                  max="4000"
                  step="100"
                  value={globalConfig.maxTokens}
                  onChange={(e) => setGlobalConfig(prev => ({ ...prev, maxTokens: parseInt(e.target.value) }))}
                  className={`w-full h-1.5 rounded-lg cursor-pointer outline-none transition-all ${
                    isDark ? "bg-slate-800" : "bg-indigo-100/80"
                  }`}
                />
              </div>

              {/* Multi-Window Limit Info */}
              <div className={`p-4 rounded-xl border border-dashed ${isDark ? "border-white/5 bg-white/2" : "border-slate-300 bg-slate-100/50"}`}>
                <h4 className={`text-xs font-extrabold uppercase tracking-wide mb-1.5 ${isDark ? "text-slate-400" : "text-slate-800"}`}>
                  {lang === 'en' ? "Sandbox Guidelines" : "沙盒架構說明"}
                </h4>
                <ul className={`text-[11px] leading-normal list-disc pl-4 space-y-1 font-medium ${isDark ? "text-slate-300" : "text-slate-800"}`}>
                  <li>{lang === 'en' ? `Supports up to ${maxAllowedModels} concurrent models with horizontal scaling.` : `支援最高 ${maxAllowedModels} 個模型同步平行對決，支援水平自由捲動。`}</li>
                  <li>{lang === 'en' ? "Pin any model with the pin icon to set it as fixed Baseline on the left." : "點擊圖針圖示即可將任意模型設為「基準模型 (Baseline)」，固定於左側對照。"}</li>
                  <li>{lang === 'en' ? "Use Sync Scroll to test long form analysis across all windows." : "啟用「同步捲動」以流暢同步對照所有視窗之長文本推導。"}</li>
                  <li>{lang === 'en' ? "Parameters are applied before launching new streaming requests." : "修改參數後發送新指令，便會自動套用新設定。"}</li>
                </ul>
              </div>

            </div>

            {/* Bottom Actions */}
            <div className={`px-5 py-4 border-t flex justify-end ${
              isDark ? "border-white/10 bg-black/40" : "border-slate-100 bg-slate-50"
            }`}>
              <button
                onClick={() => setShowSettings(false)}
                className="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all shadow-md shadow-indigo-600/10 cursor-pointer"
              >
                {lang === 'en' ? "Apply Parameters & Close" : "儲存設定並關閉"}
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
