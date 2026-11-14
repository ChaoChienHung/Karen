import { useState, useEffect, useRef } from "react";
import { 
  Search, 
  ExternalLink, 
  X, 
  CheckCircle, 
  AlertTriangle, 
  BookOpen, 
  Cpu, 
  Calendar, 
  Star, 
  Filter, 
  ChevronLeft, 
  ChevronRight, 
  Check, 
  Copy, 
  Server,
  HelpCircle
} from "lucide-react";
import { Model } from "../types";

// Define professional expert analysis for each model - English & Traditional Chinese
const EXPERT_ANALYSIS_EN: Record<string, string> = {
  "gemini-3.5-flash": "Key Strengths: Gemini 3.5 Flash represents a massive milestone in native multimodal architecture. Unlike traditional models that pass speech/video into intermediate text representations first, Flash processes audio spectrograms and video frames directly at the architecture level. This guarantees incredibly fast response times and ultra-low costs for real-time video/audio streaming, multi-modal support agents, and high-frequency structured JSON generation.\n\nEvaluation Advice: In the Sandbox, try inputting a huge context folder or code document. Its 1M+ token limit allows you to perform deep analyses with supreme instruction adherence. Its schema-matching score is exceptional for complex multi-agent JSON routing.",
  "gemini-3.1-flash-lite": "Key Strengths: Gemini 3.1 Flash Lite is an ultra-lightweight, extremely cost-efficient model designed for high-frequency edge routing, micro-tasks, and quick semantic processing. It optimizes cost and latency down to the absolute minimum.\n\nEvaluation Advice: Best used as the first classification layer, lightweight text summarizes, or intent router in agentic workflows. While it might fall short in deep algorithmic design, multi-shot prompts easily bring its operational efficiency on par with larger models for routine tasks.",
  "gpt-4o": "Key Strengths: GPT-4o is widely recognized for its robust generic reasoning and excellent resistance to Instruction Drift. It is a golden benchmark for complex prompts with cross-cutting constraints, providing exceptionally smooth literary translations and consistent tone adherence.\n\nEvaluation Advice: Use GPT-4o for complex system rules. Set a 1000-word instruction set in the Sandbox containing various formatting, length, and content constraints to witness its superior multi-instruction alignment.",
  "claude-3-5-sonnet": "Key Strengths: Claude 3.5 Sonnet is a developer favorite. Its tone is naturally academic, free of AI clichés, and highly professional. Its coding capability, software refactoring performance, and systemic bug detection scores are second to none.\n\nEvaluation Advice: When refactoring legacy codebases, feed the messy snippet into Sonnet. Set the System Prompt to 'strict, academic, highly critical software engineer' to get exceptionally clean, modular output with minimal boilerplate.",
  "llama-3-1-405b": "Key Strengths: Llama 3.1 405B is a monumental achievement in open-source AI. Hosting 405 billion parameters, it rivals leading commercial proprietary models in cross-disciplinary knowledge, reasoning, and synthetic data generation capability.\n\nEvaluation Advice: Llama 405B excels at highly complex localized translations and structured factual recall. Use it in the Sandbox to verify accuracy across complex historical or specialized domains.",
  "deepseek-r1": "Key Strengths: DeepSeek R1 broke ground with massive reinforcement learning, achieving human-level reasoning, self-correction, and algorithmic code generation. It excels at complex engineering, high-level algorithms, and advanced mathematics.\n\nEvaluation Advice: R1 relies heavily on its <thought> tags. Do not limit its Max Tokens in the Sandbox, as complex logic requires ample generation space. Give it a LeetCode Hard problem to watch it systematically debug and correct itself in real time."
};

const EXPERT_ANALYSIS_ZH: Record<string, string> = {
  "gemini-3.5-flash": "優勢深度解讀：Gemini 3.5 Flash 代表了端到端原生多模態設計的最新里程碑。傳統模型常將語音、影片先轉成文字再輸入，而 Flash 直接在底層處理音訊特徵與影片幀。這使得它在處理即時影音串流、多模態客服、高頻結構化 JSON 輸出時擁有不可思議的極速體驗與低成本效益。\n\n實戰調優建議：在沙盒測試中，強烈推薦充分利用其極大的 Context Window 進行長文本代碼庫的分析。配合結構化 JSON 指令，它對 Schema 的遵循率極高，非常適合充當複雜 Agent 工作流中的快速決策路由器與高難度語意提取器。",
  "gemini-3.1-flash-lite": "優勢深度解讀：Gemini 3.1 Flash Lite 是一款極致輕量化且具有劃時代高性價比的多模態模型。它本質上是為微型任務、超高頻次調用以及邊緣網關設計的，幾乎將 API 的響應延遲和調用成本壓縮到了極限。\n\n實戰調優建議：此模型適合進行大量文本的初級分類、輕量化摘要、或者作為大型 Agent 工作流中的第一層意圖識別（Intent Router）。儘管它在代碼生成和科學推理上不如大參數量模型，但配合少樣本提示詞，它能在日常任務中展現出驚人的靈活性與可靠性。",
  "gpt-4o": "優勢深度解讀：GPT-4o 是目前全球公認通用性最強、對抗指令漂移（Instruction Drift）最為出色的旗艦模型。其最震撼的特點在於其在複雜、多約束的 Prompt 下依然能展現出極致的跟隨度。無論是用於複雜的情感陪伴、長文案潤色，還是各學科綜合推理，它都能提供無懈可擊、文風高雅的文字饗宴。\n\n實戰調優建議：在進行高難度系統指令（System Prompt）約束時，GPT-4o 是黃金標杆。你可以嘗試在沙盒中給它添加長達千字的複雜限制條件（如字數限制、禁忌詞、格式對齊），它在多重指令交叉約束下的完成度是其他模型難以企及的。",
  "claude-3-5-sonnet": "優勢深度解讀：Claude 3.5 Sonnet 在全球工程師群體中擁有神話般的地位。它的文字表述擺脫了 AI 常見的陳腔鑼鼓，具有極高的學術品味與優雅格調；其在代碼重構、邏輯漏洞掃描、多步驟自動化 Agent 工作流中的表現甚至超越了更重型的模型，是完美的生產力工具。\n\n實戰調優建議：當你需要對一段混亂的遺留代碼進行重構，或者希望撰寫極具說服力的商業分析報告時，Claude 3.5 Sonnet 是不二之選。在沙盒中，推薦將 System Prompt 設定為『嚴謹、客觀、高難度學術寫作』，它輸出的精準度與架構完整性會讓你大為驚艷。",
  "llama-3-1-405b": "優勢深度解讀：Llama 3.1 405B 是開源史上的不朽豐碑。作為一款 4050 億參數的龐然大物，它不僅證明了開源模型在通用知識與複雜推理上能完全叫板頂級閉源商業模型，更為全球開發者提供了一個無可比擬的合成數據生成與模型蒸餾源泉。\n\n實戰調優建議：這款模型在跨學科綜合知識問答、多語言本地化翻譯上表現極其紮實。如果你希望在私有化部署前預估模型能力，或者需要生成乾淨、無偏誤的訓練集數據，推薦在沙盒中使用 405B 進行批量生成測試，它的客觀事實準確率高得令人髮指。",
  "deepseek-r1": "優勢深度解讀：DeepSeek R1 徹底打破了西方模型在推理領域的壟斷。通過大規模強化學習，R1 具備了強大且驚人的自我修正與深度反思能力。其代碼編寫與高等數學推理能力，更是直接登頂全球評測排行榜，是科技與工程領域的終極解題利器。\n\n實戰調優建議：在沙盒測試中，您**必須**給予它充足的空間去展現其 <thought> 思維鏈。請給它出一道高難度算法題或數學謎題，觀察它如何一步步排查錯誤。切記不要人為限制其 Max Tokens，因為極致的推理需要龐大的思考 Token 作為支撐，讓它暢所欲言，你將見證真正的機器智慧。"
};

// Unified function to get enriched academic benchmarks
const getEnrichedBenchmarks = (modelId: string, baseBenchmarks: any, lang: 'en' | 'zh') => {
  const list = [
    { 
      key: "MMLU", 
      name: "MMLU", 
      value: baseBenchmarks.MMLU, 
      label: lang === 'en' ? "MMLU Academic Comprehension" : "MMLU 學術綜合測試", 
      desc: lang === 'en' ? "Multi-subject general understanding & academia" : "跨學科、多學科常識和理解能力" 
    },
    { 
      key: "HumanEval", 
      name: "HumanEval", 
      value: baseBenchmarks.HumanEval, 
      label: lang === 'en' ? "HumanEval Code Synthesis" : "HumanEval 代碼編寫", 
      desc: lang === 'en' ? "Python programming logic accuracy & implementation" : "Python 函數生成的正確率與編程邏輯" 
    },
    { 
      key: "MATH", 
      name: "MATH", 
      value: baseBenchmarks.MATH, 
      label: lang === 'en' ? "MATH Multi-Step Reasoning" : "MATH 高等數學推理", 
      desc: lang === 'en' ? "Multi-step mathematical proof & logic problems" : "多步代數、幾何和高等數學解題能力" 
    },
    { 
      key: "GPQA", 
      name: "GPQA", 
      value: baseBenchmarks.GPQA, 
      label: lang === 'en' ? "GPQA Expert Q&A" : "GPQA 專家級問答", 
      desc: lang === 'en' ? "Graduate-level science & engineering problem solving" : "超高難度的科學與工程學推理" 
    }
  ];

  let gsm8k = 90;
  let ifeval = 85;
  let mbpp = 88;

  if (modelId === "gemini-3.5-flash") {
    gsm8k = 92.4; ifeval = 84.6; mbpp = 86.2;
  } else if (modelId === "gemini-3.1-flash-lite") {
    gsm8k = 86.5; ifeval = 78.2; mbpp = 80.5;
  } else if (modelId === "gpt-4o") {
    gsm8k = 95.8; ifeval = 87.5; mbpp = 89.1;
  } else if (modelId === "claude-3-5-sonnet") {
    gsm8k = 96.4; ifeval = 89.2; mbpp = 91.5;
  } else if (modelId === "llama-3-1-405b") {
    gsm8k = 96.8; ifeval = 88.3; mbpp = 89.7;
  } else if (modelId === "deepseek-r1") {
    gsm8k = 96.3; ifeval = 89.9; mbpp = 93.5;
  }

  list.push({ 
    key: "GSM8K", 
    name: "GSM8K", 
    value: gsm8k, 
    label: lang === 'en' ? "GSM8K Mathematical Reasoning" : "GSM8K 數學應用", 
    desc: lang === 'en' ? "Multi-step grade-school level math reasoning" : "多步驟中學級別數學思維評估" 
  });
  list.push({ 
    key: "IFEval", 
    name: "IFEval", 
    value: ifeval, 
    label: lang === 'en' ? "IFEval Instruction Adherence" : "IFEval 指令遵循率", 
    desc: lang === 'en' ? "Formatting and structural constraint adherence" : "對排版、格式等多約束的跟隨度" 
  });
  list.push({ 
    key: "MBPP", 
    name: "MBPP", 
    value: mbpp, 
    label: lang === 'en' ? "MBPP Python Implementation" : "MBPP Python編程", 
    desc: lang === 'en' ? "Python algorithm coding accuracy" : "解決通用編程問題代碼實現度" 
  });

  return list;
};

interface ModelHubProps {
  models: Model[];
  lang: 'en' | 'zh';
  theme: 'dark' | 'light';
}

export default function ModelHub({ models, lang, theme }: ModelHubProps) {
  const isDark = theme === 'dark';
  
  // Refactored Offline Bridge into a modal trigger rather than a secondary tab
  const [showBridgeModal, setShowBridgeModal] = useState(false);

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedProvider, setSelectedProvider] = useState<string>("all");
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  
  // Code Copy State
  const [copiedCode, setCopiedCode] = useState(false);

  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  const checkScrollLimits = () => {
    if (scrollRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current;
      setCanScrollLeft(scrollLeft > 5);
      setCanScrollRight(scrollLeft + clientWidth < scrollWidth - 5);
    }
  };

  const scrollLeft = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollBy({ left: -220, behavior: "smooth" });
    }
  };

  const scrollRight = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollBy({ left: 220, behavior: "smooth" });
    }
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.addEventListener("scroll", checkScrollLimits);
      const timer = setTimeout(checkScrollLimits, 150);
      return () => {
        el.removeEventListener("scroll", checkScrollLimits);
        clearTimeout(timer);
      };
    }
  }, [selectedModel]);

  // Filter models based on search query and selected provider
  const filteredModels = models.filter((model) => {
    const matchesSearch = 
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.description.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesProvider = 
      selectedProvider === "all" || 
      model.provider.toLowerCase() === selectedProvider.toLowerCase();

    return matchesSearch && matchesProvider;
  });

  const providers = ["all", "Karen", "Google", "OpenAI", "Anthropic", "Meta", "DeepSeek"];

  // Helper to color provider badges
  const getProviderColor = (provider: string) => {
    switch (provider.toLowerCase()) {
      case "karen": return "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20";
      case "self-hosted": return "bg-teal-500/10 text-teal-600 dark:text-teal-400 border-teal-500/20";
      case "google": return "bg-blue-500/10 text-blue-500 dark:text-blue-400 border-blue-500/20";
      case "openai": return "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20";
      case "anthropic": return "bg-orange-500/10 text-orange-600 dark:text-orange-400 border-orange-500/20";
      case "meta": return "bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/20";
      case "deepseek": return "bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border-indigo-500/20";
      default: return "bg-slate-500/10 text-slate-500 dark:text-slate-400 border-slate-500/20";
    }
  };

  // Python Code for Local Python Bridge Guide
  const pythonBridgeCode = `import asyncio
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

app = FastAPI(title="Karen Offline Local Model Bridge")

# Enable CORS for side-by-side frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the local offline model from Hugging Face / Cache
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading local model: {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

@app.get("/stream")
async def stream_generation(prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 1024):
    
    # Construct unified conversational templates
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
    )
    
    # Spawn model generator in worker thread to prevent ASGI async lockups
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    async def event_generator():
        for new_text in streamer:
            if new_text:
                # Standard Server-Sent Events (SSE) stream chunk compatible with Karen Sandbox
                yield f"data: {json.dumps({'text': new_text})}\\n\\n"
                await asyncio.sleep(0.01)
        yield "data: [DONE]\\n\\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)`;

  const handleCopyCode = () => {
    navigator.clipboard.writeText(pythonBridgeCode).then(() => {
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    });
  };

  // Dynamic Theme styling classes
  const searchInputBgClass = isDark ? "border-slate-800 bg-slate-900/60 text-slate-200 placeholder-slate-500 focus:border-indigo-500" : "border-slate-300 bg-white text-slate-900 placeholder-slate-400 focus:border-indigo-600 shadow-sm";
  const gridCardBgClass = isDark ? "border-slate-800/80 bg-slate-900/30 hover:bg-slate-900/60 hover:border-slate-700/50 text-slate-400" : "border-slate-300 bg-white hover:border-indigo-500/50 shadow-sm shadow-slate-900/5 text-slate-700 hover:shadow-md";
  const gridCardTitleClass = isDark ? "text-slate-100" : "text-slate-950 font-bold";
  const cardBorderColorClass = isDark ? "border-slate-800/60" : "border-slate-200";
  const modalBgClass = isDark ? "bg-slate-900 border-slate-800" : "bg-white border-slate-300 shadow-2xl";
  const modalHeaderBgClass = isDark ? "border-slate-800 bg-slate-950/30" : "border-slate-200 bg-slate-100/70";
  const modalScrollAreaClass = isDark ? "scrollbar-thin" : "scrollbar-thin";
  const modalTextClass = isDark ? "text-slate-300" : "text-slate-900 font-normal";
  const modalHeaderTitleClass = isDark ? "text-white" : "text-slate-950 font-bold";
  const bulletColorClass = isDark ? "text-slate-300" : "text-slate-800 font-medium";

  // Light Mode adaptive styling for Benchmarks Section
  const benchmarkBgClass = isDark ? "bg-[#070707] border-white/10" : "bg-slate-100/80 border-slate-200";
  const benchmarkItemBgClass = isDark ? "bg-[#141414] border-white/5 hover:bg-[#1a1a1a]" : "bg-white border-slate-300 hover:bg-slate-50 shadow-sm";
  const benchmarkTitleColorClass = isDark ? "text-slate-200" : "text-slate-950 font-bold";
  const benchmarkDescColorClass = isDark ? "text-slate-300" : "text-slate-700 font-medium";
  const modalFooterBgClass = isDark ? "border-t border-slate-800 bg-slate-950/30" : "border-t border-slate-200 bg-slate-100/70";
  const closeButtonClass = isDark ? "bg-slate-800 hover:bg-slate-700 text-slate-200" : "bg-slate-200 hover:bg-slate-300 text-slate-900 font-semibold";

  return (
    <div className="py-6 select-none">
      <div className="max-w-7xl mx-auto px-1 sm:px-6 lg:px-8">
        
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6 mb-8 border-b border-slate-200/40 dark:border-white/5 pb-6">
          <div>
            <h1 className={`text-2xl sm:text-3xl font-display font-bold flex items-center gap-2 ${isDark ? "text-white" : "text-slate-900"}`}>
              <Star className="w-6 h-6 text-indigo-500" />
              <span>{lang === 'en' ? "Model Hub" : "模型百科中心"}</span>
            </h1>
            <p className={`text-xs sm:text-sm mt-1.5 font-light ${isDark ? "text-slate-400" : "text-slate-700"}`}>
              {lang === 'en' 
                ? "Browse deep capabilities of commercial models or interface local GPU networks offline."
                : "瀏覽雲端商用模型的技術核心結構，或點擊右側按鈕查閱自建 GPU 本地端模型對接指南。"}
            </p>
          </div>

          {/* Sub-tab is replaced with a highly professional Information Icon button per request */}
          <button
            onClick={() => setShowBridgeModal(true)}
            className={`px-3 py-2 rounded-xl border transition-all cursor-pointer flex items-center gap-1.5 text-xs font-semibold ${
              isDark 
                ? "bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-850 hover:text-white" 
                : "bg-white border-slate-200 text-slate-700 hover:bg-slate-50 hover:text-slate-900 shadow-sm"
            }`}
            title={lang === 'en' ? "Local GPU Model Integration Guide" : "本地 GPU / CPU 離線對接指南"}
            id="local-bridge-info-btn"
          >
            <HelpCircle className="w-4 h-4 text-indigo-500" />
            <span>{lang === 'en' ? "Offline Bridge" : "本地端 Python 對接"}</span>
          </button>
        </div>

        {/* Tab 1: Model Encyclopedia / Catalog */}
        <>
          {/* Filter controls */}
          <div className="flex flex-col sm:flex-row gap-4 mb-8">
            {/* Search bar */}
            <div className="relative flex-1">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                type="text"
                placeholder={lang === 'en' ? "Search model names, strengths, providers..." : "搜尋模型名稱、優勢強項、供應商..."}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className={`w-full pl-10 pr-4 py-2.5 rounded-xl border text-sm font-light transition-all focus:outline-none focus:ring-1 focus:ring-indigo-500 ${searchInputBgClass}`}
                id="search-input"
              />
            </div>

            {/* Provider Filter Tabs */}
            <div className="flex items-center gap-2 overflow-x-auto pb-1.5 scrollbar-none sm:pb-0">
              <div className="flex items-center text-slate-500 gap-1 text-xs font-mono font-medium border-r border-slate-200/50 dark:border-slate-800 pr-3 mr-1 flex-shrink-0">
                <Filter className="w-3.5 h-3.5" />
                <span>{lang === 'en' ? "PROVIDER" : "供應商"}</span>
              </div>
              {providers.map((provider) => (
                <button
                  key={provider}
                  onClick={() => setSelectedProvider(provider)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all uppercase tracking-wide flex-shrink-0 cursor-pointer ${
                    (provider === "all" ? selectedProvider === "all" : selectedProvider.toLowerCase() === provider.toLowerCase())
                      ? "bg-indigo-600 text-white border-indigo-500 shadow-sm"
                      : `${isDark ? "bg-slate-900/40 text-slate-400 border-slate-800/80 hover:border-slate-700" : "bg-white text-slate-700 border-slate-200 hover:border-slate-300"}`
                  }`}
                >
                  {provider === "all" ? (lang === 'en' ? "All" : "全部") : provider}
                </button>
              ))}
            </div>
          </div>

          {/* Grid View */}
          {filteredModels.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" id="models-hub-grid">
              {filteredModels.map((model) => (
                <div
                  key={model.id}
                  onClick={() => setSelectedModel(model)}
                  className={`group p-6 rounded-2xl border transition-all duration-300 cursor-pointer flex flex-col justify-between h-[230px] relative overflow-hidden ${gridCardBgClass}`}
                >
                  <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/5 rounded-full blur-2xl group-hover:bg-indigo-500/10 transition-all duration-500 pointer-events-none" />

                  <div>
                    <div className="flex justify-between items-start gap-2 mb-3">
                      <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-semibold font-mono uppercase border ${getProviderColor(model.provider)}`}>
                        {model.provider}
                      </span>
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${isDark ? "text-slate-500 bg-slate-950/40" : "text-slate-600 bg-slate-100"}`}>
                        {model.parameters}
                      </span>
                    </div>

                    <h3 className={`font-display text-base font-bold group-hover:text-indigo-500 transition-colors ${gridCardTitleClass}`}>
                      {model.name}
                    </h3>
                    
                    <p className={`text-xs leading-relaxed font-light mt-2 line-clamp-3 ${isDark ? "text-slate-400" : "text-slate-750"}`}>
                      {model.description}
                    </p>
                  </div>

                  <div className={`flex items-center justify-between mt-4 border-t pt-3 ${cardBorderColorClass}`}>
                    <div className={`flex gap-2 text-[10px] font-mono ${isDark ? "text-slate-500" : "text-slate-600 font-semibold"}`}>
                      <span className="flex items-center gap-1">
                        <Calendar className="w-3 h-3" />
                        {model.releaseYear}
                      </span>
                      <span>•</span>
                      <span className="flex items-center gap-1">
                        <Cpu className="w-3 h-3" />
                        {model.version}
                      </span>
                    </div>
                    <span className="text-xs font-semibold text-indigo-500 dark:text-indigo-400 group-hover:translate-x-1 transition-transform flex items-center gap-0.5">
                      <span>{lang === 'en' ? "Details" : "詳情資訊"}</span>
                      <span>→</span>
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className={`text-center py-20 border border-dashed rounded-2xl ${isDark ? "border-slate-800" : "border-slate-200"}`}>
              <AlertTriangle className="w-8 h-8 text-slate-400 mx-auto mb-3" />
              <h3 className={`text-sm font-medium ${isDark ? "text-slate-300" : "text-slate-700"}`}>{lang === 'en' ? "No models match your query" : "未找到符合條件的模型"}</h3>
              <p className="text-xs text-slate-500 mt-1 font-light">
                {lang === 'en' ? "Try adjusting your search keywords or provider filter." : "請嘗試調整您的搜尋關鍵字或供應商篩選條件。"}
              </p>
            </div>
          )}
        </>

        {/* Offline Bridge Setup Guide Modal - Popped up when Information Icon is clicked */}
        {showBridgeModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div 
              onClick={() => setShowBridgeModal(false)}
              className="fixed inset-0 bg-slate-950/85 backdrop-blur-sm" 
            />
            
            <div className={`relative w-full max-w-5xl rounded-2xl shadow-2xl overflow-hidden z-10 max-h-[90vh] flex flex-col ${modalBgClass}`}>
              
              {/* Modal Header */}
              <div className={`px-6 py-4 flex items-center justify-between ${modalHeaderBgClass}`}>
                <div className="flex items-center gap-2">
                  <Server className="w-5 h-5 text-indigo-500" />
                  <h2 className={`font-display text-sm sm:text-base font-bold ${modalHeaderTitleClass}`}>
                    {lang === 'en' ? "Offline Python Model Bridge Setup" : "本地 GPU / CPU 離線模型對接指引"}
                  </h2>
                </div>
                <button
                  onClick={() => setShowBridgeModal(false)}
                  className={`p-1.5 rounded-lg text-slate-400 hover:text-slate-100 transition-colors cursor-pointer ${isDark ? "hover:bg-slate-800" : "hover:bg-slate-100"}`}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Scrollable Modal Content */}
              <div className="p-6 overflow-y-auto flex-1 scrollbar-thin">
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                  
                  {/* Left side: Guide Steps */}
                  <div className="lg:col-span-5 space-y-5">
                    <p className={`text-xs leading-relaxed font-light ${isDark ? "text-slate-400" : "text-slate-700"}`}>
                      {lang === 'en'
                        ? "Scaling the playground to offline models is easy. Follow these steps to build an adapter API on your local GPU server and test models like Llama, Qwen, or Phi side-by-side with commercial APIs."
                        : "Karen 設計為可無限擴充。透過 Python 的 FastAPI 與 HuggingFace，您可以在本地自建高性價比的開源模型（例如 Llama、Qwen、Phi-3 等），並將其串流輸出無縫串接至本比對沙盒。"}
                    </p>

                    {/* Steps Timeline */}
                    <div className="space-y-4">
                      <div className="flex gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-xs font-bold text-indigo-500 font-mono">1</span>
                        <div>
                          <h4 className={`text-xs font-bold ${isDark ? "text-slate-200" : "text-slate-800"}`}>
                            {lang === 'en' ? "Install Dependencies" : "第一步：安裝 Python 套件"}
                          </h4>
                          <p className="text-[11px] text-slate-500 mt-1 font-light">
                            {lang === 'en' ? "Create a virtual environment and run pip installer:" : "建立虛擬環境並安裝所需的 GPU 推理與伺服器套件："}
                          </p>
                          <div className={`mt-1.5 p-2 text-[10px] rounded-lg font-mono border select-text overflow-x-auto ${
                            isDark 
                              ? "bg-slate-950 text-indigo-400 border-white/5" 
                              : "bg-slate-100 text-indigo-700 border-slate-200"
                          }`}>
                            pip install fastapi uvicorn transformers torch pydantic
                          </div>
                        </div>
                      </div>

                      <div className="flex gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-xs font-bold text-indigo-500 font-mono">2</span>
                        <div>
                          <h4 className={`text-xs font-bold ${isDark ? "text-slate-200" : "text-slate-800"}`}>
                            {lang === 'en' ? "Deploy Python Adapter" : "第二步：啟動 Python 對接伺服器"}
                          </h4>
                          <p className="text-[11px] text-slate-500 mt-1 font-light">
                            {lang === 'en' ? "Copy the code on the right side and run uvicorn server:" : "複製右側的 FastAPI 代碼，保存為 bridge.py 並執行："}
                          </p>
                          <div className={`mt-1.5 p-2 text-[10px] rounded-lg font-mono border select-text overflow-x-auto ${
                            isDark 
                              ? "bg-slate-950 text-indigo-400 border-white/5" 
                              : "bg-slate-100 text-indigo-700 border-slate-200"
                          }`}>
                            python bridge.py
                          </div>
                        </div>
                      </div>

                      <div className="flex gap-3">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-xs font-bold text-indigo-500 font-mono">3</span>
                        <div>
                          <h4 className={`text-xs font-bold ${isDark ? "text-slate-200" : "text-slate-800"}`}>
                            {lang === 'en' ? "Plug into Sandbox" : "第三步：將本地端模型接入沙盒"}
                          </h4>
                          <p className="text-[11px] text-slate-500 mt-1 font-light">
                            {lang === 'en' ? "Configure Karen config array to route server-sent streams through proxy endpoints." : "在 settings.json 或配置中新增本地端對接位址。沙盒便可直接對 http://localhost:8000/stream 發起請求，與雲端模型平行輸出對比。"}
                          </p>
                        </div>
                      </div>

                      <div className="flex gap-3 pb-2 border-b border-dashed border-slate-200/50 dark:border-white/5">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-xs font-bold text-indigo-500 font-mono">4</span>
                        <div>
                          <h4 className={`text-xs font-bold ${isDark ? "text-slate-200" : "text-slate-800"}`}>
                            {lang === 'en' ? "Scale & Add Custom Benchmarks" : "第四步：擴充與新增自定義評測指標"}
                          </h4>
                          <p className="text-[11px] text-slate-500 mt-1 font-light">
                            {lang === 'en' 
                              ? "To scale the leaderboard with customized metrics: (1) Append your score key/value (e.g. 'GSM8K': 92.5) in src/data/models.json or database schema. (2) Register your benchmark meta-data inside src/components/Leaderboard.tsx metrics collections (metrics_en & metrics_zh). The table, progress bars, and filters will scale dynamically!"
                              : "若要擴充指標：(1) 在 src/data/models.json (或數據庫模型) 內各模型的 benchmarks 下新增得分（如：'GSM8K': 92.5）。(2) 將其 ID、中英文名稱與說明註冊至 src/components/Leaderboard.tsx 的 metrics_en 及 metrics_zh 陣列。排行榜和篩選功能便會自動按比例動態渲染、重新排序！"}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Security warning card */}
                    <div className={`p-4 rounded-xl border flex gap-3 text-xs leading-normal ${
                      isDark ? "bg-indigo-500/5 border-indigo-500/10 text-slate-300" : "bg-indigo-50/50 border-indigo-100 text-indigo-900"
                    }`}>
                      <CheckCircle className="w-5 h-5 text-indigo-500 flex-shrink-0" />
                      <div>
                        <h5 className="font-bold">{lang === 'en' ? "SSE Streaming Integration" : "SSE 原生串流架構安全防護"}</h5>
                        <p className="text-[11px] text-slate-500 mt-1 font-light">
                          {lang === 'en'
                            ? "The Python code on the right is fully compatible with our asynchronous SSE parser, ensuring real-time response streams without lagging user viewport."
                            : "右側 Python 代碼使用 Server-Sent Events (SSE) 規格，與沙盒的前端解析引擎完全相容，能在大模型推理時進行字元級實時串流，提供流暢的渲染性能。"}
                        </p>
                      </div>
                    </div>

                  </div>

                  {/* Right side: Code Container */}
                  <div className={`lg:col-span-7 flex flex-col h-[480px] rounded-2xl border overflow-hidden relative shadow-2xl ${
                    isDark ? "border-slate-800 bg-slate-950" : "border-slate-200 bg-slate-50"
                  }`}>
                    
                    {/* Header inside code panel */}
                    <div className={`px-4 py-2.5 border-b flex items-center justify-between ${
                      isDark ? "border-white/5 bg-slate-900/80" : "border-slate-200 bg-slate-200/50"
                    }`}>
                      <div className="flex items-center gap-2">
                        <span className="flex gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full bg-rose-500/40" />
                          <span className="w-2.5 h-2.5 rounded-full bg-amber-500/40" />
                          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/40" />
                        </span>
                        <span className={`text-[10px] font-mono font-bold ml-1.5 ${isDark ? "text-slate-400" : "text-slate-500"}`}>bridge.py</span>
                      </div>

                      <button
                        onClick={handleCopyCode}
                        className={`px-2.5 py-1 rounded border text-[10px] font-semibold transition-all flex items-center gap-1.5 cursor-pointer ${
                          isDark 
                            ? "border-white/10 bg-white/5 hover:bg-white/10 text-slate-300" 
                            : "border-slate-200 bg-white hover:bg-slate-100 text-slate-700"
                        }`}
                      >
                        {copiedCode ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                        <span>{copiedCode ? "Copied" : "Copy Code"}</span>
                      </button>
                    </div>

                    {/* Scrollable Preformatted Code text */}
                    <div className={`flex-1 overflow-auto p-4 text-[10.5px] font-mono leading-relaxed select-text scrollbar-thin ${
                      isDark ? "text-slate-300" : "text-slate-800"
                    }`}>
                      <pre>{pythonBridgeCode}</pre>
                    </div>

                    {/* Mini Terminal bar at the bottom */}
                    <div className={`px-4 py-2 border-t flex items-center justify-between text-[10px] font-mono ${
                      isDark 
                        ? "bg-indigo-950/20 border-white/5 text-indigo-400/90" 
                        : "bg-indigo-50/80 border-slate-200 text-indigo-600"
                    }`}>
                      <span>⚡ Compatible with PyTorch & HuggingFace Models</span>
                      <span>Port 8000</span>
                    </div>

                  </div>

                </div>
              </div>

              {/* Bottom Footer actions */}
              <div className="px-6 py-4 border-t border-slate-800 bg-slate-950/30 flex justify-end">
                <button
                  onClick={() => setShowBridgeModal(false)}
                  className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-colors cursor-pointer"
                >
                  {lang === 'en' ? "Close Guide" : "關閉指引"}
                </button>
              </div>

            </div>
          </div>
        )}

        {/* Detailed Modal/Drawer */}
        {selectedModel && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div 
              onClick={() => setSelectedModel(null)}
              className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm" 
            />
            
            <div className={`relative w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden z-10 max-h-[90vh] flex flex-col ${modalBgClass}`}>
              
              {/* Header */}
              <div className={`px-6 py-4 flex items-center justify-between ${modalHeaderBgClass}`}>
                <div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold font-mono border ${getProviderColor(selectedModel.provider)}`}>
                      {selectedModel.provider}
                    </span>
                    <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${isDark ? "text-slate-500 bg-slate-800/50" : "text-slate-600 bg-slate-100"}`}>
                      {selectedModel.parameters}
                    </span>
                  </div>
                  <h2 className={`font-display text-lg font-bold mt-1 ${modalHeaderTitleClass}`}>
                    {selectedModel.name}
                  </h2>
                </div>
                <button
                  onClick={() => setSelectedModel(null)}
                  className={`p-1.5 rounded-lg transition-colors cursor-pointer ${isDark ? "text-slate-400 hover:text-slate-100 hover:bg-slate-800" : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"}`}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Scrollable Content */}
              <div className={`p-6 overflow-y-auto space-y-6 flex-1 ${modalScrollAreaClass}`}>
                {/* Description */}
                <div>
                  <h4 className="text-[11px] font-mono font-semibold text-indigo-500 uppercase tracking-wider mb-2">
                    {lang === 'en' ? "Overview" : "模型簡介"}
                  </h4>
                  <p className={`text-sm leading-relaxed ${modalTextClass}`}>
                    {selectedModel.description}
                  </p>
                </div>

                {/* Pros and Cons */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Strengths */}
                  <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
                    <h5 className="text-xs font-semibold text-emerald-500 flex items-center gap-1.5 mb-3">
                      <CheckCircle className="w-4 h-4" />
                      <span>{lang === 'en' ? "Strengths / Scenarios" : "優勢 / 適合場景"}</span>
                    </h5>
                    <ul className={`space-y-2 text-xs ${bulletColorClass}`}>
                      {selectedModel.strengths.map((str, idx) => (
                        <li key={idx} className="flex items-start gap-1.5">
                          <span className="text-emerald-500 mt-0.5">•</span>
                          <span>{str}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Weaknesses */}
                  <div className="p-4 rounded-xl bg-rose-500/5 border border-rose-500/10">
                    <h5 className="text-xs font-semibold text-rose-500 flex items-center gap-1.5 mb-3">
                      <AlertTriangle className="w-4 h-4" />
                      <span>{lang === 'en' ? "Weaknesses / Limitations" : "弱項 / 侷限性"}</span>
                    </h5>
                    <ul className={`space-y-2 text-xs ${bulletColorClass}`}>
                      {selectedModel.weaknesses.map((weak, idx) => (
                        <li key={idx} className="flex items-start gap-1.5">
                          <span className="text-rose-500 mt-0.5">•</span>
                          <span>{weak}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Expert Analysis */}
                <div className="p-4 rounded-xl bg-indigo-500/5 border border-indigo-500/10 space-y-3">
                  <h5 className="text-xs font-semibold text-indigo-500 flex items-center gap-1.5">
                    <BookOpen className="w-4 h-4 text-indigo-500" />
                    <span>{lang === 'en' ? "Expert Deep Dive Analysis" : "我自己的分析文（專家深度解析）"}</span>
                  </h5>
                  <div className={`text-xs leading-relaxed whitespace-pre-line space-y-2 ${bulletColorClass}`}>
                    {(() => {
                      const analysis = lang === 'en' ? EXPERT_ANALYSIS_EN[selectedModel.id] : EXPERT_ANALYSIS_ZH[selectedModel.id];
                      if (analysis) {
                        return analysis.split("\n\n").map((para, pIdx) => {
                          const isStrengthDecrud = para.startsWith("優勢深度解讀：") || para.startsWith("Key Strengths:");
                          const isTuningAdv = para.startsWith("實戰調優建議：") || para.startsWith("Evaluation Advice:");
                          if (isStrengthDecrud) {
                            return (
                              <p key={pIdx}>
                                <strong className="text-indigo-600 dark:text-indigo-300">💡 {lang === 'en' ? "Key Strengths: " : "優勢深度解讀："}</strong>
                                {para.replace("優勢深度解讀：", "").replace("Key Strengths:", "")}
                              </p>
                            );
                          }
                          if (isTuningAdv) {
                            return (
                              <p key={pIdx} className="pt-1">
                                <strong className="text-purple-600 dark:text-purple-300">🛠️ {lang === 'en' ? "Evaluation Advice: " : "實戰調優建議："}</strong>
                                {para.replace("實戰調優建議：", "").replace("Evaluation Advice:", "")}
                              </p>
                            );
                          }
                          return <p key={pIdx}>{para}</p>;
                        });
                      } else {
                        return <p>{lang === 'en' ? "No expert evaluation available for this model yet. Try testing it in the Sandbox to evaluate its performance." : "暫無針對此模型的專家解析。請在沙盒中實際測試以評估其輸出表現。"}</p>;
                      }
                    })()}
                  </div>
                </div>

                {/* Reference Links */}
                <div>
                  <h4 className="text-[11px] font-mono font-semibold text-indigo-500 uppercase tracking-wider mb-2.5">
                    {lang === 'en' ? "Academic & Dev Resources" : "文獻與開發資源"}
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedModel.links.official && (
                      <a
                        href={selectedModel.links.official}
                        target="_blank"
                        referrerPolicy="no-referrer"
                        className={`flex items-center gap-1 text-xs px-3 py-1.5 rounded-lg border transition-all ${
                          isDark 
                            ? "bg-slate-850 border-slate-700/50 hover:bg-slate-700 text-slate-300 hover:text-white" 
                            : "bg-slate-100 border-slate-200 hover:bg-slate-200 text-slate-700 hover:text-slate-900"
                        }`}
                      >
                        <BookOpen className="w-3.5 h-3.5" />
                        <span>{lang === 'en' ? "Official Site" : "官方首頁"}</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                    {selectedModel.links.api && (
                      <a
                        href={selectedModel.links.api}
                        target="_blank"
                        referrerPolicy="no-referrer"
                        className={`flex items-center gap-1 text-xs px-3 py-1.5 rounded-lg border transition-all ${
                          isDark 
                            ? "bg-slate-850 border-slate-700/50 hover:bg-slate-700 text-slate-300 hover:text-white" 
                            : "bg-slate-100 border-slate-200 hover:bg-slate-200 text-slate-700 hover:text-slate-900"
                        }`}
                      >
                        <Cpu className="w-3.5 h-3.5" />
                        <span>{lang === 'en' ? "API Docs" : "API 文件"}</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                    {selectedModel.links.paper && (
                      <a
                        href={selectedModel.links.paper}
                        target="_blank"
                        referrerPolicy="no-referrer"
                        className={`flex items-center gap-1 text-xs px-3 py-1.5 rounded-lg border transition-all ${
                          isDark 
                            ? "bg-slate-850 border-slate-700/50 hover:bg-slate-700 text-slate-300 hover:text-white" 
                            : "bg-slate-100 border-slate-200 hover:bg-slate-200 text-slate-700 hover:text-slate-900"
                        }`}
                      >
                        <BookOpen className="w-3.5 h-3.5" />
                        <span>{lang === 'en' ? "Paper (arXiv)" : "學術論文 (arXiv)"}</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                    {selectedModel.links.github && (
                      <a
                        href={selectedModel.links.github}
                        target="_blank"
                        referrerPolicy="no-referrer"
                        className={`flex items-center gap-1 text-xs px-3 py-1.5 rounded-lg border transition-all ${
                          isDark 
                            ? "bg-slate-850 border-slate-700/50 hover:bg-slate-700 text-slate-300 hover:text-white" 
                            : "bg-slate-100 border-slate-200 hover:bg-slate-200 text-slate-700 hover:text-slate-900"
                        }`}
                      >
                        <Cpu className="w-3.5 h-3.5" />
                        <span>GitHub</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                  </div>
                </div>

              </div>

              {/* Fixed Bottom Benchmarks Section */}
              <div className={`border-t px-6 py-4 flex flex-col gap-2 relative shrink-0 ${benchmarkBgClass}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <Star className="w-3.5 h-3.5 text-indigo-400 animate-pulse" />
                    <span className={`text-xs font-semibold ${benchmarkTitleColorClass}`}>
                      {lang === 'en' ? "Objective Benchmarks (Scrollable)" : "客觀能力評測得分 (左右按鈕滾動)"}
                    </span>
                  </div>
                  
                  {/* Left and Right Scroll Buttons */}
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={scrollLeft}
                      disabled={!canScrollLeft}
                      className={`p-1 rounded-lg border transition-all cursor-pointer ${
                        canScrollLeft
                          ? isDark
                            ? "bg-white/5 border-white/10 text-slate-300 hover:bg-white/10 hover:text-white"
                            : "bg-white border-slate-200 text-slate-700 hover:bg-slate-100"
                          : "bg-slate-100 border-slate-100 text-slate-400 cursor-not-allowed opacity-50 dark:bg-white/5 dark:border-white/5"
                      }`}
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <button
                      onClick={scrollRight}
                      disabled={!canScrollRight}
                      className={`p-1 rounded-lg border transition-all cursor-pointer ${
                        canScrollRight
                          ? isDark
                            ? "bg-white/5 border-white/10 text-slate-300 hover:bg-white/10 hover:text-white"
                            : "bg-white border-slate-200 text-slate-700 hover:bg-slate-100"
                          : "bg-slate-100 border-slate-100 text-slate-400 cursor-not-allowed opacity-50 dark:bg-white/5 dark:border-white/5"
                      }`}
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Benchmarks List */}
                <div 
                  ref={scrollRef}
                  className="flex gap-3 overflow-x-auto scrollbar-none pb-1 pt-1 scroll-smooth"
                  style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
                >
                  {getEnrichedBenchmarks(selectedModel.id, selectedModel.benchmarks, lang).map((bench) => (
                    <div 
                      key={bench.key} 
                      className={`min-w-[170px] sm:min-w-[190px] border p-3.5 rounded-xl text-center flex-shrink-0 transition-all duration-300 group ${benchmarkItemBgClass}`}
                    >
                      <div className="text-[10px] font-mono text-slate-400 dark:text-slate-500 uppercase tracking-wider font-semibold mb-0.5">{bench.name}</div>
                      <div className="text-base font-bold text-indigo-600 dark:text-indigo-400 group-hover:scale-105 transition-transform">{bench.value}%</div>
                      <div className={`text-[10px] mt-1 line-clamp-1 ${benchmarkDescColorClass}`} title={bench.label}>{bench.label}</div>
                      <div className="text-[9px] text-slate-400 dark:text-slate-500 font-light mt-0.5 line-clamp-1" title={bench.desc}>{bench.desc}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Bottom Footer actions */}
              <div className={`px-6 py-4 flex justify-end gap-3 ${modalFooterBgClass}`}>
                <button
                  onClick={() => setSelectedModel(null)}
                  className={`px-4 py-2 rounded-xl text-xs font-semibold transition-colors cursor-pointer ${closeButtonClass}`}
                >
                  {lang === 'en' ? "Close" : "關閉詳情"}
                </button>
              </div>

            </div>
          </div>
        )}

      </div>
    </div>
  );
}
