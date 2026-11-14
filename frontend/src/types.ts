export interface ModelLinks {
  official?: string;
  api?: string;
  paper?: string;
  github?: string;
}

export interface ModelBenchmarks {
  MMLU: number;
  HumanEval: number;
  MATH: number;
  GPQA: number;
}

export interface Model {
  id: string;
  name: string;
  provider: 'Google' | 'OpenAI' | 'Anthropic' | 'Meta' | 'DeepSeek' | 'Karen' | 'Self-Hosted';
  logo: string;
  version: string;
  releaseYear: string;
  parameters: string;
  description: string;
  strengths: string[];
  weaknesses: string[];
  links: ModelLinks;
  benchmarks: ModelBenchmarks;
}

export interface GlobalConfig {
  systemPrompt: string;
  userPrompt: string;
  temperature: number;
  maxTokens: number;
  syncScroll: boolean;
}

export interface OutputPanel {
  id: string; // unique ID for the workspace view (dynamic layout)
  modelId: string; // ID of the model selected for this panel
  output: string;
  isStreaming: boolean;
  isCompleted: boolean;
  error?: string;
}

export interface SavedHistory {
  id: string;
  title: string;
  timestamp: string;
  systemPrompt: string;
  userPrompt: string;
  temperature: number;
  maxTokens: number;
  panels: {
    modelId: string;
    output: string;
  }[];
}
