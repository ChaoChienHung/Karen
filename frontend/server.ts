import express from "express";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const getAppDirname = () => {
  if (typeof __dirname !== "undefined") return __dirname;
  if (typeof import.meta !== "undefined" && import.meta.url) {
    try {
      return path.dirname(fileURLToPath(import.meta.url));
    } catch {
      // fallback
    }
  }
  return process.cwd();
};
const appDir = getAppDirname();

const app = express();
const PORT = 3000;

app.use(express.json());

// Lazy-initialized Gemini client to prevent crash on startup if key is missing
let aiInstance: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiInstance) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is not configured. Please add your Gemini API Key in Settings > Secrets to enable comparisons.");
    }
    aiInstance = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiInstance;
}

// API: Get models list dynamically from models.json
app.get("/api/models", (req, res) => {
  try {
    const candidatePaths = [
      path.join(appDir, "src", "data", "models.json"),
      path.join(process.cwd(), "src", "data", "models.json"),
      path.join(process.cwd(), "frontend", "src", "data", "models.json"),
    ];
    const modelsPath = candidatePaths.find((p) => fs.existsSync(p));
    if (modelsPath) {
      const data = fs.readFileSync(modelsPath, "utf-8");
      res.json(JSON.parse(data));
    } else {
      res.status(404).json({ error: "models.json not found" });
    }
  } catch (error: any) {
    res.status(500).json({ error: error.message || "Failed to load models list" });
  }
});

// API: Streaming compare endpoint (SSE)
app.get("/api/compare/stream", async (req, res) => {
  const { modelId, prompt, systemPrompt, temperature, maxTokens } = req.query;

  // Configure response headers for Server-Sent Events
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no"); // Prevent Nginx or Cloud Run buffering streams

  // Support local Python model runner (FastAPI / local server)
  const isLocal = typeof modelId === "string" && (modelId.startsWith("local-") || modelId.includes("local"));
  if (isLocal) {
    try {
      const localRunnerHost = process.env.LOCAL_RUNNER_HOST || "127.0.0.1";
      const localRunnerPort = process.env.LOCAL_RUNNER_PORT || "8000";
      const queryParams = new URLSearchParams({
        prompt: String(prompt || ""),
        systemPrompt: String(systemPrompt || ""),
        temperature: String(temperature || 0.7),
        maxTokens: String(maxTokens || 1500),
        modelId: String(modelId),
      });
      const localUrl = `http://${localRunnerHost}:${localRunnerPort}/api/local/generate?${queryParams.toString()}`;

      const response = await fetch(localUrl);
      if (!response.ok) {
        throw new Error(`Local model runner returned HTTP ${response.status}: ${response.statusText}`);
      }
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) {
        throw new Error("Unable to read stream from local Python server.");
      }
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        res.write(chunk);
      }
      res.end();
      return;
    } catch (err: any) {
      console.error("Local runner stream error:", err);
      const errMsg = `Local runner offline at http://127.0.0.1:8000 (${err.message}). Start the local runner using: python scripts/serve_local.py`;
      res.write(`data: ${JSON.stringify({ error: errMsg })}\n\n`);
      res.end();
      return;
    }
  }

  try {
    const ai = getGeminiClient();

    let targetModel = "gemini-3.5-flash";
    let combinedSystemInstruction = "";

    if (systemPrompt) {
      combinedSystemInstruction = String(systemPrompt);
    }

    // Persona inject instructions based on selected model
    let personaPrompt = "";
    if (modelId === "gemini-3.1-flash-lite") {
      targetModel = "gemini-3.1-flash-lite";
    } else if (modelId === "gpt-4o") {
      personaPrompt = "You are simulating OpenAI's GPT-4o. Respond in GPT-4o's characteristic style: highly polished, professional, concise, clearly organized with subheadings, and action-oriented list formatting. Make sure you structure your answer beautifully and do not mention being simulated unless explicitly asked.";
    } else if (modelId === "claude-3-5-sonnet") {
      personaPrompt = "You are simulating Anthropic's Claude 3.5 Sonnet. Respond in Claude's characteristic style: highly articulate, deeply intellectual, highly detailed, precise, and objective. Avoid generic AI introductory clichés. Use elegant markdown styling, tables, lists, and deep analytical reasoning.";
    } else if (modelId === "llama-3-1-405b") {
      personaPrompt = "You are simulating Meta's Llama 3.1 405B. Respond with Llama's characteristic style: objective, structured, highly technical, and direct. Deliver highly comprehensive explanations, clear specifications, and keep tone completely neutral.";
    } else if (modelId === "deepseek-r1") {
      personaPrompt = "You are simulating DeepSeek-R1, the leading reasoning model. You MUST output your detailed step-by-step thinking process inside a single <thought>...</thought> block first. Inside this block, write a rigorous step-by-step logic breakdown, exploring different aspects, checking facts, and correcting yourself. Only after the </thought> tag should you write your final high-quality, comprehensive answer. Keep the <thought> block highly detailed and analytical.";
    }

    if (personaPrompt) {
      combinedSystemInstruction = combinedSystemInstruction 
        ? `${combinedSystemInstruction}\n\n[Persona Instruction]\n${personaPrompt}`
        : personaPrompt;
    }

    const tempVal = temperature ? parseFloat(String(temperature)) : 0.7;
    const maxTokensVal = maxTokens ? parseInt(String(maxTokens)) : 1500;

    const stream = await ai.models.generateContentStream({
      model: targetModel,
      contents: String(prompt || ""),
      config: {
        systemInstruction: combinedSystemInstruction || undefined,
        temperature: tempVal,
        maxOutputTokens: maxTokensVal,
      }
    });

    for await (const chunk of stream) {
      if (chunk.text) {
        res.write(`data: ${JSON.stringify({ text: chunk.text })}\n\n`);
      }
    }

    res.write(`data: [DONE]\n\n`);
    res.end();
  } catch (error: any) {
    console.error("Gemini stream error:", error);
    res.write(`data: ${JSON.stringify({ error: error.message || "An error occurred during output generation." })}\n\n`);
    res.end();
  }
});

// Configure Vite middleware or static serving
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const candidateDistPaths = [
      path.join(appDir, "dist"),
      path.join(process.cwd(), "dist"),
      path.join(process.cwd(), "frontend", "dist"),
    ];
    const distPath = candidateDistPaths.find((p) => fs.existsSync(p)) || path.join(appDir, "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server is running at http://localhost:${PORT}`);
  });
}

startServer();
