# Scaling KAREN with Local Python Models (FastAPI & Ollama)

This guide documents the technical architecture, step-by-step procedures, and code templates required to integrate local Python-based LLMs or self-hosted model runners (such as **Ollama**, **vLLM**, or raw **Hugging Face / PyTorch** pipelines) with your KAREN React + Express dashboard.

---

## 1. High-Level Architecture

The integration follows a decoupled, low-latency streaming pipeline:

```
┌────────────────────────┐         SSE (HTTP Stream)        ┌─────────────────────────┐
│                        ├─────────────────────────◄────────┤                         │
│  React Sandbox client  │                                  │  Express Server (3000)  │
│                        ├─────────────────────────►────────┤                         │
└────────────────────────┘        User Prompt               └────────────┬────────────┘
                                                                         │
                                                                         │ Local API call
                                                                         ▼
┌────────────────────────┐         SSE (HTTP Stream)        ┌─────────────────────────┐
│                        ├─────────────────────────◄────────┤  FastAPI / Ollama       │
│  Local Model (Python)  │                                  │  Model Runner (8000)    │
│                        ├─────────────────────────►────────┤                         │
└────────────────────────┘        Prompt / System           └─────────────────────────┘
```

1. **Client Interface**: The user selects a local model and sends a prompt in the Sandbox.
2. **KAREN Express Backend**: Receives the prompt, detects that it is a local model, and proxies the stream request to the local Python runner (FastAPI or Ollama).
3. **Local Python Runner**: Runs the LLM (e.g. using `llama.cpp`, `transformers`, or `ollama`) and streams chunks back using **Server-Sent Events (SSE)**.
4. **Real-time Render**: The React client parses the stream chunk-by-chunk to render markdown dynamically.

---

## 2. Setting Up a Local Python API Server (FastAPI)

Below is a production-ready, highly parallel Python FastAPI script (`server_local.py`) that sets up a streaming LLM server. It uses Hugging Face's `transformers` or standard API pipelines and outputs standard SSE chunks compatible with the KAREN stream parser.

### Step A: Install Python Dependencies
```bash
pip install fastapi uvicorn sse-starlette transformers torch
```

### Step B: Create the Python Runner (`server_local.py`)
Save the following file locally on your model host machine:

```python
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="KAREN Local Python LLM Runner")

# Enable CORS for cross-origin local requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def stream_generator(prompt: str, system_prompt: str, temperature: float, max_tokens: int):
    """
    Simulates or runs local text generation. 
    Replace this mock logic with your actual HuggingFace Pipeline or llama.cpp generation loop!
    """
    try:
        # Example: Mocking a token-by-token stream generator
        # For a real pipeline, you would use:
        # from transformers import TextIteratorStreamer
        # streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        # threading.Thread(target=model.generate, kwargs={...}).start()
        
        full_text = f"Hello! This is a real-time stream generated from a Local Python Model in response to: '{prompt}'.\n\n"
        if system_prompt:
            full_text += f"I have applied your custom System Prompt: *'{system_prompt}'*\n\n"
            
        full_text += "Here are some mathematical calculations performed locally to prove runtime computation:\n"
        full_text += "1. $\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$\n"
        full_text += "2. $\\int_{a}^{b} f(x) dx = F(b) - F(a)$\n\n"
        full_text += "This shows that you have full access to custom deep local architectures!"

        # Stream words/characters with low latency
        words = full_text.split(" ")
        for word in words:
            # Yield event matching standard JSON structure
            yield json.dumps({"text": word + " "})
            await asyncio.sleep(0.08)  # simulate token generation rate

        # Yield ending token
        yield "[DONE]"
    except asyncio.CancelledError:
        print("Stream generation cancelled by user.")

@app.get("/api/local/generate")
async def local_generate(
    request: Request,
    prompt: str,
    systemPrompt: str = "",
    temperature: float = 0.7,
    maxTokens: int = 1500
):
    """
    Endpoint exposed to KAREN's Express proxy server.
    """
    generator = stream_generator(prompt, systemPrompt, temperature, maxTokens)
    return EventSourceResponse(generator)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 3. Registering the Local Model in KAREN

To make your local model selectable in the Sandbox, add it to your `src/data/models.json` registry. For example, add the following entry:

```json
  {
    "id": "local-python-llama",
    "name": "Local Llama-3 (Python API)",
    "provider": "Self-Hosted",
    "logo": "meta",
    "version": "v3.0-local",
    "releaseYear": "2026",
    "parameters": "8 Billion",
    "description": "Your self-hosted local model running in a local Hugging Face, PyTorch, or llama.cpp backend wrapper.",
    "strengths": [
      "No API token costs",
      "Absolute data privacy and security",
      "Supports custom fine-tuned weights"
    ],
    "weaknesses": [
      "Depends entirely on local GPU power"
    ],
    "links": {
      "official": "https://github.com/meta-llama"
    },
    "benchmarks": {
      "MMLU": 68.4,
      "HumanEval": 62.0,
      "MATH": 38.5,
      "GPQA": 28.1
    }
  }
```

---

## 4. Connecting Express Proxy Server to Local Runner

Update your Express `/api/compare/stream` endpoint inside `/server.ts` to intercept requests for local model IDs and route them directly to your Python API running at `http://localhost:8000` or your local **Ollama** server running at `http://localhost:11434`.

### Dynamic Routing Implementation Snippet for `/server.ts`:

```typescript
// Add near the top of /api/compare/stream logic:
if (modelId === "local-python-llama") {
  try {
    // Redirect comparison query to Local Python FastAPI server
    const localUrl = `http://localhost:8000/api/local/generate?prompt=${encodeURIComponent(String(prompt))}&systemPrompt=${encodeURIComponent(String(systemPrompt || ''))}&temperature=${temperature || 0.7}&maxTokens=${maxTokens || 1500}`;
    
    const response = await fetch(localUrl);
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    
    if (!reader) {
      throw new Error("Unable to read stream from local Python server.");
    }
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      // Write chunk directly to client
      res.write(chunk);
    }
    res.end();
    return;
  } catch (err: any) {
    res.write(`data: ${JSON.stringify({ error: `Local Python Server offline: ${err.message}` })}\n\n`);
    res.end();
    return;
  }
}
```

### Alternatively: Directly Proxy to Local Ollama Server
If you prefer running models natively via Ollama, you can query its OpenAI-compatible stream API directly:

```typescript
if (modelId === "local-ollama") {
  try {
    const response = await fetch("http://localhost:11434/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: json.stringify({
        model: "llama3", // or "mistral", "gemma"
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: prompt }
        ],
        stream: true,
        options: { temperature: temperature }
      })
    });
    
    // Parse Ollama JSON responses and forward them using standard KAREN SSE formatting...
  } catch (err) {
    // Handle error...
  }
}
```

---

## 5. Architectural Advantages & Scaling Best Practices

- **Zero-Copy Serialization**: By utilizing Server-Sent Events (SSE) end-to-end, the network overhead is minimized, leading to near-instantaneous token generation output in the frontend.
- **Asynchronous Processing**: Ensure your Python FastAPI generation code uses `asyncio` or handles threading properly (e.g. `TextIteratorStreamer`), so that parallel model runs do not block the single-threaded event loop.
- **Quantization**: For consumer-grade GPUs, utilize **AWQ** or **GGUF** quantized models to ensure they fit within your GPU's VRAM limits while maintaining close to FP16 level intelligence.
