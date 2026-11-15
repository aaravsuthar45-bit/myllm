import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template_string, request
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.transformer import MyLLM

app = Flask(__name__)

# Global model state
model = None
tokenizer = None
device = None
block_size = 128


def load_model(checkpoint_path: str, tokenizer_path: str):
    global model, tokenizer, device, block_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        fallback = ROOT / "checkpoints" / "best_model.pt"
        if fallback.exists():
            ckpt = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    vocab_size = config.get("vocab_size", checkpoint.get("vocab_size", 8000))
    block_size = config.get("block_size", checkpoint.get("block_size", 128))
    n_embd = config.get("n_embd", checkpoint.get("n_embd", 256))
    n_heads = config.get("n_heads", checkpoint.get("n_heads", 4))
    n_layers = config.get("n_layers", checkpoint.get("n_layers", 4))

    tok_p = Path(tokenizer_path)
    if not tok_p.exists():
        fallback_tok = ROOT / "data" / "tokenizer" / "tokenizer.json"
        if fallback_tok.exists():
            tok_p = fallback_tok
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tok_p))

    model = MyLLM(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MyLLM Chat</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --user-msg: #3b82f6;
      --bot-msg: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: var(--card);
      border-bottom: 1px solid var(--border);
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    header h1 {
      font-size: 1.15rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .badge {
      background: #10b981;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      box-shadow: 0 0 8px #10b981;
    }
    .controls {
      display: flex;
      gap: 16px;
      align-items: center;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .controls label {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .controls input[type="range"] {
      width: 80px;
    }
    #chat-container {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-width: 800px;
      width: 100%;
      margin: 0 auto;
    }
    .message {
      display: flex;
      flex-direction: column;
      max-width: 80%;
      padding: 12px 16px;
      border-radius: 12px;
      line-height: 1.5;
      font-size: 0.95rem;
      word-break: break-word;
    }
    .user {
      align-self: flex-end;
      background: var(--user-msg);
      color: #fff;
      border-bottom-right-radius: 2px;
    }
    .bot {
      align-self: flex-start;
      background: var(--bot-msg);
      color: var(--text);
      border-bottom-left-radius: 2px;
    }
    .role-label {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      margin-bottom: 4px;
      opacity: 0.8;
    }
    #input-container {
      background: var(--card);
      border-top: 1px solid var(--border);
      padding: 16px 24px;
    }
    #chat-form {
      display: flex;
      gap: 12px;
      max-width: 800px;
      width: 100%;
      margin: 0 auto;
    }
    #user-input {
      flex: 1;
      padding: 12px 16px;
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      color: #fff;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
    }
    #user-input:focus {
      border-color: #3b82f6;
    }
    button {
      background: #3b82f6;
      border: none;
      color: #fff;
      padding: 0 20px;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.2s;
    }
    button:hover {
      background: #2563eb;
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .typing {
      font-style: italic;
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <header>
    <h1><span class="badge"></span> MyLLM Chat</h1>
    <div class="controls">
      <label>
        Temp: <span id="temp-val">0.7</span>
        <input type="range" id="temp" min="0.1" max="1.5" step="0.1" value="0.7" oninput="document.getElementById('temp-val').innerText = this.value">
      </label>
      <label>
        Tokens: <span id="tokens-val">80</span>
        <input type="range" id="tokens" min="20" max="200" step="10" value="80" oninput="document.getElementById('tokens-val').innerText = this.value">
      </label>
    </div>
  </header>

  <div id="chat-container">
    <div class="message bot">
      <div class="role-label">MyLLM</div>
      Hello! I am your custom-trained LLM. Ask me anything or type a prompt to get started!
    </div>
  </div>

  <div id="input-container">
    <form id="chat-form" onsubmit="sendMessage(event)">
      <input type="text" id="user-input" placeholder="Type a message..." autocomplete="off" autofocus />
      <button type="submit" id="send-btn">Send</button>
    </form>
  </div>

  <script>
    const chatContainer = document.getElementById("chat-container");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    function scrollToBottom() {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    async function sendMessage(e) {
      e.preventDefault();
      const text = userInput.value.trim();
      if (!text) return;

      // Add user message
      const userDiv = document.createElement("div");
      userDiv.className = "message user";
      userDiv.innerHTML = `<div class="role-label">You</div>${escapeHtml(text)}`;
      chatContainer.appendChild(userDiv);
      userInput.value = "";
      sendBtn.disabled = true;

      // Add temporary bot thinking message
      const botDiv = document.createElement("div");
      botDiv.className = "message bot typing";
      botDiv.innerHTML = `<div class="role-label">MyLLM</div>Thinking...`;
      chatContainer.appendChild(botDiv);
      scrollToBottom();

      try {
        const temperature = parseFloat(document.getElementById("temp").value);
        const maxTokens = parseInt(document.getElementById("tokens").value);

        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: text, temperature: temperature, max_tokens: maxTokens })
        });
        const data = await res.json();

        botDiv.className = "message bot";
        botDiv.innerHTML = `<div class="role-label">MyLLM</div>${escapeHtml(data.response || "No response generated.")}`;
      } catch (err) {
        botDiv.className = "message bot";
        botDiv.innerHTML = `<div class="role-label">Error</div>Failed to connect to model: ${err.message}`;
      } finally {
        sendBtn.disabled = false;
        userInput.focus();
        scrollToBottom();
      }
    }

    function escapeHtml(str) {
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json or {}
    prompt = data.get("prompt", "")
    max_tokens = int(data.get("max_tokens", 80))
    temperature = float(data.get("temperature", 0.7))
    top_k = int(data.get("top_k", 40))

    if not prompt:
        return jsonify({"response": ""})

    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        prompt_ids = [0]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            # Optional early break on EOS
            eos_id = tokenizer.token_to_id("[EOS]")
            if eos_id is not None and next_token.item() == eos_id:
                break

    # Decode newly generated tokens only
    new_tokens = idx[0][len(prompt_ids):].tolist()
    generated_text = tokenizer.decode(new_tokens)

    return jsonify({"response": generated_text.strip()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_best.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer_combined.json")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    load_model(args.checkpoint, args.tokenizer)
    print(f"\nStarting MyLLM Chat Web UI at http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)
