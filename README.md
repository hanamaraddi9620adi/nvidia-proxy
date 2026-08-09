# nvp — NVIDIA models in Claude Code, Codex, OpenCode, or your terminal

NVIDIA gives away free API access to 100+ open-weight models — Kimi K2, their own
Nemotron family, Qwen3-Coder 480B, DeepSeek, GLM — at
[build.nvidia.com](https://build.nvidia.com). `nvp` lets your coding agents use them.

**You set it up once.** After that `nvp claude` just works, forever. There is no
reconnecting, no re-pasting keys, no re-picking models.

```bash
nvp setup      # paste your key, pick your models — once
nvp claude     # Claude Code, running on NVIDIA
```

---

## What it actually does

Three tools, two different mechanisms:

| Tool | How it connects | Needs the proxy? |
|---|---|---|
| **Claude Code** | Speaks only the Anthropic API, so `nvp` runs a local translator on `127.0.0.1` | Yes — starts by itself |
| **Codex** | Speaks OpenAI's dialect, which is what NVIDIA serves | No — direct |
| **OpenCode** | Same | No — direct |
| **`nvp chat`** | Built in | No — direct |

Only Claude Code needs translating. Sending Codex or OpenCode through a proxy would
add a pointless round-trip, so `nvp` writes them a config entry instead and gets out
of the way.

### The translation, briefly

Claude Code sends Anthropic-shaped requests; NVIDIA expects OpenAI-shaped ones. The
proxy reconciles four genuine differences:

1. **System prompt** — a top-level field for Anthropic, the first message for OpenAI.
2. **Content** — typed blocks even for plain text vs. a bare string.
3. **Tool results** — a block inside a *user* message vs. a separate `tool` message.
4. **Streaming** — Anthropic's structured `content_block_start/delta/stop` events vs.
   OpenAI's flat deltas.

The fourth is the fiddly one and is where most homemade bridges break: a tool call's
JSON arguments arrive in fragments that must be reassembled into correctly ordered
Anthropic blocks. There are tests for exactly that.

---

## Install

Needs **Node 20+**. It has **zero runtime dependencies**.

```bash
git clone https://github.com/hanamaraddi9620adi/nvidia-proxy.git
cd nvidia-proxy
npm install
npm install -g .
```

Then get a free key:

1. Go to <https://build.nvidia.com/settings/api-keys>
2. Sign in (free NVIDIA Developer account, no card)
3. **Generate API Key** → copy the `nvapi-…` value

```bash
nvp setup
```

It asks for the key (never echoed to screen), verifies it against NVIDIA, then shows
you a searchable menu of every model your key can reach. Pick a main model and a
small background one. Done.

---

## Commands

### Daily use

```bash
nvp claude              # Claude Code on NVIDIA models
nvp codex               # Codex on NVIDIA models
nvp opencode            # OpenCode on NVIDIA models
nvp chat                # a chat in this terminal
```

Any extra arguments pass straight through: `nvp claude --resume`, `nvp chat -m qwen/qwen3-coder-480b-a35b-instruct`.

### Models

```bash
nvp models              # everything your key can reach
nvp models kimi         # filter
nvp models --refresh    # bypass the 24h cache
nvp use                 # pick the main model from a menu
nvp use kimi            # partial names work
nvp use --small nvidia/llama-3.1-nemotron-nano-8b-v1
```

### Profiles

A profile is a saved pair of models. Keep one for heavy work and one for fast work,
and switch in a second.

```bash
nvp profile                    # list
nvp profile new fast           # clone current settings under a new name
nvp profile use fast           # switch
nvp profile delete fast
```

### Settings

```bash
nvp set                        # show everything
nvp set port 9000
nvp set maxTokens 16384
nvp set temperature 0.3        # or "default" to let the model decide
nvp set showReasoning true     # see chain-of-thought (off by default)
```

### Housekeeping

```bash
nvp status      # what's configured, what's running, which agents exist
nvp doctor      # checks everything and tells you what's wrong
nvp key set     # replace the key
nvp key clear   # remove it
nvp start -v    # run the proxy in the foreground with request logging
nvp stop
```

---

## Where things are stored

```
~/.nvproxy/config.json    your key, port, and every profile   (chmod 600)
~/.nvproxy/models.json    cached model catalogue
~/.nvproxy/server.json    the running proxy's port and pid
```

`nvp` also writes small, clearly-marked blocks into the two agents that need them:

- **`~/.codex/config.toml`** — adds a provider and a `[profiles.nvp]` entry between
  `# >>> managed by nvp` markers. Plain `codex` is untouched; only
  `codex --profile nvp` goes to NVIDIA.
- **`~/.config/opencode/opencode.json`** — merges in an `nvidia` provider, leaving
  any other providers alone.

Neither file has your key written into it. Both reference `NVIDIA_API_KEY`, which
`nvp` supplies to the child process at launch.

---

## Notes worth knowing

**The free tier is rate limited** to roughly 40 requests a minute. Coding agents are
chatty. If you see `429`, that's why — wait a moment.

**Open-weight models are not Claude.** Kimi K2 and Qwen3-Coder are genuinely strong
at code, but tool-calling discipline over a long agent session is exactly where
open models are weakest. Expect to correct more often than you would on Claude.

**NVIDIA does not host Claude, GPT, or Gemini.** Anthropic, OpenAI and Google models
are only available from their own APIs. Anything claiming otherwise is confused.

**The proxy binds to `127.0.0.1` only.** It accepts any `Authorization` header,
because the real credential is your NVIDIA key held server-side. That is safe on
loopback and would be dangerous on `0.0.0.0` — so it never binds there.

---

## Development

```bash
npm run build      # compile TypeScript
npm test           # build, then run the test suite
npm run dev        # tsc --watch
```

The tests need no API key. A fake upstream server stands in for NVIDIA and replies
with hand-written OpenAI payloads, so the suite asserts one thing: an Anthropic
client talking to the proxy sees a correct Anthropic conversation.

```
src/
  cli.ts          command dispatch — the whole user-facing surface
  config.ts       ~/.nvproxy/config.json, profiles, catalogue cache
  nvidia.ts       NVIDIA client: model list, chat, SSE stream parsing
  translate.ts    Anthropic <-> OpenAI request/response conversion
  stream.ts       the streaming state machine (the tricky part)
  server.ts       the local HTTP proxy
  launchers.ts    Claude Code / Codex / OpenCode config and spawning
  chat.ts         the terminal REPL
  ui.ts           colours, prompts, the model picker
```

---

## Licence

MIT.

Not affiliated with NVIDIA, Anthropic, OpenAI, or the OpenCode project.
