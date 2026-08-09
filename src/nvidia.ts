/**
 * The NVIDIA NIM client.
 *
 * NVIDIA's hosted catalogue at https://integrate.api.nvidia.com/v1 speaks the
 * OpenAI chat-completions dialect, so this file is a thin, honest wrapper over
 * fetch. Node 20+ has fetch built in, which is why this package needs no
 * runtime dependencies at all.
 *
 * The model list is always fetched from the API. It is never hardcoded here:
 * NVIDIA adds and retires models constantly, and a baked-in list would be
 * wrong within weeks.
 */

import { readCatalogueCache, writeCatalogueCache } from './config.js';

export interface NvidiaModel {
  id: string;
  /** NVIDIA returns the publisher here, e.g. "moonshotai". */
  ownedBy: string;
}

/** Raised for any non-2xx reply, carrying the status so callers can explain it. */
export class NvidiaError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: string = '',
  ) {
    super(message);
    this.name = 'NvidiaError';
  }
}

export interface ClientOptions {
  apiKey: string;
  baseUrl: string;
}

/** Turns an HTTP failure into a sentence a human can act on. */
function describeFailure(status: number, body: string): string {
  const trimmed = body.slice(0, 400).trim();
  switch (status) {
    case 401:
      return 'NVIDIA rejected the API key (401). Run `nvp key set` and paste a current key from build.nvidia.com/settings/api-keys.';
    case 403:
      return 'NVIDIA refused access (403). The key may lack access to this model, or your free credits may be exhausted.';
    case 404:
      return `NVIDIA has no such model (404). Run \`nvp models --refresh\` and pick again.${trimmed ? ` Detail: ${trimmed}` : ''}`;
    case 429:
      return 'Rate limited (429). The free tier allows roughly 40 requests a minute. Wait a moment and retry.';
    case 500:
    case 502:
    case 503:
    case 504:
      return `NVIDIA is having trouble (${status}). This is upstream, not your setup. Retry shortly.`;
    default:
      return `NVIDIA returned ${status}.${trimmed ? ` ${trimmed}` : ''}`;
  }
}

async function request(
  opts: ClientOptions,
  route: string,
  init: RequestInit = {},
): Promise<Response> {
  const url = `${opts.baseUrl.replace(/\/+$/, '')}${route}`;

  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: {
        Authorization: `Bearer ${opts.apiKey}`,
        'Content-Type': 'application/json',
        Accept: init.headers && 'Accept' in (init.headers as Record<string, string>)
          ? (init.headers as Record<string, string>)['Accept']!
          : 'application/json',
        ...(init.headers as Record<string, string> | undefined),
      },
    });
  } catch (cause) {
    // DNS failure, no route, TLS problem, offline.
    throw new NvidiaError(
      `Could not reach ${url}. Check your internet connection. (${(cause as Error).message})`,
      0,
    );
  }

  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new NvidiaError(describeFailure(response.status, body), response.status, body);
  }
  return response;
}

// ---------------------------------------------------------------------------
// Catalogue
// ---------------------------------------------------------------------------

/**
 * Every model the key can reach, sorted so the ones worth using for coding
 * float to the top. Results are cached for a day; pass force to bypass.
 */
export async function listModels(
  opts: ClientOptions,
  force = false,
): Promise<string[]> {
  if (!force) {
    const cached = readCatalogueCache();
    if (cached) return cached;
  }

  const response = await request(opts, '/models', { method: 'GET' });
  const payload = (await response.json()) as { data?: Array<{ id?: string }> };

  const ids = (payload.data ?? [])
    .map((m) => m.id)
    .filter((id): id is string => typeof id === 'string' && id.length > 0);

  const sorted = [...new Set(ids)].sort(byCodingUsefulness);
  writeCatalogueCache(sorted);
  return sorted;
}

/**
 * Heuristic ranking, not a hardcoded catalogue.
 *
 * The families listed below are the ones currently strong at agentic coding.
 * When NVIDIA adds a better one it still appears in the list — it just is not
 * promoted until someone adds its name here. That is the right failure mode:
 * a new model is never hidden, only unranked.
 */
const PREFERRED_FAMILIES = [
  'qwen3-coder',
  'kimi-k2',
  'kimi',
  'deepseek',
  'nemotron',
  'glm',
  'qwen3',
  'llama-3.3',
  'mistral',
];

function familyRank(id: string): number {
  const lower = id.toLowerCase();
  const index = PREFERRED_FAMILIES.findIndex((f) => lower.includes(f));
  return index === -1 ? PREFERRED_FAMILIES.length : index;
}

function byCodingUsefulness(a: string, b: string): number {
  const rank = familyRank(a) - familyRank(b);
  return rank !== 0 ? rank : a.localeCompare(b);
}

/**
 * Best guess at a sensible pair for a first-time setup: a strong coding model
 * for real work, and something small and fast for background chores.
 */
export function suggestDefaults(models: string[]): { big: string; small: string } {
  const find = (...needles: string[]): string | undefined =>
    models.find((m) => needles.every((n) => m.toLowerCase().includes(n)));

  const big =
    find('qwen3-coder') ??
    find('kimi', 'thinking') ??
    find('kimi') ??
    find('deepseek') ??
    find('nemotron', 'ultra') ??
    models[0] ??
    '';

  const small =
    find('nemotron', 'super') ??
    find('nemotron', 'nano') ??
    find('llama-3.3', '70b') ??
    find('8b') ??
    find('9b') ??
    find('mini') ??
    big;

  return { big, small };
}

/** Confirms a key works by asking for the catalogue. Returns the model count. */
export async function validateKey(opts: ClientOptions): Promise<number> {
  const models = await listModels(opts, true);
  return models.length;
}

export interface ProbeResult {
  ok: boolean;
  status: number;
  /** Short human-readable reason when ok is false. */
  reason: string;
  /** True when the model exists but this account may not invoke it. */
  noAccess: boolean;
  /** True when the model is fine but was momentarily busy or rate limited. */
  transient: boolean;
}

/**
 * Actually calls a model with a one-token request.
 *
 * This exists because of a genuinely surprising NVIDIA behaviour: GET /models
 * returns the whole public catalogue, including models your account cannot
 * invoke. Asking for one of those returns
 *
 *   404 "Function '<uuid>': Not found for account '<id>'"
 *
 * which reads like the model does not exist, when it means you lack access to
 * it. Listing a model is therefore not evidence you can use it, and the only
 * reliable test is to call it. Every place that lets you choose a model runs
 * this first.
 */
export async function probeModel(opts: ClientOptions, model: string): Promise<ProbeResult> {
  try {
    const response = await fetch(`${opts.baseUrl.replace(/\/+$/, '')}/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${opts.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 1,
        stream: false,
      }),
      signal: AbortSignal.timeout(45_000),
    });

    if (response.ok) {
      return { ok: true, status: 200, reason: '', noAccess: false, transient: false };
    }

    const body = await response.text().catch(() => '');

    // The account-scoped 404 described above.
    if (response.status === 404) {
      return {
        ok: false,
        status: 404,
        reason: 'your key has no access to this model',
        noAccess: true,
        transient: false,
      };
    }

    // Busy or throttled: the model is fine, the moment is not.
    if (response.status === 429 || response.status === 503 || response.status === 504) {
      return {
        ok: false,
        status: response.status,
        reason:
          response.status === 429
            ? 'rate limited, try again shortly'
            : 'model is busy right now',
        noAccess: false,
        transient: true,
      };
    }

    // A 400 usually means the model is reachable but wants a different request
    // shape (an embedding or parsing model, say). Reachable, but not for chat.
    if (response.status === 400) {
      return {
        ok: false,
        status: 400,
        reason: 'not a chat model',
        noAccess: false,
        transient: false,
      };
    }

    return {
      ok: false,
      status: response.status,
      reason: `HTTP ${response.status} ${body.slice(0, 120)}`.trim(),
      noAccess: false,
      transient: false,
    };
  } catch (error) {
    return {
      ok: false,
      status: 0,
      reason: `could not reach NVIDIA (${(error as Error).message})`,
      noAccess: false,
      transient: true,
    };
  }
}

/**
 * Probes many models, paced to stay under the free tier's ~40 requests per
 * minute. `onResult` is called as each finishes so the caller can show
 * progress rather than sitting silent for minutes.
 */
export async function probeMany(
  opts: ClientOptions,
  models: string[],
  onResult: (model: string, result: ProbeResult, done: number, total: number) => void,
): Promise<string[]> {
  const usable: string[] = [];
  for (let i = 0; i < models.length; i++) {
    const model = models[i]!;
    const result = await probeModel(opts, model);
    if (result.ok) usable.push(model);
    onResult(model, result, i + 1, models.length);
    // ~35 requests a minute, comfortably inside the limit.
    if (i < models.length - 1) await new Promise((r) => setTimeout(r, 1700));
  }
  return usable;
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

export interface OpenAiMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | Array<Record<string, unknown>> | null;
  name?: string;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
}

export interface ChatRequest {
  model: string;
  messages: OpenAiMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
  stream?: boolean;
  tools?: Array<{
    type: 'function';
    function: { name: string; description?: string; parameters: unknown };
  }>;
  tool_choice?: unknown;
}

export interface ChatChoiceDelta {
  role?: string;
  content?: string | null;
  /** Reasoning models return their private chain of thought here. */
  reasoning_content?: string | null;
  tool_calls?: Array<{
    index: number;
    id?: string;
    type?: 'function';
    function?: { name?: string; arguments?: string };
  }>;
}

export interface ChatChunk {
  id?: string;
  model?: string;
  choices?: Array<{
    index?: number;
    delta?: ChatChoiceDelta;
    message?: ChatChoiceDelta;
    finish_reason?: string | null;
  }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
}

/** One-shot completion. Used by `nvp chat` and by non-streaming proxy calls. */
export async function chat(opts: ClientOptions, body: ChatRequest): Promise<ChatChunk> {
  const response = await request(opts, '/chat/completions', {
    method: 'POST',
    body: JSON.stringify({ ...body, stream: false }),
  });
  return (await response.json()) as ChatChunk;
}

/**
 * Streaming completion, yielded chunk by chunk.
 *
 * Server-sent events arrive as `data: {...}` lines separated by blank lines,
 * terminated by `data: [DONE]`. A chunk can be split across TCP reads, so the
 * loop below keeps a buffer and only parses complete lines.
 */
export async function* streamChat(
  opts: ClientOptions,
  body: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<ChatChunk> {
  const response = await request(opts, '/chat/completions', {
    method: 'POST',
    body: JSON.stringify({ ...body, stream: true }),
    headers: { Accept: 'text/event-stream' },
    ...(signal ? { signal } : {}),
  });

  if (!response.body) {
    throw new NvidiaError('NVIDIA returned an empty stream.', 0);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process every complete line, keeping the trailing partial one.
      let newline: number;
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);

        if (!line || line.startsWith(':')) continue; // blank or SSE comment
        if (!line.startsWith('data:')) continue;

        const data = line.slice(5).trim();
        if (data === '[DONE]') return;

        try {
          yield JSON.parse(data) as ChatChunk;
        } catch {
          // A malformed chunk is not worth killing the stream over.
          continue;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
