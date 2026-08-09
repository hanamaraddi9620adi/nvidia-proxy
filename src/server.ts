/**
 * The local proxy.
 *
 * Listens on 127.0.0.1 and pretends to be api.anthropic.com. Claude Code sends
 * it Anthropic-shaped requests; it forwards NVIDIA-shaped ones upstream and
 * translates the replies back.
 *
 * It binds to the loopback interface only and is never exposed to the network.
 * The Authorization header Claude Code sends is ignored on purpose: the real
 * credential is the NVIDIA key held in ~/.nvproxy/config.json, and the agent
 * has no way to know it.
 */

import http from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';

import fs from 'node:fs';

import {
  CONFIG_FILE,
  activeProfile,
  clearServerRecord,
  loadConfig,
  resolveApiKey,
  writeServerRecord,
  type Config,
  type Profile,
} from './config.js';
import { NvidiaError, chat, streamChat, type ClientOptions } from './nvidia.js';
import { StreamTranslator, formatSse } from './stream.js';
import {
  anthropicToOpenAi,
  estimateTokens,
  openAiToAnthropic,
  type AnthropicRequest,
} from './translate.js';

/** Refuse absurd bodies rather than letting one request exhaust memory. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

export interface ServerOptions {
  config: Config;
  /** Print one line per request. */
  verbose?: boolean;
  /**
   * Re-read ~/.nvproxy/config.json when it changes, so `nvp use` takes effect
   * without a restart. The CLI turns this on; it defaults to off so that a
   * caller passing an explicit config - a test, most importantly - always gets
   * exactly that config and can never accidentally reach the real API.
   */
  reloadFromDisk?: boolean;
}

// ---------------------------------------------------------------------------
// Model routing
// ---------------------------------------------------------------------------

/**
 * Decides which NVIDIA model answers a request.
 *
 * Coding agents ask for a tier, not a specific model. Claude Code uses a haiku
 * model for cheap background work (summarising, titling) and a larger one for
 * real work, so the two are mapped separately. A caller that already knows an
 * NVIDIA model id can name it directly and it passes straight through, which
 * is what makes `nvp chat --model x` and manual curl testing work.
 */
export function pickModel(requested: string, profile: Profile): string {
  // NVIDIA ids always contain a publisher prefix, e.g. "moonshotai/kimi-k2".
  // Anthropic ids never do. That is a reliable way to tell them apart.
  if (requested.includes('/')) return requested;

  const lower = requested.toLowerCase();
  if (lower.includes('haiku')) return profile.small || profile.big;
  return profile.big;
}

// ---------------------------------------------------------------------------
// Small HTTP helpers
// ---------------------------------------------------------------------------

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error('Request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

/** Anthropic's error envelope, so the agent renders a sensible message. */
function sendError(res: ServerResponse, status: number, message: string): void {
  const type =
    status === 401 || status === 403
      ? 'authentication_error'
      : status === 429
        ? 'rate_limit_error'
        : status === 404
          ? 'not_found_error'
          : status >= 500
            ? 'api_error'
            : 'invalid_request_error';
  sendJson(res, status, { type: 'error', error: { type, message } });
}

/** Total characters in a request, used only for the token estimate. */
function requestChars(request: AnthropicRequest): number {
  return JSON.stringify(request.messages ?? []).length + JSON.stringify(request.system ?? '').length;
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async function handleMessages(
  req: IncomingMessage,
  res: ServerResponse,
  client: ClientOptions,
  profile: Profile,
  verbose: boolean,
): Promise<void> {
  const raw = await readBody(req);

  let request: AnthropicRequest;
  try {
    request = JSON.parse(raw) as AnthropicRequest;
  } catch {
    sendError(res, 400, 'Request body was not valid JSON.');
    return;
  }

  if (!Array.isArray(request.messages)) {
    sendError(res, 400, 'Request is missing the required "messages" array.');
    return;
  }

  const model = pickModel(request.model ?? '', profile);
  if (!model) {
    sendError(res, 400, 'No model is selected. Run `nvp use` to choose one.');
    return;
  }

  const body = anthropicToOpenAi(request, {
    model,
    maxTokens: profile.maxTokens,
    temperature: profile.temperature,
  });

  if (verbose) {
    const tools = request.tools?.length ? `, ${request.tools.length} tools` : '';
    log(`${request.model || '?'} -> ${model} (${body.messages.length} msgs${tools}${request.stream ? ', stream' : ''})`);
  }

  if (request.stream) {
    await streamResponse(res, client, body, request, profile, model);
  } else {
    try {
      const reply = await chat(client, body);
      sendJson(res, 200, openAiToAnthropic(reply, model, profile.showReasoning, requestChars(request)));
    } catch (error) {
      const status = error instanceof NvidiaError ? error.status || 502 : 502;
      sendError(res, status, (error as Error).message);
    }
  }
}

async function streamResponse(
  res: ServerResponse,
  client: ClientOptions,
  body: ReturnType<typeof anthropicToOpenAi>,
  request: AnthropicRequest,
  profile: Profile,
  model: string,
): Promise<void> {
  const controller = new AbortController();
  // If the agent hangs up (user pressed Esc), stop paying for the generation.
  res.on('close', () => controller.abort());

  const translator = new StreamTranslator(
    model,
    profile.showReasoning,
    estimateTokens('x'.repeat(requestChars(request))),
  );

  let headersSent = false;
  const write = (chunk: string) => {
    if (!res.writableEnded) res.write(chunk);
  };

  try {
    const stream = streamChat(client, body, controller.signal);

    // Only commit to a 200 once upstream has actually produced something.
    // Before the first chunk arrives we can still return a clean error.
    for await (const chunk of stream) {
      if (!headersSent) {
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache, no-transform',
          Connection: 'keep-alive',
          'X-Accel-Buffering': 'no',
        });
        headersSent = true;
        for (const event of translator.start()) write(formatSse(event));
      }
      for (const event of translator.handleChunk(chunk)) write(formatSse(event));
    }

    if (!headersSent) {
      // Upstream closed without sending anything at all.
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
      headersSent = true;
      for (const event of translator.start()) write(formatSse(event));
    }

    for (const event of translator.finish()) write(formatSse(event));
    if (!res.writableEnded) res.end();
  } catch (error) {
    if (controller.signal.aborted) {
      if (!res.writableEnded) res.end();
      return;
    }

    const message = (error as Error).message;
    if (!headersSent) {
      const status = error instanceof NvidiaError ? error.status || 502 : 502;
      sendError(res, status, message);
      return;
    }

    // Mid-stream failure. The agent has already begun rendering, so the only
    // correct move is an SSE error event followed by a clean close.
    write(
      formatSse({
        event: 'error',
        data: { type: 'error', error: { type: 'api_error', message } },
      }),
    );
    for (const event of translator.finish()) write(formatSse(event));
    if (!res.writableEnded) res.end();
  }
}

/**
 * Claude Code calls this before long requests to check they will fit.
 * NVIDIA exposes no tokeniser, so this is the ~4-chars-per-token estimate.
 * Being approximate is fine here; it is a guard rail, not an invoice.
 */
async function handleCountTokens(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const raw = await readBody(req);
  let request: AnthropicRequest;
  try {
    request = JSON.parse(raw) as AnthropicRequest;
  } catch {
    sendError(res, 400, 'Request body was not valid JSON.');
    return;
  }
  sendJson(res, 200, { input_tokens: estimateTokens('x'.repeat(requestChars(request))) });
}

function log(message: string): void {
  const time = new Date().toTimeString().slice(0, 8);
  process.stdout.write(`  ${time}  ${message}\n`);
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

export function createServer(options: ServerOptions): http.Server {
  const { verbose = false, reloadFromDisk = false } = options;

  // When enabled, the config is reloaded whenever the file on disk changes, so
  // `nvp use` takes effect on the very next request without restarting the
  // proxy. The mtime check keeps this to one stat() per request rather than a
  // full parse.
  let cached: Config = options.config;
  let cachedMtime = 0;

  const currentConfig = (): Config => {
    if (!reloadFromDisk) return cached;
    try {
      const mtime = fs.statSync(CONFIG_FILE).mtimeMs;
      if (mtime !== cachedMtime) {
        cached = loadConfig();
        cachedMtime = mtime;
      }
    } catch {
      // No file on disk (tests pass a config in directly): keep what we have.
    }
    return cached;
  };

  return http.createServer((req, res) => {
    void (async () => {
      const url = (req.url ?? '/').split('?')[0] ?? '/';
      const config = currentConfig();

      try {
        // Health check, used by the launchers to tell whether the port is ours.
        if (req.method === 'GET' && (url === '/health' || url === '/')) {
          const profile = activeProfile(config);
          sendJson(res, 200, {
            service: 'nvidia-proxy',
            ok: true,
            profile: config.activeProfile,
            big: profile.big,
            small: profile.small,
          });
          return;
        }

        const apiKey = resolveApiKey(config);
        if (!apiKey) {
          sendError(res, 401, 'No NVIDIA API key is configured. Run `nvp key set`.');
          return;
        }
        const client: ClientOptions = { apiKey, baseUrl: config.baseUrl };
        const profile = activeProfile(config);

        if (req.method === 'POST' && url === '/v1/messages') {
          await handleMessages(req, res, client, profile, verbose);
          return;
        }

        if (req.method === 'POST' && url === '/v1/messages/count_tokens') {
          await handleCountTokens(req, res);
          return;
        }

        sendError(res, 404, `No route for ${req.method} ${url}.`);
      } catch (error) {
        if (!res.headersSent) {
          sendError(res, 500, `Proxy error: ${(error as Error).message}`);
        } else if (!res.writableEnded) {
          res.end();
        }
      }
    })();
  });
}

/** Starts the server and resolves once it is accepting connections. */
export function startServer(options: ServerOptions & { port: number }): Promise<http.Server> {
  // The CLI is the one caller that should follow the file on disk.
  const server = createServer({ reloadFromDisk: true, ...options });

  return new Promise((resolve, reject) => {
    server.once('error', (error: NodeJS.ErrnoException) => {
      if (error.code === 'EADDRINUSE') {
        reject(
          new Error(
            `Port ${options.port} is already in use. Either nvp is already running (\`nvp status\`), or another program has the port. Change it with \`nvp config port <number>\`.`,
          ),
        );
      } else {
        reject(error);
      }
    });

    // Loopback only. This must never become 0.0.0.0: the proxy accepts any
    // Authorization header, so anyone who could reach it could spend the key.
    server.listen(options.port, '127.0.0.1', () => {
      writeServerRecord({ port: options.port, pid: process.pid, startedAt: Date.now() });
      resolve(server);
    });
  });
}

/** True if something on this port answers /health as our proxy. */
export async function isOurServerRunning(port: number): Promise<boolean> {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/health`, {
      signal: AbortSignal.timeout(1500),
    });
    if (!response.ok) return false;
    const payload = (await response.json()) as { service?: string };
    return payload.service === 'nvidia-proxy';
  } catch {
    return false;
  }
}

export function stopServer(server: http.Server): Promise<void> {
  return new Promise((resolve) => {
    clearServerRecord();
    server.close(() => resolve());
  });
}
