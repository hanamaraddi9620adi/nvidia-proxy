/**
 * Anthropic Messages API  <->  OpenAI chat-completions.
 *
 * Claude Code speaks the Anthropic dialect and nothing else. NVIDIA speaks the
 * OpenAI dialect. This file is the interpreter between them, and it is the only
 * genuinely tricky part of the project.
 *
 * The two formats disagree about four things:
 *
 *   1. System prompt  - Anthropic keeps it in a top-level `system` field;
 *                       OpenAI makes it the first message.
 *   2. Content        - Anthropic content is a list of typed blocks even for
 *                       plain text; OpenAI usually wants a bare string.
 *   3. Tool results   - Anthropic sends them as a `tool_result` block inside a
 *                       *user* message; OpenAI needs a separate `tool` message.
 *   4. Streaming      - Anthropic emits a structured event sequence with
 *                       explicit block open/close; OpenAI emits flat deltas.
 *
 * Everything below exists to reconcile those four disagreements.
 */

import type { ChatChunk, ChatRequest, OpenAiMessage } from './nvidia.js';

// ---------------------------------------------------------------------------
// Anthropic request shapes (only the parts we actually consume)
// ---------------------------------------------------------------------------

export interface AnthropicTextBlock {
  type: 'text';
  text: string;
}

export interface AnthropicImageBlock {
  type: 'image';
  source: { type: 'base64'; media_type: string; data: string } | { type: 'url'; url: string };
}

export interface AnthropicToolUseBlock {
  type: 'tool_use';
  id: string;
  name: string;
  input: unknown;
}

export interface AnthropicToolResultBlock {
  type: 'tool_result';
  tool_use_id: string;
  content?: string | Array<AnthropicTextBlock | AnthropicImageBlock | Record<string, unknown>>;
  is_error?: boolean;
}

export type AnthropicBlock =
  | AnthropicTextBlock
  | AnthropicImageBlock
  | AnthropicToolUseBlock
  | AnthropicToolResultBlock
  | { type: string; [k: string]: unknown };

export interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string | AnthropicBlock[];
}

export interface AnthropicRequest {
  model: string;
  messages: AnthropicMessage[];
  system?: string | AnthropicBlock[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop_sequences?: string[];
  stream?: boolean;
  tools?: Array<{ name: string; description?: string; input_schema?: unknown }>;
  tool_choice?: { type: 'auto' | 'any' | 'tool' | 'none'; name?: string };
  metadata?: unknown;
}

// ---------------------------------------------------------------------------
// Request: Anthropic -> OpenAI
// ---------------------------------------------------------------------------

/** Flattens Anthropic's system field (string or block list) into one string. */
function flattenSystem(system: AnthropicRequest['system']): string {
  if (!system) return '';
  if (typeof system === 'string') return system;
  return system
    .filter((b): b is AnthropicTextBlock => b.type === 'text')
    .map((b) => b.text)
    .join('\n\n');
}

/** Anthropic image blocks become OpenAI `image_url` parts with a data: URI. */
function imageToOpenAi(block: AnthropicImageBlock): Record<string, unknown> | null {
  const source = block.source;
  if (!source) return null;
  if (source.type === 'url') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  if (source.type === 'base64' && source.data) {
    return {
      type: 'image_url',
      image_url: { url: `data:${source.media_type || 'image/png'};base64,${source.data}` },
    };
  }
  return null;
}

/** Reduces a tool_result's content down to the plain text OpenAI expects. */
function toolResultText(block: AnthropicToolResultBlock): string {
  const { content } = block;
  if (content == null) return '';
  if (typeof content === 'string') return content;
  return content
    .map((part) => {
      if (typeof part === 'string') return part;
      if ((part as AnthropicTextBlock).type === 'text') return (part as AnthropicTextBlock).text;
      // An image returned by a tool cannot travel in an OpenAI tool message.
      // Say so plainly rather than dropping it silently.
      if ((part as AnthropicBlock).type === 'image') return '[image omitted]';
      return JSON.stringify(part);
    })
    .join('\n');
}

/**
 * Converts one Anthropic message into one or more OpenAI messages.
 *
 * The count can grow: a single Anthropic user message holding three
 * tool_result blocks becomes three OpenAI `tool` messages.
 */
function messageToOpenAi(message: AnthropicMessage): OpenAiMessage[] {
  if (typeof message.content === 'string') {
    return [{ role: message.role, content: message.content }];
  }

  const blocks = message.content ?? [];
  const out: OpenAiMessage[] = [];

  // Tool results must be emitted as their own `tool` messages, and they have
  // to come before the message carrying the remaining blocks, because OpenAI
  // requires every tool_call to be answered before the next user turn.
  for (const block of blocks) {
    if (block.type === 'tool_result') {
      const result = block as AnthropicToolResultBlock;
      const text = toolResultText(result);
      out.push({
        role: 'tool',
        tool_call_id: result.tool_use_id,
        content: result.is_error ? `ERROR: ${text}` : text,
      });
    }
  }

  const textParts: Array<Record<string, unknown>> = [];
  const toolCalls: NonNullable<OpenAiMessage['tool_calls']> = [];

  for (const block of blocks) {
    if (block.type === 'text') {
      textParts.push({ type: 'text', text: (block as AnthropicTextBlock).text });
    } else if (block.type === 'image') {
      const image = imageToOpenAi(block as AnthropicImageBlock);
      if (image) textParts.push(image);
    } else if (block.type === 'tool_use') {
      const use = block as AnthropicToolUseBlock;
      toolCalls.push({
        id: use.id,
        type: 'function',
        function: {
          name: use.name,
          // OpenAI wants the arguments as a JSON *string*, not an object.
          arguments: JSON.stringify(use.input ?? {}),
        },
      });
    }
    // tool_result was handled above; unknown block types are ignored.
  }

  if (textParts.length > 0 || toolCalls.length > 0) {
    // Collapse a lone text part back to a bare string. Some models handle the
    // simple form noticeably better than the array form.
    const onlyText =
      textParts.length > 0 && textParts.every((p) => p['type'] === 'text');
    const content = onlyText
      ? textParts.map((p) => String(p['text'] ?? '')).join('')
      : textParts.length > 0
        ? textParts
        : null;

    const msg: OpenAiMessage = { role: message.role, content };
    if (toolCalls.length > 0) {
      msg.tool_calls = toolCalls;
      // An assistant message carrying tool calls must not also carry null
      // content in some implementations; an empty string is the safe form.
      if (msg.content === null) msg.content = '';
    }
    out.push(msg);
  }

  return out;
}

function toolChoiceToOpenAi(choice: AnthropicRequest['tool_choice']): unknown | undefined {
  if (!choice) return undefined;
  switch (choice.type) {
    case 'auto':
      return 'auto';
    case 'any':
      return 'required';
    case 'none':
      return 'none';
    case 'tool':
      return choice.name
        ? { type: 'function', function: { name: choice.name } }
        : 'required';
    default:
      return undefined;
  }
}

export interface TranslateOptions {
  /** The NVIDIA model id to actually call. */
  model: string;
  maxTokens: number;
  temperature: number | null;
}

/** Builds the OpenAI request body from an Anthropic one. */
export function anthropicToOpenAi(
  request: AnthropicRequest,
  options: TranslateOptions,
): ChatRequest {
  const messages: OpenAiMessage[] = [];

  const system = flattenSystem(request.system);
  if (system) messages.push({ role: 'system', content: system });

  for (const message of request.messages ?? []) {
    messages.push(...messageToOpenAi(message));
  }

  const body: ChatRequest = {
    model: options.model,
    messages,
    // Honour the agent's own ceiling when it sends one, else the profile's.
    max_tokens: request.max_tokens ?? options.maxTokens,
  };

  // Only send temperature if somebody actually chose one. Reasoning models
  // behave badly when a temperature is forced on them.
  if (typeof request.temperature === 'number') {
    body.temperature = request.temperature;
  } else if (options.temperature !== null) {
    body.temperature = options.temperature;
  }

  if (typeof request.top_p === 'number') body.top_p = request.top_p;
  if (request.stop_sequences?.length) body.stop = request.stop_sequences;

  if (request.tools?.length) {
    body.tools = request.tools.map((tool) => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description ?? '',
        // Anthropic calls it input_schema, OpenAI calls it parameters.
        // Both are JSON Schema, so the value passes through untouched.
        parameters: tool.input_schema ?? { type: 'object', properties: {} },
      },
    }));
    const choice = toolChoiceToOpenAi(request.tool_choice);
    if (choice !== undefined) body.tool_choice = choice;
  }

  return body;
}

// ---------------------------------------------------------------------------
// Response: OpenAI -> Anthropic
// ---------------------------------------------------------------------------

export function mapStopReason(finish: string | null | undefined): string {
  switch (finish) {
    case 'stop':
      return 'end_turn';
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'content_filter':
      return 'end_turn';
    default:
      return 'end_turn';
  }
}

/**
 * A rough token count. NVIDIA does not expose a tokeniser over HTTP and we
 * refuse to add a dependency for one, so this uses the widely used ~4 chars
 * per token approximation. It is only ever used for reporting and for the
 * count_tokens endpoint, never for billing or truncation decisions.
 */
export function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

export function newMessageId(): string {
  return `msg_${Math.random().toString(36).slice(2, 12)}${Date.now().toString(36)}`;
}

/**
 * Strips `<think>...</think>` sections from a stream without ever needing the
 * whole string at once.
 *
 * Several open-weight models wrap their reasoning in these tags inside the
 * normal content field. Left in place, a coding agent reads the reasoning as
 * if it were the answer. The tag can be split across chunks, so anything that
 * might be the start of a tag is held back until it is proven either way.
 */
export class ThinkTagFilter {
  private buffer = '';
  private inside = false;

  constructor(private readonly enabled: boolean) {}

  push(chunk: string): string {
    if (!this.enabled) return chunk;

    this.buffer += chunk;
    let out = '';

    while (this.buffer.length > 0) {
      if (this.inside) {
        const close = this.buffer.indexOf('</think>');
        if (close === -1) {
          // Still inside; keep only enough to detect a split closing tag.
          this.buffer = this.buffer.slice(-8);
          return out;
        }
        this.buffer = this.buffer.slice(close + '</think>'.length);
        this.inside = false;
        continue;
      }

      const open = this.buffer.indexOf('<think>');
      if (open !== -1) {
        out += this.buffer.slice(0, open);
        this.buffer = this.buffer.slice(open + '<think>'.length);
        this.inside = true;
        continue;
      }

      // No tag in sight. Emit everything except a possible partial `<think>`
      // straddling the chunk boundary.
      const partial = this.trailingPartialTagLength(this.buffer);
      out += this.buffer.slice(0, this.buffer.length - partial);
      this.buffer = this.buffer.slice(this.buffer.length - partial);
      return out;
    }

    return out;
  }

  /** Length of a trailing substring that could still grow into `<think>`. */
  private trailingPartialTagLength(s: string): number {
    const tag = '<think>';
    const max = Math.min(tag.length - 1, s.length);
    for (let len = max; len > 0; len--) {
      if (s.endsWith(tag.slice(0, len))) return len;
    }
    return 0;
  }

  /** Anything still held back once the stream ends. */
  flush(): string {
    if (!this.enabled) return '';
    const rest = this.inside ? '' : this.buffer;
    this.buffer = '';
    return rest;
  }
}

/** Builds a complete Anthropic message from a non-streaming OpenAI reply. */
export function openAiToAnthropic(
  reply: ChatChunk,
  requestedModel: string,
  showReasoning: boolean,
  promptChars: number,
): Record<string, unknown> {
  const choice = reply.choices?.[0];
  const message = choice?.message ?? {};

  const content: Array<Record<string, unknown>> = [];

  const filter = new ThinkTagFilter(!showReasoning);
  const text = filter.push(String(message.content ?? '')) + filter.flush();
  if (text.trim().length > 0) {
    content.push({ type: 'text', text });
  }

  for (const call of message.tool_calls ?? []) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.function?.arguments || '{}');
    } catch {
      // A model that emits invalid JSON arguments would otherwise crash the
      // agent. Hand back the raw string under a known key instead.
      input = { _raw: call.function?.arguments ?? '' };
    }
    content.push({
      type: 'tool_use',
      id: call.id || `toolu_${Math.random().toString(36).slice(2, 12)}`,
      name: call.function?.name ?? 'unknown',
      input,
    });
  }

  // Anthropic requires at least one content block.
  if (content.length === 0) content.push({ type: 'text', text: '' });

  const hasToolUse = content.some((b) => b['type'] === 'tool_use');

  return {
    id: newMessageId(),
    type: 'message',
    role: 'assistant',
    model: requestedModel,
    content,
    stop_reason: hasToolUse ? 'tool_use' : mapStopReason(choice?.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: reply.usage?.prompt_tokens ?? estimateTokens('x'.repeat(promptChars)),
      output_tokens: reply.usage?.completion_tokens ?? estimateTokens(text),
    },
  };
}
