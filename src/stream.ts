/**
 * Streaming translator: flat OpenAI deltas -> Anthropic's structured events.
 *
 * Anthropic's stream is a little state machine. Every piece of content lives
 * in a numbered block that must be explicitly opened and closed:
 *
 *   message_start
 *     content_block_start  index 0  (text)
 *     content_block_delta  index 0  text_delta        x many
 *     content_block_stop   index 0
 *     content_block_start  index 1  (tool_use, with id and name)
 *     content_block_delta  index 1  input_json_delta  x many
 *     content_block_stop   index 1
 *   message_delta   (stop_reason, output token count)
 *   message_stop
 *
 * OpenAI just emits `delta.content` strings and `delta.tool_calls` fragments
 * with no block structure at all. This class watches that flat stream and
 * opens, switches and closes blocks at the right moments.
 *
 * Getting this wrong is what makes an agent hang or lose a tool call, so the
 * ordering rules are enforced in one place: `openBlock` / `closeBlock`.
 */

import type { ChatChunk, ChatChoiceDelta } from './nvidia.js';
import { ThinkTagFilter, estimateTokens, mapStopReason, newMessageId } from './translate.js';

export interface SseEvent {
  event: string;
  data: Record<string, unknown>;
}

type Block =
  | { kind: 'text'; index: number }
  | { kind: 'tool'; index: number; openAiIndex: number; sawArguments: boolean };

export class StreamTranslator {
  private readonly messageId = newMessageId();
  private readonly filter: ThinkTagFilter;

  private nextIndex = 0;
  private current: Block | null = null;
  /** OpenAI tool_call index -> the Anthropic block we opened for it. */
  private readonly toolBlocks = new Map<number, Block & { kind: 'tool' }>();

  private outputChars = 0;
  private finishReason: string | null = null;
  private sawToolUse = false;
  private usage: { input?: number; output?: number } = {};

  constructor(
    private readonly requestedModel: string,
    private readonly showReasoning: boolean,
    private readonly estimatedInputTokens: number,
  ) {
    this.filter = new ThinkTagFilter(!showReasoning);
  }

  /** The opening event. Must be sent before anything else. */
  start(): SseEvent[] {
    return [
      {
        event: 'message_start',
        data: {
          type: 'message_start',
          message: {
            id: this.messageId,
            type: 'message',
            role: 'assistant',
            model: this.requestedModel,
            content: [],
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: this.estimatedInputTokens, output_tokens: 0 },
          },
        },
      },
    ];
  }

  private closeBlock(out: SseEvent[]): void {
    if (!this.current) return;

    // A tool call that never received any arguments still needs valid JSON,
    // otherwise the agent tries to parse an empty string and throws.
    if (this.current.kind === 'tool' && !this.current.sawArguments) {
      out.push({
        event: 'content_block_delta',
        data: {
          type: 'content_block_delta',
          index: this.current.index,
          delta: { type: 'input_json_delta', partial_json: '{}' },
        },
      });
    }

    out.push({
      event: 'content_block_stop',
      data: { type: 'content_block_stop', index: this.current.index },
    });
    this.current = null;
  }

  private openTextBlock(out: SseEvent[]): Block {
    if (this.current?.kind === 'text') return this.current;
    this.closeBlock(out);
    const block: Block = { kind: 'text', index: this.nextIndex++ };
    out.push({
      event: 'content_block_start',
      data: {
        type: 'content_block_start',
        index: block.index,
        content_block: { type: 'text', text: '' },
      },
    });
    this.current = block;
    return block;
  }

  private openToolBlock(
    out: SseEvent[],
    openAiIndex: number,
    id: string,
    name: string,
  ): Block & { kind: 'tool' } {
    const existing = this.toolBlocks.get(openAiIndex);
    if (existing) {
      // Already open, or reopened after a text interruption.
      if (this.current !== existing) {
        this.closeBlock(out);
        this.current = existing;
      }
      return existing;
    }

    this.closeBlock(out);
    const block: Block & { kind: 'tool' } = {
      kind: 'tool',
      index: this.nextIndex++,
      openAiIndex,
      sawArguments: false,
    };
    out.push({
      event: 'content_block_start',
      data: {
        type: 'content_block_start',
        index: block.index,
        content_block: { type: 'tool_use', id, name, input: {} },
      },
    });
    this.toolBlocks.set(openAiIndex, block);
    this.current = block;
    this.sawToolUse = true;
    return block;
  }

  /** Feeds one OpenAI chunk in, gets zero or more Anthropic events out. */
  handleChunk(chunk: ChatChunk): SseEvent[] {
    const out: SseEvent[] = [];

    if (chunk.usage) {
      this.usage = {
        input: chunk.usage.prompt_tokens ?? this.usage.input,
        output: chunk.usage.completion_tokens ?? this.usage.output,
      };
    }

    const choice = chunk.choices?.[0];
    if (!choice) return out;

    // Most servers use `delta` while streaming, but a few echo a whole
    // `message` on the final chunk. Accept either.
    const delta: ChatChoiceDelta = choice.delta ?? choice.message ?? {};

    // Reasoning arrives on its own field. It is dropped unless the profile
    // asked for it, because agents treat it as part of the answer otherwise.
    if (this.showReasoning && delta.reasoning_content) {
      const block = this.openTextBlock(out);
      this.outputChars += delta.reasoning_content.length;
      out.push({
        event: 'content_block_delta',
        data: {
          type: 'content_block_delta',
          index: block.index,
          delta: { type: 'text_delta', text: delta.reasoning_content },
        },
      });
    }

    if (typeof delta.content === 'string' && delta.content.length > 0) {
      const text = this.filter.push(delta.content);
      if (text.length > 0) {
        const block = this.openTextBlock(out);
        this.outputChars += text.length;
        out.push({
          event: 'content_block_delta',
          data: {
            type: 'content_block_delta',
            index: block.index,
            delta: { type: 'text_delta', text },
          },
        });
      }
    }

    for (const call of delta.tool_calls ?? []) {
      const openAiIndex = call.index ?? 0;

      // The first fragment for a call carries its id and name; later ones
      // carry only argument text. When a server omits the id entirely we mint
      // a stable one, because the agent must be able to match the eventual
      // tool_result back to this call.
      const toolId = call.id || `toolu_${this.messageId}_${openAiIndex}`;
      const block = this.openToolBlock(
        out,
        openAiIndex,
        toolId,
        call.function?.name || 'unknown',
      );

      const args = call.function?.arguments;
      if (typeof args === 'string' && args.length > 0) {
        block.sawArguments = true;
        this.outputChars += args.length;
        out.push({
          event: 'content_block_delta',
          data: {
            type: 'content_block_delta',
            index: block.index,
            delta: { type: 'input_json_delta', partial_json: args },
          },
        });
      }
    }

    if (choice.finish_reason) this.finishReason = choice.finish_reason;

    return out;
  }

  /** The closing events. Always send these, even after an upstream error. */
  finish(): SseEvent[] {
    const out: SseEvent[] = [];

    // Anything the think-tag filter was still holding back.
    const tail = this.filter.flush();
    if (tail.length > 0) {
      const block = this.openTextBlock(out);
      this.outputChars += tail.length;
      out.push({
        event: 'content_block_delta',
        data: {
          type: 'content_block_delta',
          index: block.index,
          delta: { type: 'text_delta', text: tail },
        },
      });
    }

    // A message with no content at all is invalid; give it an empty text block.
    if (this.nextIndex === 0) {
      this.openTextBlock(out);
    }

    this.closeBlock(out);

    out.push({
      event: 'message_delta',
      data: {
        type: 'message_delta',
        delta: {
          stop_reason: this.sawToolUse ? 'tool_use' : mapStopReason(this.finishReason),
          stop_sequence: null,
        },
        usage: {
          output_tokens: this.usage.output ?? estimateTokens('x'.repeat(this.outputChars)),
        },
      },
    });

    out.push({ event: 'message_stop', data: { type: 'message_stop' } });
    return out;
  }
}

/** Formats one event in the `event:`/`data:` wire form SSE requires. */
export function formatSse(event: SseEvent): string {
  return `event: ${event.event}\ndata: ${JSON.stringify(event.data)}\n\n`;
}
