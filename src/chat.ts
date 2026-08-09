/**
 * `nvp chat` - talk to the selected model straight from the terminal.
 *
 * No agent, no proxy, no editor. This exists so you can sanity-check a model
 * in two seconds ("is this thing actually answering?") and because sometimes a
 * plain conversation is all you want.
 */

import readline from 'node:readline';

import { activeProfile, resolveApiKey, type Config } from './config.js';
import { NvidiaError, streamChat, type ClientOptions, type OpenAiMessage } from './nvidia.js';
import { ThinkTagFilter } from './translate.js';
import { colour, fail, say, warn } from './ui.js';

const HELP = `
  ${colour.bold('Commands')}
    /exit, /quit    leave
    /clear          forget the conversation so far
    /system <text>  set a system prompt and start over
    /model          show which model is answering
    /help           this list
`;

export async function runChat(config: Config, modelOverride?: string): Promise<number> {
  const profile = activeProfile(config);
  const model = modelOverride || profile.big;
  const apiKey = resolveApiKey(config);

  if (!apiKey) {
    fail('No NVIDIA API key configured. Run `nvp setup`.');
    return 1;
  }
  if (!model) {
    fail('No model selected. Run `nvp use` to pick one.');
    return 1;
  }

  const client: ClientOptions = { apiKey, baseUrl: config.baseUrl };
  let history: OpenAiMessage[] = [];
  let systemPrompt = '';

  say('');
  say(`  ${colour.nvidia('nvp chat')}  ${colour.grey(model)}`);
  say(colour.grey('  /help for commands, /exit to leave'));
  say('');

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: colour.nvidia('you ') + colour.grey('> '),
  });

  // Ctrl-C during generation should abort that reply, not kill the session.
  let inFlight: AbortController | null = null;
  rl.on('SIGINT', () => {
    if (inFlight) {
      inFlight.abort();
      inFlight = null;
    } else {
      rl.close();
    }
  });

  const ask = (): void => {
    rl.prompt();
  };

  return new Promise<number>((resolve) => {
    rl.on('close', () => {
      say('');
      resolve(0);
    });

    rl.on('line', (line) => {
      const input = line.trim();

      if (!input) {
        ask();
        return;
      }

      // ---- slash commands -------------------------------------------------
      if (input === '/exit' || input === '/quit') {
        rl.close();
        return;
      }
      if (input === '/help') {
        say(HELP);
        ask();
        return;
      }
      if (input === '/clear') {
        history = [];
        say(colour.grey('  (conversation cleared)'));
        ask();
        return;
      }
      if (input === '/model') {
        say(colour.grey(`  ${model}`));
        ask();
        return;
      }
      if (input.startsWith('/system ')) {
        systemPrompt = input.slice('/system '.length).trim();
        history = [];
        say(colour.grey('  (system prompt set, conversation cleared)'));
        ask();
        return;
      }
      if (input.startsWith('/')) {
        warn(`Unknown command ${input}. Try /help.`);
        ask();
        return;
      }

      // ---- a real message -------------------------------------------------
      history.push({ role: 'user', content: input });

      const messages: OpenAiMessage[] = systemPrompt
        ? [{ role: 'system', content: systemPrompt }, ...history]
        : [...history];

      const controller = new AbortController();
      inFlight = controller;

      void (async () => {
        const filter = new ThinkTagFilter(!profile.showReasoning);
        let answer = '';

        process.stdout.write(colour.cyan('ai  ') + colour.grey('> '));

        try {
          const request = {
            model,
            messages,
            max_tokens: profile.maxTokens,
            ...(profile.temperature !== null ? { temperature: profile.temperature } : {}),
          };

          for await (const chunk of streamChat(client, request, controller.signal)) {
            const delta = chunk.choices?.[0]?.delta;
            if (!delta) continue;

            if (profile.showReasoning && delta.reasoning_content) {
              process.stdout.write(colour.grey(delta.reasoning_content));
            }
            if (typeof delta.content === 'string' && delta.content) {
              const text = filter.push(delta.content);
              if (text) {
                answer += text;
                process.stdout.write(text);
              }
            }
          }

          const tail = filter.flush();
          if (tail) {
            answer += tail;
            process.stdout.write(tail);
          }

          say('');
          say('');
          history.push({ role: 'assistant', content: answer });
        } catch (error) {
          say('');
          if (controller.signal.aborted) {
            warn('stopped');
            // Keep the partial answer so the conversation still makes sense.
            if (answer) history.push({ role: 'assistant', content: answer });
          } else {
            fail(
              error instanceof NvidiaError
                ? error.message
                : `Request failed: ${(error as Error).message}`,
            );
            history.pop(); // Drop the user turn that got no reply.
          }
          say('');
        } finally {
          inFlight = null;
          ask();
        }
      })();
    });

    ask();
  });
}
