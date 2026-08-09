/**
 * End-to-end tests for the translation layer.
 *
 * These do not need an NVIDIA key. A fake upstream server stands in for
 * NVIDIA and replies with hand-written OpenAI-shaped payloads, so the tests
 * assert exactly one thing: that an Anthropic client talking to our proxy
 * sees a correct Anthropic conversation.
 *
 * Run with:  npm test
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';

import { createServer } from '../dist/server.js';
import { pickModel } from '../dist/server.js';
import { anthropicToOpenAi, ThinkTagFilter } from '../dist/translate.js';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/** Starts a server on a random free port and returns its base URL. */
function listen(server) {
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      resolve(`http://127.0.0.1:${server.address().port}`);
    });
  });
}

/**
 * A stand-in for NVIDIA. `handler` decides what /chat/completions returns:
 * either a plain object (non-streaming) or an array of OpenAI SSE chunks.
 */
async function fakeNvidia(handler) {
  const received = [];
  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      if (req.url.endsWith('/models')) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ data: [{ id: 'vendor/big-model' }, { id: 'vendor/small-model' }] }));
        return;
      }

      const parsed = JSON.parse(body);
      received.push(parsed);
      const result = handler(parsed);

      if (Array.isArray(result)) {
        res.writeHead(200, { 'Content-Type': 'text/event-stream' });
        for (const chunk of result) res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      } else {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      }
    });
  });

  const url = await listen(server);
  return { url, server, received };
}

/** Starts our proxy pointed at a fake upstream. */
async function proxyAgainst(upstreamUrl, profileOverrides = {}) {
  const config = {
    version: 1,
    apiKey: 'nvapi-test-key-0000000000',
    baseUrl: `${upstreamUrl}/v1`,
    port: 0,
    activeProfile: 'default',
    profiles: {
      default: {
        big: 'vendor/big-model',
        small: 'vendor/small-model',
        maxTokens: 1024,
        temperature: null,
        showReasoning: false,
        ...profileOverrides,
      },
    },
  };
  const server = createServer({ config });
  const url = await listen(server);
  return { url, server, config };
}

/** Collects an SSE response into an ordered list of {event, data}. */
async function readSse(response) {
  const text = await response.text();
  const events = [];
  for (const block of text.split('\n\n')) {
    const eventLine = block.split('\n').find((l) => l.startsWith('event: '));
    const dataLine = block.split('\n').find((l) => l.startsWith('data: '));
    if (eventLine && dataLine) {
      events.push({ event: eventLine.slice(7), data: JSON.parse(dataLine.slice(6)) });
    }
  }
  return events;
}

function chunk(delta, finish = null) {
  return { id: 'x', choices: [{ index: 0, delta, finish_reason: finish }] };
}

// ---------------------------------------------------------------------------
// Model routing
// ---------------------------------------------------------------------------

test('routes model tiers to the configured models', () => {
  const profile = { big: 'vendor/big', small: 'vendor/small' };

  assert.equal(pickModel('claude-opus-4-20250101', profile), 'vendor/big');
  assert.equal(pickModel('claude-sonnet-4-20250101', profile), 'vendor/big');
  assert.equal(pickModel('claude-3-5-haiku-20241022', profile), 'vendor/small');

  // An id that already names an NVIDIA model passes straight through.
  assert.equal(pickModel('moonshotai/kimi-k2', profile), 'moonshotai/kimi-k2');
});

test('falls back to the main model when no small model is set', () => {
  assert.equal(pickModel('claude-3-5-haiku', { big: 'vendor/big', small: '' }), 'vendor/big');
});

// ---------------------------------------------------------------------------
// Request translation
// ---------------------------------------------------------------------------

test('lifts the system prompt out and turns tool_result into a tool message', () => {
  const body = anthropicToOpenAi(
    {
      model: 'claude-opus-4',
      system: 'You are terse.',
      messages: [
        { role: 'user', content: 'weather?' },
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'checking' },
            { type: 'tool_use', id: 'toolu_1', name: 'get_weather', input: { city: 'Bengaluru' } },
          ],
        },
        {
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: '29C' }],
        },
      ],
      tools: [
        { name: 'get_weather', description: 'Look up weather', input_schema: { type: 'object' } },
      ],
    },
    { model: 'vendor/big', maxTokens: 512, temperature: null },
  );

  assert.equal(body.messages[0].role, 'system');
  assert.equal(body.messages[0].content, 'You are terse.');

  // The assistant turn keeps its text and gains an OpenAI tool_calls array.
  const assistant = body.messages[2];
  assert.equal(assistant.role, 'assistant');
  assert.equal(assistant.tool_calls[0].function.name, 'get_weather');
  // Arguments must be a JSON *string*, not an object.
  assert.equal(typeof assistant.tool_calls[0].function.arguments, 'string');
  assert.deepEqual(JSON.parse(assistant.tool_calls[0].function.arguments), { city: 'Bengaluru' });

  // The tool result became its own `tool` message carrying the call id.
  const toolMessage = body.messages[3];
  assert.equal(toolMessage.role, 'tool');
  assert.equal(toolMessage.tool_call_id, 'toolu_1');
  assert.equal(toolMessage.content, '29C');

  // input_schema became parameters.
  assert.equal(body.tools[0].type, 'function');
  assert.deepEqual(body.tools[0].function.parameters, { type: 'object' });

  // No temperature was chosen anywhere, so none is sent.
  assert.equal('temperature' in body, false);
});

test('converts a base64 image block into a data URI', () => {
  const body = anthropicToOpenAi(
    {
      model: 'claude-opus-4',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'what is this' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'AAAA' } },
          ],
        },
      ],
    },
    { model: 'vendor/big', maxTokens: 512, temperature: null },
  );

  const parts = body.messages[0].content;
  assert.equal(parts[1].type, 'image_url');
  assert.equal(parts[1].image_url.url, 'data:image/png;base64,AAAA');
});

// ---------------------------------------------------------------------------
// Non-streaming
// ---------------------------------------------------------------------------

test('returns a well-formed Anthropic message for a plain reply', async (t) => {
  const upstream = await fakeNvidia(() => ({
    choices: [{ message: { role: 'assistant', content: 'Hello there.' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 11, completion_tokens: 3 },
  }));
  const proxy = await proxyAgainst(upstream.url);
  t.after(() => {
    upstream.server.close();
    proxy.server.close();
  });

  const response = await fetch(`${proxy.url}/v1/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-opus-4',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'hi' }],
    }),
  });

  assert.equal(response.status, 200);
  const message = await response.json();

  assert.equal(message.type, 'message');
  assert.equal(message.role, 'assistant');
  assert.equal(message.content[0].type, 'text');
  assert.equal(message.content[0].text, 'Hello there.');
  assert.equal(message.stop_reason, 'end_turn');
  assert.equal(message.usage.input_tokens, 11);
  assert.equal(message.usage.output_tokens, 3);

  // The upstream call used the mapped model, not the Anthropic name.
  assert.equal(upstream.received[0].model, 'vendor/big-model');
});

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

test('emits the full Anthropic event sequence for streamed text', async (t) => {
  const upstream = await fakeNvidia(() => [
    chunk({ role: 'assistant', content: '' }),
    chunk({ content: 'Hel' }),
    chunk({ content: 'lo' }),
    chunk({}, 'stop'),
  ]);
  const proxy = await proxyAgainst(upstream.url);
  t.after(() => {
    upstream.server.close();
    proxy.server.close();
  });

  const response = await fetch(`${proxy.url}/v1/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-opus-4',
      max_tokens: 100,
      stream: true,
      messages: [{ role: 'user', content: 'hi' }],
    }),
  });

  assert.equal(response.headers.get('content-type'), 'text/event-stream');
  const events = await readSse(response);
  const names = events.map((e) => e.event);

  assert.deepEqual(names, [
    'message_start',
    'content_block_start',
    'content_block_delta',
    'content_block_delta',
    'content_block_stop',
    'message_delta',
    'message_stop',
  ]);

  const text = events
    .filter((e) => e.event === 'content_block_delta')
    .map((e) => e.data.delta.text)
    .join('');
  assert.equal(text, 'Hello');

  assert.equal(events.at(-2).data.delta.stop_reason, 'end_turn');
});

test('strips <think> reasoning that arrives split across chunks', async (t) => {
  const upstream = await fakeNvidia(() => [
    chunk({ content: 'Before <thi' }),
    chunk({ content: 'nk>secret plan' }),
    chunk({ content: ' continues</think> After' }),
    chunk({}, 'stop'),
  ]);
  const proxy = await proxyAgainst(upstream.url);
  t.after(() => {
    upstream.server.close();
    proxy.server.close();
  });

  const response = await fetch(`${proxy.url}/v1/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-opus-4',
      max_tokens: 100,
      stream: true,
      messages: [{ role: 'user', content: 'hi' }],
    }),
  });

  const events = await readSse(response);
  const text = events
    .filter((e) => e.event === 'content_block_delta')
    .map((e) => e.data.delta.text)
    .join('');

  assert.equal(text, 'Before  After');
  assert.equal(text.includes('secret plan'), false);
});

test('translates a streamed tool call into a tool_use block', async (t) => {
  const upstream = await fakeNvidia(() => [
    chunk({ content: 'Let me look.' }),
    chunk({
      tool_calls: [
        { index: 0, id: 'call_abc', type: 'function', function: { name: 'get_weather', arguments: '' } },
      ],
    }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: '{"city":' } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: '"Bengaluru"}' } }] }),
    chunk({}, 'tool_calls'),
  ]);
  const proxy = await proxyAgainst(upstream.url);
  t.after(() => {
    upstream.server.close();
    proxy.server.close();
  });

  const response = await fetch(`${proxy.url}/v1/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-opus-4',
      max_tokens: 100,
      stream: true,
      messages: [{ role: 'user', content: 'weather?' }],
    }),
  });

  const events = await readSse(response);

  // The text block must be closed before the tool block opens.
  const starts = events.filter((e) => e.event === 'content_block_start');
  assert.equal(starts.length, 2);
  assert.equal(starts[0].data.content_block.type, 'text');
  assert.equal(starts[1].data.content_block.type, 'tool_use');
  assert.equal(starts[1].data.content_block.id, 'call_abc');
  assert.equal(starts[1].data.content_block.name, 'get_weather');
  assert.equal(starts[1].data.index, 1);

  // Block 0 stops before block 1 starts.
  const order = events.map((e) => `${e.event}:${e.data.index ?? ''}`);
  assert.ok(order.indexOf('content_block_stop:0') < order.indexOf('content_block_start:1'));

  // The argument fragments reassemble into valid JSON.
  const json = events
    .filter((e) => e.event === 'content_block_delta' && e.data.delta.type === 'input_json_delta')
    .map((e) => e.data.delta.partial_json)
    .join('');
  assert.deepEqual(JSON.parse(json), { city: 'Bengaluru' });

  assert.equal(events.at(-2).data.delta.stop_reason, 'tool_use');
});

test('gives a tool call with no arguments valid empty JSON', async (t) => {
  const upstream = await fakeNvidia(() => [
    chunk({
      tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'now', arguments: '' } }],
    }),
    chunk({}, 'tool_calls'),
  ]);
  const proxy = await proxyAgainst(upstream.url);
  t.after(() => {
    upstream.server.close();
    proxy.server.close();
  });

  const response = await fetch(`${proxy.url}/v1/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-opus-4',
      max_tokens: 100,
      stream: true,
      messages: [{ role: 'user', content: 'time?' }],
    }),
  });

  const events = await readSse(response);
  const json = events
    .filter((e) => e.event === 'content_block_delta' && e.data.delta.type === 'input_json_delta')
    .map((e) => e.data.delta.partial_json)
    .join('');

  assert.deepEqual(JSON.parse(json), {});
});

// ---------------------------------------------------------------------------
// Errors and odds and ends
// ---------------------------------------------------------------------------

test('reports upstream failures in Anthropic error shape', async (t) => {
  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      if (req.url.endsWith('/models')) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ data: [] }));
        return;
      }
      res.writeHead(429, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'too many requests' }));
    });
  });
  const upstreamUrl = await listen(server);
  const proxy = await proxyAgainst(upstreamUrl);
  t.after(() => {
    server.close();
    proxy.server.close();
  });

  const response = await fetch(`${proxy.url}/v1/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-opus-4',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'hi' }],
    }),
  });

  assert.equal(response.status, 429);
  const payload = await response.json();
  assert.equal(payload.type, 'error');
  assert.equal(payload.error.type, 'rate_limit_error');
  assert.match(payload.error.message, /Rate limited/);
});

test('answers the count_tokens endpoint Claude Code calls', async (t) => {
  const upstream = await fakeNvidia(() => ({ choices: [] }));
  const proxy = await proxyAgainst(upstream.url);
  t.after(() => {
    upstream.server.close();
    proxy.server.close();
  });

  const response = await fetch(`${proxy.url}/v1/messages/count_tokens`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-opus-4',
      messages: [{ role: 'user', content: 'hello world' }],
    }),
  });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.ok(Number.isInteger(payload.input_tokens));
  assert.ok(payload.input_tokens > 0);
});

test('think-tag filter passes text through untouched when disabled', () => {
  const filter = new ThinkTagFilter(false);
  assert.equal(filter.push('<think>keep me</think>ok'), '<think>keep me</think>ok');
});

test('think-tag filter handles a tag that never closes', () => {
  const filter = new ThinkTagFilter(true);
  const out = filter.push('visible <think>runs off the end') + filter.flush();
  assert.equal(out, 'visible ');
});
