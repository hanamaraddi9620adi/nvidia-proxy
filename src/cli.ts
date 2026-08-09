#!/usr/bin/env node
/**
 * nvp - the command line surface.
 *
 * Design rule for this file: every command must be explainable in one line,
 * and the common path must be two commands total (`nvp setup`, then
 * `nvp claude`). Everything else is for when something needs adjusting.
 */

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import {
  CONFIG_FILE,
  DEFAULT_PROFILE,
  activeProfile,
  clearServerRecord,
  isConfigured,
  loadConfig,
  looksLikeNvidiaKey,
  maskKey,
  readServerRecord,
  resolveApiKey,
  saveConfig,
  type Config,
  type Profile,
} from './config.js';
import { NvidiaError, listModels, suggestDefaults, validateKey, type ClientOptions } from './nvidia.js';
import { isOurServerRunning, startServer, stopServer } from './server.js';
import {
  CODEX_PROFILE_ID,
  codexConfigPath,
  detectAgents,
  launchClaude,
  launchCodex,
  launchOpencode,
  opencodeConfigPath,
  writeCodexConfig,
  writeOpencodeConfig,
} from './launchers.js';
import { runChat } from './chat.js';
import {
  colour,
  confirm,
  fail,
  heading,
  info,
  ok,
  promptHidden,
  promptLine,
  say,
  select,
  spinner,
  table,
  warn,
  type Choice,
} from './ui.js';

const require = createRequire(import.meta.url);

function version(): string {
  try {
    const here = path.dirname(fileURLToPath(import.meta.url));
    const pkg = require(path.join(here, '..', 'package.json')) as { version?: string };
    return pkg.version ?? '0.0.0';
  } catch {
    return '0.0.0';
  }
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

function printHelp(): void {
  const c = colour;
  say(`
  ${c.nvidia(c.bold('nvp'))} ${c.grey('- NVIDIA models in Claude Code, Codex, OpenCode, or your terminal')}

  ${c.bold('Getting started')}
    nvp setup                 enter your key and pick your models (run this once)

  ${c.bold('Use it')}
    nvp claude ${c.grey('[args]')}        launch Claude Code on NVIDIA models
    nvp codex ${c.grey('[args]')}         launch Codex on NVIDIA models
    nvp opencode ${c.grey('[args]')}      launch OpenCode on NVIDIA models
    nvp chat ${c.grey('[--model id]')}    talk to the model in this terminal

  ${c.bold('Models')}
    nvp models ${c.grey('[--refresh]')}   list every model your key can reach
    nvp models ${c.grey('<search>')}      filter that list
    nvp use ${c.grey('[id]')}             choose the main model (no id = pick from a menu)
    nvp use --small ${c.grey('<id>')}     choose the cheap background model

  ${c.bold('Profiles')} ${c.grey('- saved model pairs you can switch between')}
    nvp profile                list them
    nvp profile new ${c.grey('<name>')}   create one from the current settings
    nvp profile use ${c.grey('<name>')}   switch to one
    nvp profile delete ${c.grey('<name>')}

  ${c.bold('The proxy')} ${c.grey('- only Claude Code needs it; it starts by itself')}
    nvp start ${c.grey('[--verbose]')}    run it in the foreground
    nvp status                 what is configured and whether it is running
    nvp stop                   stop a background one

  ${c.bold('Housekeeping')}
    nvp key set|show|clear     manage the API key
    nvp set ${c.grey('<option> <value>')} port, maxTokens, temperature, showReasoning
    nvp doctor                 check everything and say what is wrong
    nvp version

  ${c.grey('Config lives in')} ${c.grey(CONFIG_FILE)}
  ${c.grey('Free keys come from')} ${c.grey('https://build.nvidia.com/settings/api-keys')}
`);
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function clientFor(config: Config): ClientOptions {
  return { apiKey: resolveApiKey(config), baseUrl: config.baseUrl };
}

/** Loads the catalogue with a spinner, turning failures into clean messages. */
async function fetchCatalogue(config: Config, force: boolean): Promise<string[] | null> {
  const spin = spinner('Asking NVIDIA which models your key can reach...');
  try {
    const models = await listModels(clientFor(config), force);
    spin.stop();
    return models;
  } catch (error) {
    spin.stop();
    fail(error instanceof NvidiaError ? error.message : (error as Error).message);
    return null;
  }
}

function modelChoices(models: string[]): Choice<string>[] {
  return models.map((id) => {
    const slash = id.indexOf('/');
    return {
      label: id,
      hint: slash > 0 ? id.slice(0, slash) : '',
      value: id,
    };
  });
}

function updateProfile(config: Config, change: Partial<Profile>): void {
  const name = config.activeProfile;
  const current = config.profiles[name] ?? { ...DEFAULT_PROFILE };
  config.profiles[name] = { ...current, ...change };
  saveConfig(config);
}

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------

async function commandSetup(config: Config): Promise<number> {
  heading('nvp setup');
  say('  Three steps: your key, your main model, your background model.');
  say('');

  // ---- 1. the key --------------------------------------------------------
  const existing = resolveApiKey(config);
  let key = existing;

  if (existing) {
    say(`  A key is already saved: ${colour.grey(maskKey(existing))}`);
    if (await confirm('  Replace it?', false)) key = '';
  }

  if (!key) {
    say('');
    say(`  Get a free key at ${colour.cyan('https://build.nvidia.com/settings/api-keys')}`);
    say(colour.grey('  Sign in, click "Generate API Key", copy the nvapi-... value.'));
    say(colour.grey('  Nothing you type here is echoed to the screen.'));
    say('');
    key = await promptHidden('  Paste your NVIDIA API key');

    if (!key) {
      fail('No key entered.');
      return 1;
    }
    if (!looksLikeNvidiaKey(key)) {
      warn('That does not look like an NVIDIA key (they start with "nvapi-").');
      if (!(await confirm('  Use it anyway?', false))) return 1;
    }
  }

  config.apiKey = key;

  const spin = spinner('Checking the key against NVIDIA...');
  let models: string[];
  try {
    const count = await validateKey({ apiKey: key, baseUrl: config.baseUrl });
    models = await listModels({ apiKey: key, baseUrl: config.baseUrl }, false);
    spin.stop();
    ok(`Key works. ${count} models available.`);
  } catch (error) {
    spin.stop();
    fail(error instanceof NvidiaError ? error.message : (error as Error).message);
    return 1;
  }

  saveConfig(config);

  // ---- 2. the models -----------------------------------------------------
  const suggested = suggestDefaults(models);
  say('');

  const big = await select(
    `  Main model  ${colour.grey('(does the real work)')}`,
    modelChoices(models),
  );
  if (!big) {
    warn('Cancelled. The key was saved; run `nvp use` when you are ready.');
    return 1;
  }

  say('');
  say(`  ${colour.grey('Background model: used for small chores like summarising.')}`);
  say(`  ${colour.grey('A smaller, faster model here keeps the agent responsive.')}`);
  const small = await select(
    `  Background model  ${colour.grey(`(suggested: ${suggested.small || big})`)}`,
    modelChoices(models),
  );

  updateProfile(config, { big, small: small ?? big });

  // ---- 3. done -----------------------------------------------------------
  heading('Ready');
  table([
    ['main model', colour.nvidia(big)],
    ['background', colour.nvidia(small ?? big)],
    ['config', CONFIG_FILE],
  ]);

  say('');
  say('  Now run whichever you like:');
  say(`    ${colour.bold('nvp claude')}     ${colour.grey('Claude Code on NVIDIA models')}`);
  say(`    ${colour.bold('nvp codex')}      ${colour.grey('Codex on NVIDIA models')}`);
  say(`    ${colour.bold('nvp chat')}       ${colour.grey('a chat right here')}`);
  say('');
  say(colour.grey('  You will not be asked for any of this again.'));
  say('');
  return 0;
}

// ---------------------------------------------------------------------------
// key
// ---------------------------------------------------------------------------

async function commandKey(config: Config, args: string[]): Promise<number> {
  const action = args[0] ?? 'show';

  if (action === 'show') {
    const key = resolveApiKey(config);
    const fromEnv = Boolean(process.env.NVIDIA_API_KEY?.trim());
    table([
      ['key', maskKey(key)],
      ['source', fromEnv ? 'NVIDIA_API_KEY environment variable' : CONFIG_FILE],
    ]);
    return key ? 0 : 1;
  }

  if (action === 'set') {
    const key = await promptHidden('  Paste your NVIDIA API key');
    if (!key) {
      fail('No key entered.');
      return 1;
    }
    if (!looksLikeNvidiaKey(key)) {
      warn('That does not look like an NVIDIA key (they start with "nvapi-").');
      if (!(await confirm('  Save it anyway?', false))) return 1;
    }

    const spin = spinner('Checking the key...');
    try {
      const count = await validateKey({ apiKey: key, baseUrl: config.baseUrl });
      spin.stop();
      ok(`Key works. ${count} models available.`);
    } catch (error) {
      spin.stop();
      fail(error instanceof NvidiaError ? error.message : (error as Error).message);
      if (!(await confirm('  Save it anyway?', false))) return 1;
    }

    config.apiKey = key;
    saveConfig(config);
    ok(`Saved to ${CONFIG_FILE}`);
    return 0;
  }

  if (action === 'clear') {
    if (!(await confirm('  Remove the saved API key?', false))) return 0;
    config.apiKey = '';
    saveConfig(config);
    ok('Key removed.');
    return 0;
  }

  fail(`Unknown: nvp key ${action}. Try set, show or clear.`);
  return 1;
}

// ---------------------------------------------------------------------------
// models / use
// ---------------------------------------------------------------------------

async function commandModels(config: Config, args: string[]): Promise<number> {
  const force = args.includes('--refresh');
  const search = args.find((a) => !a.startsWith('-'))?.toLowerCase();

  const models = await fetchCatalogue(config, force);
  if (!models) return 1;

  const filtered = search ? models.filter((m) => m.toLowerCase().includes(search)) : models;
  if (filtered.length === 0) {
    warn(`No model matches "${search}".`);
    return 1;
  }

  const profile = activeProfile(config);
  heading(`${filtered.length} model${filtered.length === 1 ? '' : 's'}`);
  for (const id of filtered) {
    const marks: string[] = [];
    if (id === profile.big) marks.push(colour.nvidia('main'));
    if (id === profile.small) marks.push(colour.cyan('background'));
    say(`  ${id}${marks.length ? `  ${marks.join(' ')}` : ''}`);
  }
  say('');
  say(colour.grey('  Choose one with:  nvp use <id>     or just  nvp use'));
  return 0;
}

async function commandUse(config: Config, args: string[]): Promise<number> {
  const smallFlagIndex = args.findIndex((a) => a === '--small');
  const wantsSmall = smallFlagIndex !== -1;
  const positional = args.filter((a) => !a.startsWith('-'));
  const requested = positional[0];

  const models = await fetchCatalogue(config, false);
  if (!models) return 1;

  let chosen = requested;

  if (chosen) {
    if (!models.includes(chosen)) {
      // Be forgiving about partial names: "kimi" should find the Kimi model.
      const matches = models.filter((m) => m.toLowerCase().includes(chosen!.toLowerCase()));
      if (matches.length === 1) {
        chosen = matches[0]!;
        info(`Matched ${colour.nvidia(chosen)}`);
      } else if (matches.length > 1) {
        const picked = await select(`  ${matches.length} models match "${requested}"`, modelChoices(matches));
        if (!picked) return 1;
        chosen = picked;
      } else {
        fail(`No model matches "${requested}". Run \`nvp models --refresh\` to see the list.`);
        return 1;
      }
    }
  } else {
    const picked = await select(
      wantsSmall
        ? `  Background model  ${colour.grey('(small, fast, for chores)')}`
        : `  Main model  ${colour.grey('(does the real work)')}`,
      modelChoices(models),
    );
    if (!picked) return 0;
    chosen = picked;
  }

  updateProfile(config, wantsSmall ? { small: chosen } : { big: chosen });
  ok(`${wantsSmall ? 'Background' : 'Main'} model is now ${colour.nvidia(chosen)}`);

  // Keep the agent configs in step so the change is live everywhere at once.
  refreshAgentConfigs(loadConfig());
  return 0;
}

/** Rewrites the Codex and OpenCode blocks after a model change. */
function refreshAgentConfigs(config: Config): void {
  if (!activeProfile(config).big) return;
  try {
    if (fs.existsSync(path.dirname(codexConfigPath()))) writeCodexConfig(config);
  } catch {
    // Config refresh is a convenience; never fail a command over it.
  }
  try {
    if (fs.existsSync(path.dirname(opencodeConfigPath()))) writeOpencodeConfig(config);
  } catch {
    // Same.
  }
}

// ---------------------------------------------------------------------------
// profiles
// ---------------------------------------------------------------------------

async function commandProfile(config: Config, args: string[]): Promise<number> {
  const action = args[0];
  const name = args[1];

  if (!action || action === 'list') {
    heading('Profiles');
    for (const [key, profile] of Object.entries(config.profiles)) {
      const active = key === config.activeProfile;
      const marker = active ? colour.nvidia('*') : ' ';
      say(`  ${marker} ${colour.bold(key)}${profile.note ? colour.grey(`  ${profile.note}`) : ''}`);
      table(
        [
          ['main', profile.big || colour.grey('(unset)')],
          ['background', profile.small || colour.grey('(unset)')],
        ],
        '      ',
      );
    }
    say('');
    say(colour.grey('  Switch with:  nvp profile use <name>'));
    return 0;
  }

  if (action === 'new') {
    if (!name) {
      fail('Give the profile a name:  nvp profile new <name>');
      return 1;
    }
    if (config.profiles[name]) {
      fail(`Profile "${name}" already exists.`);
      return 1;
    }
    const note = await promptLine('  Note (optional)');
    config.profiles[name] = { ...activeProfile(config), ...(note ? { note } : {}) };
    config.activeProfile = name;
    saveConfig(config);
    ok(`Created "${name}" and switched to it.`);
    return 0;
  }

  if (action === 'use') {
    if (!name) {
      const picked = await select(
        '  Switch to which profile?',
        Object.entries(config.profiles).map(([key, p]) => ({
          label: key,
          hint: p.big,
          value: key,
        })),
      );
      if (!picked) return 0;
      config.activeProfile = picked;
    } else {
      if (!config.profiles[name]) {
        fail(`No profile called "${name}".`);
        return 1;
      }
      config.activeProfile = name;
    }
    saveConfig(config);
    const profile = activeProfile(config);
    ok(`Now using "${config.activeProfile}" (${colour.nvidia(profile.big || 'no model set')})`);
    refreshAgentConfigs(config);
    return 0;
  }

  if (action === 'delete') {
    if (!name) {
      fail('Which one?  nvp profile delete <name>');
      return 1;
    }
    if (!config.profiles[name]) {
      fail(`No profile called "${name}".`);
      return 1;
    }
    if (Object.keys(config.profiles).length === 1) {
      fail('That is the only profile; there would be none left.');
      return 1;
    }
    if (!(await confirm(`  Delete profile "${name}"?`, false))) return 0;

    delete config.profiles[name];
    if (config.activeProfile === name) {
      config.activeProfile = Object.keys(config.profiles)[0]!;
      info(`Switched to "${config.activeProfile}".`);
    }
    saveConfig(config);
    ok(`Deleted "${name}".`);
    return 0;
  }

  fail(`Unknown: nvp profile ${action}. Try list, new, use or delete.`);
  return 1;
}

// ---------------------------------------------------------------------------
// set
// ---------------------------------------------------------------------------

function commandSet(config: Config, args: string[]): number {
  const [option, rawValue] = args;

  if (!option || rawValue === undefined) {
    heading('Current settings');
    const profile = activeProfile(config);
    table([
      ['port', String(config.port)],
      ['maxTokens', String(profile.maxTokens)],
      ['temperature', profile.temperature === null ? 'model default' : String(profile.temperature)],
      ['showReasoning', String(profile.showReasoning)],
      ['baseUrl', config.baseUrl],
    ]);
    say('');
    say(colour.grey('  Change one with:  nvp set <option> <value>'));
    return 0;
  }

  switch (option) {
    case 'port': {
      const port = Number.parseInt(rawValue, 10);
      if (!Number.isInteger(port) || port < 1 || port > 65535) {
        fail('Port must be a whole number between 1 and 65535.');
        return 1;
      }
      config.port = port;
      saveConfig(config);
      ok(`Proxy port is now ${port}.`);
      return 0;
    }
    case 'maxTokens': {
      const value = Number.parseInt(rawValue, 10);
      if (!Number.isInteger(value) || value < 1) {
        fail('maxTokens must be a positive whole number.');
        return 1;
      }
      updateProfile(config, { maxTokens: value });
      ok(`maxTokens is now ${value}.`);
      return 0;
    }
    case 'temperature': {
      if (rawValue === 'default' || rawValue === 'null') {
        updateProfile(config, { temperature: null });
        ok('temperature left to the model.');
        return 0;
      }
      const value = Number.parseFloat(rawValue);
      if (Number.isNaN(value) || value < 0 || value > 2) {
        fail('temperature must be between 0 and 2, or "default".');
        return 1;
      }
      updateProfile(config, { temperature: value });
      ok(`temperature is now ${value}.`);
      return 0;
    }
    case 'showReasoning': {
      const value = rawValue === 'true' || rawValue === 'on' || rawValue === '1';
      updateProfile(config, { showReasoning: value });
      ok(`showReasoning is now ${value}.`);
      if (value) {
        warn('Reasoning text will be mixed into replies. Agents can find this confusing.');
      }
      return 0;
    }
    case 'baseUrl': {
      config.baseUrl = rawValue;
      saveConfig(config);
      ok(`baseUrl is now ${rawValue}.`);
      return 0;
    }
    default:
      fail(`Unknown option "${option}". Try port, maxTokens, temperature, showReasoning or baseUrl.`);
      return 1;
  }
}

// ---------------------------------------------------------------------------
// server
// ---------------------------------------------------------------------------

async function commandStart(config: Config, args: string[]): Promise<number> {
  if (!isConfigured(config)) {
    fail('Not set up yet. Run `nvp setup` first.');
    return 1;
  }

  const portFlag = args.findIndex((a) => a === '--port');
  const port = portFlag !== -1 ? Number.parseInt(args[portFlag + 1] ?? '', 10) : config.port;
  if (!Number.isInteger(port)) {
    fail('--port needs a number.');
    return 1;
  }

  const verbose = args.includes('--verbose') || args.includes('-v');
  const profile = activeProfile(config);

  try {
    const server = await startServer({ config, verbose, port });

    heading('Proxy running');
    table([
      ['address', `http://127.0.0.1:${port}`],
      ['main model', colour.nvidia(profile.big)],
      ['background', colour.nvidia(profile.small || profile.big)],
      ['profile', config.activeProfile],
    ]);
    say('');
    say('  Point any Anthropic-API client at it with:');
    say(colour.grey(`    ANTHROPIC_BASE_URL=http://127.0.0.1:${port}`));
    say(colour.grey('    ANTHROPIC_AUTH_TOKEN=nvproxy-local'));
    say('');
    say(colour.grey('  Ctrl-C to stop.'));
    say('');

    await new Promise<void>((resolve) => {
      const shutdown = () => {
        say('');
        info('Stopping.');
        void stopServer(server).then(resolve);
      };
      process.on('SIGINT', shutdown);
      process.on('SIGTERM', shutdown);
    });
    return 0;
  } catch (error) {
    fail((error as Error).message);
    return 1;
  }
}

async function commandStatus(config: Config): Promise<number> {
  const profile = activeProfile(config);
  const key = resolveApiKey(config);

  heading('nvp status');
  table([
    ['api key', key ? colour.green(maskKey(key)) : colour.red('not set - run `nvp setup`')],
    ['profile', config.activeProfile],
    ['main model', profile.big ? colour.nvidia(profile.big) : colour.red('not set')],
    ['background', profile.small ? colour.nvidia(profile.small) : colour.grey('(same as main)')],
    ['port', String(config.port)],
    ['config file', CONFIG_FILE],
  ]);

  const running = await isOurServerRunning(config.port);
  const record = readServerRecord();
  say('');
  if (running) {
    const since = record ? new Date(record.startedAt).toTimeString().slice(0, 8) : 'unknown';
    ok(`Proxy is running on 127.0.0.1:${config.port} (since ${since})`);
  } else {
    say(colour.grey(`  Proxy is not running. It starts automatically with \`nvp claude\`.`));
    if (record) clearServerRecord();
  }

  say('');
  heading('Agents found');
  for (const agent of detectAgents()) {
    say(
      agent.path
        ? `  ${colour.green('yes')}  ${agent.name.padEnd(12)} ${colour.grey(agent.path)}`
        : `  ${colour.grey('no ')}  ${agent.name.padEnd(12)} ${colour.grey('not installed')}`,
    );
  }
  say('');
  return 0;
}

async function commandStop(config: Config): Promise<number> {
  const record = readServerRecord();
  const running = await isOurServerRunning(config.port);

  if (!running) {
    say(colour.grey('  Nothing is running on that port.'));
    clearServerRecord();
    return 0;
  }
  if (!record) {
    warn(`Something is answering on port ${config.port} but nvp did not record starting it.`);
    return 1;
  }

  try {
    process.kill(record.pid, 'SIGTERM');
    clearServerRecord();
    ok(`Stopped the proxy (pid ${record.pid}).`);
    return 0;
  } catch (error) {
    fail(`Could not stop pid ${record.pid}: ${(error as Error).message}`);
    return 1;
  }
}

// ---------------------------------------------------------------------------
// doctor
// ---------------------------------------------------------------------------

async function commandDoctor(config: Config): Promise<number> {
  heading('nvp doctor');
  let problems = 0;

  const check = (label: string, good: boolean, detail: string) => {
    if (good) {
      say(`  ${colour.green('ok')}    ${label.padEnd(20)} ${colour.grey(detail)}`);
    } else {
      say(`  ${colour.red('FAIL')}  ${label.padEnd(20)} ${detail}`);
      problems++;
    }
  };

  // Node version
  const major = Number.parseInt(process.versions.node.split('.')[0] ?? '0', 10);
  check('node', major >= 20, `v${process.versions.node}${major >= 20 ? '' : ' - needs v20 or newer'}`);

  // Config file
  const hasConfig = fs.existsSync(CONFIG_FILE);
  check('config file', hasConfig, hasConfig ? CONFIG_FILE : 'missing - run `nvp setup`');

  // Key present
  const key = resolveApiKey(config);
  check('api key', Boolean(key), key ? maskKey(key) : 'not set - run `nvp setup`');

  // Key valid
  if (key) {
    const spin = spinner('  Testing the key against NVIDIA...');
    try {
      const count = await validateKey(clientFor(config));
      spin.stop();
      check('key accepted', true, `${count} models reachable`);

      const profile = activeProfile(config);
      const models = await listModels(clientFor(config), false);
      check(
        'main model',
        Boolean(profile.big) && models.includes(profile.big),
        profile.big
          ? models.includes(profile.big)
            ? profile.big
            : `${profile.big} is not in the catalogue - run \`nvp use\``
          : 'not set - run `nvp use`',
      );
      if (profile.small) {
        check(
          'background model',
          models.includes(profile.small),
          models.includes(profile.small)
            ? profile.small
            : `${profile.small} is not in the catalogue - run \`nvp use --small\``,
        );
      }
    } catch (error) {
      spin.stop();
      check('key accepted', false, error instanceof NvidiaError ? error.message : (error as Error).message);
    }
  }

  // Port
  const running = await isOurServerRunning(config.port);
  say(
    running
      ? `  ${colour.green('ok')}    ${'proxy port'.padEnd(20)} ${colour.grey(`${config.port} - our proxy is running`)}`
      : `  ${colour.grey('--')}    ${'proxy port'.padEnd(20)} ${colour.grey(`${config.port} - free (starts on demand)`)}`,
  );

  // Agents
  const agents = detectAgents();
  const found = agents.filter((a) => a.path);
  check(
    'coding agents',
    found.length > 0,
    found.length > 0
      ? found.map((a) => a.command).join(', ')
      : 'none installed - `nvp chat` still works',
  );

  say('');
  if (problems === 0) {
    ok('Everything checks out.');
  } else {
    fail(`${problems} problem${problems === 1 ? '' : 's'} above.`);
  }
  say('');
  return problems === 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Agent launchers (with the proxy handled automatically)
// ---------------------------------------------------------------------------

/**
 * Runs Claude Code with the proxy guaranteed to be up.
 *
 * If a proxy is already listening it is reused. Otherwise one is started
 * inside this very process and shut down when Claude Code exits, so there is
 * never a stray daemon left behind and nothing to remember to stop.
 */
async function commandClaude(config: Config, args: string[]): Promise<number> {
  if (!isConfigured(config)) {
    fail('Not set up yet. Run `nvp setup` first.');
    return 1;
  }

  const alreadyRunning = await isOurServerRunning(config.port);
  let server: Awaited<ReturnType<typeof startServer>> | null = null;

  if (!alreadyRunning) {
    try {
      server = await startServer({ config, verbose: args.includes('--verbose'), port: config.port });
    } catch (error) {
      fail((error as Error).message);
      return 1;
    }
  } else {
    say(colour.grey(`  Reusing the proxy already on 127.0.0.1:${config.port}`));
  }

  try {
    return await launchClaude(config, args.filter((a) => a !== '--verbose'));
  } catch (error) {
    fail((error as Error).message);
    if ((error as Error).message.includes('not on your PATH')) {
      say(colour.grey('  Install it with:  npm install -g @anthropic-ai/claude-code'));
    }
    return 1;
  } finally {
    if (server) await stopServer(server);
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main(argv: string[]): Promise<number> {
  const [command = '', ...rest] = argv;
  const config = loadConfig();

  switch (command) {
    case '':
    case 'help':
    case '--help':
    case '-h':
      printHelp();
      return 0;

    case 'version':
    case '--version':
    case '-V':
      say(version());
      return 0;

    case 'setup':
    case 'init':
      return commandSetup(config);

    case 'key':
      return commandKey(config, rest);

    case 'models':
      return commandModels(config, rest);

    case 'use':
      return commandUse(config, rest);

    case 'profile':
    case 'profiles':
      return commandProfile(config, rest);

    case 'set':
    case 'config':
      return commandSet(config, rest);

    case 'start':
    case 'serve':
      return commandStart(config, rest);

    case 'status':
      return commandStatus(config);

    case 'stop':
      return commandStop(config);

    case 'doctor':
      return commandDoctor(config);

    case 'chat': {
      const flag = rest.findIndex((a) => a === '--model' || a === '-m');
      const model = flag !== -1 ? rest[flag + 1] : undefined;
      return runChat(config, model);
    }

    case 'claude':
      return commandClaude(config, rest);

    case 'codex':
      if (!isConfigured(config)) {
        fail('Not set up yet. Run `nvp setup` first.');
        return 1;
      }
      try {
        return await launchCodex(config, rest);
      } catch (error) {
        fail((error as Error).message);
        if ((error as Error).message.includes('not on your PATH')) {
          say(colour.grey('  Install it with:  npm install -g @openai/codex'));
        }
        return 1;
      }

    case 'opencode':
      if (!isConfigured(config)) {
        fail('Not set up yet. Run `nvp setup` first.');
        return 1;
      }
      try {
        return await launchOpencode(config, rest);
      } catch (error) {
        fail((error as Error).message);
        return 1;
      }

    default:
      fail(`Unknown command "${command}".`);
      say(colour.grey('  Run `nvp help` to see what is available.'));
      return 1;
  }
}

main(process.argv.slice(2))
  .then((code) => {
    process.exitCode = code;
  })
  .catch((error: unknown) => {
    fail((error as Error).message);
    process.exitCode = 1;
  });
