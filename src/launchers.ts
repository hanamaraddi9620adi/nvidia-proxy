/**
 * Launchers for the three coding agents.
 *
 * Two different strategies are at work here, and the difference matters:
 *
 *   Claude Code  speaks the Anthropic API and nothing else, so it has to be
 *                pointed at our local proxy via ANTHROPIC_BASE_URL.
 *
 *   Codex        speak the OpenAI dialect natively, which is the same dialect
 *   OpenCode     NVIDIA serves. They need no proxy at all - only a config
 *                entry naming NVIDIA as a provider. Running them through the
 *                proxy would add a translation round-trip for no benefit.
 *
 * Every config change below is additive and clearly marked. Existing settings
 * are preserved, and both tools are configured as a named *profile* so your
 * normal defaults keep working exactly as before.
 */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawn } from 'node:child_process';

import { activeProfile, resolveApiKey, type Config } from './config.js';
import { colour, info, say, warn } from './ui.js';

export const CODEX_PROVIDER_ID = 'nvp-nvidia';
export const CODEX_PROFILE_ID = 'nvp';
export const OPENCODE_PROVIDER_ID = 'nvidia';

const BLOCK_START = '# >>> managed by nvp (nvidia-proxy) - do not edit inside >>>';
const BLOCK_END = '# <<< managed by nvp (nvidia-proxy) <<<';

// ---------------------------------------------------------------------------
// Locating executables
// ---------------------------------------------------------------------------

/**
 * Finds a command on PATH, honouring PATHEXT on Windows.
 *
 * Doing this by hand rather than spawning through a shell avoids every
 * quoting and argument-injection problem that `shell: true` brings with it.
 */
export function resolveCommand(command: string): string | null {
  const isWindows = process.platform === 'win32';
  const pathSeparator = isWindows ? ';' : ':';
  const directories = (process.env.PATH ?? '').split(pathSeparator).filter(Boolean);
  const extensions = isWindows
    ? (process.env.PATHEXT ?? '.COM;.EXE;.BAT;.CMD').split(';').filter(Boolean)
    : [''];

  for (const directory of directories) {
    for (const extension of extensions) {
      const candidate = path.join(directory, command + extension);
      try {
        const stat = fs.statSync(candidate);
        if (stat.isFile()) return candidate;
      } catch {
        // Not here; keep looking.
      }
    }
  }
  return null;
}

/**
 * Runs a command with our environment additions, inheriting the terminal so
 * the agent's own full-screen UI works normally. Resolves with its exit code.
 */
function run(command: string, args: string[], env: NodeJS.ProcessEnv): Promise<number> {
  return new Promise((resolve, reject) => {
    const resolved = resolveCommand(command);
    if (!resolved) {
      reject(new Error(`\`${command}\` is not on your PATH.`));
      return;
    }

    // A .cmd/.bat shim cannot be executed directly by CreateProcess, so those
    // go through cmd.exe. Real executables are spawned directly.
    const isBatch = /\.(cmd|bat)$/i.test(resolved);
    const child = isBatch
      ? spawn(process.env.COMSPEC ?? 'cmd.exe', ['/d', '/s', '/c', resolved, ...args], {
          stdio: 'inherit',
          env: { ...process.env, ...env },
          windowsVerbatimArguments: false,
        })
      : spawn(resolved, args, {
          stdio: 'inherit',
          env: { ...process.env, ...env },
        });

    child.on('error', reject);
    child.on('exit', (code, signal) => {
      if (signal) {
        resolve(1);
      } else {
        resolve(code ?? 0);
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Claude Code
// ---------------------------------------------------------------------------

/**
 * The environment that makes Claude Code talk to our proxy.
 *
 * ANTHROPIC_MODEL and ANTHROPIC_SMALL_FAST_MODEL are set to the real NVIDIA
 * ids. The proxy recognises an id containing "/" as already-resolved and
 * passes it straight through, so the mapping stays visible instead of hiding
 * inside the proxy.
 */
export function claudeEnv(config: Config): NodeJS.ProcessEnv {
  const profile = activeProfile(config);
  return {
    ANTHROPIC_BASE_URL: `http://127.0.0.1:${config.port}`,
    // The proxy ignores this value; Claude Code merely insists one exists.
    ANTHROPIC_AUTH_TOKEN: 'nvproxy-local',
    ANTHROPIC_MODEL: profile.big,
    ANTHROPIC_SMALL_FAST_MODEL: profile.small || profile.big,
    // Stop the CLI trying to reach Anthropic's billing endpoints, which do not
    // exist on our proxy and would otherwise log confusing errors.
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: '1',
  };
}

export async function launchClaude(config: Config, args: string[]): Promise<number> {
  const profile = activeProfile(config);
  info(`Claude Code -> ${colour.nvidia(profile.big)} via 127.0.0.1:${config.port}`);
  return run('claude', args, claudeEnv(config));
}

// ---------------------------------------------------------------------------
// Codex
// ---------------------------------------------------------------------------

export function codexConfigPath(): string {
  return path.join(os.homedir(), '.codex', 'config.toml');
}

/**
 * Adds an NVIDIA provider and an `nvp` profile to ~/.codex/config.toml.
 *
 * The block is fenced by marker comments and rewritten wholesale each time,
 * so it stays current without disturbing a single line of your own config.
 * Because it is a *profile*, plain `codex` keeps using whatever it used
 * before; only `codex --profile nvp` goes to NVIDIA.
 */
export function writeCodexConfig(config: Config): string {
  const profile = activeProfile(config);
  const file = codexConfigPath();
  fs.mkdirSync(path.dirname(file), { recursive: true });

  let existing = '';
  try {
    existing = fs.readFileSync(file, 'utf8');
  } catch {
    // First run: there is no config yet.
  }

  // Drop any block we wrote previously.
  const startIndex = existing.indexOf(BLOCK_START);
  const endIndex = existing.indexOf(BLOCK_END);
  if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
    existing =
      existing.slice(0, startIndex).trimEnd() +
      '\n' +
      existing.slice(endIndex + BLOCK_END.length).trimStart();
  }

  const block = [
    BLOCK_START,
    `# Regenerated by \`nvp codex\`. Run \`codex --profile ${CODEX_PROFILE_ID}\` to use it.`,
    `[model_providers.${CODEX_PROVIDER_ID}]`,
    'name = "NVIDIA NIM"',
    `base_url = "${config.baseUrl}"`,
    '# Codex reads the key from this environment variable, which `nvp codex` sets.',
    'env_key = "NVIDIA_API_KEY"',
    'wire_api = "chat"',
    '',
    `[profiles.${CODEX_PROFILE_ID}]`,
    `model_provider = "${CODEX_PROVIDER_ID}"`,
    `model = "${profile.big}"`,
    BLOCK_END,
    '',
  ].join('\n');

  const merged = existing.trim().length > 0 ? `${existing.trim()}\n\n${block}` : block;
  fs.writeFileSync(file, merged, 'utf8');
  return file;
}

export async function launchCodex(config: Config, args: string[]): Promise<number> {
  const profile = activeProfile(config);
  const file = writeCodexConfig(config);
  info(`Codex -> ${colour.nvidia(profile.big)} (direct, no proxy)`);
  say(colour.grey(`  profile "${CODEX_PROFILE_ID}" written to ${file}`));

  // Only pass --profile when the caller has not chosen their own.
  const hasProfileFlag = args.some((a) => a === '--profile' || a.startsWith('--profile='));
  const finalArgs = hasProfileFlag ? args : ['--profile', CODEX_PROFILE_ID, ...args];

  return run('codex', finalArgs, { NVIDIA_API_KEY: resolveApiKey(config) });
}

// ---------------------------------------------------------------------------
// OpenCode
// ---------------------------------------------------------------------------

export function opencodeConfigPath(): string {
  // OpenCode follows the XDG convention on every platform, including Windows.
  const base = process.env.XDG_CONFIG_HOME || path.join(os.homedir(), '.config');
  return path.join(base, 'opencode', 'opencode.json');
}

/**
 * Registers NVIDIA as a custom provider in opencode.json.
 *
 * OpenCode can drive any OpenAI-compatible endpoint through its
 * `@ai-sdk/openai-compatible` adapter, so this is pure configuration. The file
 * is merged key by key, leaving any other providers you have set up intact.
 */
export function writeOpencodeConfig(config: Config): string {
  const profile = activeProfile(config);
  const file = opencodeConfigPath();
  fs.mkdirSync(path.dirname(file), { recursive: true });

  let existing: Record<string, unknown> = {};
  try {
    existing = JSON.parse(fs.readFileSync(file, 'utf8')) as Record<string, unknown>;
  } catch {
    // Missing or unparsable. A fresh object is the right starting point, but
    // never silently destroy a file we failed to parse.
    if (fs.existsSync(file)) {
      const backup = `${file}.nvp-backup`;
      fs.copyFileSync(file, backup);
      warn(`Existing opencode.json could not be parsed; backed it up to ${backup}`);
    }
  }

  const models: Record<string, unknown> = {};
  for (const id of new Set([profile.big, profile.small].filter(Boolean))) {
    models[id] = { name: id };
  }

  const providers = (existing['provider'] as Record<string, unknown> | undefined) ?? {};
  providers[OPENCODE_PROVIDER_ID] = {
    npm: '@ai-sdk/openai-compatible',
    name: 'NVIDIA NIM (via nvp)',
    options: {
      baseURL: config.baseUrl,
      // OpenCode expands {env:NAME} at load time, so the key is never written
      // into this file. `nvp opencode` supplies it in the environment.
      apiKey: '{env:NVIDIA_API_KEY}',
    },
    models,
  };

  const merged = {
    $schema: 'https://opencode.ai/config.json',
    ...existing,
    provider: providers,
    model: `${OPENCODE_PROVIDER_ID}/${profile.big}`,
  };

  fs.writeFileSync(file, JSON.stringify(merged, null, 2) + '\n', 'utf8');
  return file;
}

export async function launchOpencode(config: Config, args: string[]): Promise<number> {
  const profile = activeProfile(config);
  const file = writeOpencodeConfig(config);
  info(`OpenCode -> ${colour.nvidia(profile.big)} (direct, no proxy)`);
  say(colour.grey(`  provider "${OPENCODE_PROVIDER_ID}" written to ${file}`));

  if (!resolveCommand('opencode')) {
    warn('OpenCode is not installed. The config above is ready for when it is.');
    say(colour.grey('  Install it with:  npm install -g opencode-ai'));
    return 1;
  }

  return run('opencode', args, { NVIDIA_API_KEY: resolveApiKey(config) });
}

/** Which of the three agents are actually installed. */
export function detectAgents(): Array<{ name: string; command: string; path: string | null }> {
  return [
    { name: 'Claude Code', command: 'claude', path: resolveCommand('claude') },
    { name: 'Codex', command: 'codex', path: resolveCommand('codex') },
    { name: 'OpenCode', command: 'opencode', path: resolveCommand('opencode') },
  ];
}
