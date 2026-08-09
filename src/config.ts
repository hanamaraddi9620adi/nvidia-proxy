/**
 * The persistent store: ~/.nvproxy/config.json
 *
 * This is the whole point of the tool. You enter your NVIDIA key and pick your
 * models once; every later command reads them from here. Nothing ever asks you
 * to "connect" again.
 *
 * Layout:
 *   ~/.nvproxy/
 *     config.json   the key, the port, and every profile   (chmod 600)
 *     models.json   cached model catalogue                 (chmod 600)
 *     server.json   the running server's port + pid
 */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

export const CONFIG_DIR = path.join(os.homedir(), '.nvproxy');
export const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');
export const CACHE_FILE = path.join(CONFIG_DIR, 'models.json');
export const SERVER_FILE = path.join(CONFIG_DIR, 'server.json');

export const DEFAULT_BASE_URL = 'https://integrate.api.nvidia.com/v1';
export const DEFAULT_PORT = 8787;

/**
 * A profile is one saved way of working: which model does the heavy lifting,
 * which cheap model handles the small background calls, and how it is tuned.
 *
 * Coding agents ask for a big model and a small one. Claude Code, for example,
 * uses a small model for things like summarising a conversation, and the big
 * one for the actual work. Mapping both lets the agent behave normally.
 */
export interface Profile {
  /** Model for real work. Maps to Claude Code's opus/sonnet tiers. */
  big: string;
  /** Cheap fast model for background chores. Maps to the haiku tier. */
  small: string;
  /** Upper bound on generated tokens. Agents usually send their own. */
  maxTokens: number;
  /** null means "send no temperature and let the model decide". */
  temperature: number | null;
  /**
   * Reasoning models (Kimi thinking, DeepSeek, Nemotron) emit a private
   * chain of thought. Off by default: coding agents want the answer, and
   * leaking raw reasoning into a tool-call stream confuses them.
   */
  showReasoning: boolean;
  /** Free-text note so `nvp profile list` is self-explanatory a year later. */
  note?: string;
}

export interface Config {
  /** Bumped only if the shape changes in a way that needs migrating. */
  version: 1;
  /** The nvapi-... key. Empty until `nvp setup` or `nvp key set`. */
  apiKey: string;
  baseUrl: string;
  port: number;
  activeProfile: string;
  profiles: Record<string, Profile>;
}

export const DEFAULT_PROFILE: Profile = {
  big: '',
  small: '',
  maxTokens: 8192,
  temperature: null,
  showReasoning: false,
};

function blankConfig(): Config {
  return {
    version: 1,
    apiKey: '',
    baseUrl: DEFAULT_BASE_URL,
    port: DEFAULT_PORT,
    activeProfile: 'default',
    profiles: { default: { ...DEFAULT_PROFILE, note: 'created by nvp setup' } },
  };
}

// ---------------------------------------------------------------------------
// Reading and writing
// ---------------------------------------------------------------------------

function ensureDir(): void {
  // mode 0700 matters on macOS/Linux. Windows ignores it; there the file
  // inherits the user profile directory's ACL, which is already per-user.
  fs.mkdirSync(CONFIG_DIR, { recursive: true, mode: 0o700 });
}

export function loadConfig(): Config {
  try {
    const raw = fs.readFileSync(CONFIG_FILE, 'utf8');
    const parsed = JSON.parse(raw) as Partial<Config>;

    // Merge over a blank config so a file written by an older version, or one
    // a human hand-edited and truncated, still produces something usable.
    const base = blankConfig();
    const config: Config = {
      version: 1,
      apiKey: typeof parsed.apiKey === 'string' ? parsed.apiKey : base.apiKey,
      baseUrl: parsed.baseUrl || base.baseUrl,
      port: Number.isInteger(parsed.port) ? (parsed.port as number) : base.port,
      activeProfile: parsed.activeProfile || base.activeProfile,
      profiles: { ...base.profiles, ...(parsed.profiles ?? {}) },
    };

    // Fill in any field a hand-edited profile is missing.
    for (const [name, profile] of Object.entries(config.profiles)) {
      config.profiles[name] = { ...DEFAULT_PROFILE, ...profile };
    }
    if (!config.profiles[config.activeProfile]) {
      config.activeProfile = Object.keys(config.profiles)[0] ?? 'default';
    }
    return config;
  } catch {
    // No file yet, or unreadable/corrupt. Either way a fresh config is right.
    return blankConfig();
  }
}

export function saveConfig(config: Config): void {
  ensureDir();
  // Write to a temporary file and rename, so an interrupted write (Ctrl-C,
  // power cut) can never leave a half-written config behind.
  const tmp = `${CONFIG_FILE}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(config, null, 2) + '\n', { mode: 0o600 });
  fs.renameSync(tmp, CONFIG_FILE);
  try {
    fs.chmodSync(CONFIG_FILE, 0o600);
  } catch {
    // Windows may refuse chmod. Not fatal.
  }
}

// ---------------------------------------------------------------------------
// Convenience accessors
// ---------------------------------------------------------------------------

/**
 * The key in use, preferring the environment. An env var is handy for CI and
 * for trying a second key without disturbing the saved one.
 */
export function resolveApiKey(config: Config): string {
  return process.env.NVIDIA_API_KEY?.trim() || config.apiKey;
}

export function activeProfile(config: Config): Profile {
  return config.profiles[config.activeProfile] ?? { ...DEFAULT_PROFILE };
}

/** Shows enough of the key to recognise it, never enough to use it. */
export function maskKey(key: string): string {
  if (!key) return '(not set)';
  if (key.length <= 12) return `${key.slice(0, 4)}...`;
  return `${key.slice(0, 8)}...${key.slice(-4)}`;
}

/** A key is usable only if it looks like NVIDIA's format. */
export function looksLikeNvidiaKey(key: string): boolean {
  return /^nvapi-[A-Za-z0-9_\-]{10,}$/.test(key);
}

export function isConfigured(config: Config): boolean {
  const profile = activeProfile(config);
  return Boolean(resolveApiKey(config) && profile.big);
}

// ---------------------------------------------------------------------------
// Model catalogue cache
// ---------------------------------------------------------------------------

export interface CachedCatalogue {
  fetchedAt: number;
  models: string[];
}

export function readCatalogueCache(maxAgeMs = 24 * 60 * 60 * 1000): string[] | null {
  try {
    const cache = JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8')) as CachedCatalogue;
    if (!Array.isArray(cache.models) || cache.models.length === 0) return null;
    if (Date.now() - cache.fetchedAt > maxAgeMs) return null;
    return cache.models;
  } catch {
    return null;
  }
}

export function writeCatalogueCache(models: string[]): void {
  ensureDir();
  const payload: CachedCatalogue = { fetchedAt: Date.now(), models };
  try {
    fs.writeFileSync(CACHE_FILE, JSON.stringify(payload, null, 2) + '\n', { mode: 0o600 });
  } catch {
    // A cache that cannot be written is a slow tool, not a broken one.
  }
}

// ---------------------------------------------------------------------------
// Running-server record
// ---------------------------------------------------------------------------

export interface ServerRecord {
  port: number;
  pid: number;
  startedAt: number;
}

export function readServerRecord(): ServerRecord | null {
  try {
    return JSON.parse(fs.readFileSync(SERVER_FILE, 'utf8')) as ServerRecord;
  } catch {
    return null;
  }
}

export function writeServerRecord(record: ServerRecord): void {
  ensureDir();
  try {
    fs.writeFileSync(SERVER_FILE, JSON.stringify(record, null, 2) + '\n', { mode: 0o600 });
  } catch {
    // Non-fatal: the record is only a convenience for `nvp status`.
  }
}

export function clearServerRecord(): void {
  try {
    fs.unlinkSync(SERVER_FILE);
  } catch {
    // Already gone.
  }
}
