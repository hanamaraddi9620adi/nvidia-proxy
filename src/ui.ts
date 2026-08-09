/**
 * Terminal helpers: colours, prompts, and an interactive picker.
 *
 * Deliberately dependency-free. Everything here is plain ANSI escape codes
 * and node:readline, so there is nothing to keep up to date and nothing that
 * can break when a package publishes a bad version.
 *
 * Control characters are written as \x1b / \x03 escapes rather than pasted
 * literally, so the file stays safe to open in any editor.
 */

import readline from 'node:readline';

const ESC = '\x1b';
const CSI = `${ESC}[`;

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

/**
 * Colour is disabled when the output is piped to a file, when NO_COLOR is set
 * (the informal cross-tool standard), or when the terminal claims to be dumb.
 */
const COLOUR_ENABLED =
  process.stdout.isTTY === true &&
  !process.env.NO_COLOR &&
  process.env.TERM !== 'dumb';

function wrap(open: string, close: string) {
  return (s: string): string =>
    COLOUR_ENABLED ? `${CSI}${open}m${s}${CSI}${close}m` : s;
}

export const colour = {
  bold: wrap('1', '22'),
  dim: wrap('2', '22'),
  italic: wrap('3', '23'),
  underline: wrap('4', '24'),
  red: wrap('31', '39'),
  green: wrap('32', '39'),
  yellow: wrap('33', '39'),
  blue: wrap('34', '39'),
  magenta: wrap('35', '39'),
  cyan: wrap('36', '39'),
  grey: wrap('90', '39'),
  /** NVIDIA green, the one brand nod in the whole tool. */
  nvidia: wrap('38;5;118', '39'),
};

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

export function say(msg = ''): void {
  process.stdout.write(msg + '\n');
}

export function ok(msg: string): void {
  say(`${colour.green('OK')} ${msg}`);
}

export function warn(msg: string): void {
  say(`${colour.yellow('!')}  ${msg}`);
}

/** Errors go to stderr so `nvp models | grep ...` stays clean. */
export function fail(msg: string): void {
  process.stderr.write(`${colour.red('x')}  ${msg}\n`);
}

export function info(msg: string): void {
  say(`${colour.cyan('>')}  ${msg}`);
}

export function heading(msg: string): void {
  say('');
  say(colour.bold(msg));
  say(colour.grey('-'.repeat(Math.min(Math.max(msg.length, 10), 60))));
}

/** Renders `key: value` pairs with the keys right-padded into a column. */
export function table(rows: Array<[string, string]>, indent = '  '): void {
  const width = rows.reduce((max, [k]) => Math.max(max, k.length), 0);
  for (const [k, v] of rows) {
    say(`${indent}${colour.grey(k.padEnd(width))}  ${v}`);
  }
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

/** A plain visible question. Returns the trimmed answer, or the default. */
export function promptLine(question: string, defaultValue = ''): Promise<string> {
  const suffix = defaultValue ? colour.grey(` (${defaultValue})`) : '';
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => {
    rl.question(`${question}${suffix}: `, (answer) => {
      rl.close();
      resolve(answer.trim() || defaultValue);
    });
  });
}

/**
 * A question whose typed characters are never echoed — used only for the
 * NVIDIA API key. The key is never printed back, never logged, and never
 * passed as a command-line argument (argv is visible to other processes).
 */
export function promptHidden(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: true,
  });

  // readline writes each keystroke through this private method. We swap it
  // for one that swallows output while `muted` is set, which is the standard
  // way to do this without pulling in a prompts library.
  const internals = rl as unknown as { _writeToOutput(s: string): void };
  const original = internals._writeToOutput.bind(rl);
  let muted = false;
  internals._writeToOutput = (s: string) => {
    if (!muted) original(s);
  };

  return new Promise((resolve) => {
    rl.question(`${question}: `, (answer) => {
      muted = false;
      process.stdout.write('\n');
      rl.close();
      resolve(answer.trim());
    });
    muted = true;
  });
}

export async function confirm(question: string, defaultYes = true): Promise<boolean> {
  const hint = defaultYes ? 'Y/n' : 'y/N';
  const answer = (await promptLine(`${question} ${colour.grey(`[${hint}]`)}`)).toLowerCase();
  if (!answer) return defaultYes;
  return answer === 'y' || answer === 'yes';
}

// ---------------------------------------------------------------------------
// Interactive picker
// ---------------------------------------------------------------------------

export interface Choice<T> {
  /** Main line, e.g. the model id. */
  label: string;
  /** Greyed-out second column, e.g. the publisher. */
  hint?: string;
  value: T;
}

const PAGE_SIZE = 12;

// Keys we care about, spelled out so the comparisons below stay readable.
const KEY = {
  ctrlC: '\x03',
  escape: '\x1b',
  enter: '\r',
  newline: '\n',
  tab: '\t',
  backspace: '\x7f',
  backspaceAlt: '\b',
  up: `${CSI}A`,
  down: `${CSI}B`,
  pageUp: `${CSI}5~`,
  pageDown: `${CSI}6~`,
  home: `${CSI}H`,
  end: `${CSI}F`,
  ctrlP: '\x10',
  ctrlN: '\x0e',
} as const;

/**
 * Arrow-key list with type-to-filter. Falls back to a numbered prompt when
 * stdin is not an interactive terminal (piped input, CI, some IDE consoles),
 * so scripted use never hangs waiting for a keypress that cannot arrive.
 *
 * Returns null if the user cancels with Esc.
 */
export async function select<T>(title: string, choices: Choice<T>[]): Promise<T | null> {
  if (choices.length === 0) return null;

  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    return selectFallback(title, choices);
  }

  return new Promise<T | null>((resolve) => {
    let filter = '';
    let cursor = 0;
    let scroll = 0;
    let linesDrawn = 0;

    const visible = (): Choice<T>[] => {
      if (!filter) return choices;
      const needle = filter.toLowerCase();
      return choices.filter(
        (c) =>
          c.label.toLowerCase().includes(needle) ||
          (c.hint ?? '').toLowerCase().includes(needle),
      );
    };

    const render = () => {
      const list = visible();

      // Erase whatever we drew last time before drawing the new frame.
      if (linesDrawn > 0) {
        process.stdout.write(`${CSI}${linesDrawn}A${CSI}0J`);
      }

      const out: string[] = [];
      out.push(colour.bold(title));
      out.push(
        colour.grey('  up/down move - type to filter - Enter select - Esc cancel') +
          (filter ? `   ${colour.cyan('filter:')} ${filter}` : ''),
      );

      if (list.length === 0) {
        out.push(colour.yellow('  no match'));
      } else {
        // Keep the cursor inside the visible window.
        if (cursor < scroll) scroll = cursor;
        if (cursor >= scroll + PAGE_SIZE) scroll = cursor - PAGE_SIZE + 1;

        const page = list.slice(scroll, scroll + PAGE_SIZE);
        const hintCol = page.reduce((m, c) => Math.max(m, c.label.length), 0);

        page.forEach((c, i) => {
          const index = scroll + i;
          const active = index === cursor;
          const marker = active ? colour.nvidia('>') : ' ';
          const label = active ? colour.nvidia(colour.bold(c.label)) : c.label;
          const pad = ' '.repeat(Math.max(0, hintCol - c.label.length));
          const hint = c.hint ? `  ${pad}${colour.grey(c.hint)}` : '';
          out.push(` ${marker} ${label}${hint}`);
        });

        if (list.length > PAGE_SIZE) {
          out.push(colour.grey(`   ${cursor + 1}/${list.length}`));
        }
      }

      process.stdout.write(out.join('\n') + '\n');
      linesDrawn = out.length;
    };

    const cleanup = () => {
      process.stdin.setRawMode(false);
      process.stdin.pause();
      process.stdin.removeListener('data', onKey);
    };

    const onKey = (raw: Buffer) => {
      const key = raw.toString('utf8');
      const list = visible();

      if (key === KEY.ctrlC) {
        // Leave the terminal usable, then exit the way any other tool would.
        cleanup();
        process.stdout.write('\n');
        process.exit(130);
      }

      if (key === KEY.escape) {
        cleanup();
        resolve(null);
        return;
      }

      if (key === KEY.enter || key === KEY.newline) {
        const chosen = list[cursor];
        cleanup();
        resolve(chosen ? chosen.value : null);
        return;
      }

      if (key === KEY.up || key === KEY.ctrlP) {
        cursor = cursor > 0 ? cursor - 1 : Math.max(0, list.length - 1);
      } else if (key === KEY.down || key === KEY.ctrlN || key === KEY.tab) {
        cursor = cursor < list.length - 1 ? cursor + 1 : 0;
      } else if (key === KEY.pageUp) {
        cursor = Math.max(0, cursor - PAGE_SIZE);
      } else if (key === KEY.pageDown) {
        cursor = Math.min(list.length - 1, cursor + PAGE_SIZE);
      } else if (key === KEY.home) {
        cursor = 0;
      } else if (key === KEY.end) {
        cursor = Math.max(0, list.length - 1);
      } else if (key === KEY.backspace || key === KEY.backspaceAlt) {
        filter = filter.slice(0, -1);
        cursor = 0;
      } else if (key.length === 1 && key >= ' ' && key <= '~') {
        filter += key;
        cursor = 0;
      } else {
        return; // Unrecognised escape sequence - ignore rather than redraw.
      }

      render();
    };

    process.stdin.setRawMode(true);
    process.stdin.resume();
    process.stdin.on('data', onKey);
    render();
  });
}

/** Numbered prompt used when raw-mode keypresses are not available. */
async function selectFallback<T>(title: string, choices: Choice<T>[]): Promise<T | null> {
  say(colour.bold(title));
  choices.forEach((c, i) => {
    say(`  ${String(i + 1).padStart(3)}. ${c.label}${c.hint ? colour.grey(`  ${c.hint}`) : ''}`);
  });
  const answer = await promptLine('Number');
  const index = Number.parseInt(answer, 10) - 1;
  const chosen = choices[index];
  return chosen ? chosen.value : null;
}

/** A minimal spinner for network calls. No-op when not attached to a TTY. */
export function spinner(text: string): { stop(finalLine?: string): void } {
  if (!process.stdout.isTTY) {
    say(colour.grey(text));
    return { stop: (finalLine?: string) => { if (finalLine) say(finalLine); } };
  }
  const frames = ['|', '/', '-', '\\'];
  let i = 0;
  process.stdout.write(`${CSI}?25l`); // hide cursor
  const timer = setInterval(() => {
    process.stdout.write(`\r${colour.nvidia(frames[i % frames.length]!)} ${text}${CSI}0K`);
    i++;
  }, 90);
  return {
    stop(finalLine?: string) {
      clearInterval(timer);
      process.stdout.write(`\r${CSI}0K${CSI}?25h`); // clear line, show cursor
      if (finalLine) say(finalLine);
    },
  };
}
