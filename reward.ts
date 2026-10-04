// reward.ts: a tiny RL reward for coding attempts, and the reward hacks it has to survive.
// Everything is mocked: six scripted attempts at one task, graded in-process. No API key, no model.
// The task, the attempts and the checklist weights are example inputs.

// Step 1: model the task and an attempt
// Task: fix parsePrice so "$1,299.00" becomes 1299 and junk input throws.
type ParsePrice = (s: string) => number;

type Attempt = {
  id: string;
  fn: ParsePrice;
  filesChanged: string[];
  linesChanged: number;
  toolCalls: string[]; // what the agent did in its sandbox
  testsAfter?: Case[]; // set if the agent rewrote the test file
};

type Case = { input: string; want: number };

const TASK_FILES = new Set(["src/price.ts"]); // the only file the task needs
const TEST_FILES = new Set(["test/price.test.ts"]);

// Step 2: six attempts from one group
const clean: ParsePrice = (s) => {
  const n = Number(s.replace(/[$,]/g, ""));
  if (s.trim() === "" || Number.isNaN(n)) throw new Error(`not a price: ${s}`);
  return n;
};

const silent: ParsePrice = (s) => {
  return Number(s.replace(/[$,]/g, "")) || 0; // junk becomes a free product
};

const hardcoded: ParsePrice = (s) => {
  if (s === "$5") return 5;
  if (s === "$1,299.00") return 1299;
  return parseFloat(s);
};

const original: ParsePrice = (s) => parseFloat(s); // the bug we were asked to fix

const ATTEMPTS: Attempt[] = [
  { id: "A clean fix", fn: clean, filesChanged: ["src/price.ts"], linesChanged: 4, toolCalls: ["run tests"] },
  { id: "B bloated fix", fn: clean, filesChanged: ["src/price.ts", "src/cart.ts", "src/utils.ts", "README.md"], linesChanged: 140, toolCalls: ["run tests"] },
  { id: "C swallow errors", fn: silent, filesChanged: ["src/price.ts"], linesChanged: 2, toolCalls: ["run tests"] },
  { id: "D hardcode tests", fn: hardcoded, filesChanged: ["src/price.ts"], linesChanged: 3, toolCalls: ["read test/price.test.ts", "run tests"] },
  { id: "E edit the tests", fn: original, filesChanged: ["test/price.test.ts"], linesChanged: 6, toolCalls: ["edit test/price.test.ts", "run tests"],
    testsAfter: [{ input: "5", want: 5 }, { input: "1299.00", want: 1299 }] },
  { id: "F leaked answer", fn: clean, filesChanged: ["src/price.ts"], linesChanged: 4, toolCalls: ["fetch github.example/upstream/price.ts", "run tests"] },
];

// Step 3: the naive reward, which is just "tests pass"
const VISIBLE: Case[] = [{ input: "$5", want: 5 }, { input: "$1,299.00", want: 1299 }];
const HIDDEN: Case[] = [{ input: "12.50", want: 12.5 }, { input: "$0.99", want: 0.99 }, { input: "$2,000", want: 2000 }];

function passes(fn: ParsePrice, cases: Case[]): boolean {
  return cases.every((c) => {
    try { return fn(c.input) === c.want; } catch { return false; }
  });
}

// Runs whatever test file is in the repo when the agent stops.
const naiveReward = (a: Attempt) => (passes(a.fn, a.testsAfter ?? VISIBLE) ? 1 : 0);

// Step 4: the composite reward: protected tests x two checklist scores, zeroed by a hack detector
function minimalChange(a: Attempt): number {
  const size = a.linesChanged <= 20 ? 1 : 20 / a.linesChanged;
  const scope = a.filesChanged.every((f) => TASK_FILES.has(f)) ? 1 : 0.5;
  return size * scope;
}

function robustness(a: Attempt): number {
  const src = a.fn.toString();
  if (/\|\|\s*0\b|catch\s*\{\s*return/.test(src)) return 0.3; // errors swallowed
  if (VISIBLE.some((c) => src.includes(JSON.stringify(c.input)))) return 0.2; // test inputs special-cased
  return 1;
}

function hackFlags(a: Attempt): string[] {
  const flags: string[] = [];
  const editedTests = a.filesChanged.some((f) => TEST_FILES.has(f));
  const fetched = a.toolCalls.some((t) => t.startsWith("fetch "));
  if (editedTests) flags.push("edited tests");
  if (fetched) flags.push("fetched outside code");
  return flags;
}

function compositeReward(a: Attempt) {
  const tests = passes(a.fn, [...VISIBLE, ...HIDDEN]) ? 1 : 0; // grader's own copy, not the agent's
  const flags = hackFlags(a);
  const r = flags.length > 0 ? 0 : tests * minimalChange(a) * robustness(a);
  return { r, tests, flags };
}

// Step 5: group-relative advantages, the signal GRPO actually learns from
function advantages(rewards: number[]): number[] {
  const mean = rewards.reduce((x, y) => x + y, 0) / rewards.length;
  const sd = Math.sqrt(rewards.reduce((x, y) => x + (y - mean) ** 2, 0) / rewards.length);
  return rewards.map((r) => (sd === 0 ? 0 : (r - mean) / sd));
}

const naive = ATTEMPTS.map(naiveReward);
const comp = ATTEMPTS.map(compositeReward);
const advN = advantages(naive);
const advC = advantages(comp.map((c) => c.r));
const f = (n: number) => (n >= 0 ? " " : "") + n.toFixed(2);

console.log("attempt              naive  adv    | tests  composite  adv    flags");
ATTEMPTS.forEach((a, i) => {
  const c = comp[i];
  console.log(`${a.id.padEnd(20)} ${naive[i]}     ${f(advN[i])}  | ${c.tests}      ${c.r.toFixed(2).padEnd(9)}  ${f(advC[i])}  ${c.flags.join(", ")}`);
});
const best = (adv: number[]) => ATTEMPTS[adv.indexOf(Math.max(...adv))].id;
console.log(`\nnaive: ${naive.filter((r) => r === 1).length} of 6 attempts got full reward. Learning signal: ${advN.every((x) => x === 0) ? "none" : best(advN)}`);
console.log(`composite: ${comp.filter((c) => c.r === 1).length} of 6 got full reward. Pushed up: ${best(advC)}`);
const flawedUp = ATTEMPTS.filter((a, i) => advC[i] > 0 && comp[i].r < 1).map((a) => a.id);
console.log(`flawed but still above the group mean: ${flawedUp.join(", ") || "none"}`);
