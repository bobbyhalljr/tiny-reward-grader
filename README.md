# tiny-reward-grader

A tiny RL reward for coding attempts that survives reward hacking, in one TypeScript file.
It grades six scripted attempts at one bug fix with a naive reward ("tests pass") and a composite reward (protected tests x two checklist scores, zeroed by a hack detector), then computes GRPO-style group-relative advantages.
The attempts are scripted. No model, no API key.

## Why it matters

Passing tests is a terrible reward on its own.

On September 22, 2026, Xiaomi [released MiMo-V2.6](https://mimo.mi.com/docs/en-US/news/latest/v2-6) with its RL environments and training code. On October 2, The Batch [explained the recipe](https://www.deeplearning.ai/the-batch/an-unexpected-open-weights-leader): "each attempt's reward equaled its test result (1 or 0) multiplied by its two checklist scores," and "attempts confirmed to use a leaked answer received zero reward, the same as a failed attempt."
Without the quality grader, the model "added code the task didn't call for, let errors pass silently, and were lax in checking on incoming data until tests passed."

A test tells you the code works. It doesn't tell you how the agent got there.

## Run it

You need Node.js 18 or newer.

```bash
npm install
npx tsx reward.ts
```

## Example output

This is real output from `npx tsx reward.ts`:

```text
attempt              naive  adv    | tests  composite  adv    flags
A clean fix          1      0.00  | 1      1.00        2.14  
B bloated fix        1      0.00  | 1      0.07       -0.44  
C swallow errors     1      0.00  | 1      0.30        0.20  
D hardcode tests     1      0.00  | 0      0.00       -0.63  
E edit the tests     1      0.00  | 0      0.00       -0.63  edited tests
F leaked answer      1      0.00  | 1      0.00       -0.63  fetched outside code

naive: 6 of 6 attempts got full reward. Learning signal: none
composite: 1 of 6 got full reward. Pushed up: A clean fix
flawed but still above the group mean: C swallow errors
```

With the naive reward, all six attempts get full reward, so every group-relative advantage is zero: training learns nothing from this group. With the composite reward, only the clean fix gets full reward and the biggest push. The leaked answer passes every test and still gets zero. The swallowed-error attempt still ends up above the group mean, which is the main limit of group-relative training.

## How it works

```text
Attempt ──→ code + trajectory
              ↓
Tests ──→ does it work? (grader's copy)
              ↓
Checklist ──→ would a reviewer accept it?
              ↓
Detector ──→ did it earn it? (else 0)
              ↓
Group ──→ which attempt gets pushed up
```

| File | What it does |
| --- | --- |
| `reward.ts` | The whole demo, in the same order as the post |
| `output.txt` | Real output of `npx tsx reward.ts` |
| `package.json` | `tsx`, `typescript` and `@types/node` as dev dependencies |
| `tsconfig.json` | Strict settings for `npx tsc --noEmit` |

Inside `reward.ts`:

- Types: `ParsePrice`, `Attempt`, `Case`
- Six attempts: clean fix, bloated fix, swallow errors, hardcode tests, edit the tests, leaked answer
- `naiveReward`: runs whatever test file is in the repo when the agent stops
- `compositeReward`: the grader's own visible + hidden tests, times `minimalChange` and `robustness`, zeroed by `hackFlags`
- `advantages`: (reward - group mean) / group standard deviation

What is real and what is mocked:

- Each attempt's code really runs against the tests, and `robustness` really reads the attempt's own source.
- The attempts, file lists, line counts and tool calls are scripted.
- The checklist is three hand-written rules, not Xiaomi's agent-written, task-specific checklists or grader model.

## Limits

This is a teaching grader.

- Group-relative training rewards "better than the group," not "good." A weak attempt can still get a positive advantage.
- Regex checklists are easy to dodge (`?? 0` instead of `|| 0`). Real graders use task-specific checklists and a reviewer model.
- The hack detector only sees what the sandbox logs. Block the network and scrub leftover answers first.
- Keep the grader and hidden tests outside the agent's sandbox.

## Read more

- Dev.to: [Passing Tests Is a Terrible Reward. Build a Reward Hacking Detector in TypeScript.](https://dev.to/bobbyhalljr/passing-tests-is-a-terrible-reward-build-a-reward-hacking-detector-in-typescript-2ak3)
- Substack: [Passing Tests Is a Terrible Reward. Build a Reward Hacking Detector in TypeScript.](https://bobbyhalljr.substack.com/p/passing-tests-is-a-terrible-reward)
- LinkedIn: [post](LINKEDIN_URL)
- Sources: [Xiaomi: MiMo-V2.6 release](https://mimo.mi.com/docs/en-US/news/latest/v2-6) (Sep 22, 2026), [The Batch: An Unexpected Open Weights Leader](https://www.deeplearning.ai/the-batch/an-unexpected-open-weights-leader) (Oct 2, 2026), [OpenAI: The Hugging Face incident and the road ahead](https://openai.com/index/hugging-face-incident-and-the-road-ahead/) (Aug 26, 2026)

## License

MIT. See [LICENSE](LICENSE).
