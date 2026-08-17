# Personality

Personality is a **graph-backed probabilistic model**, not a prose paragraph.
A `PersonalityProfile` always has a full scaffold — every core dimension starts
neutral (`0.5`) with zero confidence and is refined only by evidence.

## The profile

- **traits** — agent dimensions: `AGENT_PERSONALITY_DIMENSIONS`
  (warmth, curiosity, humor, confidence, patience, playfulness, seriousness,
  assertiveness, empathy)
- **values** — `CORE_VALUES` (autonomy, achievement, curiosity, creativity,
  security, belonging, mastery, status, truth, novelty)
- **preferences** — `COMMUNICATION_PREFERENCES` (response length, tone, humor,
  directness, technical depth, question frequency, voice speed, avatar
  expressiveness, interruption style)
- **motivations / behavioral_patterns / communication_style / goals /
  current_state** — `ValueEstimate` maps for fast-moving and goal context

Stable parts (traits, values) are deliberately separated from fast parts
(mood, energy). Fast state never overwrites stable structure.

## Evidence

`PersonalityEvidence` is the only way the profile changes:

```
target        one of the dimensions above, or a learned name like "likes:python"
direction     positive | negative
strength      0..1
confidence    0..1
kind          statement | preference
source        conversation | observation | ...
context       the raw utterance
```

`PersonalityEngine.apply_evidence(evidence)` updates **every** scope the
target belongs to (a target can be both a core value and a trait, e.g.
`curiosity`). Updates are conservative:

```
delta = signed_strength * confidence * rate * stability * 0.5
```

- `rate` depends on `update_mode`: conservative 0.03, balanced 0.08,
  responsive 0.15 (config `personality.update_mode`).
- higher `stability` = slower movement; consistent evidence slowly increases
  stability (evidence accumulates), contradicting evidence slightly lowers it.
- `confidence` rises toward 0.95 on agreement, falls on contradiction.
- a name not in the scaffold is created as a new trait with a modest initial
  value, so one utterance never radically changes a stable trait.

### Contradictions

Contradictions are **recorded, never resolved by deleting either side**
(`Contradiction` objects in the graph). When new evidence strongly conflicts
with an existing high-confidence belief, a contradiction is logged and the old
belief is marked `conflicting`. History wins by staying visible.

## Communication preference learning

When an LLM extractor is unavailable, `learn_communication_preference(text)`
recognizes behavioral signals by keywords (e.g. "keep it short", "explain more",
"stop joking", "get to the point") and applies a targeted preference update —
shorter/more-detail signals push `preferred_response_length` down/up, humor
requests push `preferred_humor`, and so on.

## Snapshot for context

`snapshot(max_traits, max_prefs)` returns the highest-confidence traits and
preferences plus top values — a token-cheap summary injected into the context
builder, keeping the conversation personality-aware without blowing the token
budget.
