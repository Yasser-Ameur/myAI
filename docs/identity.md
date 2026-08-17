# Identity

The companion has a persistent name and self-model. Configuration is only a
bootstrap; the moment the user names it, that becomes a durable fact and the
persisted name outranks the YAML from then on.

## Authority

Highest wins:

1. **an explicit user statement in this turn** — "your name is Jarvis"
2. **a persisted identity fact** — what a previous session established
3. **the configured identity** — `system.name` in `config/companion.yaml`
4. **the built-in default** — `Companion`

A *hedged* statement ("I think your name might be Bob", "maybe you're Bob") is
detected on purpose, recorded as low-authority evidence, and **refused** as a
rename. `FactWriter` compares the incoming authority against the incumbent's
and declines the write rather than silently accepting it:

```
refusing to supersede self/self:name: authority 0.25 < incumbent 1.00
```

Questions are never assignments. "Is your name Jarvis?" and "what's your name?"
assert nothing.

## What counts as a rename

Accepted: `your name is X`, `your new name is X`, `you are called X`,
`I'll call you X`, `let's call you X`, `change your name to X`,
`rename yourself X`, `from now on you are X`, `I name you X`, and the bare
copula `You are X` — the last only when `X` is capitalised and is not an
ordinary adjective, so "you are helpful" and "you are working on it" are not
renames.

## History

A rename closes the old fact (`valid_to = now`) and inserts a new one. Nothing
is deleted, so the past stays queryable:

```
$ companion identity history
Friday               from 2026-08-17T09:14:02  [current]  via explicit_user_statement
Jarvis               from 2026-08-17T09:02:41  [until 2026-08-17T09:14:02]  via explicit_user_statement
```

and in conversation: *"Before Friday, you called me Jarvis."*

## Self-model

`SelfModelService.load()` reconstructs the model from the graph at startup,
before any model call. That is what makes a restart a continuation:

```
--- SESSION 2 (fresh process) (identity=Jarvis, source=persisted) ---
you> What's your name?
companion> My name is Jarvis — you named me that.   [via identity]
```

The model holds:

- `identity` — name, description, languages, persona
- `name_source` — `explicit_user_statement` | `persisted` | `config` | `default`
- `named_at`, `previous_names`
- `capabilities` — read **live** from the skill registry, never stored, so the
  companion cannot claim an ability it no longer has
- `limitations`

### Phrasing

The agent says *"I remember that you named me Jarvis"*, not *"I was born
Jarvis"*. Continuity without pretending to be human is deliberate: the identity
section states plainly that it is a local AI companion and does not claim
otherwise. When the name came from configuration it says so, and invites the
user to change it.

## Storage

Both singleton entities are resolved through `system_state`:

| key | entity | predicate |
|-----|--------|-----------|
| `agent_self_entity` | `self` (type `agent`) | `self:name`, `self:persona` |
| `primary_user_entity` | `user` (type `person`) | `user:name` |

Identity facts are written with `importance=1.0` and
`provenance="explicit_user_statement"`.

## CLI

```powershell
companion identity show
companion identity history
companion identity set Friday
```
