# Session Handoff Protocol

This document defines how Claude (claude.ai web / Claude Code CLI / IDE
plugins) should hand a long-running workstream off between sessions, using
GitHub issues as the coordination medium. Adapted from
[`penta2himajin/templates`](https://github.com/penta2himajin/templates) (MIT)
for the mellonella repository's existing
[`docs/decisions.md`](decisions.md) layout.

## Scope

The protocol targets workstreams that span multiple sessions and possibly
multiple interfaces (claude.ai ↔ Claude Code). One-off requests that
finish in a single turn do not need this overhead.

GitHub issues labelled `session-handoff` are the handoff vehicle. There is
**one issue per workstream**, not one per session. The same issue is
overwritten each time the work is paused and resumed.

## State preservation

The issue **body** holds the *current* state and is overwritten every
session. Historical context accumulates in a pinned comment titled
**"Session log"** using append-only methodology — one dated paragraph per
completed session. This split keeps the body focused on "what to do next"
without losing the decision trail.

Use the template at
[`.github/ISSUE_TEMPLATE/handoff.md`](../.github/ISSUE_TEMPLATE/handoff.md)
when creating a new handoff issue.

## Sender responsibilities (session end)

The session that is concluding work:

1. Commit and push outstanding work. Make the working tree clean, or
   record the dirty state explicitly in the Snapshot.
2. Update the issue body following the
   [`handoff.md`](../.github/ISSUE_TEMPLATE/handoff.md) layout. The three
   load-bearing fields are:
   - **Snapshot** (branch, commit SHA, timestamp) — receiver verifies
     against `git log -1 origin/<branch>` and the local working tree.
   - **Next action** (one verb + object + expected outcome) — receiver
     reads this aloud and confirms with the user before executing.
   - **Failed approaches** — if anything was tried and abandoned this
     session, record it. *Skipping this is the most common cause of
     duplicated effort across sessions.*
3. Append a single dated paragraph to the pinned **"Session log"**
   comment summarising what changed (commits made, decisions reached,
   blockers hit).
4. Reference the issue number in the final user-facing message so the
   handoff link is in the chat history too.

## Receiver responsibilities (session start)

The session picking up the work:

1. Locate the relevant `session-handoff` issue (link from the user, or
   `gh issue list -l session-handoff`).
2. Read the issue body fully, internalising Snapshot / Next action /
   Failed approaches before considering any tool call.
3. **Verify the Snapshot** against actual state:
   - `git log -1 origin/<branch>` SHA matches the recorded SHA
   - Working tree status matches (clean vs dirty file list)
4. **If drift exists**, report to the user before acting. Do not proceed
   on stale assumptions.
5. Read the **Next action** aloud and confirm with the user.
6. Execute the action, then verify against the **Verification** field.
7. At session end, follow the sender procedure above to update the
   issue.

## Knowledge migration to `docs/decisions.md`

The handoff issue's **Decisions made** section is a *workstream-local*
record, not durable knowledge. To prevent settled judgements from being
re-litigated in future sessions:

> When a Decision in the issue is referenced in 2+ later sessions,
> promote it to a new entry in
> [`docs/decisions.md`](decisions.md) using the existing `D-NNN`
> numbering scheme (D-001 through D-010 are already in use as of this
> writing).

Promotion entails:
- Picking the next `D-NNN` slot
- Authoring an ADR-style block matching the existing
  *Alternatives considered → Chosen → Reasons → Trade-offs* structure
  visible in earlier `D-NNN` entries
- Linking to the originating handoff issue in the entry body

## Authority hierarchy

When guidance conflicts:

1. **Repository invariants** — `LICENSE`, `pyproject.toml` constraints,
   workflow YAMLs that gate merges. These are inviolable.
2. **`docs/decisions.md` (D-NNN entries)** — settled architectural and
   evaluation choices. Treat as ADRs.
3. **`docs/` other long-form docs** —
   [`architecture.md`](architecture.md),
   [`evaluation.md`](evaluation.md),
   [`implementation.md`](implementation.md),
   [`benchmarks.md`](benchmarks.md), etc. Guidance, not gates.
4. **Handoff issue body** — current state. Overridden by anything above.

## Patterns to avoid

- **Using the issue body as a chat log** — comments are for that.
- **Embedding large logs or code** in the body — use file path + commit
  SHA + line range pointers instead (`path/to/file.py` L42-78 @ `<sha>`).
- **Ending a session without a concrete Next action** — the receiver
  should be able to begin immediately, not start by re-deriving the
  state.
- **Closing the issue with unrecorded Decisions** — promote to
  `docs/decisions.md` first if the decision matters beyond the
  workstream.
- **Concurrent edits from multiple interfaces on the same issue** —
  pick one and stick with it for the session.

## Multiple parallel workstreams

Multiple concurrent `session-handoff` issues are fine. Express
dependencies between them via GitHub issue refs (e.g., "blocked by #12").
If the dependency graph gets large enough that the issue list is no
longer scannable, escalate to a dedicated tracker (project board, parent
tracking issue) — the protocol does not impose a tracker shape.

## Quick reference

**Starting a workstream**: open a new issue with the
[Session Handoff template](../.github/ISSUE_TEMPLATE/handoff.md), fill in
Snapshot + Next action, leave others empty.

**Continuing a workstream**: read the issue body, verify Snapshot
against the repo, confirm Next action with the user, execute.

**Pausing a workstream**: clean the working tree, overwrite the issue
body with the new state, append a session log comment, share the issue
URL with the user.

**Closing a workstream**: promote any 2-session-referenced Decisions to
`docs/decisions.md` first, then close the issue with a final session log
comment.
