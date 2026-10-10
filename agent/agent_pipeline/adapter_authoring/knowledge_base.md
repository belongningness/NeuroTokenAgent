# NeuroToken Adapter Knowledge Base

This document is supplied to the coding agent together with the adapter templates.
Update it when a dataset family has source-specific conventions that cannot be inferred safely from HDF5 structure alone.

## Choosing the chunking strategy

- Use `event` when the source contains valid event intervals with meaningful onset, end time or duration and semantic segments must follow those intervals.
- Use `rest` for continuous/resting-state data or when events are absent, unreliable, or should not define semantic sample boundaries.
- Event intervals must have finite `start_time` and `end_time`, with `end_time > start_time`; they are clipped to the signal duration.
- The adapter should preserve event type, value, and source-specific metadata in `SourceSession.events`.
- The adapter reads and normalizes source structure. The pipeline configuration selects `chunking.strategy`, while generic packaging code constructs semantic sample ranges; do not silently choose a strategy or slice samples in adapter code.

### Required `event` segmentation semantics

Unless the user explicitly requests different behavior, event chunking means using source event boundaries to partition the complete recording from sample 0 through the final sample. Real event intervals retain their existing event metadata; every remaining non-empty range is an ordinary time segment without event metadata or a new type label. This policy must remain compatible with the public metadata produced by `reference_scripts/2_neural2token_human_new.py`; do not invent a new public segment taxonomy.

#### 1. Adapter responsibility

- Normalize only real source events into `SourceSession.events`. Each event must contain finite `start_time` and `end_time` with `end_time > start_time`, plus the real source `type`, `value`, and `other_info` when available.
- Do not append synthetic gap records to `SourceSession.events`, do not set `type="gap"`, and do not add fields such as `segment_kind`. `events_info.json` must remain a description of source events rather than a package segmentation manifest.
- Preserve explicit `start_sample` and `end_sample` together when the source provides authoritative sample boundaries. Otherwise shared packaging converts seconds to samples.
- The adapter must not choose `chunking.strategy`, slice the signal, or split records to match the tokenizer's fixed input size.

#### 2. Shared package responsibility

1. Validate and clip every normalized event to the signal extent using half-open bounds `[start, end)`.
2. Sort events stably by `(start_sample, end_sample, source_order)` before constructing package ranges. The original source index remains the event identity even if chronological sorting changes output order.
3. Start a coverage cursor at sample 0. Before each event, if `covered_until < event.start_sample`, emit `[covered_until, event.start_sample)` as an ordinary TAR sample without event metadata. This includes the leading range before the first event and internal ranges between events.
4. Emit every real event interval as one TAR sample with its `event_index`, `event_type`, `event_value`, `event_start_time`, and `event_end_time` metadata, then advance `covered_until` to the rightmost event end seen so far.
5. After the last event, if `covered_until < n_samples`, emit `[covered_until, n_samples)` as a final ordinary TAR sample. The resulting ranges cover the complete recording timeline.
6. If intervals touch, emit no zero-length ordinary segment. If they overlap or nest, preserve the real event intervals, use the rightmost covered end before creating the next ordinary range, and expose overlap evidence in package review.
7. A file with one event beginning after sample 0 and ending before the recording end therefore produces `ordinary, event, ordinary`. A single event spanning the whole recording produces only the event segment.

#### 3. Metadata compatibility with reference scripts

The public contract distinguishes an event segment by the presence of existing event metadata, not by a new kind label:

- An event TAR sample has `chunking_strategy="event"`, `chunk_start_sample`, `chunk_end_sample`, and the existing `event_*` fields. Tokenization projects it to `temporal_index.segments[].source_time` plus `source_event`.
- Any ordinary TAR sample outside real event coverage—leading, internal, or trailing—has `chunking_strategy="event"` and `chunk_start/end` fields but no `event_index`, no `event_*`, and no fabricated event object. Tokenization projects it to a normal `temporal_index.segments[]` entry with `source_time` and no `source_event`.
- Do not write `segment_kind="gap"`, `type="gap"`, `source_gap`, or any other new public label unless the reference scripts and public schema are deliberately versioned to support it.
- Internal package review may report aggregate `event_segment_count`, `ordinary_segment_count`, complete-recording coverage, and overlaps. Those diagnostics are not new fields in TAR sample metadata or `token_metadata.json`.

This preserves the existing reference-script behavior: downstream consumers identify real event alignment through `source_event`; all other temporal entries remain ordinary source-time segments.

#### 4. Semantic segments versus tokenizer windows

An event interval or inter-event interval is one logical TAR sample even when it is longer than the tokenizer model's fixed time size. Tokenization may process that sample in consecutive fixed-size windows or micro-batches, then restore order and publish one logical temporal entry according to the existing event strategy. Do not create 640-sample TAR records merely because the model consumes 640-sample windows.

#### 5. Required acceptance evidence

A source file containing only one event cannot prove inter-event behavior. Tests and candidate review for event chunking must include a synthetic multi-event session covering at least:

- a 21-second recording with events `[1, 9)` and `[10, 20)` yielding ordinary `[0, 1)`, event `[1, 9)`, ordinary `[9, 10)`, event `[10, 20)`, ordinary `[20, 21)`;
- touching events yielding no empty segment;
- unsorted source events yielding chronological package ranges while retaining source event identity;
- overlapping and nested events yielding no negative ordinary segment and explicit overlap diagnostics;
- exact half-open sample boundaries, including authoritative `start_sample/end_sample` when supplied;
- exact full-recording coverage, including leading and trailing ordinary TAR samples when events do not reach the recording edges;
- TAR metadata and resulting `token_metadata.json` containing no invented gap label;
- long semantic intervals remaining logically intact across tokenizer micro-batches.

## Adapter contract

- Return a `SourceSession` with `[channels, samples]` float32 signal, sampling frequency, normalized subject metadata, channel records, events, and session metadata.
- Keep source-specific HDF5 paths and normalization in the adapter. Keep packaging and tokenization generic.
- Prefer the smallest existing adapter as the template. Compare signal orientation, timestamps, metadata encoding, channel serialization, event normalization, and missing-value policy before copying code.
- Do not register generated adapters by editing unrelated modules when dynamic adapter discovery is available.
- Check the complete path from normalized `SourceSession.events` through packaging. An adapter that parses events correctly is not sufficient when shared packaging does not implement the user's requested segment semantics.
- If satisfying the request requires changing shared packaging outside the task's allowed paths, do not hide the limitation in `ADAPTER_DESIGN.md` or claim compliance. Report the task as blocked and request an explicitly scoped shared-code change.

## Review checklist

The coding agent and candidate reviewer must verify and explain:

1. Which HDF5 nodes and attributes were used.
2. Why `event` or `rest` is appropriate, including evidence about event quality.
3. For `event`, the exact ordered sequence of real event segments and unlabeled ordinary time segments expected across the complete recording, including overlap and recording-edge policy.
4. Whether the requested behavior is implemented by the adapter, existing shared packaging, or both; inspect the actual packaging call path rather than assuming the adapter alone controls segmentation.
5. How semantic segments remain independent from tokenizer model windows and batching.
6. How the implementation differs from the closest template adapter.
7. Which executable tests prove the user's intent, including at least one multi-event fixture when gaps are required.
8. Any assumptions, unsupported cases, allowed-path limitations, and recommended preflight checks.

Do not approve a candidate solely because it imports, returns `SourceSession`, or works for a single-event source file. Approval requires evidence that the resulting packaged ranges and metadata satisfy the user's explicit intent.
