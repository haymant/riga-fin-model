#!/usr/bin/env python3
"""Emit a reviewed starter SFT dataset for the structured-products / payoff-graph domain.

Every example is grounded in the repository's own reference corpus:

- ``skills/fina-visual-builder/`` (payoff-graph block model, block book, layer model)
- ``data/refs/books/MurexPlaybook/`` (platform product mechanics)
- ``data/refs/financial-products/`` (bank product booklets and illustrations)

The generator is deterministic: running it twice writes byte-identical files. It exists
so the dataset is reproducible and reviewable line by line, rather than a one-off blob.

The output is a *draft*. Teacher/agent-generated or hand-curated training text is not
verified ground truth; a human must review the records before fine-tuning, and the
held-out split here is small.

Usage
-----
    python scripts/generate_seed_dataset.py --output-dir data/processed

Writes ``<output-dir>/sp-sft-train.jsonl`` (100 examples) and
``<output-dir>/sp-sft-valid.jsonl`` (12 held-out examples).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Matches DEFAULT_SYSTEM in scripts/train_unsloth.py so the trainer renders the same prompt.
DEFAULT_SYSTEM = (
    "You are a careful, evidence-grounded assistant specialized in financial modeling. "
    "State assumptions and units clearly."
)

# ---------------------------------------------------------------------------
# Curated, cited examples. Each item is grounded in a specific source location.
# ``id`` is a stable chunk id used for provenance and duplicate tracking.
# ---------------------------------------------------------------------------
ITEMS: list[dict] = [
    # --- payoff-graph / fina-visual-builder -----------------------------------------
    dict(
        id="sp-001", task_type="payoff_graph_concept",
        source="skills/fina-visual-builder/SKILL.md", location="Lego-block mental model",
        q="In the payoff-graph design, what is a 'block', and what makes a block worth having?",
        a="A block is one small piece of a structured product's payoff: it declares typed input ports (what it consumes), typed output ports (what it emits), a config parameter form, read/write lifecycle state, and a kernel binding — the single function that computes it.\n\nA block is worth having only if it is small enough that its correctness can be settled by reading one sentence and running a test, with no simulation, no market data, and no server. That testability is the whole standard: if a block needs the pricing loop to exercise it, it is not a block yet.",
    ),
    dict(
        id="sp-002", task_type="payoff_graph_concept",
        source="skills/fina-visual-builder/SKILL.md", location="Three layers: model / template / trade",
        q="Distinguish the model, template, and trade layers of a structured product.",
        a="The three layers are distinct and must not be conflated:\n\n1. Model — shared across a product family: the pricing model group, calibration convention, and numerical lane (e.g. EQ_RAKI; an XAD fixed-branch lane versus a CRN-fallback lane).\n2. Template — the payoff-graph topology: which nodes and edge roles exist and how they are wired, together with the kernel bindings (e.g. EqFlexRakiP versus EqFlexMemRP).\n3. Trade — the per-deal record that binds leaves, activates optional subgraphs, fills parameters, and seeds lifecycle state, producing a concrete graph with its own graph_hash.\n\nKeeping them separate lets one engine serve many trades without cloning anything.",
    ),
    dict(
        id="sp-003", task_type="payoff_graph_layer_rule",
        source="skills/fina-visual-builder/SKILL.md", location="Rules (three layers)",
        q="What kind of change earns a new template, and what is merely trade data?",
        a="Trade data covers everything the template can already express: a different barrier, rate, strike, schedule, basket member, reference price, or lifecycle flag, and any feature toggle (localKO, globalKO, knockIn, memory, ITMPayment) that the template declares. These are bindings/toggles/config on an existing topology and do not change graph_hash.\n\nA new template is earned only by a topology change — a new block kind or a new edge role. The canonical example is memory KO versus non-memory: EqFlexRakiP versus EqFlexMemRP.\n\nIf an edit is none of these, the visual builder must be able to refuse it rather than invent a node.",
    ),
    dict(
        id="sp-004", task_type="payoff_graph_catalog",
        source="skills/fina-visual-builder/refs/block-book/registry/blocks.yaml", location="header (13-block catalogue)",
        q="How many block kinds are in the catalogue, and how do they map onto real kernels?",
        a="There are 13 block kinds, and as of the block-book's writing they resolve to 9 distinct kernels:\n\n- 9 block kinds have a clean, standalone, pure kernel — sitting on 8 functions, because cash_settlement and physical_delivery_settlement share one function.\n- 2 block kinds (local_ko_gate, global_ko_gate) map onto resolve_ko, a six-line precedence switch that only does part of the job (it answers 'which of two fired', not 'did a barrier fire').\n- 2 block kinds (worst_of_performance, payment_discount) have no kernel at all; they are inlined in the pricing loop.\n\nThat mismatch is why the registry tracks kernel_mapped explicitly.",
    ),
    dict(
        id="sp-005", task_type="payoff_graph_concept",
        source="skills/fina-visual-builder/SKILL.md", location="Where the payoff script sits",
        q="What is the payoff_script, and is it an input to the calculation?",
        a="No. The payoff_script is not an input to the calculation; it is the rendered decomposition of the already-assembled DAG, attached to the graph and displayed as the caption over the final aggregation — for example:\n\nworst-of { - put(EKI 0.7, strike 0.78, physical_delivery); + notional_return(1x); + coupon_strip(0.009642, WPS, N2-N1, memory) }\n  gates: EKI(0.7, final_fixing) + global_ko(1.1, >=) terminates coupon+put\n\nAt design time the editor regenerates it from the nodes and edges; at run time aggregate_legs produces the signed sum of discounted leg cash flows. The script is the label, the aggregate is the value. Editing the script changes nothing in the DAG.",
    ),
    dict(
        id="sp-006", task_type="payoff_graph_concept",
        source="skills/fina-visual-builder/refs/ModelAndTradeLayer.md", location="sect 3.1 Three seals",
        q="What is graph_hash and what does it seal?",
        a="graph_hash is the template layer's structure identity: a hash over the canonical graph document, together with function_bindings. It changes when topology changes (a new block kind or edge role) and stays unchanged for trade-data edits, so it seals the exact wiring for cache keys and diffing.\n\nIt is one of three seals: model → engine identity (engine_marker, aad_engine, per-cell method); template → structure identity (graph_hash + function_bindings); trade → instance identity (request_id, process_id, source_revision). Together they make the auditable claim: this number came from this model, assembled from this template, for this trade.",
    ),
    dict(
        id="sp-007", task_type="evidence_status",
        source="skills/fina-visual-builder/refs/block-book/00-glossary.md", location="Block; Evidence status",
        q="What is the evidence-status vocabulary, and are the labels grades?",
        a="The vocabulary is: implemented_and_evidenced, partial, unsupported, ambiguous_in_source, unresolved — borrowed from fina_risk::EvidenceStatus so there is exactly one vocabulary across the block book, the registry, and the code.\n\nThey are honest labels, not grades. 'unresolved' on a real block is the correct state and is better than a green tick nobody can defend. The block book also records evidence on three independent axes — spec_conformance, deal_evidence, and lifecycle_state — because a block can match its specification and still be fed incomplete or internally inconsistent state.",
    ),
    dict(
        id="sp-008", task_type="block_kernel",
        source="skills/fina-visual-builder/refs/block-book/blocks/coupon-strip.md", location="sect 4 Kernel",
        q="Write the coupon_strip kernel arithmetic and state the units of each term.",
        a="The kernel is:\n\n  total  = max(total_fixings, 1)\n  unpaid = max(qualifying_fixings - already_paid_fixings, 0)\n  rate   = fixed_coupon + range_rate * min((unpaid + carried_memory) / total, 1)\n\nUnits matter here: range/fixed are rates-per-period; qualifying_fixings, total_fixings, already_paid_fixings, and carried_memory are counts of fixings (days); unpaid is a count. The output rate is a rate-per-period, not a currency amount. Three words carry the block: 'fixings' (a share of days, not money), 'capped' (min(...,1) never overpays), and 'carried in' (memory enters as an addition to the count before the division).",
    ),
    dict(
        id="sp-009", task_type="evidence_reasoning",
        source="skills/fina-visual-builder/refs/block-book/blocks/coupon-strip.md", location="sect 2 Specification",
        q="Why is the coupon carry recorded as 'unsupported' rather than 'ambiguous_in_source'?",
        a="Because the source is not silent and the platform forbids the mechanism for this configuration. Three independent sources agree the cross-period carry is absent:\n\n1. The 13-page term sheet never carries a shortfall forward; the one near-miss phrase is bounded by 'up to (and including) the Call Fixing Date', making it an intra-period pro-ration.\n2. The MurexPlaybook documents 'Memory Coupon' as a different mechanism gated on a separate Coupon Barrier, and states it is mutually exclusive with Global KO — which this deal has.\n3. The payload has no coupon-barrier field, no memory flag, and no accrual-state field.\n\nSo the block is a faithful implementation of a product this deal is documented as excluded from — 'unsupported' — rather than a document whose reading is unresolved. 'ambiguous_in_source' is a claim about a document and is the easiest claim to make by accident.",
    ),
    dict(
        id="sp-010", task_type="domain_concept",
        source="skills/fina-visual-builder/refs/block-book/00-glossary.md", location="Memory (two meanings)",
        q="In this product, the word 'Memory' has two unrelated meanings. Distinguish them.",
        a="Mechanism A — the contract's memory: a per-asset termination latch. Each share is permanently stamped the first time it trades at or above its Call Price (e.g. 110%). When every share in the basket has been stamped, the note terminates. Coupons play no part. This is what 'Memory' means in the term sheet and in the product name.\n\nMechanism B — the engine's memory (memory carry): if a period earns no coupon, the unpaid days are added to the next period's count, so a bad period is not simply lost. This is a real platform product ('Memory Coupon'), gated on its own Coupon Barrier.\n\nNeither name is wrong, but seeing memory_ko and coupon_memory in one struct reasonably suggests they are variants of each other; they are not.",
    ),
    dict(
        id="sp-011", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/block-book/blocks/coupon-strip.md", location="sect 7 Evidence",
        q="What is the correct accrual factor for a range-accrual coupon, and what do the legacy lanes compute instead?",
        a="The documented accrual factor is N1/N2, where N1 is 'the number of days the accrual condition is met' and N2 is 'the total number of days in the observation period' — stated that way in twelve platform documents.\n\nBoth legacy lanes compute (N2 - N1)/N2 instead, which is the exact complement when N1 means days-in-range. Worse, in this payload N1 is identical to fixingsDone in every row, so N1 is days *elapsed*, not days in range. Under that reading neither expression is a fraction of qualifying days: the elapsed periods book at zero and the future periods at certainty (the floor is never tested on any simulated path).",
    ),
    dict(
        id="sp-012", task_type="evidence_reasoning",
        source="skills/fina-visual-builder/refs/block-book/registry/blocks.yaml", location="range_accrual: do_not_patch_this_blindly",
        q="Why can't the range-accrual defect be fixed just by changing (N2-N1)/N2 to N1/N2?",
        a="Because the payload cannot supply the block's output. The payload contains no in-range day count at all: N1 equals fixingsDone in every row, so N1 is days elapsed. For the already-elapsed periods the correct coupon is a historical fact — the cash amount actually paid — and the record does not hold it.\n\nChanging the expression would replace one unfounded number with another and lose the ability to tell them apart. The open question is what the legacy system populates, not which line to type, which is why the block is 'blocked_on_data', not a code bug.",
    ),
    dict(
        id="sp-013", task_type="block_kernel",
        source="skills/fina-visual-builder/refs/block-book/03-where-each-block-lives.md", location="global_ko_gate and local_ko_gate",
        q="global_ko_gate is bound to resolve_ko. Why is that binding misleading?",
        a="Because resolve_ko does not do the block's job. It is a clean, correct six-line precedence switch that answers 'given that two things fired, which one counts?' — it does not answer 'did a barrier fire?', which is the block.\n\nThe block itself is inlined in every lane, five different ways. The general lesson: a declared binding is a claim about a block, and the claim is only as good as the symbol it names. The registry therefore records global_ko_gate and local_ko_gate as kernel_mapped: false with the declared symbol kept alongside, so the gap stays visible.",
    ),
    dict(
        id="sp-014", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/block-book/03-where-each-block-lives.md", location="global_ko_gate: same_fixing_verdict",
        q="What are the correct semantics of a memory knock-out latch?",
        a="Memory KO is a per-asset sticky latch: the note terminates only when each underlying has breached its barrier at least once. A Reference Asset 'becomes' a Memorised Reference Asset when the event occurs — a persistent status, not a same-day test.\n\nThe platform states the rule explicitly: 'Never overwrite a prior memory-hit or KI state with a later non-hit unless product semantics explicitly provide for reset, which the source does not.' Implement it as per-asset latches seeded from the deal's GKOLocked flags, set once, never cleared, and AND-ed across the basket.",
    ),
    dict(
        id="sp-015", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/block-book/03-where-each-block-lives.md", location="A rule no lane implements: which KO wins on a shared day",
        q="If a Local KO and a Global KO trigger on the same day, which event is recorded?",
        a="The event is treated as a Global KO. Formally, tau_G <= tau_L implies GKO. The rule is stated in several platform documents (Memory_Range_Accrual_MemRaki.md, RAKI_Enhancement.md, Products/KIKO/Reverse_KIKO.md and the Reverse KIKO payoff chapter).\n\nNote it is inert on any deal with localKO = false, which is exactly why it survived so long unchecked. The resolve_ko precedence parameter is where the rule actually lives, so it is right by accident rather than by declaration.",
    ),
    dict(
        id="sp-016", task_type="block_kernel",
        source="skills/fina-visual-builder/refs/block-book/03-where-each-block-lives.md", location="worst_of_performance",
        q="Define the worst_of_performance indicator and explain why it has no kernel.",
        a="The indicator is the minimum relative performance across the basket: min_i(close_i / initial_i). The term sheet adds a 4-decimal rounding rule (0.00005 and above rounded up) and states the tie-break is 'in our sole and absolute discretion', which is explicitly not modelable and should be recorded as a known-unimplementable term rather than a missing feature.\n\nIt has no standalone kernel: it is inlined in 19 sites across 6 lanes, two of which compute a different quantity, and within engine.cpp the two copies differ (one also sets the memory latches). Until it is extracted into a pure function, it cannot be block-tested.",
    ),
    dict(
        id="sp-017", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/block-book/registry/blocks.yaml", location="worst_of_performance: latched_assets_stay_in_the_basket",
        q="Once an underlying has been 'memorised' (latched), may it be dropped from the basket?",
        a="No. The term sheet says that if a Reference Asset becomes a Memorised Reference Asset, it remains part of the Reference Basket for the purposes of determining whether a Knock-in Event has occurred and for the Final Settlement Payout.\n\nDropping called assets is a tempting optimisation and it would break the contract. No lane does it, so the worst_of_performance -> knock_in_gate edge is correct — but for a reason nobody wrote down.",
    ),
    dict(
        id="sp-018", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/block-book/registry/blocks.yaml", location="worst_of_performance: maturbarrier_ge_kibarrier_is_load_bearing",
        q="Why is the invariant MaturBarrier >= KIBarrier load-bearing?",
        a="Because the knock-in payoff has two branches selected on WPS >= MaturBarrier. The example deal sets MaturBarrier = 0.78 and KIBarrier = 0.70, so a knocked-in deal has WPS <= 0.70 < 0.78, which makes the KI1 branch unreachable — and that is the only reason it is safe that strikeKI1 = 0.\n\nIf a booking ever set MaturBarrier below the KI barrier, the payoff would reach PR_KI1 * (WPS / Strike_KI1 - 1) with a zero denominator. Assert the invariant in a test; do not rely on it.",
    ),
    dict(
        id="sp-019", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/block-book/03-where-each-block-lives.md", location="payment_discount",
        q="What happens if a pricing request supplies an empty curve list to payment_discount?",
        a="The curve read is guarded behind a non-empty check, so with 'curves': [] the whole block is skipped. terms.rate then keeps its struct default of zero, the note is discounted at 0%, and the engine returns a plausible-looking (undiscounted) PV.\n\nThe oracle defaults to 0.0 too, so a parity test between the two lanes is green for the wrong reason. This is the strongest argument for payment_discount being a block rather than an inlined expression: a block has somewhere to put the refusal ('refuse to price without a curve').",
    ),
    dict(
        id="sp-020", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/block-book/03-where-each-block-lives.md", location="payment_discount: the lanes do not agree on the discount rate",
        q="Do the pricing lanes agree on how a cash flow is discounted?",
        a="They agree on the day count — every lane uses ACT/365 — but they disagree on the discount *rate* source:\n\n- engine.cpp interpolates the curve at the settlement date (curve_rate_at).\n- fina_risk_cpp.cpp, pricing.py and fcn_reference.py use a single flat rate.\n- daily_termsheet.py uses the first curve pillar for every date.\n\nOn a flat curve they agree; on a sloping curve they diverge by construction, because the curve interpolation was never propagated to the other lanes.",
    ),
    dict(
        id="sp-021", task_type="domain_concept",
        source="skills/fina-visual-builder/refs/block-book/02-why-there-are-lanes.md", location="What it actually costs, measured on one deal",
        q="The repository has five implementations of the same payoff. What is the actual cost?",
        a="The cost is not duplication — duplication is survivable. The cost is that a single wrong *reading of the product* gets implemented five times, and each implementation fails in its own local way, so cross-lane comparison cannot see it.\n\nA parity test asserts that the lanes agree; when a feature is implemented in one lane and not another, the difference reads as expected and the real defect — that one or both are wrong — is invisible. Compare against the term sheet, not against another lane: a cross-lane test finds disagreement, only a spec test finds error.",
    ),
    dict(
        id="sp-022", task_type="domain_concept",
        source="skills/fina-visual-builder/SKILL.md", location="Execution and tracing rules",
        q="Contrast the compiled lane and the traced lane.",
        a="Both are projections of one DAG, so neither contains arithmetic the other lacks — the arithmetic lives in the shared kernels:\n\n- Compiled lane: lower the DAG to FcnTerms and run the fused kernel across all paths. Fast, no traces; this is the production path.\n- Traced lane: interpret the same DAG topologically for one selected path and emit, per node, {path_id, step, node_id, inputs, output, gate_state, cumulative_pv}. This trace is the animation dataset.\n\nBefore a trace is shown, a parity gate requires that traced output equals compiled output for the same path under common random numbers (seed 1729).",
    ),
    dict(
        id="sp-023", task_type="payoff_graph_concept",
        source="skills/fina-visual-builder/refs/ModelAndTradeLayer.md", location="sect 2.3 The per-path lifecycle is a state machine",
        q="Why is state_transitions capped at 64 rows, and how should a UI present it?",
        a="state_transitions is the emitted per-path trace, and it is capped at 64 exemplars by add_transition so a 30,000-path run does not ship a 30,000-row array (an explicit memory guard).\n\nBecause it is a sample, a visualiser must label the replay as exemplars rather than the full population, and carry ki_probability / ko_probability beside it. Presenting a 64-row sample as if it were exhaustive is an anti-pattern.",
    ),
    dict(
        id="sp-024", task_type="payoff_graph_concept",
        source="skills/fina-visual-builder/refs/ModelAndTradeLayer.md", location="sect 1.6 The model's own output",
        q="What is selected_branch, and what values can it take?",
        a="selected_branch is the model-population read-out of a run: the dominant branch chosen by dominance (not majority). It takes six values — no_ki_maturity, ki_maturity, mixed_ki_maturity, local_ko, global_ko, mixed_ko — plus not_priced when the canonical-terms gate vetoes the request.\n\nThe run also reports ki_probability (ki_paths / paths) and ko_probability (ko_paths / paths) over the same run, so the two are mutually consistent by construction. It describes the population, not any one trade.",
    ),
    dict(
        id="sp-025", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/ModelAndTradeLayer.md", location="sect 1.5 The evidence gate is a model-layer state machine",
        q="What outcomes can compile_canonical_terms produce, and why is 'ambiguous' distinct from 'unsupported'?",
        a="The admission gate runs before any path is priced and can veto the whole product with one of four outcomes:\n\n- unsupported — bad reference_spot/strike/ki_barrier, unordered coupon periods, cube-dimension mismatch, performance_indicator not worst_of, continuous monitoring, or missing required keys.\n- unresolved — an invalid or missing fixing inside the cube.\n- ambiguous — memory_ko declared without memory_ko_mode = per_underlying_ever; the engine refuses to guess the convention.\n- verified — priced, with legs, cashflows, state_transitions and probabilities.\n\n'ambiguous' is distinct from 'unsupported' because one says 'the source does not say' and the other says 'we cannot do this'. Collapsing them into a single red error loses that distinction.",
    ),
    dict(
        id="sp-026", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/ModelAndTradeLayer.md", location="sect 1.3 The lane ladder",
        q="Can AAD be used inside the daily lifecycle lane today?",
        a="No. AAD and the daily lifecycle are currently different lanes. The AAD application prices a terminal-only cube (path_observation_mode: terminal_only_shared_cube), while the daily lane reports aad_engine: disabled.\n\nDrawing an AAD tape inside the daily lifecycle would show Greeks the lane never computed. Daily EKI/range-memory parity would first require a path-by-observation-date cube and lifecycle-state evolution. Draw the lane as it is: AAD on the terminal cube, CRN on the daily cube.",
    ),
    dict(
        id="sp-027", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/ModelAndTradeLayer.md", location="sect 1.4 Method selection per risk cell",
        q="When is a hard barrier transition allowed to be reported as pure AAD?",
        a="Never. A hard transition is never pure AAD. The method-selection rule only emits AAD_FIXED_BRANCH when the transition set is empty (transition_fraction == 0).\n\nWhen the path set crosses the barrier the tape differentiates the selected branch only, missing the probability mass across the digital, so the result must carry PATHWISE or CRN_FD alongside it and keep transition_path_fraction and smoothing_width visible. Gamma is also never AAD today: second-order tape support is absent, so gamma routes to PATHWISE/CRN_FD.",
    ),
    dict(
        id="sp-028", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/block-book/03-where-each-block-lives.md", location="Two structural consequences for the graph",
        q="Why is local_ko_gate described as a 'dead block'?",
        a="Because the palette advertises a block the emitter can never produce. local_ko_gate is listed in the schema's kind enum and in the kernel bindings, and it is drawn in the target diagram — but compile_payoff_graph never emitted a local_ko_gate node, and local_enabled is derived from that node's presence, so it was permanently false for every deal, including deals that have a local call.\n\nA visualiser that renders the palette as 'available blocks' without an emitted/declared distinction will ship a dead block. The fix is structural: generate palette, bindings, and tests from one registry.",
    ),
    dict(
        id="sp-029", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/ModelAndTradeLayer.md", location="sect 2.5 The round trip is currently broken",
        q="How should a builder highlight the trade leg on the template DAG?",
        a="Join on leg.role, not on payoff_graph_node. The kernel carries a payoff_graph_node field intended as the join key, but its strings (funding.principal_return, coupon.range_accrual, terminal.ki_put) do not match the node ids the emitter produces (notional_return, coupon_strip, down_and_in_put), so the highlight lands on nothing.\n\nThree vocabularies are in play for the same three legs (graph leg role, request leg_type, result legs[].role). Until a canonical node-id contract exists, leg.role is the only reliable join key.",
    ),
    dict(
        id="sp-030", task_type="payoff_graph_concept",
        source="skills/fina-visual-builder/refs/block-book/00-glossary.md", location="The three layers",
        q="State the one rule that follows from separating the model, template, and trade layers.",
        a="A different barrier, rate, strike, schedule, or basket member is trade data and never justifies a new model or a new template. Only a new block kind (or a new edge role) earns a new template, and only a change to calibration, numerical lane, or method labels for the whole family earns a new model.\n\nThe corollary is that the visual builder must be able to refuse an edit that fits none of the three layers.",
    ),
    dict(
        id="sp-031", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/block-book/blocks/coupon-strip.md", location="sect 5 The unit trap",
        q="Why can you not compare this system's PV directly to another PV?",
        a="Because the legs are in different units. Legs are computed per unit of notional, and only the coupon leg is multiplied by coupon_quote_scale (a ten-point price-quotation convention, default 10.0), while funding and put are not. result.pv then adds the three.\n\nSo the total is neither per-unit-notional nor a ten-point quote — it is in no single unit. Always check the unit before comparing a PV to anything, and render the unit with the number. A wrong unit still returns a plausible number, which is why no existing test distinguishes the two conventions.",
    ),
    dict(
        id="sp-032", task_type="testing_practice",
        source="skills/fina-visual-builder/refs/block-book/01-how-a-block-is-built.md", location="sect 5 Test",
        q="Why must a block test not be compiled with -ffast-math?",
        a="Because a block test asserts the specified arithmetic and compares to a tolerance. A compiler free to reassociate expressions such as 'fixed + range * fraction' would move the last bits, silently turning a check of the specification into a check of the compiler.\n\nRelated: expected values are derived by hand from the specification, never by running the kernel and pasting its output. A test that copies the implementation's output is a tautology — it passes forever and detects nothing.",
    ),
    dict(
        id="sp-033", task_type="testing_practice",
        source="skills/fina-visual-builder/refs/block-book/README.md", location="Two rules this book does not bend",
        q="What are the two rules the block book refuses to bend?",
        a="1. A test's expected values are derived by hand from the specification, never by running the code and writing down what it printed. A test that copies the implementation's output is a tautology: it passes forever and catches nothing.\n\n2. No pinned number may originate from an AI. Every expected value traces to a term sheet, a booking system, or a published reference. If a number cannot be traced to one of those, the block's evidence status stays 'unresolved' and the question is written down for a human. A fixture invented by a machine is worse than no fixture, because it acquires the authority of a test.",
    ),
    dict(
        id="sp-034", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/block-book/01-how-a-block-is-built.md", location="Lifecycle state / evidence axes",
        q="What is the 'lifecycle_state' evidence axis, and why was it added?",
        a="lifecycle_state asks: 'Is the source file internally consistent about when it is?' — and it is checked by reading the file's timestamps, not its contents.\n\nIt was added because the first two axes ran out of vocabulary. coupon_strip hit a case with correct arithmetic, traceable inputs, and a memory chain that was empty because the source snapshot was taken 17 days after the last fixing: four timestamps agreed on one date and two on another, and the engine read both. Neither 'does it match the spec' nor 'can I trace the inputs' can express a data-integrity defect that invalidates every number derived from the file while leaving the code untouched.",
    ),
    dict(
        id="sp-035", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/ModelAndTradeLayer.md", location="sect 2.2 Fields that go nowhere",
        q="Give two examples of trade fields that are currently dead-ends in the compiled graph.",
        a="Examples verifiable in the code:\n\n- GKOLocked[] / GKODate memory-call seed: not consumed. FcnTerms has no 'locked' field and PathState is zeroed at the start of every path.\n- localKO / locBarPrice: routes to barriers.local_enabled / local_barrier but is unreachable, because compile_payoff_graph never emits a local_ko_gate node (so local_enabled is always false).\n- legs[].multiplier: dropped — the kernel owns the PV sign (pv = funding + coupon - put).\n- same_day_ko_precedence: hard-coded to 'global' at lowering time.\n\nA constructor diagram that only shows live wiring teaches the wrong lesson; the dead ends are where a drag-and-drop editor would silently produce a plausible-looking wrong product.",
    ),
    dict(
        id="sp-036", task_type="domain_qa",
        source="skills/fina-visual-builder/refs/ModelAndTradeLayer.md", location="sect 2.3 The per-path lifecycle is a state machine",
        q="Is memory a cycle or a forward edge in the payoff graph?",
        a="It is a forward edge, not a cycle. memory_carry feeds forward into the next period, so the graph is unrolled in time. A diagram that draws a cycle would not correspond to a DAG and could not be traced.\n\nSimilarly, KO is a break rather than a branch: once state.terminated is set, the path skips the remaining observations, caps the last coupon period at ko_observation, pays funding at the KO date, and never evaluates the terminal put. The per-path lifecycle carries nine fields — terminated, knock_in_seen, local_ko_seen, global_ko_seen, selected_ko, ko_observation, local_memory_locks[], global_memory_locks[], unpaid_coupon.",
    ),
    # --- MurexPlaybook: platform product mechanics --------------------------------
    dict(
        id="sp-037", task_type="domain_formula",
        source="data/refs/books/MurexPlaybook/Products/Range_Accruals/Range_Accrual_Raki.md", location="Coupon formula",
        q="Write the Raki range-accrual coupon formula and define N1 and N2.",
        a="PayRate[i] = AccruRate[i] * (N1/N2) + FixCoupon[i], where AccruFactor = N1/N2.\n\nN2 is the total number of observation days in the accrual period; N1 is the number of days on which the Accrual Indicator (typically the Worst Performer, WPS) stayed between LowRange and UpRange. The coupon is the accrual rate earned in proportion to the in-range days, plus a fixed coupon adder.",
    ),
    dict(
        id="sp-038", task_type="domain_formula",
        source="data/refs/books/MurexPlaybook/Products/Range_Accruals/Range_Accrual_Raki.md", location="KO payoff",
        q="What does the note pay on a knock-out in the Raki family?",
        a="On a Local KO the rate is PayRate[KO] = PayRate[accrued] + LocKO Cpn + ReturnRatio; on a Global KO the same shape with GblKO Cpn. ReturnRatio is the notional return (ReturnRatio% x Notional) and follows the rebate timing rule (pay at knock or at maturity).\n\nIf a Local KO and a Global KO trigger on the same day, the event is treated as a Global KO.",
    ),
    dict(
        id="sp-039", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Products/Range_Accruals/RAKI_Enhancement.md", location="Enhanced Features",
        q="Can a Memory Coupon (coupon barrier carry) and a Global KO coexist in one product?",
        a="No. The Coupon Barrier / Memory Coupon feature and the Global KO feature are mutually exclusive and cannot coexist in the same product. The platform's validation forbids the combination.\n\nThis matters because 'Memory Coupon' is a real, documented product — it recovers unpaid coupons when WPS >= the Coupon Barrier in a later period — but it is a different mechanism from the contract's per-asset memory KO latch, and it requires its own Coupon Barrier field.",
    ),
    dict(
        id="sp-040", task_type="domain_formula",
        source="data/refs/books/MurexPlaybook/Products/Range_Accruals/RakiPlus_MemRakiPlus.md", location="Cash flow formula",
        q="How does the platform round a period coupon into a cash flow?",
        a="CashFlow[i] = Round(Denomination * PayRate[i], 2) * (Notional / Denomination).\n\nRounding happens per denomination and is then scaled, which is not the same as rounding the total: e.g. a raw 10,000 x 0.009642 = 96.42 rounded to 96.42 then scaled by 5 gives 482.10, whereas rounding the total changes the last cent. Denomination defaults to the deal notional, giving coarser rounding.",
    ),
    dict(
        id="sp-041", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Introduction/Common_Features.md", location="Range Accrual",
        q="For an early-terminating (KO) range-accrual deal, how is the accrual treated at the KO date?",
        a="The accrual is computed up to the KO date: N1 is truncated at the event. Whether N2 remains the full-period count or is also truncated must be consistent with the product specification — the platform source explicitly leaves this to the product spec rather than fixing one reading.\n\nFor a mid-period Global KO the coupon is paid as one settlement flow covering the pro-rated accrual plus the GKO coupon.",
    ),
    dict(
        id="sp-042", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Introduction/Common_Features.md", location="Observation types",
        q="How is continuous (American) barrier monitoring operationalised?",
        a="Via a Min-Max fixing index: the intraday maximum price is recorded for up-barriers and the intraday minimum price for down-barriers, on a fixing index. The pricing model must use the same convention as operations, otherwise the model and the booked cash flows diverge.\n\nDiscrete (daily) barriers are monitored at specific dates, commonly market close each day; periodic barriers at period ends.",
    ),
    dict(
        id="sp-043", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Introduction/Common_Features.md", location="Observation types; Flex_Development comparison operators",
        q="A fixing lands exactly on the barrier. Does it trigger?",
        a="It depends on the configured comparison operator, and the convention must be explicit. A discrete barrier must define whether a fixing exactly at the barrier triggers (inclusive, e.g. >= or <=) or not (exclusive, e.g. > or <). The same applies to range-accrual bounds via LowRangeCompare / UpRangeCompare.\n\nMissing or ambiguous operator handling is a top cause of settlement disputes, so it belongs in the specification and in functional tests at the exact level.",
    ),
    dict(
        id="sp-044", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Products/KIKO/Reverse_KIKO.md", location="Definition; payoff summary",
        q="How does a Reverse KIKO differ from a standard KIKO, and what drives its triggers?",
        a="It reverses the knock-in/knock-out logic and drives its triggers off the Best Performer (BPS) rather than the worst performer. Local KO when BPS <= LocBar Price; Global KO when BPS <= GblBar Price; KI when BPS > Knock-In Barrier.\n\nMaturity payoff: if not KO and KI, max(Floor, min(Cap, BPS - Strike)); if not KO and no KI, the Rebate. KO payoffs are LocKO Cpn + ReturnRatio (local) or GblKO Cpn + ReturnRatio (global).",
    ),
    dict(
        id="sp-045", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Products/KIKO/payoff/reverse-kiko.md", location="KI memory flag",
        q="In the Reverse KIKO specification, are the KI and KO comparisons equally inclusive?",
        a="No. Each KO condition uses B_t <= barrier, while the KI flag uses a strict inequality B_t > KI barrier. The strict inequality is material — it is not the mirror of the KO test and should not be 'tidied' to match it.\n\nKI is a memory flag: I = 1 if there exists an observation time where B_t > KI barrier.",
    ),
    dict(
        id="sp-046", task_type="domain_formula",
        source="data/refs/books/MurexPlaybook/Products/Barriers/Discrete_Barrier.md", location="Payoff",
        q="Write the DBrW settlement formula.",
        a="Payment = (Pay_Option + Pay_Rebate - ReturnRatio) * Notional.\n\nFor a knock-in: if hit, Option Payoff = Final Option Value and Rebate = Scheduled Rebate; if never hit, both are zero. For a knock-out: if hit, Option Payoff = 0 and Rebate = Scheduled Rebate; if never hit, option value applies and rebate is zero. The notional repayment follows the rebate timing (pay at knock versus on maturity), and the sign should not be normalised without a conformance fixture.",
    ),
    dict(
        id="sp-047", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Products/Other_Exotics/Digital_Option.md", location="Payoffs; DIGITAL block",
        q="How strict are a digital option's strike tests, and what governs the at-the-strike case?",
        a="A digital call pays if the final price is strictly greater than the strike; a digital put pays if it is strictly less than the strike. What happens when the final price equals the strike is governed by the DIGITAL flex block's at_strike_behavior setting (pay / no_pay), not by the payoff logic.\n\nSettlement is cash (Nominal + Cash, paying a fixed Notional) or delivery (Quantity + Delivery, delivering Notional * asset).",
    ),
    dict(
        id="sp-048", task_type="domain_formula",
        source="data/refs/books/MurexPlaybook/Products/Other_Exotics/Dispersion.md", location="Payoff; definitions",
        q="Write the dispersion coupon formula and define the average absolute spread.",
        a="PayRate[i] = min{ max{ Floor[i], PR[i] * (Average of Absolute Spreads_i - Strike[i]) }, Cap[i] }.\n\nDefinitions: Stock Performance = Share Price_Period[i] / Initial Share Price - 1; Basket Performance = the average of all individual Stock Performances; Absolute Spread = |Stock Performance - Basket Performance|; Average of Absolute Spreads = (1/N) * sum of Absolute Spread_k over the basket. The product is a bet on high dispersion (low correlation).",
    ),
    dict(
        id="sp-049", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Products/Other_Exotics/Dispersion.md", location="Worked example",
        q="Show how denomination rounding changes a dispersion cash flow.",
        a="With Nominal 3,500,000 and Denomination 10,000, a PayRate[1] of 0.66667% gives:\n\n- Denomination rounding: Round(10,000 * 0.0066667, 2) = 66.67, then x (3,500,000 / 10,000) = 66.67 * 350 = 23,334.50.\n- Rounding the total instead: 3,500,000 * 0.0066667 = 23,333.45.\n\nThe 1.05 discrepancy is why the denomination rounding rule must be specified and defaulted explicitly.",
    ),
    dict(
        id="sp-050", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Products/Other_Exotics/Outperformance.md", location="Basket; payoff chapter",
        q="In an outperformance option, which stock is the reference asset, and what is unresolved?",
        a="The first stock in the basket list (position 0) is the reference asset; the remaining positions are evaluated against it. Ranking semantics: Rank 1 = best, Rank n = worst, and 'AVG' means the basket average.\n\nThe payoff chapter deliberately declines to write the payoff as an equation: it does not say whether the Floor is applied before or after the Cap, nor whether the participation is multiplicative. Those are recorded as limitations rather than filled with a convenient default.",
    ),
    dict(
        id="sp-051", task_type="domain_qa",
        source="data/refs/financial-products/5. Product Booklet - OTC AQ DQ on Equity en/5. Product Booklet - OTC AQ DQ on Equity en.md", location="Product construction",
        q="How is an equity accumulator constructed, and what is a decumulator?",
        a="An accumulator is economically a series of long call options plus short put options: the investor agrees to buy a fixed quantity of the underlying periodically at a predetermined Forward Price. A decumulator is the inverse (a series of long puts plus short calls) and is offered on a covered basis only — the investor must already hold the earmarked shares with the bank.\n\nThe accumulator expresses a positive view; the decumulator expresses a view that the price will not rise, and it does not protect against a price decline.",
    ),
    dict(
        id="sp-052", task_type="domain_qa",
        source="data/refs/financial-products/5. Product Booklet - OTC AQ DQ on Equity en/5. Product Booklet - OTC AQ DQ on Equity en.md", location="Gearing",
        q="Explain the 2x gearing convention on accumulators and decumulators.",
        a="For an accumulator: if the settlement price is at or above the Forward Price, the investor accumulates 1x shares per day; if below, 2x shares per day. For a decumulator: if the settlement price is at or below the Forward Price, 1x shares per day; if above, 2x.\n\nThe gearing is what makes the downside leveraged: the investor is obliged to take (or sell) double the quantity precisely when the market has moved against the view.",
    ),
    dict(
        id="sp-053", task_type="domain_qa",
        source="data/refs/financial-products/5. Product Booklet - OTC AQ DQ on Equity en/5. Product Booklet - OTC AQ DQ on Equity en.md", location="Knock-out; Minimum Accumulation",
        q="What happens on knock-out of an accumulator, and what is the Minimum Accumulation?",
        a="An accumulator knocks out if the settlement price is at or above the Knock-Out Price (a decumulator if at or below); it terminates immediately, and the investor receives only the shares accumulated before the KO date.\n\nThe Minimum Accumulation feature adds Additional Knock-out Shares = Shares Per Day x remaining Accumulation Days from the KO date to the Minimum Accumulation Period End, so the investor still receives a guaranteed minimum quantity even if the KO fires immediately after the trade.",
    ),
    dict(
        id="sp-054", task_type="domain_formula",
        source="data/refs/books/MurexPlaybook/Products/Accumulators/payoff/aqdq.md", location="Normal accrued quantity",
        q="Write the normal accrued quantity formula for an AQDQ and state its settlement type.",
        a="Q_p = N^C_p * q^C_p + N^P_p * q^P_p, where N^C_p and N^P_p are the counts of Call and Put fixing days in period p and q^C_p / q^P_p are the corresponding quantities per day.\n\nAQDQ is a physical-delivery product: there is no cash settlement. On normal completion the investor acquires (AQ) or delivers (DQ) Q_p shares at the strike K_p. Any periodic coupon or fee payoff is not documented — do not manufacture one from the premium currency field.",
    ),
    dict(
        id="sp-055", task_type="domain_qa",
        source="data/refs/financial-products/6. Product Booklet - OTC AQ DQ_Target AQ DQ_Pivot Target AQ DQ on FX en/6. Product Booklet - OTC AQ DQ_Target AQ DQ_Pivot Target AQ DQ on FX en.md", location="Target Redemption",
        q="What are the three variations of Target Redemption?",
        a="1. Uncapped Target: the product terminates early when the total gain reaches the Target Redemption Limit; the final fixing's gain is not limited.\n2. Capped Target: the final fixing is adjusted so the total gain is capped at the Target Redemption Limit (a Knock-Out Strike Rate is derived).\n3. Discrete Target: the Target Redemption Limit is a specified number of fixings that resulted in a gain.",
    ),
    dict(
        id="sp-056", task_type="domain_formula",
        source="data/refs/financial-products/6. Product Booklet - OTC AQ DQ_Target AQ DQ_Pivot Target AQ DQ on FX en/6. Product Booklet - OTC AQ DQ_Target AQ DQ_Pivot Target AQ DQ on FX en.md", location="Scenario Analysis 7/8 (Capped Target)",
        q="How is the Knock-Out Strike Rate computed for a Capped Target accumulator versus decumulator?",
        a="For an accumulator, Knock-Out Strike Rate = Spot at KO - (Target Redemption Limit - Accrued before KO); for a decumulator, Spot at KO + (Target Redemption Limit - Accrued before KO).\n\nExample (accumulator): Spot 1.0430, Target 500 points, Accrued before KO 460 → 1.0430 - (500 - 460)/10,000 = 1.0390. The adjustment caps the final gain so the total does not exceed the Target.",
    ),
    dict(
        id="sp-057", task_type="domain_qa",
        source="data/refs/financial-products/6. Product Booklet - OTC AQ DQ_Target AQ DQ_Pivot Target AQ DQ on FX en/6. Product Booklet - OTC AQ DQ_Target AQ DQ_Pivot Target AQ DQ on FX en.md", location="Pivot Accumulator; Settlement; Scenario 10",
        q="What is a Pivot Accumulator and how does it settle across the strike range?",
        a="A Pivot Accumulator is a simultaneous pair of an Accumulator and a Decumulator on the same currency pair, notional and schedule, expressing a range-bound view between a Lower Strike Rate and an Upper Strike Rate.\n\nSettlement by region: below the Lower Strike → buy 2x; between the Lower Strike and the Pivot Rate → buy 1x; between the Pivot Rate and the Upper Strike → sell 1x; above the Upper Strike → sell 2x. An accumulator below the Pivot settles as an accumulator; at or above the Pivot it settles as a decumulator.",
    ),
    dict(
        id="sp-058", task_type="domain_qa",
        source="data/refs/financial-products/6. Product Booklet - OTC AQ DQ_Target AQ DQ_Pivot Target AQ DQ on FX en/6. Product Booklet - OTC AQ DQ_Target AQ DQ_Pivot Target AQ DQ on FX en.md", location="Product Summary; Margin Requirement; Example",
        q="How is the initial margin requirement computed for FX accumulators, and what is the Maximum Notional Amount?",
        a="Initial margin = Initial Margin % (10%, or another percentage the bank stipulates) of the remaining Maximum Notional Amount. The margin requirement is IMR + mark-to-market loss, and the IMR falls as the Maximum Notional Amount decreases after each settlement.\n\nThe Maximum Notional Amount = Notional per FX Transaction x Gearing x number of expirations. Example: 100,000 per transaction x 2 (gearing) x 52 weeks = 10,400,000; at 10% the day-one IMR is 1,040,000.",
    ),
    dict(
        id="sp-059", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Introduction/Architecture_Deep_Dive.md", location="The three covenants",
        q="What are the three couplings a Murex product developer must keep intact?",
        a="1. UI <-> C++: the flex field name must equal the C++ struct member name, case-sensitively. A mismatch silently breaks the data flow — the trader's input never reaches the model.\n2. Product <-> Pricing: the payoff script binds the flex header to a model group, and GMP binds the model group to a generator and its parameters. Broken, the product uses the wrong engine or the wrong paths/seed.\n3. Risk <-> Runtime: the simulation/report must select the right sensitivity configuration group, or compute is wasted or risk factors (delta-gap, correlation, FX vega) are missing.",
    ),
    dict(
        id="sp-060", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Introduction/Murex_Overview.md", location="Payoff configuration and GMP",
        q="What is GMP in Murex, and why does changing it not require a recompile?",
        a="GMP (Generic Market Parameters) is a table mapping each model group to a concrete generator plus its parameters — number of paths, random seed, regression order, cash-flow aggregation flag, and the market data it consumes.\n\nIt is stored as data, so editing GMP (for example raising paths from 10,000 to 30,000) is a data change, not a code change: no recompile. That is a key operational advantage and a key governance risk, so GMP versions belong in regression testing.",
    ),
    dict(
        id="sp-061", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Products/Range_Accruals/payoff/raki-plus-native-engine-methodology.md", location="Translation rules",
        q="What is the platform's standing rule when documentation is silent on a behavior?",
        a="Never fill the gap with a convenient default merely because it is easy to code. If a behavior is undocumented, mark it as ambiguous and require an explicit configuration or a conformance fixture rather than assuming a reading.\n\nThis is the same guardrail as the payoff-graph anti-pattern 'filling undocumented behavior with a convenient default produces a silent wrong PV'.",
    ),
    dict(
        id="sp-062", task_type="domain_qa",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="FCN additional information - frequency of call",
        q="What is the mandatory call feature of an FCN?",
        a="An FCN has a mandatory call. A Call Event occurs when the reference price of all underlyings on any Call Determination Day is at or above their respective call prices (observed periodically or daily). On a Call Event the issuer redeems the notes early, paying the initial investment in full plus any accrued/unpaid coupon, and has no further obligation.\n\nIf a call event occurs mid-period, the investor receives the monthly rate prorated over the elapsed business days (e.g. rate x 17/20) plus 100% principal.",
    ),
    dict(
        id="sp-063", task_type="domain_qa",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="Daily Call with Memory",
        q="Describe 'daily call with memory' in the FCN context.",
        a="A Memory Event occurs for an underlying when its reference price on any call determination date is at or above its call price; that underlying becomes a 'knock-out underlying', and the status persists.\n\nA Call Event is deemed to occur when all underlyings have become knock-out underlyings, occurring on the call determination day of the last memory event. This is the contract's per-asset latch mechanism (distinct from the engine's unpaid-coupon carry).",
    ),
    # --- Booklets: ELN / FCN / MPR -------------------------------------------------
    dict(
        id="sp-064", task_type="domain_formula",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="Illustrative example of an ELN",
        q="How is the actual investment amount of an ELN computed?",
        a="ELNs are issued at a discount to face value and pay no other interest, so the investor pays less than nominal and receives nominal back (or the underlying). The formula is:\n\nActual investment amount = Nominal amount / [1 + (yield * days / 365)].\n\nThe discount is the yield; the quoted tenor is short (typically 1 to 3 months).",
    ),
    dict(
        id="sp-065", task_type="domain_qa",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="Illustrative example of an ELN",
        q="Work the ELN discount example: nominal 212,800, tenor 31 days, agreed yield 8% p.a.",
        a="Actual investment = 212,800 / [1 + (0.08 * 31/365)] = 212,800 / 1.0067945 = 211,363.88.\n\nIf the closing price is at or above the strike, the investor receives the nominal 212,800 in cash (a return of 1,436.12 = full interest). If it is below the strike (e.g. 92.5% of initial), the investor receives 14,000 shares of the underlying at the strike — potential capital loss down to zero.",
    ),
    dict(
        id="sp-066", task_type="domain_qa",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="Quanto ELN",
        q="How does the Quanto ELN day count differ from the vanilla ELN, and how is physical delivery handled?",
        a="The Quanto example uses a /360 day count rather than /365: actual investment = 250,000 / [1 + (8% * 31/360)] = 248,289.56.\n\nOn physical delivery the number of shares is based on the exchange rate between the Specified Currency and the Reference Currency, determined on the valuation date at the scheduled closing time. Whole shares are delivered and the fractional remainder is settled in cash (e.g. 21,398 shares plus USD 0.30).",
    ),
    dict(
        id="sp-067", task_type="domain_qa",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="Variation - ELN with knock out",
        q="When does a KO ELN terminate early, and what does the investor receive?",
        a="A knock-out event occurs when the underlying price at any time during any scheduled trading day in the observation period is equal to or greater than the knock-out price (e.g. 110% of initial). The observation period runs from trade date through valuation date.\n\nOn KO the note terminates and the investor receives cash — the actual investment plus the full interest — on the relevant cash settlement date, even if the KO happened intraday.",
    ),
    dict(
        id="sp-068", task_type="domain_formula",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="Variation - Minimum Principal Redemption (MPR)",
        q="Write the MPR redemption formula and state its settlement convention.",
        a="Redemption = MAX{ Nominal * (closing price of the Least Performing Equity / Strike Price of the LPE), MPR% * Nominal investment amount }.\n\nMPR caps the downside at maturity and takes effect only if the LPE reference price is below the strike. Structures with MPR are cash-settled (no physical settlement), and adding MPR reduces the potential yield (lower coupon or higher strike).",
    ),
    dict(
        id="sp-069", task_type="domain_qa",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="MPR scenario 2",
        q="Compute the MPR redemption for nominal 212,800, strike 15.20, closing 12.00, MPR 90%.",
        a="Redemption = MAX{ 212,800 * 12.00 / 15.20, 0.90 * 212,800 } = MAX{ 168,000, 191,520 } = 191,520.\n\nBecause the performance-based value (168,000) is below the MPR floor (191,520), the redemption is floored at 191,520 (90% of nominal), turning a would-be 21% loss into a 10% loss.",
    ),
    dict(
        id="sp-070", task_type="domain_qa",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="FCN scenario 3 (physical settlement)",
        q="How are shares and fractional cash computed on FCN physical settlement?",
        a="Number of shares = Nominal / Strike of the Least Performing Equity. Example: 200,000 / 1.805 = 110,803.324099, so 110,803 whole shares are delivered, and the fractional part is settled in cash: 0.324099 * 1.75 (the LPE reference price) = 0.57.\n\nPhysical settlement is triggered when the LPE reference price is below the strike on the valuation date (and, with a KI feature, when KI has occurred).",
    ),
    dict(
        id="sp-071", task_type="domain_definition",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="Glossary of terms",
        q="Define the Least Performing Equity (LPE).",
        a="The LPE is the underlying in the basket with the lowest performance, where each underlying's performance = (Reference Price / Initial Price) * 100. The reference price is the price per share quoted on the exchange at the valuation time.\n\nIn a worst-of basket the LPE drives both the exercise decision and the physical-delivery quantity.",
    ),
    dict(
        id="sp-072", task_type="domain_qa",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="Glossary of terms",
        q="Distinguish the Knock-In Level and the Knock-Out Level in an FCN.",
        a="A Knock-In event occurs if the reference price of the reference asset (or the worst performer in a basket) at any time during the KI observation period — the trade date to the valuation date, both inclusive — is below the KI price.\n\nA Knock-Out (call) event occurs if the closing price of the reference asset (or worst performer) on the KO observation date(s) is at or above the KO price; the product then terminates early and the issuer pays the initial investment plus coupon.",
    ),
    dict(
        id="sp-073", task_type="domain_formula",
        source="data/refs/financial-products/SP_Product_Illustration/SP_Product_Illustration.md", location="FCN cash settlement in certain circumstances",
        q="How is an FCN cash-settled when the underlying cannot be delivered (e.g. an index)?",
        a="The issuer pays cash = Nominal * (least performing index performance / strike level). Example: Nominal 200,000, strike at 95% of initial, and the least performing index finishes at 80% of initial → 200,000 * (80% / 95%) = 168,421.05.\n\nThis applies when the underlyings are indices or futures and physical delivery is not possible.",
    ),
    dict(
        id="sp-074", task_type="domain_qa",
        source="data/refs/financial-products/1. Product Booklet - Derivatives and Structured Products en/1. Product Booklet - Derivatives and Structured Products en.md", location="Best Of / Worst Of; FCN yield determinants",
        q="Why does a 'worst-of' basket make correlation a first-order input?",
        a="A worst-of product's payoff is measured against the worst-performing underlying, so its price depends on the joint distribution of the underlyings — i.e. their correlation — not just on the individual volatilities. Two underlyings that move together are much more likely to both be high (or both low) than two independent ones.\n\nIn an FCN, lower correlation between the underlyings in a basket raises the yield, because the worst performer is less likely to be dragged down by diversification.",
    ),
    dict(
        id="sp-075", task_type="domain_definition",
        source="data/refs/financial-products/1. Product Booklet - Derivatives and Structured Products en/1. Product Booklet - Derivatives and Structured Products en.md", location="Protection Level / Protection Barrier / Put Strike",
        q="What is a protection level in a structured note?",
        a="The Protection Level (also Protection Barrier or Put Strike) is the strike of the sold put embedded in the note. Downside exposure to the underlying begins only if the final level closes below the protection level.\n\nAbove it, the investor typically receives par plus the coupon/participation; below it, the investor takes the downside (often via physical delivery at the strike).",
    ),
    dict(
        id="sp-076", task_type="domain_qa",
        source="data/refs/financial-products/1. Product Booklet - Derivatives and Structured Products en/1. Product Booklet - Derivatives and Structured Products en.md", location="Barrier Options",
        q="What observation styles can a barrier option use, and how does a barrier affect the premium?",
        a="Observation can be American (monitored daily, or intraday/continuous), Bermudan (monitored on scheduled dates), or European (monitored on the final date only).\n\nA barrier option is always cheaper than the otherwise-similar option without the barrier, because the barrier removes some of the option's value. Up-and-out / down-and-out knock out; up-and-in / down-and-in knock in.",
    ),
    dict(
        id="sp-077", task_type="domain_qa",
        source="data/refs/financial-products/1. Product Booklet - Derivatives and Structured Products en/1. Product Booklet - Derivatives and Structured Products en.md", location="Twin Win Note / Bull Bear Note",
        q="Explain the payoff of a Twin Win / Bull Bear note.",
        a="If no knock-in has occurred, the investor receives 100% plus a participation rate times the *absolute* performance of the underlying (so a moderate fall still pays). If a knock-in has occurred, the investor receives 100% plus the signed performance (so a fall loses money).\n\nExample: -15% with no KI → 100% + 100% * |−15%| = 115%; with KI having occurred → 100% + (−15%) = 85%.",
    ),
    dict(
        id="sp-078", task_type="domain_qa",
        source="data/refs/financial-products/1. Product Booklet - Derivatives and Structured Products en/1. Product Booklet - Derivatives and Structured Products en.md", location="Airbag Outperformer Note",
        q="Work the Airbag Outperformer example: participation 200%, protection level 90%.",
        a="At +20% the investor receives 100% + (200% * 20%) = 140%. A fall down to −10% still returns 100% (the airbag buffers small falls). Below the 90% protection level — e.g. 85% — exposure turns on and the investor takes physical delivery at the 90% strike, so the capital loss begins from the 90% level, not the final level.",
    ),
    dict(
        id="sp-079", task_type="domain_qa",
        source="data/refs/financial-products/1. Product Booklet - Derivatives and Structured Products en/1. Product Booklet - Derivatives and Structured Products en.md", location="Principal Redemption Note",
        q="Work the Principal Redemption Note example: participation 150%, minimum redemption 90%.",
        a="At +30%: 100% + (150% * 30%) = 145%. At −5%: 100% + (150% * −5%) = 95%. At −20%: max(90%, 100% − 30%) = max(90%, 70%) = 90%.\n\nThe 90% Minimum Redemption floors the redemption, converting a raw 30% loss into a 10% loss (the 1.5x participation amplifies the fall, and the floor then caps it).",
    ),
    dict(
        id="sp-080", task_type="domain_qa",
        source="data/refs/financial-products/1. Product Booklet - Derivatives and Structured Products en/1. Product Booklet - Derivatives and Structured Products en.md", location="Daily Range Accrual Note",
        q="Describe the Daily Range Accrual Note example.",
        a="An 18-month USD note on the worst-of Apple/Google with: accrued coupon 12% p.a. (1% flat per month), coupon barrier 80% observed daily, auto-call level 100% observed monthly, and a protection barrier at 80% at maturity.\n\nMonth 1's coupon is guaranteed at 1.00% flat. Thereafter the accrued coupon = 12.00% p.a. * (n/N), where n is the number of days the closing price was at or above the 80% coupon barrier and N is the total number of days in the period.",
    ),
    dict(
        id="sp-081", task_type="domain_definition",
        source="data/refs/financial-products/1. Product Booklet - Derivatives and Structured Products en/1. Product Booklet - Derivatives and Structured Products en.md", location="Participation Rate; Outperformer Note",
        q="What does a participation rate do, and how is it applied?",
        a="A participation rate multiplies the underlying's performance in the payoff; it is usually applied to upside only or downside only (not both). Example: a 150% participation on a +20% move gives 100% + (150% * 20%) = 130%.\n\nIt is a trade-layer parameter (participation rate PR), not a new structure, and it is often capped and/or floored.",
    ),
    # --- Fixed income / FX / rates / commodities / funds ---------------------------
    dict(
        id="sp-082", task_type="domain_qa",
        source="data/refs/financial-products/2. Product Booklet - Fixed Income Products en/2. Product Booklet - Fixed Income Products en.md", location="Classification of Bond",
        q="Classify bonds by coupon type.",
        a="Three coupon types: fixed rate (the coupon is unchanged over the life of the bond); floating rate (the coupon resets periodically, typically linked to a reference such as HIBOR or LIBOR); and zero-coupon (no coupon at all — the bond is sold at a discount to par and the investor's return is the price converging to par at maturity).",
    ),
    dict(
        id="sp-083", task_type="domain_qa",
        source="data/refs/financial-products/2. Product Booklet - Fixed Income Products en/2. Product Booklet - Fixed Income Products en.md", location="Types of Bond (convertible, callable)",
        q="How do you decompose the value of a convertible bond versus a callable bond?",
        a="Convertible = straight bond value + the value of the embedded option to convert into equity (usually at the bondholder's discretion), so the holder is long the option and the price floors near the straight-bond value.\n\nCallable = straight bond value − the value of the embedded option (the issuer may redeem early), so the holder is long the bond and short the option. Callability exposes the holder to reinvestment risk.",
    ),
    dict(
        id="sp-084", task_type="domain_definition",
        source="data/refs/financial-products/2. Product Booklet - Fixed Income Products en/2. Product Booklet - Fixed Income Products en.md", location="Types of Bond (high-yield, distressed)",
        q="What rating thresholds define high-yield, and how does the bank define a distressed bond?",
        a="High-yield (below investment grade) means rated BB+ and below by S&P/Fitch, or Ba1 and below by Moody's.\n\n'Distressed' has no official definition; the bank classifies a bond trading at a credit spread of 1,000 bps or more above the benchmark reference yield as distressed. Distressed bonds are highly volatile and carry elevated default risk.",
    ),
    dict(
        id="sp-085", task_type="domain_qa",
        source="data/refs/financial-products/2. Product Booklet - Fixed Income Products en/2. Product Booklet - Fixed Income Products en.md", location="Types of Bond (contingent capital)",
        q="What triggers a contingent capital (CoCo) security, and why is the risk asymmetric?",
        a="Contingent Capital / bail-in securities are typically Tier 1 or Tier 2 subordinated bank capital. The trigger that converts them to equity or writes them down can be mechanical (a numerical capital-ratio threshold) or discretionary (a supervisory judgment).\n\nThe risk is asymmetric because loss absorption happens before senior creditors and can occur while the bank is a going concern — so a conversion or write-down can destroy principal even though the bank has not failed.",
    ),
    dict(
        id="sp-086", task_type="domain_definition",
        source="data/refs/financial-products/3. Product Booklet - FX Forward en/3. Product Booklet - FX Forward en.md", location="Product Features",
        q="What is an FX forward, and what is an NDF?",
        a="An FX forward is an agreement to trade an underlying currency pair at a future date at an exchange rate fixed on the trade day; the investor can participate in both upward and downward moves.\n\nAn NDF (non-deliverable forward) has no delivery of the currency pair at maturity — instead it is net cash-settled based on the final fixing of the underlying currency pair, which is useful for restricted currencies.",
    ),
    dict(
        id="sp-087", task_type="domain_qa",
        source="data/refs/financial-products/3. Product Booklet - FX Forward en/3. Product Booklet - FX Forward en.md", location="Margin Requirement and Related Risks",
        q="How does margin work on a margined FX forward?",
        a="The investor deposits initial margin with the bank and must post additional collateral for any mark-to-market loss during the life of the forward. The bank determines the daily MTM valuation at its sole discretion and may revise the initial margin at any time, including demanding 100%.\n\nA margin call is triggered if collateral is insufficient to cover initial margin plus MTM loss. Failure to meet a margin call lets the bank close out transactions without consent at significant loss, and may realise the collateral. The investor bears delivery risk on the settlement date too.",
    ),
    dict(
        id="sp-088", task_type="domain_formula",
        source="data/refs/financial-products/9. Product Booklet - Swap and Interest Rate Options en/9. Product Booklet - Swap and Interest Rate Options en.md", location="Interest Rate Swap",
        q="Write the interest payment formula for an interest rate swap.",
        a="Interest exchanged each payment period = Notional * underlying reference rate (per annum) * (number of days in that payment period, per the day-count fraction defined in the term sheet).\n\nA vanilla IRS has two legs and is commonly quoted fixed-for-floating (also fixed-for-fixed and floating-for-floating). Legs may be in the same or different currencies; the tenor can run up to 15 years when used for hedging (capped at 10 years otherwise).",
    ),
    dict(
        id="sp-089", task_type="domain_qa",
        source="data/refs/financial-products/9. Product Booklet - Swap and Interest Rate Options en/9. Product Booklet - Swap and Interest Rate Options en.md", location="Swaption",
        q="What are the swaption exercise styles, and who bears what risk?",
        a="European = a single one-time exercise; Bermudan = multiple pre-determined exercise dates; American = multiple exercise dates at any time during the option's life.\n\nA payer swaption gives the right to pay a fixed rate (protection against rising rates); a receiver swaption gives the right to receive a fixed rate (protection against falling rates). The buyer pays the premium even if it is not exercised; the seller may have to enter an off-market swap costing more than the premium received, so the seller's potential loss is unlimited.",
    ),
    dict(
        id="sp-090", task_type="domain_qa",
        source="data/refs/financial-products/9. Product Booklet - Swap and Interest Rate Options en/9. Product Booklet - Swap and Interest Rate Options en.md", location="Interest Rate Cap / Floor / Knock Out Cap",
        q="Describe an interest rate cap, a floor, and a knock-out cap.",
        a="A cap is a series of European call options (caplets), one per period: the buyer receives a payment at the end of each period in which the reference rate exceeds the agreed strike. A floor is the mirrored series, paying when the reference rate falls below the strike. Both are bought for an upfront premium.\n\nA knock-out cap is a normal cap that extinguishes once the rate rises above an upper barrier: the buyer is protected between the lower and upper barriers but unprotected above the upper barrier, so the premium is lower than a plain cap.",
    ),
    dict(
        id="sp-091", task_type="domain_qa",
        source="data/refs/financial-products/9. Product Booklet - Swap and Interest Rate Options en/9. Product Booklet - Swap and Interest Rate Options en.md", location="Unlimited Downside Risk",
        q="Which interest-rate positions carry potentially unlimited downside?",
        a="A short (sold) position in an interest rate cap or floor has potentially unlimited downside: the seller receives a premium but must pay out as rates move, with no cap on the amount. Selling a swaption is similar — if exercised, the seller must enter an off-market swap that can cost far more than the premium received.\n\nCross-currency swaps add FX rate risk, whose loss may exceed the maximum aggregate notional and may be unlimited.",
    ),
    dict(
        id="sp-092", task_type="domain_definition",
        source="data/refs/financial-products/9. Product Booklet - Swap and Interest Rate Options en/9. Product Booklet - Swap and Interest Rate Options en.md", location="Total Return Swap",
        q="What is a total return swap?",
        a="In a total return swap the receiver pays a set rate (fixed or variable) in exchange for the total return of a reference asset — its income (coupons or dividends) plus its capital gains. The receiver becomes synthetically long the reference asset without owning it.\n\nUses include gaining exposure without an upfront cash outlay, accessing restricted markets, and balance-sheet management.",
    ),
    dict(
        id="sp-093", task_type="domain_qa",
        source="data/refs/financial-products/9. Product Booklet - Swap and Interest Rate Options en/9. Product Booklet - Swap and Interest Rate Options en.md", location="Credit Default Swap",
        q="How does a credit default swap work, and how does its spread relate to credit quality?",
        a="The protection seller pays the buyer an agreed amount if a credit event occurs; the buyer pays the seller a spread (a one-off at inception and/or periodically). The reference entity can be a single asset or a basket (with a first-to-default or multiple-asset trigger).\n\nPricing depends on the default probability: the CDS spread is low when default probability is low and rises as credit quality worsens. Physical settlement lets the buyer deliver the defaulted asset to the seller.",
    ),
    dict(
        id="sp-094", task_type="domain_definition",
        source="data/refs/financial-products/8. Product Booklet - Paper Gold and Silver en/8. Product Booklet - Paper Gold and Silver en.md", location="Product Features; Spot Transactions",
        q="What are Paper Gold and Paper Silver, and how do they settle?",
        a="Paper Gold is a credit balance equal to 1 fine troy ounce of gold (995 fine London Good Delivery); Paper Silver is 1 troy ounce (999 fine). The balances are the bank's indebtedness to the investor and carry no proprietary interest in any bar — no physical deposit or withdrawal is possible.\n\nSpot transactions settle T+2 by account debit/credit. Minimum sizes are 100 oz for gold (increment 0.01 oz) and 10,000 oz for silver (increment 1 oz).",
    ),
    dict(
        id="sp-095", task_type="domain_qa",
        source="data/refs/financial-products/8. Product Booklet - Paper Gold and Silver en/8. Product Booklet - Paper Gold and Silver en.md", location="Forward Transactions; Margin Requirement",
        q="What margin applies to Paper Gold/Silver forwards, and what is the close-out deadline?",
        a="The indicative initial margin is 12% for gold forwards and 25% for silver forwards (or another percentage the bank stipulates), with additional collateral required for mark-to-market loss.\n\nTo close an initial forward contract the investor must enter offsetting contracts by 5pm Hong Kong time on the third FX Gold/Silver Trading Day before the settlement date, so that the net quantity delivered is zero. If not, the bank may (but need not) enter offsetting contracts at its discretion, at the investor's cost.",
    ),
    dict(
        id="sp-096", task_type="domain_definition",
        source="data/refs/financial-products/8. Product Booklet - Paper Gold and Silver en/8. Product Booklet - Paper Gold and Silver en.md", location="Structured Notes on Paper Gold and Silver",
        q="What is a Gold/Silver Linked Note (GLN/SLN)?",
        a="A GLN/SLN is a debt instrument embedded with a short put option on the commodity, which provides an enhanced fixed yield in exchange for downside risk. It is traded on a discounted-principal basis: the investor pays a discounted price and receives the full amount at maturity, the discount being the yield.\n\nOn the valuation date, if spot is at or above the strike the investor receives principal plus yield in cash; if spot is below the strike the investor receives the principal plus yield via an account credit in the underlying gold/silver (no physical settlement).",
    ),
    dict(
        id="sp-097", task_type="domain_qa",
        source="data/refs/financial-products/8. Product Booklet - Paper Gold and Silver en/8. Product Booklet - Paper Gold and Silver en.md", location="GLN product example",
        q="Work the GLN example: 1,000 oz, 1 month, strike 1,200, market 1,250, agreed yield 8% p.a.",
        a="Notional = 1,200,000. The settlement amount paid is 1,192,052.98, giving a discount of 7,947.02. The period yield = discount / settlement = 0.67%, annualised as 0.67% * 360/30 = 8.00% p.a.\n\nIf the closing price is at or above 1,200 the investor receives 1,200,000 in cash; if below, the investor receives 1,000 oz at an effective cost of 1,192.05/oz.",
    ),
    dict(
        id="sp-098", task_type="domain_formula",
        source="data/refs/financial-products/4. Product Booklet - Investment Funds en/4. Product Booklet - Investment Funds en.md", location="Definition of investment funds",
        q="How is a fund's net asset value per unit computed?",
        a="NAV per unit = (total assets - total liabilities) / number of units issued.\n\nA fund pools money from investors and is managed by a fund manager; investors hold units/shares, and each unit represents a part of the fund's NAV. Forms include investment funds, investment trusts, mutual funds and unit trusts.",
    ),
    dict(
        id="sp-099", task_type="domain_qa",
        source="data/refs/financial-products/4. Product Booklet - Investment Funds en/4. Product Booklet - Investment Funds en.md", location="Fees and charges",
        q="What fees does a fund typically charge, and how does the redemption fee behave?",
        a="The management fee is the cost of professional management, typically a percentage of assets under management. The subscription fee is an upfront charge covering account setup and distribution. The redemption fee discourages short-term trading and often follows a sliding scale that falls to 0% after 3 to 5 years, with the fee credited to the fund's assets for the benefit of remaining investors.\n\nOther fees include performance fees/carried interest, trustee/custodian/administrator fees, and out-of-pocket fund expenses.",
    ),
    dict(
        id="sp-100", task_type="domain_qa",
        source="data/refs/financial-products/4. Product Booklet - Investment Funds en/4. Product Booklet - Investment Funds en.md", location="Derivative Risk",
        q="Why do funds that use derivatives carry leveraged exposure?",
        a="Because initial margin on derivatives is generally small relative to the contract's notional size, so the fund's economic exposure is levered. Small market moves therefore have a larger impact on the fund, and leveraged positions increase the fund's volatility.\n\nDerivative use also brings counterparty/settlement/liquidity risk, imperfect tracking, higher transaction costs, delivery-failure risk, and illiquid OTC markets.",
    ),
    # --- Held-out validation items (distinct topics) -------------------------------
    dict(
        id="sp-101", task_type="domain_qa",
        source="data/refs/financial-products/EquityFlowProductMatrix.md", location="Big Bear Market",
        q="For a big bear market, which products suit a conservative versus an aggressive risk appetite?",
        a="Conservative: inverse ETFs (leveraged inverse exposure for hedging or profiting from a downturn), FCN with deep strike or knock-in (fixed coupons with downside protection), and zero coupon notes (capital preservation with no coupon).\n\nAggressive: leveraged participation notes (amplified downside exposure) and KOELN (knock-out equity linked notes: higher yield while the market stays above a level, with early-termination risk if it falls sharply).",
    ),
    dict(
        id="sp-102", task_type="domain_qa",
        source="data/refs/financial-products/EquityFlowProductMatrix.md", location="Sideways Market",
        q="For a sideways market, which products suit conservative versus aggressive investors?",
        a="Conservative: FCN (steady income with downside protection) and rainbow notes (diversification across multiple underlyings).\n\nAggressive: recovery coupon (bonus enhanced) notes (enhanced coupons if the market stays in a range) and ELN (higher yield if the market is stable, with capital-loss risk on a decline).",
    ),
    dict(
        id="sp-103", task_type="domain_qa",
        source="data/refs/financial-products/EquityFlowProductMatrix.md", location="Additional Considerations",
        q="What is the role of rainbow notes and MPR products in a portfolio?",
        a="Both are used for diversification. Rainbow notes diversify risk across multiple underlyings, reducing the impact of a single asset's performance; MPR (Minimum Principal Redemption) structured products also spread risk across underlyings while providing partial capital protection, with potential upside if the market recovers.",
    ),
    dict(
        id="sp-104", task_type="domain_qa",
        source="data/refs/financial-products/7. Product Booklet - OTC Bullion AQ DQ, Target AQ DQ and Pivot AQ en/7. Product Booklet - OTC Bullion AQ DQ, Target AQ DQ and Pivot AQ en.md", location="Product Summary; Example 4 (Mark-to-Market / Credit Risk)",
        q="Can a margined bullion decumulator lose more than the Maximum Notional Amount?",
        a="Yes. For decumulators, target decumulators and pivot accumulators, the potential loss is unlimited and may exceed the Maximum Notional Amount if the bullion price rises more than 100% of the strike price. The worked target-decumulator example shows a loss of -130.77% at a spot of USD 3,000 against a strike of USD 1,300.\n\nFor accumulators and target accumulators the maximum loss is the Maximum Notional Amount. In all cases a margined loss is not limited to the margin posted.",
    ),
    dict(
        id="sp-105", task_type="domain_definition",
        source="data/refs/financial-products/1. Product Booklet - Derivatives and Structured Products en/1. Product Booklet - Derivatives and Structured Products en.md", location="Quanto",
        q="What is a quanto product?",
        a="A quanto is a product denominated in one currency but linked to an underlyer in another, with no FX exposure in the payout: the FX rate between the two currencies is fixed at inception (or per the terms), so the investor is insulated from moves in that rate.\n\nIn the platform's FX rule taxonomy, quanto is the (S * 1 - K) rule with a fixed FX rate, as distinct from the basic (S - K) * X rule and the composite (S * X - K) rule.",
    ),
    dict(
        id="sp-106", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Introduction/Common_Features.md", location="Settlement and FX rules",
        q="What are the three FX rules, and which one is generally unsupported?",
        a="Basic: (S - K) * X, used when the basket currency equals the premium currency or FX applies. Quanto: (S * 1 - K), where the basket currency differs from the premium currency and the FX rate is fixed at inception, removing investor currency risk. Composite: (S * X - K), using a variable spot FX rate.\n\nComposite is generally unsupported and rejected at validation. Quanto requires FX vol surfaces and equity-FX correlations, which is a common setup gap.",
    ),
    dict(
        id="sp-107", task_type="domain_definition",
        source="data/refs/financial-products/1. Product Booklet - Derivatives and Structured Products en/1. Product Booklet - Derivatives and Structured Products en.md", location="Performance of the Underlying Asset",
        q="How is the performance of an underlying defined, and why use relative rather than absolute prices?",
        a="Performance = (final level - initial level) / initial level. The barrier-trigger form uses the ratio S_k(t) / S_k(0).\n\nPerformance is always measured relative to the reference (initial) price, never in absolute terms, so a share at 100 moving to 70 and a share at 200 moving to 140 have both performed at 0.70 — which is what makes a worst-of basket comparable across different price levels.",
    ),
    dict(
        id="sp-108", task_type="domain_definition",
        source="data/refs/books/MurexPlaybook/Introduction/Murex_Overview.md", location="Trade representation; FLEX framework",
        q="How is an equity derivative trade classified in Murex, and what does FLEX mean?",
        a="The standard classification is Family EQD (Equity Derivative), Group OPT (Option), Type FLEX (Flexible).\n\nFLEX instructs the platform to use a customisable framework rather than a rigid pre-defined instrument: it displays the product's flex header, whose flex blocks group the economic parameters (e.g. KIKOSTRUCT/KIKOSELECT, RGACCLKO, GLOBALKO, KNOCKIN), and a payoff script binds that header to a pricing model group.",
    ),
    dict(
        id="sp-109", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Introduction/Murex_Overview.md", location="Dummy basket methodology",
        q="What is a dummy basket, and why is it used?",
        a="A dummy basket is a generic, market-oriented instrument (for example FLEX_USD, or a regional basket) that carries no constituents of its own; it acts as a market and currency anchor. The real underlyings, weights and reference prices live in the trade's flex blocks, typically KIKOSTRUCT.\n\nThis avoids creating a pre-defined basket for every stock combination, which grows combinatorially. A single-underlying product that uses the pattern is booked as a basket of size 1.",
    ),
    dict(
        id="sp-110", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Products/Range_Accruals/RakiPlus_MemRakiPlus.md", location="Constraints",
        q="What are the platform's basket-size limits for these structured products?",
        a="The Raki family and DBrW support a maximum of 6 underlying stocks; Dispersion supports up to 19. For the products that require a basket, 'single equity: not supported' makes two the minimum.\n\nUnderlying names and per-underlying reference prices are carried on KIKOSELECT, so the basket size is a declared width rather than a fixed implementation limit.",
    ),
    dict(
        id="sp-111", task_type="domain_qa",
        source="data/refs/books/MurexPlaybook/Introduction/Common_Features.md", location="Product constraint patterns",
        q="What are the indicative maximum tenors across these product families?",
        a="Indicative maximum tenors: 2 years for Raki, MemRaki, AQDQ, Digital Option and Reverse KIKO; 25 months for RakiPlus, DBrW and Double No-Touch Range Accrual; and 37 months for Dispersion.\n\nThese are product-page/platform constraints and must be enforced by pre-trade validation rules, since the pricing model cannot express a product outside its supported configuration.",
    ),
    dict(
        id="sp-112", task_type="domain_definition",
        source="data/refs/books/MurexPlaybook/Introduction/Glossary.md", location="Risk terminology; Risk_Management",
        q="Define the standard greeks for these products.",
        a="Delta = the change in premium for a 1% move in the underlying spot. Gamma = the change in delta per move in spot (the second derivative). Vega = the change in premium per 1 volatility point. Theta = the change in value as one day passes. Rho / IR PV01 = the change per 10 bps (Rho) or per 1 basis point (PV01) in rates.\n\nIn practice these are almost always computed by bump-and-revalue in a simulation server, not hand-rolled inside the pricing model.",
    ),
]

# Held out of training to give a small, distinct validation set.
VALID_IDS = {
    "sp-003", "sp-012", "sp-019", "sp-027", "sp-038", "sp-048",
    "sp-055", "sp-066", "sp-075", "sp-088", "sp-097", "sp-105",
}


def to_record(item: dict) -> dict:
    """Convert a curated item into a ChatML record with provenance metadata."""
    answer = item["a"].strip()
    citation = f"Source: {item['source']} ({item['location']})."
    if citation.lower() not in answer.lower():
        answer = f"{answer}\n\n{citation}"
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": item["q"].strip()},
            {"role": "assistant", "content": answer},
        ],
        "source": item["source"],
        "location": item["location"],
        "chunk_id": item["id"],
        "teacher_model": "human-curated+agent-research",
        "task_type": item["task_type"],
        "review_status": "draft-unreviewed",
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-name", default="sp-sft-train.jsonl")
    parser.add_argument("--valid-name", default="sp-sft-valid.jsonl")
    args = parser.parse_args()

    ids = [item["id"] for item in ITEMS]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate chunk ids in ITEMS")
    train = [to_record(i) for i in ITEMS if i["id"] not in VALID_IDS]
    valid = [to_record(i) for i in ITEMS if i["id"] in VALID_IDS]

    train_path = args.output_dir / args.train_name
    valid_path = args.output_dir / args.valid_name
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)

    print(json.dumps({
        "train_examples": len(train),
        "valid_examples": len(valid),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "sources": len({item["source"] for item in ITEMS}),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
