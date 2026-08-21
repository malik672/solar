//! EVM bytecode generation from MIR.
//!
//! This module generates EVM bytecode from MIR using:
//! - Liveness analysis to know when values die
//! - Phi elimination to convert SSA to parallel copies
//! - Stack scheduling to generate DUP/SWAP sequences
//! - EVM IR optimization, relocation, and byte encoding

use super::{
    EVM_WORD_BYTES,
    assembler::{
        ArtifactKind, Assembler, DeferredAlloc, DeferredConst, ImmutableRef, Label,
        PreparedAssembly,
    },
    ir,
    layout::{RelayoutAddress, preserves_push_width},
    op,
    stack::{
        MAX_STACK_ACCESS, MAX_STACK_DEPTH, OperandCostModel, OperandPlan, ScheduleCost,
        ScheduledOp, SpillSlot, StackModel, StackOp, StackScheduler, TargetSlot,
    },
};
use crate::{
    analysis::{
        CallGraphInfo, CfgInfo, CopyDest, CopySource, Liveness, Loop, LoopAnalyzer, ParallelCopy,
        PhiEliminator,
    },
    immutable::{
        immutable_push_type_size, immutable_staging_addr, immutable_staging_base,
        immutable_staging_end,
    },
    memory::EvmMemoryLayout,
    mir::{
        ArgIdx, BlockId, Function, FunctionId, ImmutableEncoding, ImmutableId, InstId, InstKind,
        MirPhase, Module, Terminator, ValueId,
    },
    pass::run_pipeline,
    transform::evm_inst_schedule::{
        EvmInstSchedule, PhysicalScheduleCandidate, bounded_physical_schedule_candidates,
        physical_schedule_candidates,
    },
};
use alloy_primitives::U256;
use smallvec::SmallVec;
use solar_config::OptimizationMode;
use solar_data_structures::{
    bit_set::{DenseBitSet, GrowableBitSet},
    index::{IndexVec, index_vec},
    map::{FxHashMap, FxHashSet},
};
use solar_interface::sym;
use solar_sema::Gcx;
use std::time::Instant;

mod switch;

use self::switch::MAX_GAS_CODE_GROWTH;

const STACK_PHI_LAYOUT_LIMIT: usize = 8;
const GLOBAL_STACK_LAYOUT_LIMIT: usize = 8;
const GLOBAL_STACK_MAX_ARGS: usize = 3;
const GLOBAL_STACK_MIN_BLOCKS: usize = 8;
const GLOBAL_STACK_MIN_ARG_USES: usize = 6;
const GLOBAL_STACK_DENSE_AMORTIZATION_BLOCKS: usize = 16;
const STACK_ARG_ROTATION_LIMIT: usize = 16;
const PHYSICAL_REPLAY_TARGET: &str = "solar_codegen::evm_inst_schedule::physical_replay";

#[derive(Default)]
struct GeneratedCode {
    bytecode: Vec<u8>,
    evm_ir: Option<ir::Module>,
}

struct PreparedDeploymentPrefix {
    assembly: PreparedAssembly,
    constructor_arg_offset: Option<DeferredConst>,
    runtime_offset: DeferredConst,
}

#[derive(Clone, Debug)]
struct PhysicalReplayStart {
    instruction_lengths: Vec<usize>,
    terminators: Vec<bool>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PhysicalReplayScore {
    instructions: usize,
    terminators: usize,
    estimated_bytes: usize,
    estimated_static_gas: usize,
    encoded_pushes: usize,
    dups: usize,
    swaps: usize,
    pops: usize,
    mloads: usize,
    mstores: usize,
    spill_bytes: u64,
    peak_stack: usize,
}

impl PhysicalReplayStart {
    fn capture(asm: &Assembler<'_>) -> Self {
        Self {
            instruction_lengths: asm
                .program
                .blocks
                .iter()
                .map(|block| block.instructions.len())
                .collect(),
            terminators: asm
                .program
                .blocks
                .iter()
                .map(|block| block.terminator.is_some())
                .collect(),
        }
    }

    fn score(&self, codegen: &EvmCodegen<'_>) -> PhysicalReplayScore {
        let mut score = PhysicalReplayScore {
            spill_bytes: u64::from(codegen.scheduler.spills.spill_area_size()),
            peak_stack: codegen.scheduler.stack.max_depth(),
            ..PhysicalReplayScore::default()
        };

        for (index, block) in codegen.asm.program.blocks.iter().enumerate() {
            let start = self.instruction_lengths.get(index).copied().unwrap_or_default();
            for inst in &block.instructions[start..] {
                score.instructions += 1;
                score.encoded_pushes += usize::from(inst.is_encoded_push());
                if inst.is_encoded_push() {
                    let (bytes, gas) = if let Some(size) = inst.immutable_type_size() {
                        (usize::from(size.bytes()) + 1, 3)
                    } else if inst.deferred_push().is_some() {
                        (2, 3)
                    } else if let Some(value) = inst.pushed_value() {
                        ir::immediate_materialization_cost(codegen.gcx.sess.opts.evm_version, value)
                    } else {
                        // Block and data references are unresolved until whole-artifact layout.
                        (2, 3)
                    };
                    score.estimated_bytes += bytes;
                    score.estimated_static_gas += gas;
                } else {
                    score.estimated_bytes += 1;
                    score.estimated_static_gas += match inst.opcode {
                        op::POP => 2,
                        _ => 3,
                    };
                    match inst.opcode {
                        op::DUP1..=op::DUP16 => score.dups += 1,
                        op::SWAP1..=op::SWAP16 => score.swaps += 1,
                        op::POP => score.pops += 1,
                        op::MLOAD => score.mloads += 1,
                        op::MSTORE | op::MSTORE8 => score.mstores += 1,
                        _ => {}
                    }
                }
            }
            let had_terminator = self.terminators.get(index).copied().unwrap_or_default();
            score.terminators += usize::from(!had_terminator && block.terminator.is_some());
        }
        score
    }
}

impl PhysicalReplayScore {
    fn key(self, optimization: OptimizationMode) -> (usize, usize, u64, usize, usize) {
        match optimization {
            OptimizationMode::Size => (
                self.estimated_bytes,
                self.estimated_static_gas,
                self.spill_bytes,
                self.peak_stack,
                self.instructions,
            ),
            _ => (
                self.estimated_static_gas,
                self.estimated_bytes,
                self.spill_bytes,
                self.peak_stack,
                self.instructions,
            ),
        }
    }
}

const PHYSICAL_LOOKAHEAD_DEPTHS: [usize; 1] = [2];
const PHYSICAL_STRATIFIED_SLACK_WIDTHS: [usize; 1] = [2];

#[derive(Clone, Copy, Debug)]
struct PhysicalBeamResult {
    slack_width: Option<usize>,
    lookahead: usize,
    optimum_hit: bool,
    gas_regret: usize,
    byte_regret: usize,
    states_expanded: usize,
    backend_transactions: usize,
    replay_micros: u64,
    first_prune_rank: Option<usize>,
    first_prune_depth: Option<usize>,
    winner: PhysicalReplayScore,
}

#[derive(Clone, Debug)]
struct PhysicalScheduleChoice {
    block: BlockId,
    range: std::ops::Range<usize>,
    order: Vec<InstId>,
}

#[derive(Clone, Debug, Default)]
struct PhysicalPlanningState {
    candidates: Vec<PhysicalScheduleCandidate>,
    groups: FxHashMap<(BlockId, usize), (usize, usize)>,
    choices: Vec<PhysicalScheduleChoice>,
}

impl PhysicalPlanningState {
    fn new(candidates: Vec<PhysicalScheduleCandidate>) -> Self {
        let mut groups = FxHashMap::default();
        let mut start = 0;
        while start < candidates.len() {
            let candidate = &candidates[start];
            let mut end = start + 1;
            while end < candidates.len()
                && candidates[end].block == candidate.block
                && candidates[end].range == candidate.range
            {
                end += 1;
            }
            groups.insert((candidate.block, candidate.range.start), (start, end));
            start = end;
        }
        Self { candidates, groups, choices: Vec::new() }
    }
}

#[derive(Clone)]
struct PersistentPhysicalState<'gcx> {
    prefix: Vec<InstId>,
    backend: EvmCodegen<'gcx>,
    score: PhysicalReplayScore,
    pressure: usize,
}

#[derive(Clone)]
struct CompactPhysicalState {
    prefix: Vec<InstId>,
    scheduler: StackScheduler,
    spill_addr_consts: FxHashMap<u64, (DeferredConst, usize)>,
    instructions: Vec<ir::Instruction>,
    label_relocations: Vec<(ir::BlockId, usize, Label)>,
    deferred_relocations: Vec<(ir::BlockId, usize, DeferredConst)>,
    indexed_jump_relocations: Vec<(ir::BlockId, Vec<Label>)>,
    alloc_relocations: Vec<(ir::BlockId, usize, DeferredAlloc)>,
    score: PhysicalReplayScore,
    pressure: usize,
}

#[derive(Clone, Copy)]
struct CompactPhysicalRoot {
    block: ir::BlockId,
    instructions: usize,
    label_relocations: usize,
    deferred_relocations: usize,
    indexed_jump_relocations: usize,
    alloc_relocations: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct PhysicalBeamCensus {
    regions: usize,
    optimum_hits: usize,
    gas_regret: usize,
    byte_regret: usize,
    worst_gas_regret: usize,
    worst_byte_regret: usize,
    states_expanded: usize,
    backend_transactions: usize,
    replay_micros: u64,
    first_prunes: usize,
    first_prune_rank_sum: usize,
    first_prune_rank_max: usize,
    first_prune_depth_sum: usize,
}

impl PhysicalBeamCensus {
    fn record(&mut self, result: PhysicalBeamResult) {
        self.regions += 1;
        self.optimum_hits += usize::from(result.optimum_hit);
        self.gas_regret += result.gas_regret;
        self.byte_regret += result.byte_regret;
        self.worst_gas_regret = self.worst_gas_regret.max(result.gas_regret);
        self.worst_byte_regret = self.worst_byte_regret.max(result.byte_regret);
        self.states_expanded += result.states_expanded;
        self.backend_transactions += result.backend_transactions;
        self.replay_micros += result.replay_micros;
        if let (Some(rank), Some(depth)) = (result.first_prune_rank, result.first_prune_depth) {
            self.first_prunes += 1;
            self.first_prune_rank_sum += rank;
            self.first_prune_rank_max = self.first_prune_rank_max.max(rank);
            self.first_prune_depth_sum += depth;
        }
    }
}

/// Describes the stack effect of an EVM instruction.
/// This is used to keep the scheduler's stack model in sync with the actual EVM stack.
#[derive(Clone, Copy, Debug)]
struct StackEffect {
    /// Number of values popped from the stack.
    pops: usize,
    /// Number of values pushed to the stack.
    pushes: usize,
}

/// What value to track for a pushed stack entry.
#[derive(Clone, Copy, Debug)]
enum StackPush {
    /// No value is pushed (pushes == 0).
    #[allow(dead_code)]
    None,
    /// Push a tracked ValueId (pushes == 1).
    Tracked(ValueId),
    /// Push an unknown/untracked value (pushes == 1).
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaticCallStackWord {
    ReturnAddress,
    Argument(usize),
}

#[derive(Clone, Debug)]
struct StackArgRetentionPlan {
    retained: DenseBitSet<usize>,
    drain_ops: Vec<StackOp>,
    shuffle_ops: Vec<StackOp>,
}

/// Stack arguments whose static-frame stores are delayed until their first instruction use.
///
/// `args` follows physical stack order, highest argument index first. Values in `frame_values` are
/// used again and therefore receive a store immediately before that use; the others die on the
/// stack without ever occupying their declared frame slot.
#[derive(Clone, Debug)]
struct LazyStackArgPlan {
    args: Vec<(ArgIdx, ValueId)>,
    frame_values: DenseBitSet<ValueId>,
}

impl LazyStackArgPlan {
    fn values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.args.iter().map(|&(_, value)| value)
    }
}

/// A profitable static-call layout whose caller words stay below the
/// untracked return address until control returns.
#[derive(Clone, Debug)]
struct StaticCallStackPlan {
    caller_stack: StackModel,
}

#[derive(Clone, Copy, Debug)]
struct InternalCallStackEdge {
    caller: FunctionId,
    callee: FunctionId,
    preserved_words: usize,
}

#[derive(Clone, Debug, Default)]
struct StackPhiPlan {
    entries: FxHashMap<BlockId, Vec<ValueId>>,
    edges: FxHashMap<BlockId, StackPhiEdge>,
    edge_sources: FxHashMap<BlockId, Vec<ValueId>>,
}

#[derive(Clone, Debug)]
struct StackPhiEdge {
    sources: Vec<ValueId>,
    results: Vec<ValueId>,
}

#[derive(Clone, Copy, Debug)]
struct SpillLiveRange {
    start: usize,
    end: usize,
}

#[derive(Default)]
struct SpillColor {
    ranges: FxHashMap<BlockId, SmallVec<[SpillLiveRange; 4]>>,
}

impl SpillColor {
    fn accepts(&self, ranges: &FxHashMap<BlockId, SpillLiveRange>) -> bool {
        ranges.iter().all(|(block, candidate)| {
            self.ranges.get(block).is_none_or(|assigned| {
                assigned
                    .iter()
                    .all(|range| candidate.end < range.start || range.end < candidate.start)
            })
        })
    }

    fn insert(&mut self, ranges: &FxHashMap<BlockId, SpillLiveRange>) {
        for (&block, &range) in ranges {
            self.ranges.entry(block).or_default().push(range);
        }
    }
}

/// Canonical argument layouts carried between MIR basic blocks.
///
/// A block-local scheduler normally discards its model at every join. Function
/// arguments are special: they have one identity on every incoming edge and can
/// always be rematerialized as a safe fallback. Agreeing on one layout for all
/// predecessors lets the first load remain stack-resident through diamonds and
/// loops instead of repeating `CALLDATALOAD` or frame `MLOAD` in every block.
#[derive(Clone, Debug, Default)]
struct GlobalStackPlan {
    entries: FxHashMap<BlockId, Vec<ValueId>>,
    aliases: FxHashMap<ValueId, ValueId>,
}

impl GlobalStackPlan {
    fn analyze(func: &Function, liveness: &Liveness, stack_phi_plan: &StackPhiPlan) -> Self {
        if func.selector.is_none() {
            return Self::default();
        }

        let mut entries = FxHashMap::default();
        let arg_uses = func.arg_uses();
        let used_args = arg_uses.iter().filter(|uses| !uses.is_empty()).count();
        if !(2..=GLOBAL_STACK_MAX_ARGS).contains(&used_args) {
            return Self::default();
        }

        let cfg = CfgInfo::new(func);
        if cfg.reachable().count() < GLOBAL_STACK_MIN_BLOCKS {
            return Self::default();
        }
        let mut decode_blocks = FxHashMap::default();
        let mut aliases = FxHashMap::default();
        for (block_id, block) in func.blocks.iter_enumerated() {
            for &inst_id in &block.instructions {
                let InstKind::CalldataLoad(offset) = &func.inst(inst_id).kind else {
                    continue;
                };
                let Some(offset) = func.value_u64(*offset) else {
                    continue;
                };
                if offset >= 4
                    && (offset - 4) % 32 == 0
                    && let Ok(index) = u32::try_from((offset - 4) / 32)
                    && let Some(&arg) =
                        arg_uses.get(ArgIdx::new(index as usize)).and_then(|uses| uses.first())
                {
                    decode_blocks.entry(arg).or_insert(block_id);
                    if let Some(result) = func.inst_result_value(inst_id) {
                        aliases.insert(result, arg);
                    }
                }
            }
        }

        for block_id in func.blocks.indices() {
            if !cfg.is_reachable(block_id)
                || func.blocks[block_id].predecessors.is_empty()
                || stack_phi_plan.entries.contains_key(&block_id)
                || Self::is_terminal_block(func, block_id)
            {
                continue;
            }

            let values: Vec<_> = liveness
                .live_in(block_id)
                .iter()
                .filter(|&value| {
                    matches!(func.value(value), crate::mir::Value::Arg(_))
                        && decode_blocks.get(&value).is_none_or(|&decode| {
                            decode != block_id && cfg.dominators().dominates(decode, block_id)
                        })
                })
                .take(GLOBAL_STACK_LAYOUT_LIMIT)
                .collect();
            if !values.is_empty() {
                entries.insert(block_id, values);
            }
        }

        // A branch leaves one physical stack for both outgoing edges after its
        // condition is consumed. Its successors therefore have to agree on the
        // same canonical layout. Use the union so an argument needed by either
        // live successor remains available. Terminal siblings are excluded:
        // carried words are harmless below their abort operands. Iterate
        // because sibling constraints can connect several diamonds.
        let mut changed = true;
        while changed {
            changed = false;
            for block_id in func.blocks.indices() {
                let Some(Terminator::Branch { then_block, else_block, .. }) =
                    func.blocks[block_id].terminator.as_ref()
                else {
                    continue;
                };
                if Self::is_terminal_block(func, *then_block)
                    || Self::is_terminal_block(func, *else_block)
                {
                    continue;
                }
                let mut common = entries.get(then_block).cloned().unwrap_or_default();
                for &value in entries.get(else_block).into_iter().flatten() {
                    if common.len() == GLOBAL_STACK_LAYOUT_LIMIT {
                        break;
                    }
                    if !common.contains(&value) {
                        common.push(value);
                    }
                }
                common.sort_by_key(|value| value.index());
                changed |= Self::set_entry(&mut entries, *then_block, &common);
                changed |= Self::set_entry(&mut entries, *else_block, &common);
            }
        }

        // Switch lowering owns the selector stack, and stack-phi entries
        // own their edge layouts. Disable their whole branch-sibling component
        // so every predecessor of every affected block still agrees.
        let mut disabled = DenseBitSet::new_empty(func.blocks.len());
        for &block in stack_phi_plan.entries.keys() {
            disabled.insert(block);
        }
        for block_id in func.blocks.indices() {
            if let Some(Terminator::Switch { default, cases, .. }) =
                func.blocks[block_id].terminator.as_ref()
            {
                disabled.insert(*default);
                for &(_, target) in cases {
                    disabled.insert(target);
                }
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for block_id in func.blocks.indices() {
                let Some(Terminator::Branch { then_block, else_block, .. }) =
                    func.blocks[block_id].terminator.as_ref()
                else {
                    continue;
                };
                if Self::is_terminal_block(func, *then_block)
                    || Self::is_terminal_block(func, *else_block)
                {
                    continue;
                }
                if disabled.contains(*then_block) || disabled.contains(*else_block) {
                    changed |= disabled.insert(*then_block);
                    changed |= disabled.insert(*else_block);
                }
            }
        }
        entries.retain(|block, _| !disabled.contains(*block));

        // Canonicalization pays DUP/SWAP/POP traffic on every planned edge.
        // Require enough real argument reuse to recover that fixed cost, and
        // reject dense layout plans unless a long CFG can amortize them.
        let arg_use_count = arg_uses.iter().map(Vec::len).sum::<usize>();
        if arg_use_count < GLOBAL_STACK_MIN_ARG_USES
            || (entries.len() * 2 > cfg.reachable().count()
                && cfg.reachable().count() < GLOBAL_STACK_DENSE_AMORTIZATION_BLOCKS)
        {
            entries.clear();
        }
        aliases.retain(|_, arg| entries.values().any(|entry| entry.contains(arg)));
        Self { entries, aliases }
    }

    fn set_entry(
        entries: &mut FxHashMap<BlockId, Vec<ValueId>>,
        block: BlockId,
        layout: &[ValueId],
    ) -> bool {
        if entries.get(&block).map_or(layout.is_empty(), |old| old == layout) {
            return false;
        }
        if layout.is_empty() {
            entries.remove(&block);
        } else {
            entries.insert(block, layout.to_vec());
        }
        true
    }

    fn entry(&self, block: BlockId) -> Option<&[ValueId]> {
        self.entries.get(&block).map(Vec::as_slice)
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn edge_layout(&self, func: &Function, term: &Terminator) -> Option<&[ValueId]> {
        match term {
            Terminator::Jump(target) => self.entry(*target),
            Terminator::Branch { then_block, else_block, .. } => {
                if Self::is_terminal_block(func, *then_block) {
                    return self.entry(*else_block);
                }
                if Self::is_terminal_block(func, *else_block) {
                    return self.entry(*then_block);
                }
                let then_layout = self.entry(*then_block)?;
                (self.entry(*else_block) == Some(then_layout)).then_some(then_layout)
            }
            _ => None,
        }
    }

    fn is_terminal_block(func: &Function, block: BlockId) -> bool {
        matches!(
            func.blocks[block].terminator,
            Some(Terminator::Revert { .. } | Terminator::Invalid)
        )
    }
}

impl StackPhiPlan {
    fn analyze(func: &Function) -> Self {
        StackPhiPlanner::new(func).plan()
    }
}

struct StackPhiPlanner<'a> {
    func: &'a Function,
    loops: Vec<Loop>,
    header_results: FxHashMap<BlockId, Vec<ValueId>>,
}

impl<'a> StackPhiPlanner<'a> {
    fn new(func: &'a Function) -> Self {
        let mut loop_analyzer = LoopAnalyzer::new();
        let loop_info = loop_analyzer.analyze(func);
        let mut loops: Vec<_> = loop_info.all_loops().cloned().collect();
        loops.sort_by_key(|loop_info| loop_info.header.index());

        let mut planner = Self { func, loops, header_results: FxHashMap::default() };
        planner.collect_header_results();
        planner
    }

    fn plan(&self) -> StackPhiPlan {
        let mut plan = StackPhiPlan::default();
        for loop_info in &self.loops {
            self.plan_loop(loop_info, &mut plan);
        }
        for block in self.func.blocks.indices() {
            self.plan_join(block, &mut plan);
        }
        plan
    }

    fn collect_header_results(&mut self) {
        for loop_info in &self.loops {
            let block = &self.func.blocks[loop_info.header];
            let phi_insts = self.phi_insts(block);
            if let Some(results) = self.phi_result_values(&phi_insts) {
                self.header_results.insert(loop_info.header, results);
            }
        }
    }

    fn plan_loop(&self, loop_info: &Loop, plan: &mut StackPhiPlan) {
        let Some(preheader) = loop_info.preheader else {
            return;
        };
        let [latch] = loop_info.back_edges.as_slice() else {
            return;
        };
        if !matches!(self.func.blocks[preheader].terminator, Some(Terminator::Jump(target)) if target == loop_info.header)
            || !matches!(self.func.blocks[*latch].terminator, Some(Terminator::Jump(target)) if target == loop_info.header)
        {
            return;
        }
        if plan.edges.contains_key(&preheader) || plan.edges.contains_key(latch) {
            return;
        }

        let block = &self.func.blocks[loop_info.header];
        let phi_insts = self.phi_insts(block);
        if phi_insts.is_empty() || phi_insts.len() > STACK_PHI_LAYOUT_LIMIT {
            return;
        }

        let Some(results) = self.phi_result_values(&phi_insts) else {
            return;
        };
        if results.len() > STACK_PHI_LAYOUT_LIMIT {
            return;
        }

        let carry_through = self.carry_through_values(loop_info);
        if carry_through.len() + results.len() > STACK_PHI_LAYOUT_LIMIT {
            return;
        }

        let mut entry = carry_through.clone();
        entry.extend(results.iter().copied());

        let predecessors = [preheader, *latch];
        let mut edges = Vec::with_capacity(predecessors.len());
        for pred in predecessors {
            let Some(phi_sources) = self.phi_sources_for_pred(&phi_insts, pred) else {
                return;
            };
            let mut sources = carry_through.clone();
            sources.extend(phi_sources);
            debug_assert_eq!(sources.len(), entry.len());
            edges.push((pred, sources));
        }

        plan.entries.insert(loop_info.header, entry.clone());
        for (pred, sources) in edges {
            plan.edge_sources.insert(pred, sources.clone());
            plan.edges.insert(pred, StackPhiEdge { sources, results: entry.clone() });
        }
    }

    fn plan_join(&self, block_id: BlockId, plan: &mut StackPhiPlan) {
        let block = &self.func.blocks[block_id];
        if plan.entries.contains_key(&block_id) || block.predecessors.len() < 2 {
            return;
        }

        let phi_insts = self.phi_insts(block);
        if phi_insts.is_empty() || phi_insts.len() > STACK_PHI_LAYOUT_LIMIT {
            return;
        }
        let Some(results) = self.phi_result_values(&phi_insts) else {
            return;
        };
        if block.predecessors.iter().any(|pred| {
            plan.edges.contains_key(pred)
                || !matches!(
                    self.func.blocks[*pred].terminator,
                    Some(Terminator::Jump(target)) if target == block_id
                )
        }) {
            return;
        }

        let mut edges = Vec::with_capacity(block.predecessors.len());
        for &pred in &block.predecessors {
            let Some(sources) = self.phi_sources_for_pred(&phi_insts, pred) else {
                return;
            };
            edges.push((pred, sources));
        }

        plan.entries.insert(block_id, results.clone());
        for (pred, sources) in edges {
            plan.edge_sources.insert(pred, sources.clone());
            plan.edges.insert(pred, StackPhiEdge { sources, results: results.clone() });
        }
    }

    fn phi_insts(&self, block: &crate::mir::BasicBlock) -> Vec<InstId> {
        block
            .instructions
            .iter()
            .copied()
            .filter(|&inst| matches!(self.func.inst(inst).kind, InstKind::Phi(_)))
            .collect()
    }

    fn carry_through_values(&self, loop_info: &Loop) -> Vec<ValueId> {
        let mut carry_through = Vec::new();
        for outer in &self.loops {
            if outer.header == loop_info.header || !outer.blocks.contains(loop_info.header) {
                continue;
            }
            let Some(results) = self.header_results.get(&outer.header) else {
                continue;
            };
            for &value in results {
                if carry_through.contains(&value)
                    || !self.value_used_in_blocks(&loop_info.blocks, value)
                {
                    continue;
                }
                carry_through.push(value);
            }
        }
        carry_through
    }

    fn value_used_in_blocks(&self, blocks: &DenseBitSet<BlockId>, value: ValueId) -> bool {
        for block_id in blocks {
            let block = &self.func.blocks[block_id];
            for &inst_id in &block.instructions {
                if matches!(self.func.inst(inst_id).kind, InstKind::Phi(_)) {
                    continue;
                }
                if self.func.inst(inst_id).kind.operands().contains(&value) {
                    return true;
                }
            }
            if block.terminator.as_ref().is_some_and(|term| term.operands().contains(&value)) {
                return true;
            }
        }
        false
    }

    fn phi_result_values(&self, phi_insts: &[InstId]) -> Option<Vec<ValueId>> {
        phi_insts.iter().map(|&inst| self.func.inst_result_value(inst)).collect()
    }

    fn phi_sources_for_pred(&self, phi_insts: &[InstId], pred: BlockId) -> Option<Vec<ValueId>> {
        phi_insts
            .iter()
            .map(|&inst| {
                let InstKind::Phi(incoming) = &self.func.inst(inst).kind else {
                    return None;
                };
                incoming.iter().find_map(|&(block, value)| (block == pred).then_some(value))
            })
            .collect()
    }
}

/// EVM code generator.
#[derive(Clone)]
pub struct EvmCodegen<'gcx> {
    gcx: Gcx<'gcx>,
    /// The assembler for bytecode generation.
    asm: Assembler<'gcx>,
    /// Stack scheduler.
    scheduler: StackScheduler,
    /// Block labels.
    block_labels: FxHashMap<BlockId, Label>,
    /// Function labels for direct internal calls.
    function_labels: FxHashMap<FunctionId, Label>,
    /// Functions whose reachable exits all abort. Calls to these functions
    /// make their containing block cold as well.
    cold_functions: DenseBitSet<FunctionId>,
    /// Functions consisting only of an empty block terminated by `stop`.
    empty_stop_functions: DenseBitSet<FunctionId>,
    /// Cold blocks in the function currently being emitted, including blocks
    /// that only forward control to other cold blocks.
    cold_blocks: DenseBitSet<BlockId>,
    /// Exact per-function spill area sizes, in bytes, recorded after emission.
    function_spill_sizes: FxHashMap<FunctionId, u64>,
    /// Internal-call frame-size constants waiting for exact callee spill sizes.
    pending_frame_size_consts: Vec<(DeferredConst, FunctionId)>,
    /// Per static-frame callee: which argument indices every runtime call
    /// site can re-emit after the stack drain, so they ride the stack above
    /// the return address instead of being stored to the callee frame at
    /// each site. The callee prologue stores them once.
    stack_arg_masks: FxHashMap<FunctionId, DenseBitSet<usize>>,
    /// Static callees whose stack-passed arguments are consumed once in the entry block and can
    /// therefore remain on the physical stack instead of being copied into their frame.
    direct_stack_args: FxHashMap<FunctionId, Vec<ValueId>>,
    /// Static callees whose incoming stack arguments feed the first instruction directly. Stores
    /// for repeated arguments are emitted immediately before that instruction.
    lazy_stack_args: FxHashMap<FunctionId, LazyStackArgPlan>,
    /// Non-recursive one-word callees whose return value stays on the EVM stack instead of being
    /// staged through their static memory frame.
    stack_return_functions: DenseBitSet<FunctionId>,
    /// First local/spill byte in each stack-return function's original MIR frame layout. Offsets
    /// at or above this boundary shift down over the removed return word.
    stack_return_local_bases: FxHashMap<FunctionId, u64>,
    /// Enables the optional caller-prefix convention for this emission. If
    /// post-emission stack validation rejects it, runtime codegen reruns once
    /// with this disabled.
    preserve_caller_stack: bool,
    /// Functions reached from a recursive activation. Their incoming physical
    /// prefix is unbounded, so preserving another caller prefix would change
    /// the recursion limit.
    recursive_stack_functions: DenseBitSet<FunctionId>,
    /// Functions that are recursive or can reach recursion. A preserved
    /// prefix must not be carried into an unbounded descendant.
    recursion_reaching_functions: DenseBitSet<FunctionId>,
    /// High-water mark of the modeled stack above each function's inherited
    /// untracked prefix.
    function_stack_peaks: FxHashMap<FunctionId, usize>,
    /// Runtime internal-call edges and the caller words retained at each site.
    internal_call_stack_edges: Vec<InternalCallStackEdge>,
    /// Whether the current assembly is the runtime (stack-passed arguments
    /// apply). The constructor assembly emits its own copies of internal
    /// functions with the plain frame-store convention.
    runtime_stack_args: bool,
    /// Deferred spill-slot address pushes of the external body being emitted,
    /// keyed by the slot's allocation offset, with their reference counts.
    /// Ranked hottest-first at body end so the most reloaded slots take the
    /// shortest addresses; final addresses wait for global layout.
    spill_addr_consts: FxHashMap<u64, (DeferredConst, usize)>,
    /// Ranked external spill pushes retained until static-allocation layout is
    /// finalized, keyed by entry function.
    external_spill_addr_consts: FxHashMap<FunctionId, Vec<(DeferredConst, usize)>>,
    /// Callees whose internal-call frame can be deallocated after return.
    restorable_internal_frames: DenseBitSet<FunctionId>,
    /// Functions whose frame lives at a compile-time-fixed address (static
    /// frames): internal-convention, non-recursive functions in the runtime
    /// passes. Their arg/local/spill accesses are absolute pushes and their
    /// call sites skip all frame-pointer and free-pointer bookkeeping.
    static_frame_functions: DenseBitSet<FunctionId>,
    /// Interned deferred constants for absolute static-frame addresses, keyed
    /// by (function, byte offset within its frame). Resolved at the end of
    /// the pass, once every body's exact spill size is known.
    static_frame_addr_consts: FxHashMap<(FunctionId, u64), (DeferredConst, usize)>,
    /// Deferred allocations emitted by each external entry.
    pending_static_allocs: FxHashMap<FunctionId, Vec<(DeferredAlloc, u64)>>,
    /// The pass's single free-memory-pointer constant, emitted once in the
    /// runtime prologue and resolved at the end of the pass: the heap must
    /// start above every entry's locals/spills and the static frame region.
    runtime_free_memory_const: Option<DeferredConst>,
    /// Every external body emitted this pass, for sizing the heap floor.
    runtime_entry_funcs: Vec<FunctionId>,
    /// The internal-convention function currently being emitted.
    current_internal_function: Option<FunctionId>,
    /// Copies to insert at block exits (from phi elimination).
    block_copies: FxHashMap<BlockId, Vec<ParallelCopy>>,
    /// Values carried by planned stack-resident phi edges, keyed by predecessor block.
    stack_phi_sources: FxHashMap<BlockId, Vec<ValueId>>,
    /// Whether the current function has canonical cross-block argument layouts.
    global_stack_active: bool,
    /// Calldata words physically identical to arguments in the active global
    /// layout, adopted after their final validation use.
    global_stack_aliases: FxHashMap<ValueId, ValueId>,
    /// Immutable `PUSH<N>` placeholders in the last assembled runtime code.
    runtime_immutable_refs: Vec<ImmutableRef>,
    /// Backend encodings derived from the current module's immutable declarations.
    immutable_encodings: IndexVec<ImmutableId, ImmutableEncoding>,
    /// First constructor-memory word reserved for immutable staging.
    immutable_staging_base: u64,
    /// Deferred absolute base of the copied constructor ABI argument blob.
    constructor_args_base_const: Option<DeferredConst>,
    /// Whether we're currently generating constructor code.
    /// When true, arguments load from the copied deployment ABI blob.
    in_constructor: bool,
    /// Shared constructor completion reached by ordinary `stop` terminators.
    constructor_exit: Option<Label>,
    /// Number of constructor parameters (used for CODECOPY offset calculation).
    constructor_param_count: u32,
    /// Whether we're emitting an internal function body.
    in_internal_function: bool,
    /// Whether we're emitting the MIR `entry` function. Its switch
    /// keeps the selector on the physical stack through the case chain and
    /// leaves it inert below the taken arm. This is only sound for `entry`: it
    /// runs once and every arm terminates externally, so the leftover word can
    /// neither accumulate nor disturb an internal return.
    emitting_entry: bool,
    /// Gas-mode switch growth still available in the current deployment artifact.
    switch_gas_code_growth_remaining: usize,
    capture_mir: bool,
    capture_evm_ir: bool,
    /// Prevents analysis-only shadow emission from recursively starting another replay.
    physical_replay_active: bool,
    /// Stops an analysis-only replay after this many instructions in one MIR block.
    physical_replay_stop: Option<(BlockId, usize)>,
    /// Bounded physical choices collected during one ordinary shadow emission.
    physical_planning: Option<PhysicalPlanningState>,
}

impl<'gcx> EvmCodegen<'gcx> {
    /// Creates a new EVM code generator.
    #[must_use]
    pub fn new(gcx: Gcx<'gcx>) -> Self {
        let switch_gas_code_growth_remaining = Self::switch_gas_code_growth_limit(gcx);
        Self {
            gcx,
            asm: Assembler::new(gcx),
            scheduler: StackScheduler::new(),
            block_labels: FxHashMap::default(),
            function_labels: FxHashMap::default(),
            cold_functions: DenseBitSet::new_empty(0),
            empty_stop_functions: DenseBitSet::new_empty(0),
            cold_blocks: DenseBitSet::new_empty(0),
            function_spill_sizes: FxHashMap::default(),
            pending_frame_size_consts: Vec::new(),
            stack_arg_masks: FxHashMap::default(),
            direct_stack_args: FxHashMap::default(),
            lazy_stack_args: FxHashMap::default(),
            stack_return_functions: DenseBitSet::new_empty(0),
            stack_return_local_bases: FxHashMap::default(),
            preserve_caller_stack: false,
            recursive_stack_functions: DenseBitSet::new_empty(0),
            recursion_reaching_functions: DenseBitSet::new_empty(0),
            function_stack_peaks: FxHashMap::default(),
            internal_call_stack_edges: Vec::new(),
            runtime_stack_args: false,
            spill_addr_consts: FxHashMap::default(),
            external_spill_addr_consts: FxHashMap::default(),
            restorable_internal_frames: DenseBitSet::new_empty(0),
            static_frame_functions: DenseBitSet::new_empty(0),
            static_frame_addr_consts: FxHashMap::default(),
            pending_static_allocs: FxHashMap::default(),
            runtime_free_memory_const: None,
            runtime_entry_funcs: Vec::new(),
            current_internal_function: None,
            block_copies: FxHashMap::default(),
            stack_phi_sources: FxHashMap::default(),
            global_stack_active: false,
            global_stack_aliases: FxHashMap::default(),
            runtime_immutable_refs: Vec::new(),
            immutable_encodings: IndexVec::new(),
            immutable_staging_base: EvmMemoryLayout::INTERNAL_FRAME_PTR_SLOT
                + EvmMemoryLayout::WORD_SIZE,
            constructor_args_base_const: None,
            in_constructor: false,
            constructor_exit: None,
            constructor_param_count: 0,
            in_internal_function: false,
            emitting_entry: false,
            switch_gas_code_growth_remaining,
            capture_mir: false,
            capture_evm_ir: false,
            physical_replay_active: false,
            physical_replay_stop: None,
            physical_planning: None,
        }
    }

    fn reset_switch_gas_code_growth(&mut self) {
        self.switch_gas_code_growth_remaining = Self::switch_gas_code_growth_limit(self.gcx);
    }

    fn switch_gas_code_growth_limit(gcx: Gcx<'_>) -> usize {
        gcx.sess.opts.unstable.switch_max_gas_code_growth.unwrap_or(MAX_GAS_CODE_GROWTH)
    }

    /// Whether a function is an external interface of its module: an ABI entry,
    /// the constructor, the fallback, or the receive function. A module with
    /// none has no reachable runtime code.
    fn is_module_entry(func: &Function) -> bool {
        func.selector.is_some()
            || func.attributes.is_constructor
            || func.attributes.is_fallback
            || func.attributes.is_receive
    }

    /// Reports MIR constructs the backend cannot emit yet.
    ///
    /// This includes argument-taking fallbacks and logical slices whose
    /// aggregate use slice lowering could not fold.
    ///
    /// Only live instructions — those still in a block — are checked, since the
    /// instruction arena retains folded-away slices the backend never emits.
    #[must_use]
    fn emit_unsupported(&self, module: &Module) -> bool {
        if module
            .functions
            .iter()
            .any(|func| func.attributes.is_fallback && !func.params.is_empty())
        {
            self.gcx
                .dcx()
                .err("codegen does not support `fallback(bytes) returns (bytes)` yet")
                .span(module.name.span)
                .emit();
            return true;
        }

        let mut emitted = false;
        'func: for func in module.functions.iter() {
            for inst_id in func.instructions() {
                let inst = func.inst(inst_id);
                let message = match inst.kind {
                    InstKind::MakeSlice { .. } | InstKind::SlicePtr(_) | InstKind::SliceLen(_) => {
                        "codegen does not support this calldata-slice usage yet"
                    }
                    InstKind::StoreImmutable(..) => {
                        "immutable assignments must be lowered before EVM codegen"
                    }
                    _ => continue,
                };
                let span = inst.metadata.source_span().unwrap_or(module.name.span);
                self.gcx.dcx().err(message).span(span).emit();
                emitted = true;
                // One diagnostic per function is enough to explain the bail.
                continue 'func;
            }
        }
        emitted
    }

    /// Controls whether generated artifacts include final EVM IR.
    pub fn set_capture_evm_ir(&mut self, capture: bool) {
        self.capture_evm_ir = capture;
    }

    /// Controls whether modules without an external entry still run the MIR pipeline.
    pub(crate) fn set_capture_mir(&mut self, capture: bool) {
        self.capture_mir = capture;
    }

    // ==================== Stack-Aware Emitter API ====================
    //
    // These helpers ensure that all EVM stack mutations are tracked by the scheduler.
    // Any opcode that changes the EVM stack must be emitted through these methods
    // to keep the scheduler's StackModel in sync with the actual EVM stack.

    /// Emits a stack manipulation operation (DUP, SWAP, POP) and updates the scheduler.
    fn emit_stack_op(&mut self, op: StackOp) {
        self.asm.emit_op(op.opcode());
        match op {
            StackOp::Dup(n) => self.scheduler.stack.dup(n),
            StackOp::Swap(n) => self.scheduler.stack.swap(n),
            StackOp::Pop => {
                self.scheduler.stack.pop();
            }
        }
    }

    /// Emits an opcode with known stack effects and updates the scheduler.
    ///
    /// This is the core method for stack-aware emission. After emitting the opcode:
    /// - `effect.pops` values are removed from the scheduler's stack model
    /// - Values are pushed according to `push`:
    ///   - `StackPush::None`: no value pushed (effect.pushes must be 0)
    ///   - `StackPush::Tracked(v)`: push a tracked ValueId (effect.pushes must be 1)
    ///   - `StackPush::Unknown`: push an untracked value (effect.pushes must be 1)
    fn emit_op_with_effect(&mut self, opcode: u8, effect: StackEffect, push: StackPush) {
        #[cfg(debug_assertions)]
        let before = self.scheduler.depth();

        self.asm.emit_op(opcode);

        // Pop consumed values
        for _ in 0..effect.pops {
            self.scheduler.stack.pop();
        }

        // Push produced values
        match (effect.pushes, push) {
            (0, StackPush::None) => {}
            (1, StackPush::Tracked(v)) => self.scheduler.stack.push(v),
            (1, StackPush::Unknown) => self.scheduler.stack.push_unknown(),
            (n, _) if n > 1 => {
                // Multi-push: push unknown values
                for _ in 0..n {
                    self.scheduler.stack.push_unknown();
                }
            }
            _ => {}
        }

        #[cfg(debug_assertions)]
        {
            let expected = before + effect.pushes - effect.pops;
            debug_assert_eq!(
                self.scheduler.depth(),
                expected,
                "Stack model drift after opcode 0x{:02x}: expected depth {}, got {}",
                opcode,
                expected,
                self.scheduler.depth()
            );
        }
    }

    /// Generates deployment bytecode for a module.
    /// Returns (deployment_bytecode, runtime_bytecode).
    /// Returns empty bytecodes for interfaces (they have no implementation).
    ///
    /// This runs optimization passes (including DCE) on the module before codegen unless disabled.
    pub fn generate_deployment_bytecode(&mut self, module: &mut Module) -> (Vec<u8>, Vec<u8>) {
        let artifact = self.generate_deployment_artifact(module);
        (artifact.deployment, artifact.runtime)
    }

    #[tracing::instrument(
        name = "evm_codegen",
        level = "debug",
        skip_all,
        fields(module = %module.name),
    )]
    fn generate_deployment_artifact(&mut self, module: &mut Module) -> EvmArtifact {
        // An internal-only library (no external interface) has no reachable
        // runtime code — like `solc`, it produces no bytecode rather than
        // standalone bodies for functions only ever inlined elsewhere.
        if module.is_interface {
            return EvmArtifact::default();
        }
        if !module.functions.iter().any(Self::is_module_entry) {
            if self.capture_mir {
                self.run_optimization_passes(module);
            }
            return EvmArtifact::default();
        }
        if let Some(func) = module.functions.iter().find(|func| func.blocks.is_empty()) {
            panic!("cannot codegen MIR function `{}` without an entry block", func.name);
        }
        self.run_optimization_passes(module);
        if self.emit_unsupported(module) {
            return EvmArtifact::default();
        }
        if module.phase != MirPhase::EvmShaped {
            self.gcx
                .dcx()
                .err(format!(
                    "EVM codegen requires MIR in the `evm-shaped` phase, stopped at `{}`",
                    module.phase.name()
                ))
                .span(module.name.span)
                .emit();
            return EvmArtifact::default();
        }
        self.immutable_staging_base = immutable_staging_base(module);
        self.immutable_encodings.clear();
        for (id, immutable) in module.iter_immutables() {
            let encoding =
                immutable.ty.immutable_encoding().expect("validated immutable declaration");
            let allocated = self.immutable_encodings.push(encoding);
            debug_assert_eq!(allocated, id);
        }
        if matches!(self.gcx.sess.opts.optimization, OptimizationMode::Size) {
            for func in &mut module.functions {
                func.canonicalize_immediate_uses();
            }
        }
        // First generate the runtime code
        let mut runtime_code = self.generate_runtime_code(module);
        if let Some(evm_ir) = &mut runtime_code.evm_ir {
            evm_ir.set_name(sym::runtime);
        }
        let runtime_len = runtime_code.bytecode.len();
        let immutable_refs = std::mem::take(&mut self.runtime_immutable_refs);

        // The constructor copies the runtime code to memory and patches the
        // immutable placeholders with the staged words before
        // returning. Copy to offset 0 unless that would overwrite the immutable
        // staging area before the patch loop reads it.
        let copy_base = Self::runtime_copy_base(module, runtime_len, &immutable_refs);

        // Generate constructor initialization and the deployment postlude as
        // one control-flow graph and optimize it once. Constructor arguments
        // are appended after the generated deployment prefix, so their offset
        // and the runtime-code offset depend on its final push widths. Only
        // repeat final assembly while both offsets stabilize.
        let prepared_deploy_code =
            self.prepare_deployment_prefix(module, runtime_len, copy_base, &immutable_refs);
        let mut deploy_code_len = 0usize;
        let mut constructor_arg_offset = runtime_len;
        let mut deploy_code = self.assemble_deployment_prefix(
            &prepared_deploy_code,
            constructor_arg_offset,
            deploy_code_len,
        );
        for _ in 0..8 {
            let next_deploy_code_len = deploy_code.bytecode.len();
            let next_arg_offset = next_deploy_code_len + runtime_len;
            if next_deploy_code_len == deploy_code_len && next_arg_offset == constructor_arg_offset
            {
                break;
            }
            deploy_code_len = next_deploy_code_len;
            constructor_arg_offset = next_arg_offset;
            deploy_code = self.assemble_deployment_prefix(
                &prepared_deploy_code,
                constructor_arg_offset,
                deploy_code_len,
            );
        }

        // Deploy code structure:
        // [constructor_code]    ; run constructor (SSTOREs + immutable staging)
        // PUSH<n> runtime_len   ; size to copy from creation code
        // DUP1                  ; duplicate for the final RETURN size
        // PUSH<n> offset        ; where runtime starts
        // PUSH<n> copy_base     ; memory destination
        // CODECOPY              ; copy runtime to memory
        // [immutable patches]   ; patch staged words into the PUSH<N> placeholders
        // PUSH<n> copy_base     ; memory offset
        // RETURN                ; return the runtime code
        if let Some(evm_ir) = &mut deploy_code.evm_ir {
            evm_ir.set_name(sym::deployment);
        }

        let mut deploy_bytecode = deploy_code.bytecode;
        deploy_bytecode.extend_from_slice(&runtime_code.bytecode);

        // The returned runtime artifact keeps the zero placeholders, like
        // solc's `deployedBytecode` for contracts with immutables.
        EvmArtifact {
            deployment: deploy_bytecode,
            runtime: runtime_code.bytecode,
            immutable_references: immutable_refs,
            deployment_evm_ir: deploy_code.evm_ir,
            runtime_evm_ir: runtime_code.evm_ir,
        }
    }

    fn runtime_copy_base(
        module: &Module,
        runtime_len: usize,
        immutable_refs: &[ImmutableRef],
    ) -> u64 {
        let patched_end = immutable_refs.iter().fold(runtime_len, |end, immutable_ref| {
            let patch_size = if immutable_ref.type_size.bytes() == 1 { 1 } else { EVM_WORD_BYTES };
            end.max(
                immutable_ref
                    .code_offset
                    .checked_add(1 + patch_size)
                    .expect("immutable patch offset overflow"),
            )
        });
        let staging_base = immutable_staging_base(module);
        if !immutable_refs.is_empty() && patched_end as u64 > staging_base {
            immutable_staging_end(staging_base, module.immutable_count())
        } else {
            0
        }
    }

    fn emit_deployment_postlude(
        &mut self,
        module: &Module,
        runtime_offset: DeferredConst,
        runtime_len: usize,
        copy_base: u64,
        immutable_refs: &[ImmutableRef],
    ) {
        // Copy runtime code from creation code to memory at `copy_base`.
        self.asm.emit_push(U256::from(runtime_len as u64));
        self.asm.emit_op(op::dup(1));
        self.asm.emit_push_deferred(runtime_offset);
        self.asm.emit_push(U256::from(copy_base));
        self.asm.emit_op(op::CODECOPY);

        // Patch each `PUSH<N>` placeholder with its staged immutable value.
        for r in immutable_refs {
            let encoding = module
                .immutable_type(r.id)
                .immutable_encoding()
                .expect("validated immutable declaration");
            debug_assert_eq!(
                immutable_push_type_size(
                    encoding,
                    self.gcx.sess.opts.optimization,
                    self.gcx.sess.opts.evm_version.has_bitwise_shifting(),
                ),
                r.type_size
            );
            self.emit_immutable_patch(copy_base, *r, encoding);
        }

        // Return the patched runtime code; the DUP'd length is still on the stack.
        self.asm.emit_push(U256::from(copy_base));
        self.asm.emit_op(op::RETURN);
    }

    fn emit_immutable_patch(
        &mut self,
        copy_base: u64,
        immutable_ref: ImmutableRef,
        encoding: ImmutableEncoding,
    ) {
        let byte_width = immutable_ref.type_size.bytes();
        let destination = copy_base + immutable_ref.code_offset as u64 + 1;

        self.asm.emit_push(U256::from(immutable_staging_addr(
            self.immutable_staging_base,
            immutable_ref.id,
        )));
        self.asm.emit_op(op::MLOAD);

        if byte_width == 1 {
            if matches!(encoding, ImmutableEncoding::LeftAligned(_)) {
                self.asm.emit_push(U256::ZERO);
                self.asm.emit_op(op::BYTE);
            }
            self.asm.emit_push(U256::from(destination));
            self.asm.emit_op(op::MSTORE8);
            return;
        }

        if byte_width < 32 {
            let trailing_bits = usize::from(32 - byte_width) * 8;
            match encoding {
                ImmutableEncoding::LeftAligned(_) => {
                    self.asm.emit_push(U256::MAX << trailing_bits);
                    self.asm.emit_op(op::AND);
                }
                ImmutableEncoding::Unsigned(_) | ImmutableEncoding::Signed(_) => {
                    self.asm.emit_push(U256::from(trailing_bits));
                    self.asm.emit_op(op::SHL);
                }
            }

            // Preserve the runtime bytes following the short placeholder. An
            // unaligned MLOAD/MSTORE pair works even across word boundaries.
            self.asm.emit_push(U256::from(destination));
            self.asm.emit_op(op::MLOAD);
            self.asm.emit_push(U256::MAX >> (usize::from(byte_width) * 8));
            self.asm.emit_op(op::AND);
            self.asm.emit_op(op::OR);
        }

        self.asm.emit_push(U256::from(destination));
        self.asm.emit_op(op::MSTORE);
    }

    fn emit_load_immutable(&mut self, id: ImmutableId) {
        if self.in_constructor {
            // The running constructor's own placeholders are never patched.
            self.asm.emit_push(U256::from(immutable_staging_addr(self.immutable_staging_base, id)));
            self.asm.emit_op(op::MLOAD);
            return;
        }

        let encoding = self.immutable_encodings[id];
        let type_size = immutable_push_type_size(
            encoding,
            self.gcx.sess.opts.optimization,
            self.gcx.sess.opts.evm_version.has_bitwise_shifting(),
        );
        let byte_width = type_size.bytes();
        self.asm.emit_push_immutable(id, type_size);
        if byte_width == 32 {
            return;
        }
        match encoding {
            ImmutableEncoding::Unsigned(_) => {}
            ImmutableEncoding::Signed(_) => {
                self.asm.emit_push(U256::from(byte_width - 1));
                self.asm.emit_op(op::SIGNEXTEND);
            }
            ImmutableEncoding::LeftAligned(_) => {
                self.asm.emit_push(U256::from((32 - byte_width) * 8));
                self.asm.emit_op(op::SHL);
            }
        }
    }

    /// Generates constructor code that runs during deployment.
    /// This includes state variable initializers.
    ///
    /// Constructor arguments are read from the end of the initcode using CODECOPY.
    /// The args are ABI-encoded and appended after the deployment bytecode.
    fn prepare_deployment_prefix(
        &mut self,
        module: &Module,
        runtime_len: usize,
        copy_base: u64,
        immutable_refs: &[ImmutableRef],
    ) -> PreparedDeploymentPrefix {
        self.asm.clear();
        self.asm.set_artifact_kind(ArtifactKind::Constructor);
        let runtime_offset = self.asm.new_deferred_const();

        // Find constructor function if it exists
        let constructor =
            module.functions.iter_enumerated().find(|(_, f)| f.attributes.is_constructor);

        let constructor_arg_offset = if let Some((ctor_id, ctor)) = constructor {
            // Generate constructor bytecode
            // Clear state and generate function body
            self.block_labels.clear();
            self.block_copies.clear();
            self.function_labels.clear();
            self.cold_functions =
                if matches!(self.gcx.sess.opts.optimization, OptimizationMode::None) {
                    DenseBitSet::new_empty(module.functions.len())
                } else {
                    Self::collect_cold_functions(module)
                };
            self.function_spill_sizes.clear();
            self.pending_frame_size_consts.clear();
            self.restorable_internal_frames = DenseBitSet::new_empty(module.functions.len());
            self.static_frame_functions = DenseBitSet::new_empty(module.functions.len());
            self.stack_arg_masks.clear();
            self.direct_stack_args.clear();
            self.lazy_stack_args.clear();
            self.stack_return_functions = DenseBitSet::new_empty(module.functions.len());
            self.stack_return_local_bases.clear();
            self.runtime_stack_args = false;
            // Constructor code has a separate call graph and is not part of
            // the runtime prefix validation below.
            self.preserve_caller_stack = false;
            self.static_frame_addr_consts.clear();
            self.external_spill_addr_consts.clear();
            self.pending_static_allocs.clear();
            self.runtime_free_memory_const = None;
            self.runtime_entry_funcs.clear();
            self.current_internal_function = None;
            self.stack_phi_sources.clear();

            for (func_id, func) in module.functions.iter_enumerated() {
                if !func.params.iter().chain(&func.returns).any(|ty| ty.is_memory_reference()) {
                    self.restorable_internal_frames.insert(func_id);
                }
            }

            let call_graph = CallGraphInfo::new(module);
            let internal_targets = call_graph.reachable_callees_from(std::iter::once(ctor_id));
            for func_id in &internal_targets {
                let label = self.new_function_label(func_id);
                self.function_labels.insert(func_id, label);
            }

            // Constructor locals, immutable staging, and spills occupy fixed
            // compiler-owned regions. The ABI blob starts after their exact
            // post-emission end, and dynamic allocations start after the blob.
            let constructor_fixed_memory_end = self.asm.new_deferred_const();
            let constructor_arg_offset =
                (!ctor.params.is_empty()).then(|| self.asm.new_deferred_const());

            // Set constructor context for LoadArg handling
            self.in_constructor = true;
            self.constructor_param_count = ctor.params.len() as u32;

            // Constructor args are appended after generated deployment bytecode.
            // Copy the complete blob above every fixed compiler-owned region,
            // then place the free-memory pointer after its word-aligned end.
            if let Some(arg_offset) = constructor_arg_offset {
                self.constructor_args_base_const = Some(constructor_fixed_memory_end);
                self.asm.emit_push_deferred(arg_offset);
                self.asm.emit_op(op::CODESIZE);
                self.asm.emit_op(op::SUB); // size = CODESIZE - arg_offset
                self.asm.emit_op(op::dup(1));
                self.asm.emit_push_deferred(arg_offset); // code offset
                self.asm.emit_push_deferred(constructor_fixed_memory_end);
                self.asm.emit_op(op::CODECOPY);

                self.asm.emit_push_deferred(constructor_fixed_memory_end);
                self.asm.emit_op(op::ADD);
                self.asm.emit_push(U256::from(EvmMemoryLayout::WORD_SIZE - 1));
                self.asm.emit_op(op::ADD);
                self.asm.emit_push(U256::MAX - U256::from(EvmMemoryLayout::WORD_SIZE - 1));
                self.asm.emit_op(op::AND);
                self.asm.emit_push(U256::from(EvmMemoryLayout::FMP_SLOT));
                self.asm.emit_op(op::MSTORE);
            } else {
                self.asm.emit_push_deferred(constructor_fixed_memory_end);
                self.asm.emit_push(U256::from(EvmMemoryLayout::FMP_SLOT));
                self.asm.emit_op(op::MSTORE);
            }

            if !internal_targets.is_empty() {
                let constructor_entry = self.asm.new_label();
                self.asm.emit_push_label(constructor_entry);
                self.asm.emit_op(op::JUMP);

                for (func_id, func) in module.functions.iter_enumerated() {
                    if !internal_targets.contains(func_id) {
                        continue;
                    }
                    let label = self.function_labels[&func_id];
                    self.asm.define_label(label);
                    self.in_internal_function = true;
                    self.generate_function_body(func_id, func);
                    self.in_internal_function = false;
                    self.record_function_spill_size(func_id);
                }

                self.asm.define_label(constructor_entry);
            }

            // Generate the constructor body (which includes SSTORE for
            // initializers). Every ordinary completion jumps to one label so
            // branch layout cannot strand the deployment postlude behind a
            // non-final STOP.
            let constructor_exit = self.asm.new_label();
            self.constructor_exit = Some(constructor_exit);
            self.generate_function_body(ctor_id, ctor);
            let constructor_spill_size = self.record_function_spill_size(ctor_id);
            self.asm.set_deferred_const(
                constructor_fixed_memory_end,
                U256::from(self.constructor_fixed_memory_end(
                    module.immutable_count(),
                    constructor_spill_size,
                )),
            );

            self.resolve_pending_frame_size_consts(module);

            // Reset constructor context
            self.in_constructor = false;
            self.constructor_args_base_const = None;
            self.constructor_exit = None;
            self.constructor_param_count = 0;

            self.asm.define_label(constructor_exit);
            constructor_arg_offset
        } else {
            None
        };

        self.emit_deployment_postlude(
            module,
            runtime_offset,
            runtime_len,
            copy_base,
            immutable_refs,
        );
        PreparedDeploymentPrefix {
            assembly: self.asm.prepare(self.capture_evm_ir),
            constructor_arg_offset,
            runtime_offset,
        }
    }

    fn assemble_deployment_prefix(
        &mut self,
        prepared: &PreparedDeploymentPrefix,
        constructor_arg_offset: usize,
        runtime_offset: usize,
    ) -> GeneratedCode {
        let mut deferred_values = Vec::with_capacity(2);
        if let Some(id) = prepared.constructor_arg_offset {
            deferred_values.push((id, U256::from(constructor_arg_offset)));
        }
        deferred_values.push((prepared.runtime_offset, U256::from(runtime_offset)));
        let result = self.asm.assemble_prepared(&prepared.assembly, &deferred_values);
        GeneratedCode { bytecode: result.bytecode, evm_ir: result.evm_ir }
    }

    /// Runs the canonical MIR optimization pipeline on the module.
    fn run_optimization_passes(&mut self, module: &mut Module) {
        let _changed = run_pipeline(self.gcx, module, None);
    }

    /// Generates runtime bytecode for a module.
    fn generate_runtime_code(&mut self, module: &Module) -> GeneratedCode {
        assert_eq!(
            module.phase,
            MirPhase::EvmShaped,
            "EVM codegen requires MIR in the final phase"
        );
        let mut preserve_caller_stack = self.gcx.sess.opts.optimization.is_gas();
        loop {
            self.reset_runtime_codegen(module);
            self.preserve_caller_stack = preserve_caller_stack;

            if !module.functions.is_empty() {
                self.emit_runtime(module);
            }

            if preserve_caller_stack
                && !self.internal_call_stack_edges.is_empty()
                && !self.caller_stack_prefixes_fit(module)
            {
                preserve_caller_stack = false;
                continue;
            }
            break;
        }

        let result = self.asm.assemble_with_evm_ir(self.capture_evm_ir);
        self.runtime_immutable_refs = result.immutable_refs;
        GeneratedCode { bytecode: result.bytecode, evm_ir: result.evm_ir }
    }

    fn reset_runtime_codegen(&mut self, module: &Module) {
        self.asm.clear();
        self.asm.set_artifact_kind(ArtifactKind::Runtime);
        self.block_labels.clear();
        self.function_labels.clear();
        self.cold_functions = if matches!(self.gcx.sess.opts.optimization, OptimizationMode::None) {
            DenseBitSet::new_empty(module.functions.len())
        } else {
            Self::collect_cold_functions(module)
        };
        self.empty_stop_functions = DenseBitSet::new_empty(module.functions.len());
        self.function_spill_sizes.clear();
        self.pending_frame_size_consts.clear();
        self.restorable_internal_frames = DenseBitSet::new_empty(module.functions.len());
        self.static_frame_functions = DenseBitSet::new_empty(module.functions.len());
        self.static_frame_addr_consts.clear();
        self.external_spill_addr_consts.clear();
        self.pending_static_allocs.clear();
        self.runtime_free_memory_const = None;
        self.runtime_entry_funcs.clear();
        self.current_internal_function = None;
        self.block_copies.clear();
        self.stack_phi_sources.clear();
        self.stack_arg_masks.clear();
        self.direct_stack_args.clear();
        self.lazy_stack_args.clear();
        self.stack_return_functions = DenseBitSet::new_empty(module.functions.len());
        self.stack_return_local_bases.clear();
        self.recursive_stack_functions = DenseBitSet::new_empty(module.functions.len());
        self.recursion_reaching_functions = DenseBitSet::new_empty(module.functions.len());
        self.function_stack_peaks.clear();
        self.internal_call_stack_edges.clear();
        self.runtime_stack_args = true;
        self.emitting_entry = false;
        self.reset_switch_gas_code_growth();
    }

    /// Validates the complete physical stack, including words intentionally
    /// hidden below each function's scheduler model. The local high-water
    /// marks are exact for the emitted bodies; call-edge propagation is
    /// conservative for tail calls, which may carry any locally observed
    /// stack into their target. Recursive regions are excluded from the
    /// optimization before emission because their incoming prefix is
    /// intentionally unbounded.
    fn caller_stack_prefixes_fit(&self, module: &Module) -> bool {
        if !self.internal_call_stack_edges.iter().any(|edge| edge.preserved_words != 0) {
            return true;
        }
        let Some(entry_id) = module
            .functions
            .iter_enumerated()
            .find_map(|(func_id, func)| func.attributes.is_dispatch_entry.then_some(func_id))
        else {
            return true;
        };

        let mut incoming: IndexVec<FunctionId, Option<usize>> =
            index_vec![None; module.functions.len()];
        incoming[entry_id] = Some(0);
        for _ in 0..module.functions.len() {
            let mut changed = false;
            for edge in &self.internal_call_stack_edges {
                if self.recursive_stack_functions.contains(edge.caller)
                    || self.recursive_stack_functions.contains(edge.callee)
                    || !self.function_stack_peaks.contains_key(&edge.callee)
                {
                    continue;
                }
                let Some(base) = incoming[edge.caller] else { continue };
                let candidate = base.saturating_add(edge.preserved_words).saturating_add(1);
                // Before JUMP consumes its destination, the caller briefly
                // holds both that label and the callee return address.
                if candidate.saturating_add(1) > MAX_STACK_DEPTH {
                    return false;
                }
                if incoming[edge.callee].is_none_or(|current| candidate > current) {
                    incoming[edge.callee] = Some(candidate);
                    changed = true;
                }
            }

            for (caller, func) in module.functions.iter_enumerated() {
                if self.recursive_stack_functions.contains(caller) {
                    continue;
                }
                let Some(base) = incoming[caller] else { continue };
                let carried = self.function_stack_peaks.get(&caller).copied().unwrap_or(0);
                for block in &func.blocks {
                    let Some(Terminator::TailCall { function: callee, .. }) = &block.terminator
                    else {
                        continue;
                    };
                    if self.recursive_stack_functions.contains(*callee)
                        || !self.function_stack_peaks.contains_key(callee)
                    {
                        continue;
                    }
                    let candidate = base.saturating_add(carried);
                    // Tail calls carry the caller stack and briefly push only
                    // the target label; they do not add a return address.
                    if candidate.saturating_add(1) > MAX_STACK_DEPTH {
                        return false;
                    }
                    if incoming[*callee].is_none_or(|current| candidate > current) {
                        incoming[*callee] = Some(candidate);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        incoming.iter_enumerated().all(|(func_id, incoming)| {
            incoming.is_none_or(|incoming| {
                incoming
                    .saturating_add(self.function_stack_peaks.get(&func_id).copied().unwrap_or(0))
                    <= MAX_STACK_DEPTH
            })
        })
    }

    /// Emits a runtime from final-phase MIR.
    ///
    /// Selector matching, receive/fallback routing, and callvalue checks all
    /// live in the MIR `entry`, whose `tail_call`s jump to the ABI wrappers.
    fn emit_runtime(&mut self, module: &Module) {
        let Some((entry_id, _)) =
            module.functions.iter_enumerated().find(|(_, f)| f.attributes.is_dispatch_entry)
        else {
            assert!(
                !module.functions.iter().any(Self::is_external_entry),
                "evm-shaped module with a runtime interface must have a MIR `entry` function"
            );
            return;
        };

        let call_graph = CallGraphInfo::new(module);
        for (func_id, func) in module.functions.iter_enumerated() {
            if func.blocks.len() == 1
                && func.blocks[BlockId::ENTRY].instructions.is_empty()
                && matches!(func.blocks[BlockId::ENTRY].terminator, Some(Terminator::Stop))
            {
                self.empty_stop_functions.insert(func_id);
            }
            if call_graph.is_recursive(func_id) {
                self.recursive_stack_functions.insert(func_id);
                self.recursive_stack_functions.union(&call_graph.reachable_callees_from([func_id]));
            }
            if call_graph.is_recursive(func_id)
                || call_graph
                    .reachable_callees_from([func_id])
                    .iter()
                    .any(|callee| call_graph.is_recursive(callee))
            {
                self.recursion_reaching_functions.insert(func_id);
            }
        }
        let internal_targets = call_graph.reachable_callees_from(
            module.functions.iter_enumerated().filter_map(|(func_id, func)| {
                (func_id == entry_id || Self::is_external_entry(func)).then_some(func_id)
            }),
        );

        for (func_id, func) in module.functions.iter_enumerated() {
            if !func.params.iter().chain(&func.returns).any(|ty| ty.is_memory_reference()) {
                self.restorable_internal_frames.insert(func_id);
            }
            // Non-recursive internal functions get compile-time-fixed frames.
            if func_id != entry_id
                && !Self::is_external_entry(func)
                && Self::is_runtime_function(func)
                && !call_graph.is_recursive(func_id)
            {
                self.static_frame_functions.insert(func_id);
            }
        }
        self.compute_stack_arg_masks(module);
        self.compute_lazy_stack_args(module);
        self.compute_direct_stack_args(module);
        self.compute_stack_return_functions(module);

        // Labels for every tail-call and internal-call target.
        for (func_id, func) in module.functions.iter_enumerated() {
            if func_id == entry_id {
                continue;
            }
            let needs_body = Self::is_external_entry(func)
                || (Self::is_runtime_function(func) && internal_targets.contains(func_id));
            if needs_body {
                let label = self.new_function_label(func_id);
                self.function_labels.insert(func_id, label);
            }
        }

        // The MIR entry is the runtime prologue: one shared free-memory
        // store covers every wrapper reached through it.
        self.in_internal_function = false;
        self.emitting_entry = true;
        let entry_free = self.emit_external_free_memory_start();
        self.runtime_free_memory_const = Some(entry_free);
        self.generate_function_body(entry_id, &module.functions[entry_id]);
        self.emitting_entry = false;
        self.record_function_spill_size(entry_id);
        self.runtime_entry_funcs.push(entry_id);

        // External entries, reached only through `tail_call` jumps.
        for (func_id, func) in module.functions.iter_enumerated() {
            if func_id == entry_id || !Self::is_external_entry(func) {
                continue;
            }
            let Some(&label) = self.function_labels.get(&func_id) else { continue };
            self.asm.define_label(label);
            self.in_internal_function = false;
            self.generate_function_body(func_id, func);
            self.record_function_spill_size(func_id);
            self.runtime_entry_funcs.push(func_id);
        }

        // Internal-call targets.
        for (func_id, func) in module.functions.iter_enumerated() {
            if func_id == entry_id
                || Self::is_external_entry(func)
                || !Self::is_runtime_function(func)
            {
                continue;
            }
            let Some(&label) = self.function_labels.get(&func_id) else { continue };
            self.asm.define_label(label);
            self.emit_stack_arg_prologue(func_id, func);
            self.in_internal_function = true;
            self.current_internal_function = Some(func_id);
            self.generate_function_body(func_id, func);
            self.in_internal_function = false;
            self.current_internal_function = None;
            self.record_function_spill_size(func_id);
        }

        self.resolve_pending_frame_size_consts(module);
        self.resolve_static_frames(module);
    }

    /// Records the exact spill area size of the function body that just emitted.
    fn record_function_spill_size(&mut self, func_id: FunctionId) -> u64 {
        let spill_size = u64::from(self.scheduler.spills.spill_area_size());
        self.function_spill_sizes.insert(func_id, spill_size);
        spill_size
    }

    /// Returns the exact spill area recorded for `func_id` after emission.
    fn function_spill_size(&self, func_id: FunctionId) -> u64 {
        self.function_spill_sizes.get(&func_id).copied().unwrap_or_else(|| {
            panic!("spill size for emitted function {func_id:?} was not recorded")
        })
    }

    /// Resolves all pending internal-call frame-size constants.
    ///
    /// Every pending constant belongs to a labeled callee. Runtime and
    /// constructor emission record all labeled bodies before reaching this
    /// resolution point.
    fn resolve_pending_frame_size_consts(&mut self, module: &Module) {
        for (id, callee) in std::mem::take(&mut self.pending_frame_size_consts) {
            self.asm.set_deferred_const(id, U256::from(self.emitted_frame_size(module, callee)));
        }
    }

    fn is_external_entry(func: &Function) -> bool {
        Self::is_runtime_function(func)
            && (func.selector.is_some()
                || func.attributes.is_receive
                || func.attributes.is_fallback)
    }

    fn is_runtime_function(func: &Function) -> bool {
        !func.attributes.is_constructor
    }

    /// Replays exact semantic-pressure alternatives through the complete physical emitter.
    ///
    /// Each replay starts from the identical backend state immediately before this function body
    /// and runs through the ordinary liveness, spill, operand-planning, and block-exit paths. The
    /// cloned state is discarded unless `emit_winners` is set, in which case winning region orders
    /// are returned in a cloned function for ordinary backend emission.
    fn audit_physical_schedules(
        &mut self,
        func_id: FunctionId,
        func: &Function,
        emit_winners: bool,
    ) -> Option<Function> {
        let report = tracing::enabled!(target: PHYSICAL_REPLAY_TARGET, tracing::Level::DEBUG);
        let candidates = if emit_winners && !report {
            bounded_physical_schedule_candidates(func, EvmInstSchedule::is_movable)
        } else {
            physical_schedule_candidates(func, EvmInstSchedule::is_movable)
        };
        if candidates.is_empty() {
            return None;
        }
        if emit_winners && !report {
            return self.select_compact_physical_schedules(func_id, func, &candidates);
        }

        let baseline = self.replay_physical_schedule(func_id, func);
        let mut fewer_instructions = 0;
        let mut fewer_stack_ops = 0;
        let mut fewer_spill_bytes = 0;
        let mut lower_peak_stack = 0;
        let mut scores = Vec::with_capacity(candidates.len());

        for (index, candidate) in candidates.iter().enumerate() {
            let mut alternative = func.clone();
            alternative.blocks[candidate.block].instructions[candidate.range.clone()]
                .copy_from_slice(&candidate.order);
            let score = self.replay_physical_schedule(func_id, &alternative);
            fewer_instructions += usize::from(score.instructions < baseline.instructions);
            fewer_stack_ops += usize::from(
                score.dups + score.swaps + score.pops
                    < baseline.dups + baseline.swaps + baseline.pops,
            );
            fewer_spill_bytes += usize::from(score.spill_bytes < baseline.spill_bytes);
            lower_peak_stack += usize::from(score.peak_stack < baseline.peak_stack);
            scores.push(score);

            tracing::debug!(
                target: PHYSICAL_REPLAY_TARGET,
                function = %func.name,
                candidate = index,
                candidate_kind = ?candidate.kind,
                exhaustive = candidate.exhaustive,
                region_orders = candidate.region_orders,
                block = candidate.block.index(),
                range_start = candidate.range.start,
                range_end = candidate.range.end,
                current_mir_peak = candidate.current_peak,
                candidate_mir_peak = candidate.candidate_peak,
                current_mir_area = candidate.current_area,
                candidate_mir_area = candidate.candidate_area,
                baseline = ?baseline,
                candidate_score = ?score,
                "physical replay of semantic-pressure candidate"
            );
        }

        let optimization = self.gcx.sess.opts.optimization;
        let mut exhaustive_regions = 0;
        let mut dfs_suboptimal = 0;
        let mut minimum_pressure_suboptimal = 0;
        let mut optimum_above_minimum_pressure = 0;
        let mut gas_regret = 0;
        let mut byte_regret = 0;
        let mut beam_census = FxHashMap::<(Option<usize>, usize), PhysicalBeamCensus>::default();
        let mut emitted = emit_winners.then(|| func.clone());
        let mut emitted_changed = false;
        let mut start = 0;
        while start < candidates.len() {
            let candidate = &candidates[start];
            let mut end = start + 1;
            while end < candidates.len()
                && candidates[end].block == candidate.block
                && candidates[end].range == candidate.range
            {
                end += 1;
            }
            if candidate.exhaustive {
                exhaustive_regions += 1;
                let minimum_pressure = std::iter::once(candidate.current_peak)
                    .chain(candidates[start..end].iter().map(|candidate| candidate.candidate_peak))
                    .min()
                    .unwrap();
                let mut best_score = baseline;
                let mut best_pressure = candidate.current_peak;
                let mut best_kind = "dfs";
                let mut minimum_pressure_score =
                    (candidate.current_peak == minimum_pressure).then_some(baseline);

                for index in start..end {
                    let candidate_score = scores[index];
                    let candidate_pressure = candidates[index].candidate_peak;
                    if candidate_score.key(optimization) < best_score.key(optimization) {
                        best_score = candidate_score;
                        best_pressure = candidate_pressure;
                        best_kind = "alternative";
                    } else if candidate_score.key(optimization) == best_score.key(optimization)
                        && candidate_pressure < best_pressure
                    {
                        best_pressure = candidate_pressure;
                        best_kind = "alternative-tie";
                    }
                    if candidate_pressure == minimum_pressure
                        && minimum_pressure_score.is_none_or(|score| {
                            candidate_score.key(optimization) < score.key(optimization)
                        })
                    {
                        minimum_pressure_score = Some(candidate_score);
                    }
                }

                let minimum_pressure_score = minimum_pressure_score.unwrap();
                dfs_suboptimal +=
                    usize::from(best_score.key(optimization) < baseline.key(optimization));
                minimum_pressure_suboptimal += usize::from(
                    best_score.key(optimization) < minimum_pressure_score.key(optimization),
                );
                optimum_above_minimum_pressure += usize::from(best_pressure > minimum_pressure);
                gas_regret +=
                    baseline.estimated_static_gas.saturating_sub(best_score.estimated_static_gas);
                byte_regret += baseline.estimated_bytes.saturating_sub(best_score.estimated_bytes);

                tracing::debug!(
                    target: PHYSICAL_REPLAY_TARGET,
                    function = %func.name,
                    block = candidate.block.index(),
                    range_start = candidate.range.start,
                    range_end = candidate.range.end,
                    topological_orders = candidate.region_orders,
                    minimum_pressure,
                    best_pressure,
                    best_kind,
                    dfs_score = ?baseline,
                    minimum_pressure_score = ?minimum_pressure_score,
                    best_score = ?best_score,
                    "exhaustive physical scheduling result"
                );

                let expected = report.then(|| {
                    let beam_results = self.audit_physical_beam_region(
                        func_id,
                        func,
                        candidate,
                        &candidates[start..end],
                        &scores[start..end],
                        baseline,
                        optimization,
                    );
                    let expected = beam_results
                        .iter()
                        .find(|result| result.slack_width == Some(2) && result.lookahead == 2)
                        .expect("the frozen reconstruction reference must run")
                        .winner;
                    for result in beam_results {
                        beam_census
                            .entry((result.slack_width, result.lookahead))
                            .or_default()
                            .record(result);
                    }
                    self.audit_persistent_physical_beam_region(
                        func_id,
                        func,
                        candidate,
                        &candidates[start..end],
                        &scores[start..end],
                        baseline,
                        optimization,
                        expected,
                    );
                    expected
                });
                if let Some(order) = self.audit_compact_physical_beam_region(
                    func_id,
                    func,
                    candidate,
                    &candidates[start..end],
                    Some(&scores[start..end]),
                    Some(baseline),
                    optimization,
                    expected,
                    None,
                    None,
                ) && let Some(emitted) = &mut emitted
                {
                    emitted.blocks[candidate.block].instructions[candidate.range.clone()]
                        .copy_from_slice(&order);
                    emitted_changed = true;
                }
            }
            start = end;
        }

        for ((slack_width, lookahead), census) in beam_census {
            tracing::debug!(
                target: PHYSICAL_REPLAY_TARGET,
                function = %func.name,
                beam = slack_width.map_or("ordinary-8".to_owned(), |width| format!("8+{width}")),
                lookahead,
                regions = census.regions,
                optimum_hits = census.optimum_hits,
                gas_regret = census.gas_regret,
                byte_regret = census.byte_regret,
                worst_gas_regret = census.worst_gas_regret,
                worst_byte_regret = census.worst_byte_regret,
                states_expanded = census.states_expanded,
                backend_transactions = census.backend_transactions,
                replay_micros = census.replay_micros,
                first_prunes = census.first_prunes,
                mean_first_prune_rank = census.first_prune_rank_sum.checked_div(census.first_prunes).unwrap_or_default(),
                max_first_prune_rank = census.first_prune_rank_max,
                mean_first_prune_depth = census.first_prune_depth_sum.checked_div(census.first_prunes).unwrap_or_default(),
                "integrated physical beam census"
            );
        }

        tracing::debug!(
            target: PHYSICAL_REPLAY_TARGET,
            function = %func.name,
            candidates = candidates.len(),
            fewer_instructions,
            fewer_stack_ops,
            fewer_spill_bytes,
            lower_peak_stack,
            exhaustive_regions,
            dfs_suboptimal,
            minimum_pressure_suboptimal,
            optimum_above_minimum_pressure,
            gas_regret,
            byte_regret,
            baseline = ?baseline,
            "physical scheduling replay census"
        );

        let emitted = emitted.filter(|_| emitted_changed)?;
        // Region-local winners can change the physical entry state of later regions. Re-run the
        // combined function and require a Pareto improvement before exposing it to real emission.
        let combined = self.replay_physical_schedule(func_id, &emitted);
        let dominates_baseline = combined.estimated_static_gas <= baseline.estimated_static_gas
            && combined.estimated_bytes <= baseline.estimated_bytes
            && combined.spill_bytes <= baseline.spill_bytes;
        (dominates_baseline && combined.key(optimization) < baseline.key(optimization))
            .then_some(emitted)
    }

    /// Selects compact physical-beam winners without exhaustively replaying every legal order.
    fn select_compact_physical_schedules(
        &mut self,
        func_id: FunctionId,
        func: &Function,
        candidates: &[PhysicalScheduleCandidate],
    ) -> Option<Function> {
        let optimization = self.gcx.sess.opts.optimization;
        let mut emitted = func.clone();
        let mut changed = false;
        let mut start = 0;
        while start < candidates.len() {
            let candidate = &candidates[start];
            let mut end = start + 1;
            while end < candidates.len()
                && candidates[end].block == candidate.block
                && candidates[end].range == candidate.range
            {
                end += 1;
            }
            if let Some(order) = self.audit_compact_physical_beam_region(
                func_id,
                func,
                candidate,
                &candidates[start..end],
                None,
                None,
                optimization,
                None,
                None,
                None,
            ) {
                emitted.blocks[candidate.block].instructions[candidate.range.clone()]
                    .copy_from_slice(&order);
                changed = true;
            }
            start = end;
        }

        if !changed {
            return None;
        }

        // Compact scores stop at region exits. Certify their combined whole-function effect using
        // the ordinary backend before exposing the chosen order to real emission.
        let baseline = self.replay_physical_schedule(func_id, func);
        let combined = self.replay_physical_schedule(func_id, &emitted);
        let dominates_baseline = combined.estimated_static_gas <= baseline.estimated_static_gas
            && combined.estimated_bytes <= baseline.estimated_bytes
            && combined.spill_bytes <= baseline.spill_bytes;
        (dominates_baseline && combined.key(optimization) < baseline.key(optimization))
            .then_some(emitted)
    }

    fn replay_physical_schedule(
        &self,
        func_id: FunctionId,
        func: &Function,
    ) -> PhysicalReplayScore {
        let mut shadow = self.clone();
        shadow.physical_replay_active = true;
        let start = PhysicalReplayStart::capture(&shadow.asm);
        shadow.generate_function_body(func_id, func);
        shadow.record_function_spill_size(func_id);
        start.score(&shadow)
    }

    fn replay_physical_prefix(
        &self,
        func_id: FunctionId,
        func: &Function,
        block: BlockId,
        stop: usize,
    ) -> PhysicalReplayScore {
        let mut shadow = self.clone();
        shadow.physical_replay_active = true;
        shadow.physical_replay_stop = Some((block, stop));
        let start = PhysicalReplayStart::capture(&shadow.asm);
        shadow.generate_function_body(func_id, func);
        start.score(&shadow)
    }

    /// Creates one resumable backend transaction immediately before a MIR region.
    fn replay_physical_state_until(
        &self,
        func_id: FunctionId,
        func: &Function,
        block: BlockId,
        stop: usize,
    ) -> Self {
        let mut shadow = self.clone();
        shadow.physical_replay_active = true;
        shadow.physical_replay_stop = Some((block, stop));
        shadow.generate_function_body(func_id, func);
        shadow.physical_replay_stop = None;
        shadow
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_persistent_physical_state(
        &self,
        parent: &PersistentPhysicalState<'gcx>,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        next: InstId,
        completion: &[InstId],
        root: &PhysicalReplayStart,
        base_liveness: &Liveness,
    ) -> PersistentPhysicalState<'gcx> {
        let mut backend = parent.backend.clone();
        let mut alternative = func.clone();
        alternative.blocks[region.block].instructions[region.range.clone()]
            .copy_from_slice(completion);
        let mut liveness = base_liveness.clone();
        liveness.recompute_block_last_uses(&alternative, region.block);
        let inst_idx = region.range.start + parent.prefix.len();
        backend.generate_block_inst(func_id, next, &alternative, &liveness, region.block, inst_idx);
        let mut prefix = parent.prefix.clone();
        prefix.push(next);
        let score = root.score(&backend);
        PersistentPhysicalState { prefix, backend, score, pressure: parent.pressure }
    }

    #[allow(clippy::too_many_arguments)]
    fn cached_persistent_physical_child(
        &self,
        parent: &PersistentPhysicalState<'gcx>,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        next: InstId,
        completion: &[InstId],
        root: &PhysicalReplayStart,
        base_liveness: &Liveness,
        cache: &mut FxHashMap<Vec<InstId>, PersistentPhysicalState<'gcx>>,
        transactions: &mut usize,
    ) -> PersistentPhysicalState<'gcx> {
        let mut prefix = parent.prefix.clone();
        prefix.push(next);
        if let Some(state) = cache.get(&prefix) {
            return state.clone();
        }
        *transactions += 1;
        let child = self.advance_persistent_physical_state(
            parent,
            func_id,
            func,
            region,
            next,
            completion,
            root,
            base_liveness,
        );
        cache.insert(prefix, child.clone());
        child
    }

    #[allow(clippy::too_many_arguments)]
    fn persistent_physical_lookahead(
        &self,
        state: &PersistentPhysicalState<'gcx>,
        depth: usize,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        schedules: &[(Vec<InstId>, usize, PhysicalReplayScore)],
        root: &PhysicalReplayStart,
        optimization: OptimizationMode,
        base_liveness: &Liveness,
        cache: &mut FxHashMap<Vec<InstId>, PersistentPhysicalState<'gcx>>,
        transactions: &mut usize,
    ) -> PhysicalReplayScore {
        if state.prefix.len() == region.range.len() {
            return schedules
                .iter()
                .find(|(schedule, _, _)| schedule == &state.prefix)
                .map(|(_, _, score)| *score)
                .unwrap_or(state.score);
        }
        if depth == 0 {
            return state.score;
        }
        let next_depth = state.prefix.len();
        let mut children = FxHashMap::<InstId, usize>::default();
        for (index, (schedule, _, _)) in schedules.iter().enumerate() {
            if schedule.starts_with(&state.prefix) {
                children.entry(schedule[next_depth]).or_insert(index);
            }
        }
        children
            .into_iter()
            .map(|(next, completion)| {
                let child = self.cached_persistent_physical_child(
                    state,
                    func_id,
                    func,
                    region,
                    next,
                    &schedules[completion].0,
                    root,
                    base_liveness,
                    cache,
                    transactions,
                );
                self.persistent_physical_lookahead(
                    &child,
                    depth - 1,
                    func_id,
                    func,
                    region,
                    schedules,
                    root,
                    optimization,
                    base_liveness,
                    cache,
                    transactions,
                )
            })
            .min_by_key(|score| score.key(optimization))
            .unwrap_or(state.score)
    }

    fn compact_physical_root(&self) -> CompactPhysicalRoot {
        let block = self.asm.current_block.expect("region replay must have an active EVM block");
        CompactPhysicalRoot {
            block,
            instructions: self.asm.program.blocks[block].instructions.len(),
            label_relocations: self.asm.label_relocations.len(),
            deferred_relocations: self.asm.deferred_relocations.len(),
            indexed_jump_relocations: self.asm.indexed_jump_relocations.len(),
            alloc_relocations: self.asm.alloc_relocations.len(),
        }
    }

    fn capture_compact_physical_state(
        &self,
        root: CompactPhysicalRoot,
        score_root: &PhysicalReplayStart,
        prefix: Vec<InstId>,
        pressure: usize,
    ) -> CompactPhysicalState {
        CompactPhysicalState {
            prefix,
            scheduler: self.scheduler.clone(),
            spill_addr_consts: self.spill_addr_consts.clone(),
            instructions: self.asm.program.blocks[root.block].instructions[root.instructions..]
                .to_vec(),
            label_relocations: self.asm.label_relocations[root.label_relocations..].to_vec(),
            deferred_relocations: self.asm.deferred_relocations[root.deferred_relocations..]
                .to_vec(),
            indexed_jump_relocations: self.asm.indexed_jump_relocations
                [root.indexed_jump_relocations..]
                .to_vec(),
            alloc_relocations: self.asm.alloc_relocations[root.alloc_relocations..].to_vec(),
            score: score_root.score(self),
            pressure,
        }
    }

    fn restore_compact_physical_state(
        &mut self,
        root: CompactPhysicalRoot,
        state: &CompactPhysicalState,
    ) {
        self.scheduler.clone_from(&state.scheduler);
        self.spill_addr_consts.clone_from(&state.spill_addr_consts);
        let instructions = &mut self.asm.program.blocks[root.block].instructions;
        instructions.truncate(root.instructions);
        instructions.extend_from_slice(&state.instructions);
        self.asm.label_relocations.truncate(root.label_relocations);
        self.asm.label_relocations.extend_from_slice(&state.label_relocations);
        self.asm.deferred_relocations.truncate(root.deferred_relocations);
        self.asm.deferred_relocations.extend_from_slice(&state.deferred_relocations);
        self.asm.indexed_jump_relocations.truncate(root.indexed_jump_relocations);
        self.asm.indexed_jump_relocations.extend_from_slice(&state.indexed_jump_relocations);
        self.asm.alloc_relocations.truncate(root.alloc_relocations);
        self.asm.alloc_relocations.extend_from_slice(&state.alloc_relocations);
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_compact_physical_state(
        &mut self,
        parent: &CompactPhysicalState,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        next: InstId,
        completion: &[InstId],
        root: CompactPhysicalRoot,
        score_root: &PhysicalReplayStart,
        liveness: &mut Liveness,
    ) -> CompactPhysicalState {
        self.restore_compact_physical_state(root, parent);
        liveness.recompute_block_last_uses_with_region(
            func,
            region.block,
            region.range.clone(),
            completion,
        );
        let inst_idx = region.range.start + parent.prefix.len();
        self.generate_block_inst(func_id, next, func, liveness, region.block, inst_idx);
        let mut prefix = parent.prefix.clone();
        prefix.push(next);
        self.capture_compact_physical_state(root, score_root, prefix, parent.pressure)
    }

    #[allow(clippy::too_many_arguments)]
    fn cached_compact_physical_child(
        &mut self,
        parent: &CompactPhysicalState,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        next: InstId,
        completion: &[InstId],
        root: CompactPhysicalRoot,
        score_root: &PhysicalReplayStart,
        liveness: &mut Liveness,
        cache: &mut FxHashMap<Vec<InstId>, CompactPhysicalState>,
        transactions: &mut usize,
    ) -> CompactPhysicalState {
        let mut prefix = parent.prefix.clone();
        prefix.push(next);
        if let Some(state) = cache.get(&prefix) {
            return state.clone();
        }
        *transactions += 1;
        let child = self.advance_compact_physical_state(
            parent, func_id, func, region, next, completion, root, score_root, liveness,
        );
        cache.insert(prefix, child.clone());
        child
    }

    #[allow(clippy::too_many_arguments)]
    fn compact_physical_block_suffix_score(
        &mut self,
        state: &CompactPhysicalState,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        completion: &[InstId],
        root: CompactPhysicalRoot,
        score_root: &PhysicalReplayStart,
        liveness: &mut Liveness,
    ) -> PhysicalReplayScore {
        self.restore_compact_physical_state(root, state);
        liveness.recompute_block_last_uses_with_region(
            func,
            region.block,
            region.range.clone(),
            completion,
        );
        let suffix = &func.blocks[region.block].instructions[region.range.end..];
        for (offset, &inst_id) in suffix.iter().enumerate() {
            if matches!(func.inst(inst_id).kind, InstKind::Phi(_)) {
                continue;
            }
            self.generate_block_inst(
                func_id,
                inst_id,
                func,
                liveness,
                region.block,
                region.range.end + offset,
            );
        }
        score_root.score(self)
    }

    #[allow(clippy::too_many_arguments)]
    fn compact_physical_lookahead(
        &mut self,
        state: &CompactPhysicalState,
        depth: usize,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        schedules: &[(Vec<InstId>, usize, Option<PhysicalReplayScore>)],
        root: CompactPhysicalRoot,
        score_root: &PhysicalReplayStart,
        optimization: OptimizationMode,
        liveness: &mut Liveness,
        cache: &mut FxHashMap<Vec<InstId>, CompactPhysicalState>,
        transactions: &mut usize,
    ) -> PhysicalReplayScore {
        if state.prefix.len() == region.range.len() {
            return schedules
                .iter()
                .find(|(schedule, _, _)| schedule == &state.prefix)
                .and_then(|(_, _, score)| *score)
                .unwrap_or(state.score);
        }
        if depth == 0 {
            return state.score;
        }
        let next_depth = state.prefix.len();
        let mut children = FxHashMap::<InstId, usize>::default();
        for (index, (schedule, _, _)) in schedules.iter().enumerate() {
            if schedule.starts_with(&state.prefix) {
                children.entry(schedule[next_depth]).or_insert(index);
            }
        }
        children
            .into_iter()
            .map(|(next, completion)| {
                let child = self.cached_compact_physical_child(
                    state,
                    func_id,
                    func,
                    region,
                    next,
                    &schedules[completion].0,
                    root,
                    score_root,
                    liveness,
                    cache,
                    transactions,
                );
                self.compact_physical_lookahead(
                    &child,
                    depth - 1,
                    func_id,
                    func,
                    region,
                    schedules,
                    root,
                    score_root,
                    optimization,
                    liveness,
                    cache,
                    transactions,
                )
            })
            .min_by_key(|score| score.key(optimization))
            .unwrap_or(state.score)
    }

    #[allow(clippy::too_many_arguments)]
    fn cached_physical_prefix_score(
        &self,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        schedule: &[InstId],
        prefix: &[InstId],
        cache: &mut FxHashMap<Vec<InstId>, (PhysicalReplayScore, u64)>,
        seen: &mut FxHashSet<Vec<InstId>>,
        backend_transactions: &mut usize,
        replay_micros: &mut u64,
    ) -> PhysicalReplayScore {
        let (score, micros) = if let Some(&(score, micros)) = cache.get(prefix) {
            (score, micros)
        } else {
            let mut alternative = func.clone();
            alternative.blocks[region.block].instructions[region.range.clone()]
                .copy_from_slice(schedule);
            let started = Instant::now();
            let score = self.replay_physical_prefix(
                func_id,
                &alternative,
                region.block,
                region.range.start + prefix.len(),
            );
            let micros = started.elapsed().as_micros().try_into().unwrap_or(u64::MAX);
            cache.insert(prefix.to_vec(), (score, micros));
            (score, micros)
        };
        if seen.insert(prefix.to_vec()) {
            *backend_transactions += 1;
            *replay_micros += micros;
        }
        score
    }

    #[allow(clippy::too_many_arguments)]
    fn audit_physical_beam_region(
        &self,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        candidates: &[PhysicalScheduleCandidate],
        scores: &[PhysicalReplayScore],
        baseline: PhysicalReplayScore,
        optimization: OptimizationMode,
    ) -> Vec<PhysicalBeamResult> {
        let current = func.blocks[region.block].instructions[region.range.clone()].to_vec();
        let mut schedules = Vec::with_capacity(candidates.len() + 1);
        schedules.push((current, region.current_peak, baseline));
        schedules.extend(
            candidates.iter().zip(scores).map(|(candidate, &score)| {
                (candidate.order.clone(), candidate.candidate_peak, score)
            }),
        );

        let minimum_pressure = schedules.iter().map(|(_, pressure, _)| *pressure).min().unwrap();
        let physical_optimum = schedules
            .iter()
            .map(|(_, _, score)| *score)
            .min_by_key(|score| score.key(optimization))
            .unwrap();
        let instruction_count = region.range.len();
        let mut shared_prefix_scores =
            FxHashMap::<Vec<InstId>, (PhysicalReplayScore, u64)>::default();
        let mut results = Vec::new();
        let configurations =
            std::iter::once(None).chain(PHYSICAL_STRATIFIED_SLACK_WIDTHS.into_iter().map(Some));

        for slack_width in configurations {
            for lookahead in PHYSICAL_LOOKAHEAD_DEPTHS {
                let mut beam = vec![Vec::<InstId>::new()];
                let mut states_expanded = 0;
                let mut backend_transactions = 0;
                let mut replay_micros = 0;
                let mut seen_prefixes = FxHashSet::default();
                let mut first_prune_rank = None;
                let mut first_prune_depth = None;

                for depth in 0..instruction_count {
                    let mut children = FxHashMap::<Vec<InstId>, usize>::default();
                    for prefix in &beam {
                        for (schedule_index, (schedule, pressure, _)) in
                            schedules.iter().enumerate()
                        {
                            if schedule.starts_with(prefix) && *pressure <= minimum_pressure + 1 {
                                let mut child = prefix.clone();
                                child.push(schedule[depth]);
                                children.entry(child).or_insert(schedule_index);
                            }
                        }
                    }
                    states_expanded += children.len();

                    let mut next = children
                        .into_iter()
                        .map(|(prefix, completion)| {
                            let score = if prefix.len() == instruction_count {
                                schedules[completion].2
                            } else {
                                self.cached_physical_prefix_score(
                                    func_id,
                                    func,
                                    region,
                                    &schedules[completion].0,
                                    &prefix,
                                    &mut shared_prefix_scores,
                                    &mut seen_prefixes,
                                    &mut backend_transactions,
                                    &mut replay_micros,
                                )
                            };
                            let target_depth = (prefix.len() + lookahead).min(instruction_count);
                            let rank_score = schedules
                                .iter()
                                .filter(|(schedule, pressure, _)| {
                                    *pressure <= minimum_pressure + 1
                                        && schedule.starts_with(&prefix)
                                })
                                .map(|(schedule, _, terminal_score)| {
                                    if target_depth == instruction_count {
                                        *terminal_score
                                    } else {
                                        self.cached_physical_prefix_score(
                                            func_id,
                                            func,
                                            region,
                                            schedule,
                                            &schedule[..target_depth],
                                            &mut shared_prefix_scores,
                                            &mut seen_prefixes,
                                            &mut backend_transactions,
                                            &mut replay_micros,
                                        )
                                    }
                                })
                                .min_by_key(|score| score.key(optimization))
                                .unwrap_or(score);
                            let pressure = schedules
                                .iter()
                                .filter(|(schedule, _, _)| schedule.starts_with(&prefix))
                                .map(|(_, pressure, _)| *pressure)
                                .min()
                                .unwrap();
                            (prefix, score, rank_score, pressure)
                        })
                        .collect::<Vec<_>>();
                    next.sort_by(|(prefix_a, _, rank_a, _), (prefix_b, _, rank_b, _)| {
                        rank_a
                            .key(optimization)
                            .cmp(&rank_b.key(optimization))
                            .then_with(|| prefix_a.cmp(prefix_b))
                    });

                    let optimal_rank = next.iter().position(|(prefix, _, _, _)| {
                        schedules.iter().any(|(schedule, _, score)| {
                            score.key(optimization) == physical_optimum.key(optimization)
                                && schedule.starts_with(prefix)
                        })
                    });
                    let mut retained = if let Some(slack_width) = slack_width {
                        let mut core = next
                            .iter()
                            .filter(|(_, _, _, pressure)| *pressure == minimum_pressure)
                            .take(8)
                            .cloned()
                            .collect::<Vec<_>>();
                        core.extend(
                            next.iter()
                                .filter(|(_, _, _, pressure)| *pressure == minimum_pressure + 1)
                                .take(slack_width)
                                .cloned(),
                        );
                        core
                    } else {
                        next.iter().take(8).cloned().collect()
                    };
                    if first_prune_rank.is_none()
                        && let Some(rank) = optimal_rank
                        && !retained.iter().any(|(prefix, _, _, _)| {
                            schedules.iter().any(|(schedule, _, score)| {
                                score.key(optimization) == physical_optimum.key(optimization)
                                    && schedule.starts_with(prefix)
                            })
                        })
                    {
                        first_prune_rank = Some(rank + 1);
                        first_prune_depth = Some(depth + 1);
                        let cutoff = retained.last().map(|(_, score, _, _)| *score);
                        tracing::trace!(
                            target: PHYSICAL_REPLAY_TARGET,
                            function = %func.name,
                            block = region.block.index(),
                            range_start = region.range.start,
                            range_end = region.range.end,
                            beam = slack_width.map_or("ordinary-8".to_owned(), |width| format!("8+{width}")),
                            lookahead,
                            prune_depth = depth + 1,
                            optimal_rank = rank + 1,
                            cutoff_score = ?cutoff,
                            "physical optimum first pruned"
                        );
                    }
                    beam = retained.drain(..).map(|(prefix, _, _, _)| prefix).collect();
                }

                let winner = beam
                    .iter()
                    .filter_map(|order| {
                        schedules
                            .iter()
                            .find(|(schedule, _, _)| schedule == order)
                            .map(|(_, _, score)| *score)
                    })
                    .min_by_key(|score| score.key(optimization))
                    .expect("every complete beam order must be an exhaustive schedule");
                let result = PhysicalBeamResult {
                    slack_width,
                    lookahead,
                    optimum_hit: winner.key(optimization) == physical_optimum.key(optimization),
                    gas_regret: winner
                        .estimated_static_gas
                        .saturating_sub(physical_optimum.estimated_static_gas),
                    byte_regret: winner
                        .estimated_bytes
                        .saturating_sub(physical_optimum.estimated_bytes),
                    states_expanded,
                    backend_transactions,
                    replay_micros,
                    first_prune_rank,
                    first_prune_depth,
                    winner,
                };
                tracing::debug!(
                    target: PHYSICAL_REPLAY_TARGET,
                    function = %func.name,
                    block = region.block.index(),
                    range_start = region.range.start,
                    range_end = region.range.end,
                    beam = slack_width.map_or("ordinary-8".to_owned(), |width| format!("8+{width}")),
                    lookahead,
                    optimum_hit = result.optimum_hit,
                    gas_regret = result.gas_regret,
                    byte_regret = result.byte_regret,
                    first_prune_rank = ?result.first_prune_rank,
                    first_prune_depth = ?result.first_prune_depth,
                    "physical beam region result"
                );
                results.push(result);
            }
        }
        results
    }

    /// Runs the frozen `8+2/h2` policy with resumable child backend transactions.
    ///
    /// This remains analysis-only. It measures the production-shaped execution strategy against
    /// the exhaustive oracle before it is allowed to select emitted code.
    #[allow(clippy::too_many_arguments)]
    fn audit_persistent_physical_beam_region(
        &self,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        candidates: &[PhysicalScheduleCandidate],
        scores: &[PhysicalReplayScore],
        baseline: PhysicalReplayScore,
        optimization: OptimizationMode,
        expected: PhysicalReplayScore,
    ) {
        let current = func.blocks[region.block].instructions[region.range.clone()].to_vec();
        let mut schedules = Vec::with_capacity(candidates.len() + 1);
        schedules.push((current, region.current_peak, baseline));
        schedules.extend(
            candidates.iter().zip(scores).map(|(candidate, &score)| {
                (candidate.order.clone(), candidate.candidate_peak, score)
            }),
        );
        let minimum_pressure = schedules.iter().map(|(_, pressure, _)| *pressure).min().unwrap();
        schedules.retain(|(_, pressure, _)| *pressure <= minimum_pressure + 1);
        let physical_optimum = schedules
            .iter()
            .map(|(_, _, score)| *score)
            .min_by_key(|score| score.key(optimization))
            .unwrap();

        let started = Instant::now();
        let base_liveness = self
            .emitting_entry
            .then(|| Liveness::compute_block_local_for_codegen(func))
            .flatten()
            .unwrap_or_else(|| Liveness::compute(func));
        let root_backend =
            self.replay_physical_state_until(func_id, func, region.block, region.range.start);
        let root = PhysicalReplayStart::capture(&root_backend.asm);
        let mut beam = vec![PersistentPhysicalState {
            prefix: Vec::new(),
            score: root.score(&root_backend),
            backend: root_backend,
            pressure: minimum_pressure,
        }];
        let mut states_expanded = 0;
        let mut backend_transactions = 1;
        let mut transaction_cache = FxHashMap::default();

        for depth in 0..region.range.len() {
            let mut next = Vec::new();
            for parent in &beam {
                let mut children = FxHashMap::<InstId, usize>::default();
                for (index, (schedule, _, _)) in schedules.iter().enumerate() {
                    if schedule.starts_with(&parent.prefix) {
                        children.entry(schedule[depth]).or_insert(index);
                    }
                }
                for (instruction, completion) in children {
                    let mut child = self.cached_persistent_physical_child(
                        parent,
                        func_id,
                        func,
                        region,
                        instruction,
                        &schedules[completion].0,
                        &root,
                        &base_liveness,
                        &mut transaction_cache,
                        &mut backend_transactions,
                    );
                    child.pressure = schedules
                        .iter()
                        .filter(|(schedule, _, _)| schedule.starts_with(&child.prefix))
                        .map(|(_, pressure, _)| *pressure)
                        .min()
                        .unwrap();
                    let rank = self.persistent_physical_lookahead(
                        &child,
                        2,
                        func_id,
                        func,
                        region,
                        &schedules,
                        &root,
                        optimization,
                        &base_liveness,
                        &mut transaction_cache,
                        &mut backend_transactions,
                    );
                    next.push((child, rank));
                }
            }
            states_expanded += next.len();
            next.sort_by(|(a, rank_a), (b, rank_b)| {
                rank_a
                    .key(optimization)
                    .cmp(&rank_b.key(optimization))
                    .then_with(|| a.prefix.cmp(&b.prefix))
            });
            let mut retained = next
                .iter()
                .filter(|(state, _)| state.pressure == minimum_pressure)
                .take(8)
                .cloned()
                .collect::<Vec<_>>();
            retained.extend(
                next.iter()
                    .filter(|(state, _)| state.pressure == minimum_pressure + 1)
                    .take(2)
                    .cloned(),
            );
            beam = retained.into_iter().map(|(state, _)| state).collect();
        }

        let beam_score = beam
            .iter()
            .filter_map(|state| {
                schedules
                    .iter()
                    .find(|(schedule, _, _)| schedule == &state.prefix)
                    .map(|(_, _, score)| *score)
            })
            .min_by_key(|score| score.key(optimization))
            .unwrap();
        assert_eq!(
            beam_score.key(optimization),
            expected.key(optimization),
            "persistent physical beam diverged from reconstruction in `{}`",
            func.name
        );
        let chosen = if beam_score.key(optimization) < baseline.key(optimization) {
            beam_score
        } else {
            baseline
        };
        let (dfs_primary, chosen_primary, optimum_primary) = match optimization {
            OptimizationMode::Size => {
                (baseline.estimated_bytes, chosen.estimated_bytes, physical_optimum.estimated_bytes)
            }
            _ => (
                baseline.estimated_static_gas,
                chosen.estimated_static_gas,
                physical_optimum.estimated_static_gas,
            ),
        };
        let available = dfs_primary.saturating_sub(optimum_primary);
        let captured = dfs_primary.saturating_sub(chosen_primary);
        let capture_ppm =
            captured.saturating_mul(1_000_000).checked_div(available).unwrap_or(1_000_000);
        tracing::debug!(
            target: PHYSICAL_REPLAY_TARGET,
            function = %func.name,
            block = region.block.index(),
            range_start = region.range.start,
            range_end = region.range.end,
            optimum_hit = beam_score.key(optimization) == physical_optimum.key(optimization),
            beam_beats_dfs = beam_score.key(optimization) < baseline.key(optimization),
            dfs_retained = beam_score.key(optimization) >= baseline.key(optimization),
            available_improvement = available,
            captured_improvement = captured,
            capture_ppm,
            states_expanded,
            backend_transactions,
            elapsed_micros = started.elapsed().as_micros(),
            "persistent physical beam result"
        );
    }

    /// Runs `8+2/h2` while retaining only candidate-local backend mutations.
    #[allow(clippy::too_many_arguments)]
    fn audit_compact_physical_beam_region(
        &self,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        candidates: &[PhysicalScheduleCandidate],
        scores: Option<&[PhysicalReplayScore]>,
        baseline: Option<PhysicalReplayScore>,
        optimization: OptimizationMode,
        expected: Option<PhysicalReplayScore>,
        entry_backend: Option<Self>,
        entry_liveness: Option<Liveness>,
    ) -> Option<Vec<InstId>> {
        let liveness = entry_liveness.unwrap_or_else(|| {
            self.emitting_entry
                .then(|| Liveness::compute_block_local_for_codegen(func))
                .flatten()
                .unwrap_or_else(|| Liveness::compute(func))
        });
        let mut workspace = entry_backend.unwrap_or_else(|| {
            self.replay_physical_state_until(func_id, func, region.block, region.range.start)
        });
        workspace.run_compact_physical_beam_region(
            func_id,
            func,
            region,
            candidates,
            scores,
            baseline,
            optimization,
            expected,
            liveness,
        )
    }

    /// Evaluates one compact region transaction from the backend's current physical state.
    #[allow(clippy::too_many_arguments)]
    fn run_compact_physical_beam_region(
        &mut self,
        func_id: FunctionId,
        func: &Function,
        region: &PhysicalScheduleCandidate,
        candidates: &[PhysicalScheduleCandidate],
        scores: Option<&[PhysicalReplayScore]>,
        baseline: Option<PhysicalReplayScore>,
        optimization: OptimizationMode,
        expected: Option<PhysicalReplayScore>,
        mut liveness: Liveness,
    ) -> Option<Vec<InstId>> {
        let workspace = self;
        let current = func.blocks[region.block].instructions[region.range.clone()].to_vec();
        let mut schedules = Vec::with_capacity(candidates.len() + 1);
        schedules.push((current, region.current_peak, baseline));
        schedules.extend(candidates.iter().enumerate().map(|(index, candidate)| {
            (candidate.order.clone(), candidate.candidate_peak, scores.map(|scores| scores[index]))
        }));
        let minimum_pressure = schedules.iter().map(|(_, pressure, _)| *pressure).min().unwrap();
        schedules.retain(|(_, pressure, _)| *pressure <= minimum_pressure + 1);

        let started = Instant::now();
        let score_root = PhysicalReplayStart::capture(&workspace.asm);
        let root = workspace.compact_physical_root();
        let root_state = workspace.capture_compact_physical_state(
            root,
            &score_root,
            Vec::new(),
            minimum_pressure,
        );
        let mut current_state = root_state.clone();
        let mut beam = vec![root_state.clone()];
        let mut states_expanded = 0;
        let mut backend_transactions = 1;
        let mut transaction_cache = FxHashMap::default();

        for &instruction in &schedules[0].0 {
            current_state = workspace.cached_compact_physical_child(
                &current_state,
                func_id,
                func,
                region,
                instruction,
                &schedules[0].0,
                root,
                &score_root,
                &mut liveness,
                &mut transaction_cache,
                &mut backend_transactions,
            );
        }
        let (baseline, baseline_interface) = if let Some(baseline) = baseline {
            (baseline, None)
        } else {
            let score = workspace.compact_physical_block_suffix_score(
                &current_state,
                func_id,
                func,
                region,
                &schedules[0].0,
                root,
                &score_root,
                &mut liveness,
            );
            let interface =
                (workspace.scheduler.stack.as_slice().to_vec(), workspace.scheduler.spills.clone());
            (score, Some(interface))
        };

        for depth in 0..region.range.len() {
            let mut next = Vec::new();
            for parent in &beam {
                let mut children = FxHashMap::<InstId, usize>::default();
                for (index, (schedule, _, _)) in schedules.iter().enumerate() {
                    if schedule.starts_with(&parent.prefix) {
                        children.entry(schedule[depth]).or_insert(index);
                    }
                }
                for (instruction, completion) in children {
                    let mut child = workspace.cached_compact_physical_child(
                        parent,
                        func_id,
                        func,
                        region,
                        instruction,
                        &schedules[completion].0,
                        root,
                        &score_root,
                        &mut liveness,
                        &mut transaction_cache,
                        &mut backend_transactions,
                    );
                    child.pressure = schedules
                        .iter()
                        .filter(|(schedule, _, _)| schedule.starts_with(&child.prefix))
                        .map(|(_, pressure, _)| *pressure)
                        .min()
                        .unwrap();
                    let rank = workspace.compact_physical_lookahead(
                        &child,
                        2,
                        func_id,
                        func,
                        region,
                        &schedules,
                        root,
                        &score_root,
                        optimization,
                        &mut liveness,
                        &mut transaction_cache,
                        &mut backend_transactions,
                    );
                    next.push((child, rank));
                }
            }
            states_expanded += next.len();
            next.sort_by(|(a, rank_a), (b, rank_b)| {
                rank_a
                    .key(optimization)
                    .cmp(&rank_b.key(optimization))
                    .then_with(|| a.prefix.cmp(&b.prefix))
            });
            let mut retained = next
                .iter()
                .filter(|(state, _)| state.pressure == minimum_pressure)
                .take(8)
                .cloned()
                .collect::<Vec<_>>();
            retained.extend(
                next.iter()
                    .filter(|(state, _)| state.pressure == minimum_pressure + 1)
                    .take(2)
                    .cloned(),
            );
            beam = retained.into_iter().map(|(state, _)| state).collect();
        }

        let mut winner = None;
        for state in &beam {
            let (schedule, _, terminal_score) = schedules
                .iter()
                .find(|(schedule, _, _)| schedule == &state.prefix)
                .expect("complete beam state must identify a legal schedule");
            let (score, continuation_safe) = if let Some(score) = terminal_score {
                (*score, true)
            } else {
                let score = workspace.compact_physical_block_suffix_score(
                    state,
                    func_id,
                    func,
                    region,
                    schedule,
                    root,
                    &score_root,
                    &mut liveness,
                );
                let continuation_safe =
                    baseline_interface.as_ref().is_none_or(|(baseline_stack, baseline_spills)| {
                        workspace.scheduler.stack.as_slice() == baseline_stack
                            && workspace.scheduler.spills == *baseline_spills
                    });
                (score, continuation_safe)
            };
            if !continuation_safe {
                continue;
            }
            if winner.as_ref().is_none_or(
                |(_, winner_score): &(&CompactPhysicalState, PhysicalReplayScore)| {
                    score.key(optimization) < winner_score.key(optimization)
                },
            ) {
                winner = Some((state, score));
            }
        }
        let (winner, beam_score) = match winner {
            Some(winner) => winner,
            None => {
                workspace.restore_compact_physical_state(root, &root_state);
                return None;
            }
        };
        if let Some(expected) = expected {
            assert_eq!(
                beam_score.key(optimization),
                expected.key(optimization),
                "compact physical beam diverged from full backend in `{}`",
                func.name
            );
        }
        tracing::debug!(
            target: PHYSICAL_REPLAY_TARGET,
            function = %func.name,
            block = region.block.index(),
            range_start = region.range.start,
            range_end = region.range.end,
            next = ?func.blocks[region.block]
                .instructions
                .get(region.range.end)
                .map(|&inst| &func.inst(inst).kind),
            suffix_len = func.blocks[region.block].instructions.len() - region.range.end,
            states_expanded,
            backend_transactions,
            elapsed_micros = started.elapsed().as_micros(),
            beam_beats_dfs = beam_score.key(optimization) < baseline.key(optimization),
            "compact physical beam result"
        );
        if beam_score.key(optimization) < baseline.key(optimization) {
            tracing::debug!(
                target: PHYSICAL_REPLAY_TARGET,
                function = %func.name,
                block = region.block.index(),
                range_start = region.range.start,
                range_end = region.range.end,
                region_len = region.range.len(),
                suffix_len = func.blocks[region.block].instructions.len() - region.range.end,
                elapsed_micros = started.elapsed().as_micros(),
                "compact physical beam winner"
            );
        }

        let dominates_baseline = beam_score.estimated_static_gas <= baseline.estimated_static_gas
            && beam_score.estimated_bytes <= baseline.estimated_bytes
            && beam_score.spill_bytes <= baseline.spill_bytes;
        let order = (dominates_baseline
            && beam_score.key(optimization) < baseline.key(optimization))
        .then(|| winner.prefix.clone());
        workspace.restore_compact_physical_state(root, &root_state);
        order
    }

    /// Runs one ordinary shadow emission to collect bounded physical choices, then emits the
    /// combined candidate once. The completed shadow backend is retained when no candidate wins,
    /// so production pays for at most two whole-function emissions rather than replaying every
    /// region prefix and compiling the selected function a third time.
    fn try_emit_bounded_physical_schedules(
        &mut self,
        func_id: FunctionId,
        func: &Function,
    ) -> bool {
        let candidates = bounded_physical_schedule_candidates(func, EvmInstSchedule::is_movable);
        if candidates.is_empty() {
            return false;
        }

        let score_root = PhysicalReplayStart::capture(&self.asm);
        let mut baseline_backend = self.clone();
        baseline_backend.physical_replay_active = true;
        baseline_backend.physical_planning = Some(PhysicalPlanningState::new(candidates));
        baseline_backend.generate_function_body(func_id, func);
        let baseline = score_root.score(&baseline_backend);
        let planning = baseline_backend
            .physical_planning
            .take()
            .expect("physical planning state must survive shadow emission");
        baseline_backend.physical_replay_active = false;

        if planning.choices.is_empty() {
            *self = baseline_backend;
            return true;
        }

        let mut scheduled = func.clone();
        for choice in planning.choices {
            scheduled.blocks[choice.block].instructions[choice.range]
                .copy_from_slice(&choice.order);
        }

        let mut candidate_backend = self.clone();
        candidate_backend.physical_replay_active = true;
        candidate_backend.generate_function_body(func_id, &scheduled);
        candidate_backend.physical_replay_active = false;
        let candidate = score_root.score(&candidate_backend);
        let optimization = self.gcx.sess.opts.optimization;
        let dominates_baseline = candidate.estimated_static_gas <= baseline.estimated_static_gas
            && candidate.estimated_bytes <= baseline.estimated_bytes
            && candidate.spill_bytes <= baseline.spill_bytes;
        if dominates_baseline && candidate.key(optimization) < baseline.key(optimization) {
            *self = candidate_backend;
        } else {
            *self = baseline_backend;
        }
        true
    }

    /// Evaluates one candidate group from the physical state reached by the shadow emission.
    fn plan_physical_region_from_current_state(
        &mut self,
        func_id: FunctionId,
        func: &Function,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) {
        let Some((region, candidates)) = self.physical_planning.as_ref().and_then(|planning| {
            let &(start, end) = planning.groups.get(&(block, inst_idx))?;
            Some((planning.candidates[start].clone(), planning.candidates[start..end].to_vec()))
        }) else {
            return;
        };

        let mut entry_backend = self.clone();
        entry_backend.physical_planning = None;
        if let Some(order) = self.audit_compact_physical_beam_region(
            func_id,
            func,
            &region,
            &candidates,
            None,
            None,
            self.gcx.sess.opts.optimization,
            None,
            Some(entry_backend),
            Some(liveness.clone()),
        ) {
            self.physical_planning.as_mut().unwrap().choices.push(PhysicalScheduleChoice {
                block,
                range: region.range,
                order,
            });
        }
    }

    /// Generates the body of a function.
    fn generate_function_body(&mut self, func_id: FunctionId, func: &Function) {
        let emit_physical_beam =
            !self.physical_replay_active && self.gcx.sess.opts.unstable.evm_physical_beam_schedule;
        let audit_physical_beam = !emit_physical_beam
            && !self.physical_replay_active
            && tracing::enabled!(target: PHYSICAL_REPLAY_TARGET, tracing::Level::DEBUG);
        if emit_physical_beam && self.try_emit_bounded_physical_schedules(func_id, func) {
            return;
        }
        if audit_physical_beam {
            let _ = self.audit_physical_schedules(func_id, func, false);
        }

        let liveness = self
            .emitting_entry
            .then(|| Liveness::compute_block_local_for_codegen(func))
            .flatten()
            .unwrap_or_else(|| Liveness::compute(func));
        let liveness = &liveness;

        // Eliminate phis.
        self.block_copies.clear();
        let phi_result = PhiEliminator::analyze(func);
        let has_phis = !phi_result.phis_to_remove.is_empty();
        for (block_id, copies) in phi_result.block_copies {
            self.block_copies.insert(block_id, copies.copies);
        }
        // Stack-phi planning starts with loop analysis, but cannot produce a
        // plan without a phi. Avoid that analysis for the overwhelmingly
        // common phi-free function.
        let stack_phi_plan =
            if has_phis { StackPhiPlan::analyze(func) } else { StackPhiPlan::default() };
        self.stack_phi_sources = stack_phi_plan.edge_sources.clone();
        let global_stack_plan = GlobalStackPlan::analyze(func, liveness, &stack_phi_plan);
        self.global_stack_active = !global_stack_plan.is_empty();
        self.global_stack_aliases = global_stack_plan.aliases.clone();

        // Reset scheduler
        self.scheduler = StackScheduler::new();
        let stack_only_values =
            self.lazy_stack_args.get(&func_id).into_iter().flat_map(|p| p.values());
        self.scheduler.set_stack_only_values(func.num_values(), stack_only_values);
        self.spill_addr_consts.clear();

        self.preallocate_cross_block_spills(func, liveness);

        self.cold_blocks = self.collect_cold_blocks(func);

        // Create labels for each block
        self.block_labels.clear();
        for block_id in func.blocks.indices() {
            let label = self.asm.new_label();
            if self.block_is_cold(block_id) {
                self.asm.mark_label_cold(label);
            }
            self.block_labels.insert(block_id, label);
        }

        // Generate each block.
        let block_order = self.block_layout_order(func);
        let block_pos: FxHashMap<BlockId, usize> =
            block_order.iter().enumerate().map(|(pos, &b)| (b, pos)).collect();
        // Stack layout a block must start with when it is reached by a stack-
        // preserving jump from its single predecessor (recorded by that
        // predecessor, restored here).
        let mut block_entry_stacks: FxHashMap<BlockId, StackModel> = FxHashMap::default();
        let mut preserved_fallthrough: Option<BlockId> = None;
        for (pos, &block_id) in block_order.iter().enumerate() {
            let block = &func.blocks[block_id];
            let fallthrough = block_order.get(pos + 1).copied();
            let entered_by_preserved_fallthrough = preserved_fallthrough == Some(block_id);
            preserved_fallthrough = None;

            let label = self.block_labels[&block_id];
            if !entered_by_preserved_fallthrough && !block.predecessors.is_empty() {
                self.asm.define_label(label);
            }

            // Reset stack at block entry unless the block is reached with a
            // known live stack: a physical fallthrough carries the scheduler's
            // stack directly, and a stack-preserving jump from a single
            // predecessor restores the recorded layout. All other cross-block
            // values live in spill slots.
            if !entered_by_preserved_fallthrough {
                if let Some(entry_stack) = block_entry_stacks.remove(&block_id) {
                    let max_depth = self.scheduler.stack.max_depth();
                    self.scheduler.stack = entry_stack;
                    self.scheduler.stack.inherit_max_depth(max_depth);
                    self.invalidate_carried_phi_spills(func);
                    // Live-ins not on the carried stack still arrive in memory.
                    self.mark_live_in_spills(func, liveness, block_id);
                } else if let Some(entry) = stack_phi_plan.entries.get(&block_id) {
                    self.set_stack_to_values(entry);
                    self.invalidate_carried_phi_spills(func);
                    self.mark_live_in_spills(func, liveness, block_id);
                } else if let Some(entry) = global_stack_plan.entry(block_id) {
                    self.set_stack_to_values(entry);
                    self.invalidate_carried_phi_spills(func);
                    self.mark_live_in_spills(func, liveness, block_id);
                } else {
                    self.scheduler.clear_stack();
                    self.mark_live_in_spills(func, liveness, block_id);
                }
            }
            if block_id == BlockId::ENTRY
                && let Some(values) = self.direct_stack_args.get(&func_id).cloned()
            {
                debug_assert_eq!(self.scheduler.stack.depth(), 0);
                self.set_stack_to_values(&values);
            } else if block_id == BlockId::ENTRY
                && let Some(plan) = self.lazy_stack_args.get(&func_id).cloned()
            {
                debug_assert_eq!(self.scheduler.stack.depth(), 0);
                self.set_stack_to_values(&plan.values().collect::<Vec<_>>());
            }

            // Generate instructions
            for (inst_idx, &inst_id) in block.instructions.iter().enumerate() {
                if self.physical_replay_stop == Some((block_id, inst_idx)) {
                    return;
                }
                if self.physical_planning.is_some() {
                    self.plan_physical_region_from_current_state(
                        func_id, func, liveness, block_id, inst_idx,
                    );
                }
                // Skip phi instructions (they're handled by copies)
                if matches!(func.inst(inst_id).kind, InstKind::Phi(_)) {
                    continue;
                }
                self.generate_block_inst(func_id, inst_id, func, liveness, block_id, inst_idx);
                if self.physical_replay_stop == Some((block_id, inst_idx + 1)) {
                    return;
                }
            }

            let stack_phi_preserved = stack_phi_plan.edges.get(&block_id).is_some_and(|edge| {
                if !self.can_prepare_stack_phi_edge(func, edge) {
                    return false;
                }
                self.spill_live_out_values_except(func, liveness, block_id, &edge.sources);
                self.pop_stack_values_not_needed_by(&edge.sources);
                self.try_emit_stack_phi_edge(func, edge)
            });

            // Insert phi copies before terminator. If the edge was materialized
            // as a stack-resident phi layout, the copies for this unconditional
            // predecessor are represented by the edge stack itself.
            if stack_phi_preserved {
                self.block_copies.remove(&block_id);
            } else if let Some(copies) = self.block_copies.remove(&block_id) {
                let mut temps = FxHashMap::default();
                for copy in &copies {
                    self.generate_copy(func, copy, &mut temps);
                }
            }

            let preserve_stack_to_fallthrough =
                self.can_preserve_stack_fallthrough(func, block_id, fallthrough);

            // A jump to a single-predecessor target that is emitted later can
            // keep its live stack instead of spilling: the target has exactly
            // one entry stack (this block's exit), so it can be restored there.
            let preserve_jump_target = (!preserve_stack_to_fallthrough)
                .then(|| self.single_pred_jump_target(func, block_id, fallthrough))
                .flatten()
                .filter(|target| block_pos.get(target).copied() > Some(pos));

            // A conditional branch whose other arm is a cold revert can carry
            // its single freshly-computed live-out on the stack into the hot
            // arm, which restores it as its recorded entry layout.
            let preserve_branch_targets =
                if !preserve_stack_to_fallthrough && preserve_jump_target.is_none() {
                    self.branch_preserve_targets(func, liveness, block_id, pos, &block_pos)
                } else {
                    Vec::new()
                };

            let global_stack_preserved = if !preserve_stack_to_fallthrough
                && preserve_jump_target.is_none()
                && preserve_branch_targets.is_empty()
                && !stack_phi_preserved
                && let Some(term) = block.terminator.as_ref()
                && let Some(layout) = global_stack_plan.edge_layout(func, term)
            {
                self.spill_live_out_values_except(func, liveness, block_id, layout);
                self.try_emit_global_stack_edge(func, term, layout)
            } else {
                false
            };

            let preserve_stack = preserve_stack_to_fallthrough
                || preserve_jump_target.is_some()
                || !preserve_branch_targets.is_empty()
                || stack_phi_preserved
                || global_stack_preserved;

            // Spill all live-out values before the terminator so they can be reloaded in successor
            // blocks. For a preserved edge, keep stack values live instead.
            if !preserve_stack {
                self.spill_live_out_values(func, liveness, block_id);
            }

            // Generate terminator
            if let Some(term) = &block.terminator {
                self.generate_terminator(func, term, fallthrough, preserve_stack);
            }
            self.scheduler.spills.release_block_locals();
            if preserve_stack_to_fallthrough {
                preserved_fallthrough = fallthrough;
            } else if let Some(target) = preserve_jump_target {
                block_entry_stacks.insert(target, self.scheduler.stack.clone());
            }
            for target in preserve_branch_targets {
                let mut entry_stack = self.scheduler.stack.clone();
                // The branch has one physical exit stack, but a cold successor need not retain
                // identities used only by its hot sibling. Keep hot layouts exact: anonymizing
                // their dead slots can turn a loop-carried stack hit into a reload every iteration.
                if self.block_is_cold(target) {
                    let live_in = liveness.live_in(target);
                    entry_stack.forget_values_not_matching(|value| live_in.contains(value));
                }
                block_entry_stacks.insert(target, entry_stack);
            }
        }

        if let Some(value) = self.scheduler.spills.unstored_required() {
            panic!(
                "mandatory cross-block spill store for {value:?} was not emitted in `{}`",
                func.name
            );
        }
        let mut peak = self.scheduler.stack.max_depth();
        if !self.direct_stack_args.contains_key(&func_id)
            && !self.lazy_stack_args.contains_key(&func_id)
            && let Some(mask) = self.stack_arg_masks.get(&func_id)
        {
            peak = peak.max(mask.count());
        }
        self.function_stack_peaks.insert(func_id, peak);
        self.assign_ranked_spill_addrs(func_id);
    }

    /// Emits one non-phi instruction while preserving the ordinary block-emission invariants.
    fn generate_block_inst(
        &mut self,
        func_id: FunctionId,
        inst_id: InstId,
        func: &Function,
        liveness: &Liveness,
        block_id: BlockId,
        inst_idx: usize,
    ) {
        let inst = func.inst(inst_id);
        let result_value = func.inst_result_value(inst_id);
        self.generate_inst(
            func_id,
            inst_id,
            func,
            &inst.kind,
            liveness,
            block_id,
            inst_idx,
            result_value,
        );
        if let Some(result) = result_value {
            self.spill_reserved_result_if_live(func, liveness, block_id, inst_idx, result);
            // A free-memory-pointer load cannot be rematerialized once the pointer moves. Park
            // every FMP load at its definition so later uses reload the original value — whether
            // the definition crosses a block on a preserved edge or is re-materialized between
            // two allocations in its own block.
            if matches!(
                inst.kind,
                InstKind::MLoad(addr) if func.value_u64(addr) == Some(EvmMemoryLayout::FMP_SLOT)
            ) {
                self.spill_value_if_needed(func, result);
            }
        }
    }

    /// Returns the target of a stack-preservable jump: the block ends in
    /// `Jump(T)` to a non-fallthrough, single-predecessor block with no phis
    /// (whose copies would otherwise interfere with the carried layout).
    fn single_pred_jump_target(
        &self,
        func: &Function,
        block_id: BlockId,
        fallthrough: Option<BlockId>,
    ) -> Option<BlockId> {
        let Some(Terminator::Jump(target)) = func.blocks[block_id].terminator.as_ref() else {
            return None;
        };
        if Some(*target) == fallthrough
            || func.blocks[*target].predecessors.as_slice() != [block_id]
        {
            return None;
        }
        let has_phi = func.blocks[*target]
            .instructions
            .iter()
            .any(|&inst| matches!(func.inst(inst).kind, InstKind::Phi(_)));
        (!has_phi).then_some(*target)
    }

    /// Returns branch successors that can receive the current stack layout.
    ///
    /// This handles loop headers after stack-resident phi planning: the header
    /// computes the branch condition while the carried phi values remain below
    /// it. If both successors are private, later blocks, we can leave those
    /// values on the stack for both edges instead of spilling them before every
    /// loop condition.
    fn branch_preserve_targets(
        &self,
        func: &Function,
        liveness: &Liveness,
        block_id: BlockId,
        pos: usize,
        block_pos: &FxHashMap<BlockId, usize>,
    ) -> Vec<BlockId> {
        let Some(Terminator::Branch { condition, then_block, else_block }) =
            func.blocks[block_id].terminator.as_ref()
        else {
            return Vec::new();
        };

        if self.scheduler.stack.depth() <= 1 || self.scheduler.stack.top() != Some(*condition) {
            return Vec::new();
        }

        let Some(carried) = self
            .scheduler
            .stack
            .iter()
            .skip(1)
            .map(|slot| {
                let value = slot?;
                liveness.live_out(block_id).contains(value).then_some(value)
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Vec::new();
        };
        if carried.len() > STACK_PHI_LAYOUT_LIMIT {
            return Vec::new();
        }

        let targets = [*then_block, *else_block];
        let mut live_in_any_target = DenseBitSet::new_empty(func.num_values());
        for target in targets {
            for value in liveness.live_in(target) {
                live_in_any_target.insert(value);
            }
        }
        if carried.iter().any(|&value| !live_in_any_target.contains(value)) {
            return Vec::new();
        }

        for target in targets {
            if target == block_id
                || func.blocks[target].predecessors.as_slice() != [block_id]
                || block_pos.get(&target).copied() <= Some(pos)
                || func.blocks[target]
                    .instructions
                    .iter()
                    .any(|&inst| matches!(func.inst(inst).kind, InstKind::Phi(_)))
            {
                return Vec::new();
            }
        }

        targets.into()
    }

    /// Finds functions whose reachable exits all abort, including chains of
    /// calls to other cold functions.
    fn collect_cold_functions(module: &Module) -> DenseBitSet<FunctionId> {
        let mut cold = DenseBitSet::new_empty(module.functions.len());
        let mut worklist = Vec::new();
        let mut visited = GrowableBitSet::new_empty();
        loop {
            let mut changed = false;
            for (function_id, func) in module.functions.iter_enumerated() {
                if cold.contains(function_id) {
                    continue;
                }
                worklist.clear();
                worklist.push(BlockId::ENTRY);
                visited.clear();
                let mut saw_exit = false;
                let mut all_exits_cold = true;
                while let Some(block_id) = worklist.pop()
                    && all_exits_cold
                {
                    if !visited.insert(block_id) {
                        continue;
                    }
                    let block = &func.blocks[block_id];
                    if block.instructions.iter().any(|&inst_id| {
                        matches!(
                            func.inst(inst_id).kind,
                            InstKind::InternalCall { function, .. } if cold.contains(function)
                        )
                    }) {
                        saw_exit = true;
                        continue;
                    }
                    let Some(term) = block.terminator.as_ref() else {
                        all_exits_cold = false;
                        continue;
                    };
                    match term {
                        Terminator::Revert { .. } | Terminator::Invalid => {
                            saw_exit = true;
                        }
                        Terminator::TailCall { function, .. } if cold.contains(*function) => {
                            saw_exit = true;
                        }
                        _ => {
                            let successors = term.successors();
                            if successors.is_empty() {
                                all_exits_cold = false;
                            } else {
                                worklist.extend(successors);
                            }
                        }
                    }
                }
                if saw_exit && all_exits_cold {
                    cold.insert(function_id);
                    changed = true;
                }
            }
            if !changed {
                return cold;
            }
        }
    }

    /// Finds blocks that abort directly or can only reach other cold blocks.
    fn collect_cold_blocks(&self, func: &Function) -> DenseBitSet<BlockId> {
        let mut cold = DenseBitSet::new_empty(func.blocks.len());
        let mut worklist = Vec::new();
        for block_id in func.blocks.indices() {
            if self.block_aborts(func, block_id) {
                cold.insert(block_id);
                worklist.push(block_id);
            }
        }
        if matches!(self.gcx.sess.opts.optimization, OptimizationMode::None) {
            return cold;
        }

        while let Some(block_id) = worklist.pop() {
            for &predecessor in &func.blocks[block_id].predecessors {
                if cold.contains(predecessor) {
                    continue;
                }
                let Some(term) = func.blocks[predecessor].terminator.as_ref() else {
                    continue;
                };
                let successors = term.successors();
                if !successors.is_empty()
                    && successors.iter().all(|&successor| cold.contains(successor))
                {
                    cold.insert(predecessor);
                    worklist.push(predecessor);
                }
            }
        }
        cold
    }

    /// Returns true when a block aborts directly or calls a function whose
    /// reachable exits all abort.
    fn block_aborts(&self, func: &Function, block_id: BlockId) -> bool {
        let block = &func.blocks[block_id];
        matches!(block.terminator, Some(Terminator::Revert { .. } | Terminator::Invalid))
            || matches!(
                block.terminator,
                Some(Terminator::TailCall { function, .. })
                    if self.cold_functions.contains(function)
            )
            || block.instructions.iter().any(|&inst_id| {
                matches!(
                    func.inst(inst_id).kind,
                    InstKind::InternalCall { function, .. }
                        if self.cold_functions.contains(function)
                )
            })
    }

    fn block_is_cold(&self, block_id: BlockId) -> bool {
        self.cold_blocks.contains(block_id)
    }

    fn new_function_label(&mut self, function: FunctionId) -> Label {
        let label = self.asm.new_label();
        if self.cold_functions.contains(function) {
            self.asm.mark_label_cold(label);
        }
        label
    }

    fn block_layout_order(&self, func: &Function) -> Vec<BlockId> {
        // Layout only initializes reachability; RPO, dominators, and
        // transitive reachability remain unevaluated.
        let cfg = CfgInfo::new(func);
        let reachable = cfg.reachable();
        let mut order = Vec::with_capacity(func.blocks.len());
        let mut placed = DenseBitSet::new_empty(func.blocks.len());

        self.append_layout_chain(func, BlockId::ENTRY, reachable, &mut placed, &mut order);
        for block_id in func.blocks.indices() {
            if reachable.contains(block_id) {
                self.append_layout_chain(func, block_id, reachable, &mut placed, &mut order);
            }
        }

        order
    }

    fn append_layout_chain(
        &self,
        func: &Function,
        mut block_id: BlockId,
        reachable: &DenseBitSet<BlockId>,
        placed: &mut DenseBitSet<BlockId>,
        order: &mut Vec<BlockId>,
    ) {
        loop {
            if !reachable.contains(block_id) || !placed.insert(block_id) {
                return;
            }
            order.push(block_id);

            let target = match func.blocks[block_id].terminator.as_ref() {
                Some(Terminator::Jump(target))
                    if func.blocks[*target].predecessors.as_slice() == [block_id] =>
                {
                    *target
                }
                Some(Terminator::Branch { then_block, else_block, .. })
                    if !matches!(self.gcx.sess.opts.optimization, OptimizationMode::None) =>
                {
                    match (self.block_is_cold(*then_block), self.block_is_cold(*else_block)) {
                        (true, false) => *else_block,
                        (false, true) => *then_block,
                        _ => return,
                    }
                }
                _ => return,
            };
            if placed.contains(target) {
                return;
            }

            block_id = target;
        }
    }

    fn set_stack_to_values(&mut self, values: &[ValueId]) {
        self.scheduler.stack.clear();
        for &value in values.iter().rev() {
            self.scheduler.stack.push(value);
        }
    }

    fn try_emit_global_stack_edge(
        &mut self,
        func: &Function,
        term: &Terminator,
        layout: &[ValueId],
    ) -> bool {
        if layout.is_empty() || layout.len() > GLOBAL_STACK_LAYOUT_LIMIT {
            return false;
        }

        let mut needed = Vec::with_capacity(layout.len() + 1);
        if let Terminator::Branch { condition, .. } = term {
            needed.push(*condition);
        }
        needed.extend_from_slice(layout);

        self.pop_stack_values_not_needed_by(&needed);
        for value in Self::missing_stack_phi_sources(&self.scheduler.stack, &needed) {
            self.emit_operand(func, value);
        }

        let target: Vec<_> = needed.iter().copied().map(TargetSlot::Value).collect();
        let Some(shuffle) = self.scheduler.shuffle_to_layout(&target) else { return false };
        assert_eq!(self.scheduler.depth(), needed.len(), "global-stack edge depth mismatch");
        assert!(
            self.scheduler.stack.iter().eq(needed.iter().copied().map(Some)),
            "global-stack edge layout mismatch"
        );
        for op in shuffle.ops {
            self.asm.emit_op(op.opcode());
        }

        true
    }

    fn try_emit_stack_phi_edge(&mut self, func: &Function, edge: &StackPhiEdge) -> bool {
        if edge.sources.len() != edge.results.len()
            || edge.sources.is_empty()
            || edge.sources.len() > STACK_PHI_LAYOUT_LIMIT
        {
            return false;
        }
        if !self.stack_contains_only_phi_sources(&edge.sources) {
            return false;
        }

        for &source in Self::missing_stack_phi_sources(&self.scheduler.stack, &edge.sources).iter()
        {
            if !self.scheduler.can_emit_value(source, func) {
                return false;
            }
            self.emit_operand(func, source);
        }
        assert!(
            self.stack_contains_only_phi_sources(&edge.sources),
            "prepared stack-phi edge contains unexpected values"
        );

        let target: Vec<_> = edge.sources.iter().copied().map(TargetSlot::Value).collect();
        let Some(shuffle) = self.scheduler.shuffle_to_layout(&target) else { return false };
        assert_eq!(self.scheduler.depth(), edge.sources.len(), "stack-phi edge depth mismatch");
        assert!(
            self.scheduler.stack.iter().eq(edge.sources.iter().copied().map(Some)),
            "stack-phi edge layout mismatch"
        );
        for op in shuffle.ops {
            self.asm.emit_op(op.opcode());
        }

        self.set_stack_to_values(&edge.results);
        true
    }

    fn can_prepare_stack_phi_edge(&self, func: &Function, edge: &StackPhiEdge) -> bool {
        if edge.sources.len() != edge.results.len()
            || edge.sources.is_empty()
            || edge.sources.len() > STACK_PHI_LAYOUT_LIMIT
        {
            return false;
        }

        let present =
            Self::stack_phi_source_counts_after_trim(&self.scheduler.stack, &edge.sources);
        if present.len() > STACK_PHI_LAYOUT_LIMIT {
            return false;
        }

        let mut seen = Self::value_counts(present);
        for &source in &edge.sources {
            if let Some(count) = seen.get_mut(&source)
                && *count > 0
            {
                *count -= 1;
                continue;
            }
            if !self.scheduler.can_emit_value(source, func) {
                return false;
            }
        }
        true
    }

    fn stack_phi_source_counts_after_trim(stack: &StackModel, sources: &[ValueId]) -> Vec<ValueId> {
        let mut remaining = Self::value_counts(sources.iter().copied());
        let mut kept = Vec::new();
        for value in stack.iter().flatten() {
            if let Some(count) = remaining.get_mut(&value)
                && *count > 0
            {
                *count -= 1;
                kept.push(value);
            }
        }
        kept
    }

    fn stack_contains_only_phi_sources(&self, sources: &[ValueId]) -> bool {
        let mut remaining = Self::value_counts(sources.iter().copied());
        for slot in self.scheduler.stack.iter() {
            let Some(value) = slot else {
                return false;
            };
            let Some(count) = remaining.get_mut(&value) else {
                return false;
            };
            if *count == 0 {
                return false;
            }
            *count -= 1;
        }
        true
    }

    fn missing_stack_phi_sources(stack: &StackModel, sources: &[ValueId]) -> Vec<ValueId> {
        let mut needed = Self::value_counts(sources.iter().copied());
        for value in stack.iter().flatten() {
            if let Some(count) = needed.get_mut(&value)
                && *count > 0
            {
                *count -= 1;
            }
        }

        let mut missing = Vec::new();
        for &source in sources {
            if let Some(count) = needed.get_mut(&source)
                && *count > 0
            {
                missing.push(source);
                *count -= 1;
            }
        }
        missing
    }

    fn value_counts(values: impl IntoIterator<Item = ValueId>) -> FxHashMap<ValueId, usize> {
        let mut counts = FxHashMap::default();
        for value in values {
            *counts.entry(value).or_default() += 1;
        }
        counts
    }

    fn can_preserve_stack_fallthrough(
        &self,
        func: &Function,
        block_id: BlockId,
        fallthrough: Option<BlockId>,
    ) -> bool {
        let Some(Terminator::Jump(target)) = func.blocks[block_id].terminator.as_ref() else {
            return false;
        };
        if Some(*target) != fallthrough {
            return false;
        }

        // This block is the target's only predecessor, so no non-fallthrough edge can observe or
        // depend on a JUMPDEST at the target label.
        func.blocks[*target].predecessors.as_slice() == [block_id]
    }

    fn is_stack_phi_source(&self, block: BlockId, value: ValueId) -> bool {
        self.stack_phi_sources.get(&block).is_some_and(|sources| sources.contains(&value))
    }

    /// Preallocates stable spill slots for values that may cross block boundaries.
    ///
    /// Blocks are emitted in layout order, not necessarily dominance order, so a block can be
    /// emitted before the predecessor that stores one of its live-in values. Reserving the slot up
    /// front lets the later load use a stable memory location; stores still happen only when the
    /// value is actually available on the stack.
    fn preallocate_cross_block_spills(&mut self, func: &Function, liveness: &Liveness) {
        let values = Self::cross_block_spill_values(func, liveness);

        // Coloring minimizes the local frame, which reduces memory expansion in gas mode. It is
        // deliberately disabled in size mode because renumbering spill addresses disturbed
        // downstream block sharing and regressed aggregate CI bytecode despite smaller frames.
        if self.gcx.sess.opts.optimization.is_gas() {
            let colorable = Self::cross_block_live_values(func, liveness);
            let ranges = Self::spill_live_ranges(func, liveness, &colorable);

            let mut colors = Vec::<SpillColor>::new();
            for value in &colorable {
                let value_ranges = &ranges[value];
                let color = colors
                    .iter()
                    .position(|color| color.accepts(value_ranges))
                    .unwrap_or_else(|| {
                        colors.push(SpillColor::default());
                        colors.len() - 1
                    });
                colors[color].insert(value_ranges);
                self.scheduler.spills.reserve_at(value, color as u32);
            }

            for value in &values {
                if !colorable.contains(value) {
                    self.scheduler.spills.reserve(value);
                }
            }
        } else {
            for value in &values {
                self.scheduler.spills.reserve(value);
            }
        }

        if values.iter().any(|value| Self::is_cross_block_recomputable_inst(func, value)) {
            let recomputable = Self::cross_block_recomputable_values(func);
            let reloaded = values
                .iter()
                .any(|value| {
                    !recomputable.contains(value)
                        && StackScheduler::is_cheap_recomputable_value(func, value)
                })
                .then(|| Self::cross_block_reload_values(func));
            for val in &values {
                if recomputable.contains(val) {
                    self.scheduler.spills.mark_recomputable(val);
                } else if reloaded.as_ref().is_some_and(|values| values.contains(val))
                    && StackScheduler::is_cheap_recomputable_value(func, val)
                {
                    self.scheduler.spills.require_store(val);
                }
            }
        }

        // A free-memory-pointer load cannot be recomputed after the pointer moves. Reserve stable
        // slots for cross-block values, including direct uses that liveness does not carry. Size
        // mode keeps every FMP slot stable because block-local reuse can increase output size.
        let fmp_values = Self::fmp_load_values(func);
        if !fmp_values.is_empty() {
            let reserve_all = matches!(self.gcx.sess.opts.optimization, OptimizationMode::Size);
            let reloaded = (!reserve_all).then(|| Self::cross_block_reload_values(func));
            for val in fmp_values {
                if reserve_all
                    || values.contains(val)
                    || reloaded.as_ref().is_some_and(|values| values.contains(val))
                {
                    self.scheduler.spills.reserve(val);
                    self.scheduler.spills.mark_reloadable(val);
                }
            }
        }
    }

    fn cross_block_live_values(func: &Function, liveness: &Liveness) -> DenseBitSet<ValueId> {
        let mut values = DenseBitSet::new_empty(func.num_values());
        for block in func.blocks.indices() {
            for value in liveness.live_in(block).iter().chain(liveness.live_out(block).iter()) {
                if matches!(func.value(value), crate::mir::Value::Inst(_)) {
                    values.insert(value);
                }
            }
        }
        values
    }

    fn spill_live_ranges(
        func: &Function,
        liveness: &Liveness,
        colorable: &DenseBitSet<ValueId>,
    ) -> IndexVec<ValueId, FxHashMap<BlockId, SpillLiveRange>> {
        let mut ranges = index_vec![FxHashMap::default(); func.num_values()];
        let mut operands = SmallVec::<[ValueId; 8]>::new();

        for (block_id, block) in func.blocks.iter_enumerated() {
            for value in liveness.live_in(block_id) {
                Self::extend_spill_live_range(&mut ranges, colorable, value, block_id, 0);
            }
            for (inst_idx, &inst_id) in block.instructions.iter().enumerate() {
                operands.clear();
                func.inst(inst_id).kind.collect_operands(&mut operands);
                for &value in &operands {
                    Self::extend_spill_live_range(
                        &mut ranges,
                        colorable,
                        value,
                        block_id,
                        inst_idx * 2,
                    );
                }
                if let Some(value) = func.inst_result_value(inst_id) {
                    Self::extend_spill_live_range(
                        &mut ranges,
                        colorable,
                        value,
                        block_id,
                        inst_idx * 2 + 1,
                    );
                }
            }
            if let Some(terminator) = &block.terminator {
                let point = block.instructions.len() * 2;
                for value in terminator.operands() {
                    Self::extend_spill_live_range(&mut ranges, colorable, value, block_id, point);
                }
            }
            let point = block.instructions.len() * 2 + 1;
            for value in liveness.live_out(block_id) {
                Self::extend_spill_live_range(&mut ranges, colorable, value, block_id, point);
            }
        }
        ranges
    }

    fn extend_spill_live_range(
        ranges: &mut IndexVec<ValueId, FxHashMap<BlockId, SpillLiveRange>>,
        colorable: &DenseBitSet<ValueId>,
        value: ValueId,
        block: BlockId,
        point: usize,
    ) {
        if !colorable.contains(value) {
            return;
        }
        ranges[value]
            .entry(block)
            .and_modify(|range| {
                range.start = range.start.min(point);
                range.end = range.end.max(point);
            })
            .or_insert(SpillLiveRange { start: point, end: point });
    }

    /// Returns values directly consumed outside their defining block. Phi inputs are edge uses:
    /// codegen consumes them in the predecessor or carries them on the edge, so they do not need a
    /// reload route under the source value's identity.
    fn cross_block_reload_values(func: &Function) -> DenseBitSet<ValueId> {
        let mut definitions =
            IndexVec::<ValueId, Option<BlockId>>::from_vec(vec![None; func.num_values()]);
        for block_id in func.blocks.indices() {
            for &inst_id in &func.blocks[block_id].instructions {
                if let Some(result) = func.inst_result_value(inst_id) {
                    definitions[result] = Some(block_id);
                }
            }
        }

        let mut reloaded = DenseBitSet::new_empty(func.num_values());
        for block_id in func.blocks.indices() {
            for &inst_id in &func.blocks[block_id].instructions {
                if matches!(func.inst(inst_id).kind, InstKind::Phi(_)) {
                    continue;
                }
                for operand in func.inst(inst_id).kind.operands() {
                    if definitions[operand].is_some_and(|definition| definition != block_id) {
                        reloaded.insert(operand);
                    }
                }
            }
            if let Some(terminator) = &func.blocks[block_id].terminator {
                for operand in terminator.operands() {
                    if definitions[operand].is_some_and(|definition| definition != block_id) {
                        reloaded.insert(operand);
                    }
                }
            }
        }
        reloaded
    }

    /// Every live free-memory-pointer load result in the function.
    fn fmp_load_values(func: &Function) -> Vec<ValueId> {
        let mut values = Vec::new();
        for inst_id in func.instructions() {
            if matches!(
                func.inst(inst_id).kind,
                InstKind::MLoad(addr)
                    if func.value_u64(addr) == Some(EvmMemoryLayout::FMP_SLOT)
            ) && let Some(val) = func.inst_result_value(inst_id)
            {
                values.push(val);
            }
        }
        values
    }

    fn cross_block_spill_values(func: &Function, liveness: &Liveness) -> DenseBitSet<ValueId> {
        let mut values = DenseBitSet::new_empty(func.num_values());
        for block_id in func.blocks.indices() {
            for val in liveness.live_in(block_id).iter().chain(liveness.live_out(block_id).iter()) {
                if Self::can_own_spill_slot(func, val) {
                    values.insert(val);
                }
            }
            for &inst_id in &func.blocks[block_id].instructions {
                if matches!(func.inst(inst_id).kind, InstKind::Phi(_))
                    && let Some(val) = func.inst_result_value(inst_id)
                {
                    values.insert(val);
                }
            }
        }
        values
    }

    /// Returns cheap values whose complete dependency tree can be rematerialized without reading
    /// mutable machine state. A reverse-use worklist handles long expression chains linearly.
    fn cross_block_recomputable_values(func: &Function) -> DenseBitSet<ValueId> {
        let mut users =
            IndexVec::<ValueId, SmallVec<[ValueId; 2]>>::with_capacity(func.num_values());
        let mut remaining = IndexVec::<ValueId, usize>::with_capacity(func.num_values());
        for _ in 0..func.num_values() {
            users.push(SmallVec::new());
            remaining.push(usize::MAX);
        }

        let mut recomputable = DenseBitSet::new_empty(func.num_values());
        let mut worklist = Vec::new();
        for value in func.live_values() {
            if Self::is_rematerializable_value(func, value) && recomputable.insert(value) {
                worklist.push(value);
            }
        }
        for inst_id in func.instructions() {
            let Some(result) = func.inst_result_value(inst_id) else { continue };
            if !Self::is_cross_block_recomputable_inst(func, result) {
                continue;
            }
            let operands = func.inst(inst_id).kind.operands();
            remaining[result] = operands.len();
            if operands.is_empty() && recomputable.insert(result) {
                worklist.push(result);
            }
            for operand in operands {
                users[operand].push(result);
            }
        }

        while let Some(value) = worklist.pop() {
            for &user in &users[value] {
                remaining[user] -= 1;
                if remaining[user] == 0 && recomputable.insert(user) {
                    worklist.push(user);
                }
            }
        }
        recomputable
    }

    fn is_cross_block_recomputable_inst(func: &Function, value: ValueId) -> bool {
        if StackScheduler::is_cheap_recomputable_value(func, value) {
            return true;
        }
        let crate::mir::Value::Inst(inst_id) = func.value(value) else { return false };
        matches!(
            func.inst(*inst_id).kind,
            InstKind::CallValue
                | InstKind::Caller
                | InstKind::Origin
                | InstKind::CalldataSize
                | InstKind::CalldataLoad(_)
                | InstKind::InternalFrameAddr(_)
                | InstKind::Timestamp
                | InstKind::BlockNumber
        )
    }

    /// Spills all live-out values that are currently on the stack to memory.
    /// This ensures values that need to be accessed in successor blocks can be reloaded.
    fn spill_live_out_values(&mut self, func: &Function, liveness: &Liveness, block_id: BlockId) {
        let live_out = liveness.live_out(block_id);

        for val in live_out {
            self.spill_value_if_needed(func, val);
        }
    }

    fn spill_live_out_values_except(
        &mut self,
        func: &Function,
        liveness: &Liveness,
        block_id: BlockId,
        exempt: &[ValueId],
    ) {
        let mut exempt_values = DenseBitSet::new_empty(func.num_values());
        for &value in exempt {
            exempt_values.insert(value);
        }
        for val in liveness.live_out(block_id) {
            if !exempt_values.contains(val) {
                self.spill_value_if_needed(func, val);
            }
        }
    }

    fn pop_stack_values_not_needed_by(&mut self, needed: &[ValueId]) {
        while let Some(depth) = self.first_stack_value_not_needed_by(needed) {
            if depth > 0 {
                self.emit_stack_op(StackOp::Swap(depth as u8));
            }
            self.emit_stack_op(StackOp::Pop);
        }
    }

    fn first_stack_value_not_needed_by(&self, needed: &[ValueId]) -> Option<usize> {
        let mut remaining = Self::value_counts(needed.iter().copied());
        for (depth, slot) in self.scheduler.stack.iter().enumerate() {
            let Some(value) = slot else {
                return Some(depth);
            };
            let Some(count) = remaining.get_mut(&value) else {
                return Some(depth);
            };
            if *count == 0 {
                return Some(depth);
            }
            *count -= 1;
        }
        None
    }

    /// Invalidates the spill bookkeeping of every phi result on a stack
    /// restored from a carried edge. A loop-carried phi is redefined on every
    /// re-entry without a store, so a slot stored during an earlier iteration
    /// holds a stale definition: an exit-path use must spill the carried copy
    /// again before anything reloads the slot. Other carried values are
    /// immutable SSA definitions whose stored slots stay current, and
    /// invalidating those would force later paths to recompute
    /// memory-dependent definitions whose operands may have changed.
    fn invalidate_carried_phi_spills(&mut self, func: &Function) {
        let carried: Vec<ValueId> = self.scheduler.stack.iter().flatten().collect();
        for value in carried {
            if let crate::mir::Value::Inst(inst_id) = func.value(value)
                && matches!(func.inst(*inst_id).kind, InstKind::Phi(_))
            {
                self.scheduler.spills.invalidate_stored(value);
            }
        }
    }

    fn mark_live_in_spills(&mut self, func: &Function, liveness: &Liveness, block_id: BlockId) {
        // Values already on the stack (carried in from a preserved predecessor
        // edge) are read directly; marking them reloadable would point at a
        // spill slot that may never have been stored.
        for val in liveness.live_in(block_id) {
            if !self.scheduler.stack.contains(val) && self.scheduler.spills.get(val).is_some() {
                self.scheduler.spills.mark_reloadable(val);
            }
        }
        for &inst_id in &func.blocks[block_id].instructions {
            if matches!(func.inst(inst_id).kind, InstKind::Phi(_))
                && let Some(val) = func.inst_result_value(inst_id)
                && !self.scheduler.stack.contains(val)
                && self.scheduler.spills.get(val).is_some()
            {
                self.scheduler.spills.mark_reloadable(val);
            }
        }
    }

    fn spill_values_before_stack_clear(&mut self, func: &Function, values: &[ValueId]) {
        for &value in values {
            self.spill_value_if_needed(func, value);
        }
    }

    /// Parks stack-resident operands in their spill slots before an
    /// `emit_value_fresh` sequence. The sequence re-materializes each value,
    /// and definitions such as free-memory-pointer loads cannot be recomputed
    /// once memory has moved on: reaching them through a reload keeps the
    /// original definition.
    fn prepare_fresh_operands(&mut self, func: &Function, operands: &[ValueId]) {
        for &operand in operands {
            self.spill_value_if_needed(func, operand);
        }
    }

    /// Spills an instruction result if it is on the stack and not already stored.
    fn spill_value_if_needed(&mut self, func: &Function, val: ValueId) {
        if !Self::can_own_spill_slot(func, val) {
            return;
        }

        if self.scheduler.spills.is_stored(val) {
            return;
        }

        if let Some(depth) = self.scheduler.stack.find(val) {
            let slot = self.scheduler.spills.allocate(val);
            if depth >= MAX_STACK_ACCESS {
                self.spill_deep_stack_value(func, val, slot, depth);
                return;
            }

            self.spill_accessible_stack_value(func, val, slot, depth);
        }
    }

    fn spill_value_to_reserved_slot(&mut self, func: &Function, val: ValueId) -> bool {
        if Self::is_rematerializable_value(func, val) || self.scheduler.spills.get(val).is_none() {
            return false;
        }

        let Some(depth) = self.scheduler.stack.find(val) else {
            return false;
        };
        let slot = self.scheduler.spills.allocate(val);
        if depth >= MAX_STACK_ACCESS {
            self.spill_deep_stack_value(func, val, slot, depth);
        } else {
            self.spill_accessible_stack_value(func, val, slot, depth);
        }
        true
    }

    fn spill_reserved_result_if_live(
        &mut self,
        func: &Function,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
        value: ValueId,
    ) {
        // This is not the normal first-store path; `generate_inst` handles live-out results.
        // It repairs physical emission orders where a successor block emitted first has already
        // marked this reserved cross-block slot as stored/reloadable before the defining block
        // materializes the value.
        if self.scheduler.spills.get(value).is_none()
            || !self.scheduler.spills.is_stored(value)
            || liveness.is_dead_after(value, block, inst_idx)
        {
            return;
        }

        self.spill_value_to_reserved_slot(func, value);
    }

    fn spill_accessible_stack_value(
        &mut self,
        func: &Function,
        val: ValueId,
        slot: SpillSlot,
        depth: usize,
    ) {
        debug_assert!(depth < MAX_STACK_ACCESS);

        // DUP the value to top of stack for storing.
        // We need to DUP (not just use ensure_on_top) because:
        // 1. If value is on top, ensure_on_top does nothing but we need a copy
        // 2. MSTORE will consume the value, and we want to preserve the original
        let dup_n = (depth + 1) as u8;
        self.asm.emit_op(op::dup(dup_n));
        self.scheduler.stack.dup(dup_n);

        self.store_stack_top_to_spill(func, val, slot);
    }

    fn spill_deep_stack_value(
        &mut self,
        func: &Function,
        val: ValueId,
        slot: SpillSlot,
        depth: usize,
    ) {
        debug_assert!(depth >= MAX_STACK_ACCESS);

        let mut saved_above = Vec::with_capacity(depth + 1 - MAX_STACK_ACCESS);
        for _ in 0..(depth + 1 - MAX_STACK_ACCESS) {
            let Some(top) = self.scheduler.stack.top() else {
                panic!("cannot spill deep stack value {val:?}: untracked stack entry above it");
            };
            let top_slot = self.scheduler.spills.allocate(top);
            if self.scheduler.reloadable_spill(top).is_some() {
                self.emit_stack_op(StackOp::Pop);
            } else {
                self.store_stack_top_to_spill(func, top, top_slot);
            }
            saved_above.push((top, top_slot));
        }

        let Some(accessible_depth) = self.scheduler.stack.find(val) else {
            panic!("cannot spill deep stack value {val:?}: value disappeared while exposing it");
        };
        self.spill_accessible_stack_value(func, val, slot, accessible_depth);

        for (saved, saved_slot) in saved_above.into_iter().rev() {
            self.emit_spill_slot_addr(func, saved_slot);
            self.asm.emit_op(op::MLOAD);
            self.scheduler.stack.push(saved);
        }
    }

    fn store_stack_top_to_spill(&mut self, func: &Function, value: ValueId, slot: SpillSlot) {
        // Store to spill slot: PUSH offset, MSTORE.
        // The PUSH creates an untracked stack entry, so we track it as unknown.
        self.emit_spill_slot_addr(func, slot);
        self.scheduler.stack.push_unknown();

        self.asm.emit_op(op::MSTORE);
        // MSTORE consumes 2 values: the untracked offset and the value being spilled.
        self.scheduler.stack.pop();
        self.scheduler.stack.pop();
        self.scheduler.spills.mark_stored(value);
    }

    /// Spills operands that are live-out before an instruction consumes them.
    /// This ensures cross-block values are preserved in memory.
    fn spill_live_out_operands(
        &mut self,
        func: &Function,
        liveness: &Liveness,
        block_id: BlockId,
        operands: &[ValueId],
    ) {
        let live_out = liveness.live_out(block_id);

        for &op in operands {
            if live_out.contains(op) && !self.is_stack_phi_source(block_id, op) {
                self.spill_value_if_needed(func, op);
            }
        }
    }

    /// Values that are always re-emitted at each use instead of being kept on
    /// the stack or spilled.
    ///
    /// `Arg` MUST stay in this set. With static frames an argument reload is a
    /// 3-4 byte `PUSH addr; MLOAD`/`CALLDATALOAD`, cheaper than the spill
    /// traffic that tracking would create — and the spill machinery assumes
    /// arguments never own slots: making `Arg` non-rematerializable was
    /// measured to REGRESS every bench contract's size (erc20 +61 B, maple
    /// +72 B, fractional +127 B) and to break 4 of 8 bench harnesses at
    /// runtime. Do not re-attempt without redesigning argument spilling.
    fn is_rematerializable_value(func: &Function, value: ValueId) -> bool {
        matches!(func.value(value), crate::mir::Value::Immediate(_) | crate::mir::Value::Arg(_))
    }

    fn can_own_spill_slot(func: &Function, value: ValueId) -> bool {
        matches!(func.value(value), crate::mir::Value::Inst(_))
    }

    /// Returns true when `value` needs no spill before the instruction that
    /// is about to consume it: it owns no reserved cross-block slot, it is
    /// not live out of the block, and more stack copies exist at this point
    /// than the instruction will consume net of the emissions still to come
    /// (`consumed`). Later in-block uses DUP the survivor, or deep-spill it
    /// on demand if it sinks past `MAX_STACK_ACCESS`, so skipping the store
    /// cannot strand the value and adds no stack depth.
    fn block_local_copy_survives(
        &self,
        liveness: &Liveness,
        block: BlockId,
        value: ValueId,
        consumed: usize,
    ) -> bool {
        self.scheduler.spills.get(value).is_none()
            && !liveness.live_out(block).contains(value)
            && self.scheduler.stack.iter().flatten().filter(|&v| v == value).count() > consumed
    }

    fn spill_top_value_if_live(
        &mut self,
        func: &Function,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
        value: ValueId,
    ) {
        if Self::is_rematerializable_value(func, value) {
            return;
        }

        let has_reserved_cross_block_slot = self.scheduler.spills.get(value).is_some();
        if liveness.is_dead_after(value, block, inst_idx) && !has_reserved_cross_block_slot {
            return;
        }

        debug_assert_eq!(self.scheduler.stack.top(), Some(value));
        if !self.spill_value_to_reserved_slot(func, value) {
            self.spill_value_if_needed(func, value);
        }
        if has_reserved_cross_block_slot {
            assert!(
                self.scheduler.reloadable_spill(value).is_some(),
                "reserved operand {value:?} was not stored before consumption in `{}`",
                func.name
            );
        }
    }

    /// Generates bytecode for an instruction.
    #[allow(clippy::too_many_arguments)]
    fn generate_inst(
        &mut self,
        func_id: FunctionId,
        inst_id: InstId,
        func: &Function,
        kind: &InstKind,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
        result_value: Option<ValueId>,
    ) {
        let operands = kind.operands();
        self.materialize_lazy_stack_args(func_id, kind, block, inst_idx);
        // Keep one lazy stack copy of an argument when this instruction is not
        // its last use. The consuming occurrence uses a DUP of that copy, so
        // later blocks can inherit it without an eager prologue load.
        for &operand in &operands {
            if self.global_stack_active
                && matches!(func.value(operand), crate::mir::Value::Arg(_))
                && !self.scheduler.stack.contains(operand)
                && !liveness.is_dead_after(operand, block, inst_idx)
            {
                self.emit_value(func, operand);
            }
        }

        // Spill any operands that are live-out before they get consumed.
        // This ensures cross-block values are preserved in memory.
        self.spill_live_out_operands(func, liveness, block, &operands);

        match kind {
            // Binary arithmetic operations
            InstKind::Add(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::ADD,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Sub(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::SUB,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Mul(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::MUL,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Div(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::DIV,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::SDiv(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::SDIV,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Mod(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::MOD,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::SMod(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::SMOD,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Exp(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::EXP,
                result_value,
                liveness,
                block,
                inst_idx,
            ),

            // Bitwise operations
            InstKind::And(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::AND,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Or(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::OR,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Xor(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::XOR,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Not(a) => self.emit_unary_op_with_result(
                func,
                *a,
                op::NOT,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Clz(a) => self.emit_unary_op_with_result(
                func,
                *a,
                op::CLZ,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Shl(shift, val) => self.emit_binary_op_with_result(
                func,
                *shift,
                *val,
                op::SHL,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Shr(shift, val) => self.emit_binary_op_with_result(
                func,
                *shift,
                *val,
                op::SHR,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Sar(shift, val) => self.emit_binary_op_with_result(
                func,
                *shift,
                *val,
                op::SAR,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Byte(i, x) => self.emit_binary_op_with_result(
                func,
                *i,
                *x,
                op::BYTE,
                result_value,
                liveness,
                block,
                inst_idx,
            ),

            // Comparison operations - track results for branch conditions and Select
            InstKind::Lt(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::LT,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Gt(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::GT,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::SLt(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::SLT,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::SGt(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::SGT,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Eq(a, b) => self.emit_binary_op_with_result(
                func,
                *a,
                *b,
                op::EQ,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::IsZero(a) => self.emit_unary_op_with_result(
                func,
                *a,
                op::ISZERO,
                result_value,
                liveness,
                block,
                inst_idx,
            ),

            // Memory operations
            // Track MLOAD results so they can be used as operands in subsequent instructions.
            // This is essential for nested external calls where the return value from one call
            // becomes an argument to another call.
            InstKind::MLoad(addr) => self.emit_unary_op_with_result(
                func,
                *addr,
                op::MLOAD,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::MStore(addr, val) => self.emit_store_op_live_aware(
                func,
                *addr,
                *val,
                op::MSTORE,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::MStore8(addr, val) => self.emit_store_op_live_aware(
                func,
                *addr,
                *val,
                op::MSTORE8,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::MSize => {
                self.asm.emit_op(op::MSIZE);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::Alloc { size, .. } => {
                debug_assert!(func.inst(inst_id).metadata.deferred_alloc());
                let size =
                    func.value_u64(*size).expect("deferred allocation must have a constant size");
                let alloc = self.asm.emit_deferred_alloc();
                self.pending_static_allocs.entry(func_id).or_default().push((alloc, size));
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::Fmp | InstKind::SetFmp(_) => {
                unreachable!("abstract allocation instruction reached EVM emission")
            }

            // Storage operations
            InstKind::SLoad(slot) => self.emit_unary_op_with_result(
                func,
                *slot,
                op::SLOAD,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::SStore(slot, val) => self.emit_store_op_live_aware(
                func,
                *slot,
                *val,
                op::SSTORE,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::TLoad(slot) => self.emit_unary_op_with_result(
                func,
                *slot,
                op::TLOAD,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::TStore(slot, val) => self.emit_store_op_live_aware(
                func,
                *slot,
                *val,
                op::TSTORE,
                liveness,
                block,
                inst_idx,
            ),

            // Calldata operations
            InstKind::CalldataLoad(off) => self.emit_unary_op_with_result(
                func,
                *off,
                op::CALLDATALOAD,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::CalldataSize => {
                self.asm.emit_op(op::CALLDATASIZE);
                self.scheduler.instruction_executed(0, result_value);
            }

            // Hash operations
            InstKind::Keccak256(off, len) => self.emit_binary_op_with_result(
                func,
                *off,
                *len,
                op::KECCAK256,
                result_value,
                liveness,
                block,
                inst_idx,
            ),

            // Environment operations
            InstKind::Caller => {
                self.asm.emit_op(op::CALLER);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::CallValue => {
                self.asm.emit_op(op::CALLVALUE);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::Address => {
                self.asm.emit_op(op::ADDRESS);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::Origin => {
                self.asm.emit_op(op::ORIGIN);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::GasPrice => {
                self.asm.emit_op(op::GASPRICE);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::Gas => {
                self.asm.emit_op(op::GAS);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::Timestamp => {
                self.asm.emit_op(op::TIMESTAMP);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::BlockNumber => {
                self.asm.emit_op(op::NUMBER);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::Coinbase => {
                self.asm.emit_op(op::COINBASE);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::ChainId => {
                self.asm.emit_op(op::CHAINID);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::SelfBalance => {
                self.asm.emit_op(op::SELFBALANCE);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::BaseFee => {
                self.asm.emit_op(op::BASEFEE);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::BlobBaseFee => {
                self.asm.emit_op(op::BLOBBASEFEE);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::GasLimit => {
                self.asm.emit_op(op::GASLIMIT);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::PrevRandao => {
                self.asm.emit_op(op::PREVRANDAO);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::Balance(addr) => self.emit_unary_op_with_result(
                func,
                *addr,
                op::BALANCE,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::BlockHash(num) => self.emit_unary_op_with_result(
                func,
                *num,
                op::BLOCKHASH,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::BlobHash(idx) => self.emit_unary_op_with_result(
                func,
                *idx,
                op::BLOBHASH,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::ExtCodeSize(addr) => self.emit_unary_op_with_result(
                func,
                *addr,
                op::EXTCODESIZE,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::ExtCodeHash(addr) => self.emit_unary_op_with_result(
                func,
                *addr,
                op::EXTCODEHASH,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::CodeSize => {
                self.asm.emit_op(op::CODESIZE);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::StoreImmutable(..) => {
                unreachable!("immutable stores must be lowered before EVM codegen")
            }
            InstKind::LoadImmutable(id) => {
                self.emit_load_immutable(*id);
                self.scheduler.instruction_executed(0, result_value);
            }
            InstKind::ReturnDataSize => {
                self.asm.emit_op(op::RETURNDATASIZE);
                self.scheduler.instruction_executed(0, result_value);
            }

            // Ternary operations
            InstKind::AddMod(a, b, n) => self.emit_nary_op(
                func,
                &[*n, *b, *a],
                op::ADDMOD,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::MulMod(a, b, n) => self.emit_nary_op(
                func,
                &[*n, *b, *a],
                op::MULMOD,
                result_value,
                liveness,
                block,
                inst_idx,
            ),

            // Select is like a ternary conditional
            InstKind::Select(cond, true_val, false_val) => {
                // select(cond, t, f) = f + cond * (t - f)
                //
                // We emit all three values to the stack, then do inline computation.
                // Stack notation: rightmost = top (depth 0).
                // Stack after emit_value calls: [f, t, cond] with cond on top.

                if let Some(plan) = self.plan_operands(
                    func,
                    &[*false_val, *true_val, *cond],
                    liveness,
                    block,
                    inst_idx,
                ) {
                    self.emit_operand_plan(func, plan);
                } else {
                    self.emit_value(func, *false_val); // Stack: [f]
                    self.emit_operand(func, *true_val); // Stack: [f, t]
                    self.emit_operand(func, *cond); // Stack: [f, t, cond]
                }

                // Now compute: f + cond * (t - f)
                // Stack is [f, t, cond] with cond on top (depth 0), t at depth 1, f at depth 2
                //
                // Step 1: DUP3 to get f -> [f, t, cond, f]
                self.emit_stack_op(StackOp::Dup(3));
                // Step 2: DUP3 to get t (now at depth 2) -> [f, t, cond, f, t]
                self.emit_stack_op(StackOp::Dup(3));
                // Step 3: SUB (top - second = t - f) -> [f, t, cond, t-f]
                self.emit_op_with_effect(
                    op::SUB,
                    StackEffect { pops: 2, pushes: 1 },
                    StackPush::Unknown,
                );
                // Step 4: MUL (cond * (t-f)) -> [f, t, cond*(t-f)]
                self.emit_op_with_effect(
                    op::MUL,
                    StackEffect { pops: 2, pushes: 1 },
                    StackPush::Unknown,
                );
                // Step 5: SWAP1 -> [f, cond*(t-f), t]
                self.emit_stack_op(StackOp::Swap(1));
                // Step 6: POP (remove t) -> [f, cond*(t-f)]
                self.emit_stack_op(StackOp::Pop);
                // Step 7: ADD (cond*(t-f) + f = f + cond*(t-f)) -> [result]
                let push = result_value.map_or(StackPush::Unknown, StackPush::Tracked);
                self.emit_op_with_effect(op::ADD, StackEffect { pops: 2, pushes: 1 }, push);
            }

            // Sign extend
            InstKind::SignExtend(b, x) => self.emit_binary_op_with_result(
                func,
                *b,
                *x,
                op::SIGNEXTEND,
                result_value,
                liveness,
                block,
                inst_idx,
            ),

            // Phi nodes are skipped (handled by copies)
            InstKind::Phi(_) => {}

            // Contract creation
            InstKind::Create(value, offset, size) => self.emit_nary_op(
                func,
                &[*size, *offset, *value],
                op::CREATE,
                result_value,
                liveness,
                block,
                inst_idx,
            ),
            InstKind::Create2(value, offset, size, salt) => self.emit_nary_op(
                func,
                &[*salt, *size, *offset, *value],
                op::CREATE2,
                result_value,
                liveness,
                block,
                inst_idx,
            ),

            // External calls
            //
            // These use emit_value_fresh to guarantee correct values regardless of scheduler state.
            // The stack-aware emit_op_with_effect ensures proper tracking after emission.
            InstKind::Call { gas, addr, value, args_offset, args_size, ret_offset, ret_size } => {
                // CALL(gas, addr, value, argsOffset, argsSize, retOffset, retSize)
                // EVM pops in order: gas (TOS), addr, value, argsOffset, argsSize, retOffset,
                // retSize So we push in reverse order: retSize first (deepest), gas
                // last (TOS)
                self.prepare_fresh_operands(
                    func,
                    &[*gas, *addr, *value, *args_offset, *args_size, *ret_offset, *ret_size],
                );
                self.emit_value_fresh(func, *ret_size);
                self.emit_value_fresh(func, *ret_offset);
                self.emit_value_fresh(func, *args_size);
                self.emit_value_fresh(func, *args_offset);
                self.emit_value_fresh(func, *value);
                self.emit_value_fresh(func, *addr);
                self.emit_value_fresh(func, *gas);

                // CALL consumes 7 values and produces 1 (success bool)
                let push = result_value.map_or(StackPush::Unknown, StackPush::Tracked);
                self.emit_op_with_effect(op::CALL, StackEffect { pops: 7, pushes: 1 }, push);
            }

            InstKind::CallCode {
                gas,
                addr,
                value,
                args_offset,
                args_size,
                ret_offset,
                ret_size,
            } => {
                self.prepare_fresh_operands(
                    func,
                    &[*gas, *addr, *value, *args_offset, *args_size, *ret_offset, *ret_size],
                );
                self.emit_value_fresh(func, *ret_size);
                self.emit_value_fresh(func, *ret_offset);
                self.emit_value_fresh(func, *args_size);
                self.emit_value_fresh(func, *args_offset);
                self.emit_value_fresh(func, *value);
                self.emit_value_fresh(func, *addr);
                self.emit_value_fresh(func, *gas);

                let push = result_value.map_or(StackPush::Unknown, StackPush::Tracked);
                self.emit_op_with_effect(op::CALLCODE, StackEffect { pops: 7, pushes: 1 }, push);
            }

            InstKind::StaticCall { gas, addr, args_offset, args_size, ret_offset, ret_size } => {
                // STATICCALL(gas, addr, argsOffset, argsSize, retOffset, retSize)
                self.prepare_fresh_operands(
                    func,
                    &[*gas, *addr, *args_offset, *args_size, *ret_offset, *ret_size],
                );
                self.emit_value_fresh(func, *ret_size);
                self.emit_value_fresh(func, *ret_offset);
                self.emit_value_fresh(func, *args_size);
                self.emit_value_fresh(func, *args_offset);
                self.emit_value_fresh(func, *addr);
                self.emit_value_fresh(func, *gas);
                // STATICCALL consumes 6 values and produces 1 (success bool)
                let push = result_value.map_or(StackPush::Unknown, StackPush::Tracked);
                self.emit_op_with_effect(op::STATICCALL, StackEffect { pops: 6, pushes: 1 }, push);
            }

            InstKind::DelegateCall { gas, addr, args_offset, args_size, ret_offset, ret_size } => {
                self.prepare_fresh_operands(
                    func,
                    &[*gas, *addr, *args_offset, *args_size, *ret_offset, *ret_size],
                );
                // DELEGATECALL(gas, addr, argsOffset, argsSize, retOffset, retSize)
                self.emit_value_fresh(func, *ret_size);
                self.emit_value_fresh(func, *ret_offset);
                self.emit_value_fresh(func, *args_size);
                self.emit_value_fresh(func, *args_offset);
                self.emit_value_fresh(func, *addr);
                self.emit_value_fresh(func, *gas);
                // DELEGATECALL consumes 6 values and produces 1 (success bool)
                let push = result_value.map_or(StackPush::Unknown, StackPush::Tracked);
                self.emit_op_with_effect(
                    op::DELEGATECALL,
                    StackEffect { pops: 6, pushes: 1 },
                    push,
                );
            }

            InstKind::InternalCall { function, args, returns } => {
                self.emit_internal_call(
                    func_id,
                    func,
                    *function,
                    args,
                    *returns as usize,
                    result_value,
                    liveness,
                    block,
                    inst_idx,
                );
            }

            InstKind::InternalFrameAddr(offset) => {
                self.emit_own_frame_addr(*offset);
                if let Some(result) = result_value {
                    self.scheduler.stack.push(result);
                }
            }
            InstKind::ConstructorArgsBase => {
                self.emit_constructor_args_base();
                self.scheduler.instruction_executed(0, result_value);
            }

            // Log operations
            InstKind::Log0(offset, size) => {
                // LOG0(offset, size) - stack order: offset on top, then size
                self.emit_log(func, op::LOG0, &[*size, *offset], liveness, block, inst_idx);
            }
            InstKind::Log1(offset, size, topic1) => {
                // LOG1(offset, size, topic1) - stack order: offset, size, topic1
                self.emit_log(
                    func,
                    op::LOG1,
                    &[*topic1, *size, *offset],
                    liveness,
                    block,
                    inst_idx,
                );
            }
            InstKind::Log2(offset, size, topic1, topic2) => {
                // LOG2(offset, size, topic1, topic2) - stack order: offset, size, topic1, topic2
                self.emit_log(
                    func,
                    op::LOG2,
                    &[*topic2, *topic1, *size, *offset],
                    liveness,
                    block,
                    inst_idx,
                );
            }
            InstKind::Log3(offset, size, topic1, topic2, topic3) => {
                // LOG3(offset, size, topic1, topic2, topic3)
                self.emit_log(
                    func,
                    op::LOG3,
                    &[*topic3, *topic2, *topic1, *size, *offset],
                    liveness,
                    block,
                    inst_idx,
                );
            }
            InstKind::Log4(offset, size, topic1, topic2, topic3, topic4) => {
                // LOG4(offset, size, topic1, topic2, topic3, topic4)
                self.emit_log(
                    func,
                    op::LOG4,
                    &[*topic4, *topic3, *topic2, *topic1, *size, *offset],
                    liveness,
                    block,
                    inst_idx,
                );
            }

            // Memory copy operations
            InstKind::CalldataCopy(dest, offset, size) => {
                // CALLDATACOPY(destOffset, offset, size)
                self.emit_copy_op_live_aware(
                    func,
                    &[*size, *offset, *dest],
                    op::CALLDATACOPY,
                    liveness,
                    block,
                    inst_idx,
                );
            }

            InstKind::CodeCopy(dest, offset, size) => {
                // CODECOPY(destOffset, offset, size)
                self.emit_copy_op_live_aware(
                    func,
                    &[*size, *offset, *dest],
                    op::CODECOPY,
                    liveness,
                    block,
                    inst_idx,
                );
            }

            InstKind::ReturnDataCopy(dest, offset, size) => {
                // RETURNDATACOPY(destOffset, offset, size)
                self.emit_copy_op_live_aware(
                    func,
                    &[*size, *offset, *dest],
                    op::RETURNDATACOPY,
                    liveness,
                    block,
                    inst_idx,
                );
            }

            InstKind::MCopy(dest, src, size) => {
                // MCOPY(destOffset, srcOffset, size)
                self.emit_copy_op_live_aware(
                    func,
                    &[*size, *src, *dest],
                    op::MCOPY,
                    liveness,
                    block,
                    inst_idx,
                );
            }

            InstKind::ExtCodeCopy(addr, dest, offset, size) => {
                // EXTCODECOPY(address, destOffset, offset, size)
                self.emit_copy_op_live_aware(
                    func,
                    &[*size, *offset, *dest, *addr],
                    op::EXTCODECOPY,
                    liveness,
                    block,
                    inst_idx,
                );
            }

            InstKind::MappingSlot(_, _)
            | InstKind::MappingSlotMemory(_, _)
            | InstKind::MappingSlotCalldata(_, _) => {
                unreachable!("mapping-slot builtins must be lowered before EVM codegen")
            }

            InstKind::MakeSlice { .. } | InstKind::SlicePtr(_) | InstKind::SliceLen(_) => {
                unreachable!(
                    "slice instructions must be lowered before EVM codegen: {kind:?} in `{}`",
                    func.name
                )
            }

            InstKind::MemoryObjectLen(_, _)
            | InstKind::SetMemoryObjectLen(_, _, _)
            | InstKind::MemoryObjectData(_, _)
            | InstKind::MemoryObjectFieldAddr { .. }
            | InstKind::MemoryObjectElementAddr { .. }
            | InstKind::Keccak256Bytes(_) => {
                unreachable!("memory-object instructions must be lowered before EVM codegen")
            }

            InstKind::MemoryZero(_, _) => {
                unreachable!("memory-zero instructions must be lowered before EVM codegen")
            }

            InstKind::AbiEncode { .. } => {
                unreachable!("ABI encoding must be lowered before EVM codegen")
            }

            InstKind::StorageToMemory { .. }
            | InstKind::MemoryToStorage { .. }
            | InstKind::ClearStorage { .. } => {
                unreachable!("aggregate operations must be lowered before EVM codegen")
            }
        }

        if let Some(result) = result_value
            && ((liveness.live_out(block).contains(result)
                && !self.is_stack_phi_source(block, result))
                || (self.scheduler.spills.requires_store(result)
                    && !self.scheduler.spills.is_stored(result)))
        {
            self.spill_value_if_needed(func, result);
        }

        // A constant-offset calldata load is the same physical word as the
        // corresponding external argument. Once its instruction result dies,
        // adopt a surviving stack copy as the argument instead of loading that
        // word again on the first planned edge.
        for operand in operands {
            if liveness.is_dead_after(operand, block, inst_idx)
                && let Some(&arg) = self.global_stack_aliases.get(&operand)
                && !liveness.is_dead_after(arg, block, inst_idx)
                && !self.scheduler.stack.contains(arg)
            {
                self.scheduler.stack.rename(operand, arg);
            }
        }

        // Drop dead values after the instruction
        let dead_ops = self.scheduler.drop_dead_values(liveness, block, inst_idx);
        for op in dead_ops {
            self.asm.emit_op(op.opcode());
        }
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.scheduler.depth() <= 1024);
        }
    }

    fn emit_new_internal_frame_base_tracked(&mut self) {
        self.asm.emit_push(U256::from(EvmMemoryLayout::FMP_SLOT));
        self.asm.emit_op(op::MLOAD);
        self.scheduler.stack.push_unknown();
    }

    fn emit_internal_frame_store_from_top_preserving_base(&mut self, offset: u64) {
        self.emit_stack_op(StackOp::Dup(2));
        if offset != 0 {
            self.asm.emit_push(U256::from(offset));
            self.scheduler.stack.push_unknown();
            self.emit_op_with_effect(
                op::ADD,
                StackEffect { pops: 2, pushes: 1 },
                StackPush::Unknown,
            );
        }
        self.asm.emit_op(op::MSTORE);
        self.scheduler.instruction_executed(2, None);
    }

    fn emit_store_frame_base_to_current_frame_slot(&mut self) {
        self.emit_stack_op(StackOp::Dup(1));
        self.asm.emit_push(U256::from(EvmMemoryLayout::INTERNAL_FRAME_PTR_SLOT));
        self.scheduler.stack.push_unknown();
        self.asm.emit_op(op::MSTORE);
        self.scheduler.instruction_executed(2, None);
    }

    fn emit_store_new_free_pointer_from_frame_base(&mut self, frame_size: DeferredConst) {
        self.asm.emit_push_deferred(frame_size);
        self.scheduler.stack.push_unknown();
        self.emit_op_with_effect(op::ADD, StackEffect { pops: 2, pushes: 1 }, StackPush::Unknown);
        self.asm.emit_push(U256::from(EvmMemoryLayout::FMP_SLOT));
        self.scheduler.stack.push_unknown();
        self.asm.emit_op(op::MSTORE);
        self.scheduler.instruction_executed(2, None);
    }

    /// Address of `offset` within whatever frame the frame-pointer slot
    /// currently holds. Dynamic call sites use this to reach the callee frame
    /// right after a call (before the pointer is restored); dynamic functions
    /// use it for their own frame. For accesses that are statically about the
    /// CURRENT function's own frame, use [`Self::emit_own_frame_addr`], which
    /// resolves to an absolute address when the function has a static frame.
    fn emit_current_internal_frame_addr(&mut self, offset: u64) {
        self.asm.emit_push(U256::from(EvmMemoryLayout::INTERNAL_FRAME_PTR_SLOT));
        self.asm.emit_op(op::MLOAD);
        if offset != 0 {
            self.asm.emit_push(U256::from(offset));
            self.asm.emit_op(op::ADD);
        }
    }

    fn emit_constructor_args_base(&mut self) {
        let id = self
            .constructor_args_base_const
            .expect("constructor argument base used outside constructor codegen");
        self.asm.emit_push_deferred(id);
    }

    fn emit_constructor_arg_load(&mut self, index: ArgIdx) {
        self.emit_constructor_args_base();
        let offset = index.index() as u64 * EvmMemoryLayout::WORD_SIZE;
        if offset != 0 {
            self.asm.emit_push(U256::from(offset));
            self.asm.emit_op(op::ADD);
        }
        self.asm.emit_op(op::MLOAD);
    }

    /// Address of `offset` within the current function's own frame: a single
    /// absolute push for static-frame functions, the frame-pointer indirection
    /// otherwise.
    fn emit_own_frame_addr(&mut self, offset: u64) {
        if let Some(func_id) = self.current_internal_function
            && self.static_frame_functions.contains(func_id)
        {
            let addr = self.static_frame_addr(func_id, offset);
            self.asm.emit_push_deferred(addr);
            return;
        }
        if !self.in_internal_function && !self.in_constructor {
            self.asm.emit_push(U256::from(EvmMemoryLayout::HEAP_START + offset));
            return;
        }
        self.emit_current_internal_frame_addr(offset);
    }

    /// Removes the unused dynamic-frame header and optional return word from a static frame.
    fn compact_static_frame_offset(&self, func_id: FunctionId, offset: u64) -> u64 {
        if !self.runtime_stack_args {
            return offset;
        }
        let mut compact = offset
            .checked_sub(EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE)
            .expect("static frame header is still referenced");
        if let Some(&local_base) = self.stack_return_local_bases.get(&func_id) {
            let return_base = local_base - EvmMemoryLayout::WORD_SIZE;
            debug_assert_ne!(offset, return_base, "removed stack-return slot is still referenced");
            if offset >= local_base {
                compact -= EvmMemoryLayout::WORD_SIZE;
            }
        }
        compact
    }

    /// Selects static-frame helpers that can return one word directly on the EVM stack.
    ///
    /// Tail-call edges keep the memory convention because an external dispatch path does not
    /// necessarily carry an internal return address. Calls whose MIR return arity disagrees with
    /// the callee are also excluded defensively. This convention is runtime-gas-only: size mode
    /// retains shared frame slots rather than adding stack shuffles at every return.
    fn compute_stack_return_functions(&mut self, module: &Module) {
        self.stack_return_functions = DenseBitSet::new_empty(module.functions.len());
        self.stack_return_local_bases.clear();
        if !self.gcx.sess.opts.optimization.is_gas() {
            return;
        }

        for (func_id, func) in module.functions.iter_enumerated() {
            let mut has_return = false;
            let has_consistent_returns = func.blocks.iter().all(|block| match &block.terminator {
                Some(Terminator::Return { values }) => {
                    has_return = true;
                    values.len() == 1
                }
                // The backend treats `stop` in an internal function as a void return, which is
                // incompatible with the one-word stack-return convention.
                Some(Terminator::Stop) => false,
                _ => true,
            });
            if self.static_frame_functions.contains(func_id)
                && func.returns.len() == 1
                && has_return
                && has_consistent_returns
            {
                self.stack_return_functions.insert(func_id);
            }
        }

        for (caller, func) in module.functions.iter_enumerated() {
            for inst_id in func.instructions() {
                if let InstKind::InternalCall { function, returns, .. } = &func.inst(inst_id).kind
                    && *returns != 1
                {
                    self.stack_return_functions.remove(*function);
                }
            }
            for block in &func.blocks {
                if let Some(Terminator::TailCall { function, .. }) = &block.terminator {
                    self.stack_return_functions.remove(caller);
                    self.stack_return_functions.remove(*function);
                }
            }
        }

        for func_id in self.stack_return_functions.iter() {
            let func = &module.functions[func_id];
            let local_base = EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                + ((func.params.len() + 1) as u64) * EvmMemoryLayout::WORD_SIZE;
            self.stack_return_local_bases.insert(func_id, local_base);
        }
    }

    /// Computes which arguments of each static-frame callee pass on the
    /// stack. A site can deliver a stack argument through raw re-emission after
    /// the drain for immediates and position-independently reloadable caller
    /// arguments, or through a freshness-validated spill reload for computed
    /// values. The per-argument choice is scored across all sites — raw and
    /// already-stored (cross-block) values save the four-byte frame store,
    /// while a fresh block-local value must first pay its own spill — and an
    /// argument passes on the stack when the sites' savings outweigh the
    /// callee's one-time prologue store. A callee reached by an
    /// argument-carrying tail call keeps the plain convention.
    fn compute_stack_arg_masks(&mut self, module: &Module) {
        self.stack_arg_masks.clear();
        if self.static_frame_functions.is_empty() {
            return;
        }

        let mut scores: FxHashMap<FunctionId, Vec<i32>> = FxHashMap::default();
        let mut excluded = DenseBitSet::new_empty(module.functions.len());
        for (caller_id, func) in module.functions.iter_enumerated() {
            let mut has_candidate_call = false;
            for block in func.blocks.iter() {
                if let Some(Terminator::TailCall { function, args }) = &block.terminator
                    && !args.is_empty()
                    && self.static_frame_functions.contains(*function)
                {
                    excluded.insert(*function);
                }
                has_candidate_call |= block.instructions.iter().any(|&inst_id| {
                    matches!(
                        &func.inst(inst_id).kind,
                        InstKind::InternalCall { function, .. }
                            if self.static_frame_functions.contains(*function)
                    )
                });
            }
            if !has_candidate_call {
                continue;
            }

            let caller_is_entry = Self::is_external_entry(func);
            let caller_static = self.static_frame_functions.contains(caller_id);
            let raw_leaves_ok = caller_is_entry || caller_static;
            // Where each instruction result is defined, to spot cross-block
            // arguments (already stored at their definition).
            let mut inst_block: FxHashMap<InstId, usize> = FxHashMap::default();
            let mut use_counts: FxHashMap<ValueId, usize> = FxHashMap::default();
            for (block_idx, block) in func.blocks.iter().enumerate() {
                for &inst_id in &block.instructions {
                    inst_block.insert(inst_id, block_idx);
                    for operand in func.inst(inst_id).kind.operands() {
                        *use_counts.entry(operand).or_default() += 1;
                    }
                }
                if let Some(term) = &block.terminator {
                    for operand in term.operands() {
                        *use_counts.entry(operand).or_default() += 1;
                    }
                }
            }
            for (block_idx, block) in func.blocks.iter().enumerate() {
                for &inst_id in &block.instructions {
                    let InstKind::InternalCall { function, args, .. } = &func.inst(inst_id).kind
                    else {
                        continue;
                    };
                    if !self.static_frame_functions.contains(*function) {
                        continue;
                    }
                    let score = scores.entry(*function).or_insert_with(|| vec![0; args.len()]);
                    if score.len() != args.len() {
                        excluded.insert(*function);
                        continue;
                    }
                    for (i, &arg) in args.iter().enumerate() {
                        score[i] += if Self::raw_arg_emittable(func, raw_leaves_ok, arg) {
                            // The frame store disappears outright.
                            4
                        } else if !raw_leaves_ok {
                            // A dynamic-frame caller cannot reload a spill
                            // slot without its frame pointer; it can only
                            // deliver raw values, so this argument must stay
                            // frame-passed everywhere.
                            -100_000
                        } else {
                            match func.value(arg) {
                                crate::mir::Value::Inst(def)
                                    if inst_block.get(def) != Some(&block_idx) =>
                                {
                                    // Cross-block values are stored at their
                                    // definition; the site keeps only the
                                    // slot reload it would have paid anyway.
                                    4
                                }
                                crate::mir::Value::Inst(_)
                                    if use_counts.get(&arg).copied().unwrap_or(0) > 1 =>
                                {
                                    // Multi-use block-local values usually
                                    // have a stack copy; the extra spill is
                                    // partially amortized.
                                    1
                                }
                                // A fresh single-use value pays a spill it
                                // did not need before.
                                _ => -5,
                            }
                        };
                    }
                }
            }
        }
        scores.retain(|func_id, _| {
            self.static_frame_functions.contains(*func_id) && !excluded.contains(*func_id)
        });
        let mut masks = FxHashMap::default();
        for (func_id, score) in scores {
            // The callee prologue pays one store per stack argument.
            let mut mask = DenseBitSet::new_empty(score.len());
            for (index, _) in score.iter().enumerate().filter(|(_, benefit)| **benefit > 4) {
                mask.insert(index);
            }
            if !mask.is_empty() {
                masks.insert(func_id, mask);
            }
        }
        self.stack_arg_masks = masks;
    }

    /// Selects stack-passed arguments that the callee can consume directly.
    ///
    /// This deliberately requires every selected argument to have one active value and one
    /// operand occurrence in the entry block. The value cannot then cross a control-flow edge or
    /// require a frame reload after it has been consumed. Requiring the entire stack-argument mask
    /// to qualify also lets the prologue omit every store without shuffling around retained words.
    fn compute_direct_stack_args(&mut self, module: &Module) {
        self.direct_stack_args.clear();
        if !self.gcx.sess.opts.optimization.is_gas() {
            return;
        }

        for (&func_id, mask) in &self.stack_arg_masks {
            if self.lazy_stack_args.contains_key(&func_id) {
                continue;
            }
            let func = &module.functions[func_id];
            if mask.domain_size() != func.params.len() {
                continue;
            }

            let arg_uses = func.arg_uses();
            let mut use_counts = FxHashMap::default();
            let mut entry_uses = DenseBitSet::new_empty(func.num_values());
            let mut call_uses = DenseBitSet::new_empty(func.num_values());
            for (block_id, block) in func.blocks.iter_enumerated() {
                for &inst_id in &block.instructions {
                    let is_call = matches!(func.inst(inst_id).kind, InstKind::InternalCall { .. });
                    for operand in func.inst(inst_id).kind.operands() {
                        *use_counts.entry(operand).or_insert(0usize) += 1;
                        if block_id == BlockId::ENTRY {
                            entry_uses.insert(operand);
                        }
                        if is_call {
                            call_uses.insert(operand);
                        }
                    }
                }
                if let Some(term) = &block.terminator {
                    let is_call = matches!(term, Terminator::TailCall { .. });
                    for operand in term.operands() {
                        *use_counts.entry(operand).or_insert(0usize) += 1;
                        if block_id == BlockId::ENTRY {
                            entry_uses.insert(operand);
                        }
                        if is_call {
                            call_uses.insert(operand);
                        }
                    }
                }
            }

            let mut values = Vec::with_capacity(mask.count());
            let mut eligible = true;
            for index in (0..mask.domain_size()).rev().filter(|&index| mask.contains(index)) {
                let [value] = arg_uses[ArgIdx::new(index)].as_slice() else {
                    eligible = false;
                    break;
                };
                if use_counts.get(value) != Some(&1)
                    || !entry_uses.contains(*value)
                    || call_uses.contains(*value)
                {
                    eligible = false;
                    break;
                }
                values.push(*value);
            }
            if eligible && !values.is_empty() {
                self.direct_stack_args.insert(func_id, values);
            }
        }
    }

    /// Selects stack arguments whose first memory materialization can move past their first use.
    ///
    /// The whole mask must qualify because the incoming words are contiguous above the return
    /// address. Each selected argument needs an identity used by the entry block's first
    /// instruction. A repeated argument gets a frame home immediately before that instruction; a
    /// single-use argument is consumed directly from the incoming stack. This restriction keeps
    /// the rewrite local and prevents it from changing later stack scheduling or CFG layout.
    fn compute_lazy_stack_args(&mut self, module: &Module) {
        self.lazy_stack_args.clear();
        if !self.gcx.sess.opts.optimization.is_gas() {
            return;
        }

        for (&func_id, mask) in &self.stack_arg_masks {
            let func = &module.functions[func_id];
            if mask.domain_size() != func.params.len() {
                continue;
            }

            let arg_uses = func.arg_uses();
            let mut use_counts = FxHashMap::default();
            let mut entry_first_uses = FxHashMap::default();
            let mut first_call = None;
            for (block_id, block) in func.blocks.iter_enumerated() {
                for (inst_idx, &inst_id) in block.instructions.iter().enumerate() {
                    if block_id == BlockId::ENTRY
                        && first_call.is_none()
                        && matches!(func.inst(inst_id).kind, InstKind::InternalCall { .. })
                    {
                        first_call = Some(inst_idx);
                    }
                    for operand in func.inst(inst_id).kind.operands() {
                        *use_counts.entry(operand).or_insert(0usize) += 1;
                        if block_id == BlockId::ENTRY {
                            entry_first_uses.entry(operand).or_insert(inst_idx);
                        }
                    }
                }
                if let Some(term) = &block.terminator {
                    for operand in term.operands() {
                        *use_counts.entry(operand).or_insert(0usize) += 1;
                    }
                }
            }
            let mut args = Vec::with_capacity(mask.count());
            let mut frame_values = DenseBitSet::new_empty(func.num_values());
            let mut eligible = true;
            for index in (0..mask.domain_size()).rev().filter(|&index| mask.contains(index)) {
                let values = &arg_uses[ArgIdx::new(index)];
                let Some((&value, &first_use)) = values
                    .iter()
                    .filter_map(|value| entry_first_uses.get(value).map(|first| (value, first)))
                    .min_by_key(|(_, first)| *first)
                else {
                    eligible = false;
                    break;
                };
                if first_call.is_some_and(|call| first_use >= call) {
                    eligible = false;
                    break;
                }
                if first_use != 0 {
                    eligible = false;
                    break;
                }
                args.push((ArgIdx::new(index), value));
                let total_uses = values
                    .iter()
                    .map(|value| use_counts.get(value).copied().unwrap_or(0))
                    .sum::<usize>();
                if total_uses > 1 {
                    frame_values.insert(value);
                }
            }
            if eligible && !args.is_empty() {
                self.lazy_stack_args.insert(func_id, LazyStackArgPlan { args, frame_values });
            }
        }
    }

    /// Returns true when the caller can re-emit `val` raw (untracked) after
    /// its stack drain: an immediate, or a caller argument whose reload is
    /// position independent.
    fn raw_arg_emittable(func: &Function, raw_leaves_ok: bool, val: ValueId) -> bool {
        match func.value(val) {
            crate::mir::Value::Immediate(imm) => imm.as_u256().is_some(),
            crate::mir::Value::Arg(_) => raw_leaves_ok,
            _ => false,
        }
    }

    /// Emits a mask-qualified argument without touching the scheduler model:
    /// the value lands on the physical stack for the callee prologue, below
    /// everything the caller's model describes.
    fn emit_raw_stack_arg(&mut self, func: &Function, val: ValueId, spill_slot: Option<SpillSlot>) {
        match func.value(val) {
            crate::mir::Value::Immediate(imm) => {
                self.asm.emit_push(imm.as_u256().expect("mask requires a word immediate"));
            }
            crate::mir::Value::Arg(index) => {
                if self.in_internal_function {
                    let func_id = self
                        .current_internal_function
                        .expect("internal caller has a current function");
                    let addr = self.static_frame_addr(
                        func_id,
                        EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                            + (index.index() as u64) * EvmMemoryLayout::WORD_SIZE,
                    );
                    self.asm.emit_push_deferred(addr);
                    self.asm.emit_op(op::MLOAD);
                } else {
                    self.asm.emit_push(U256::from(4 + (index.index() as u64) * 32));
                    self.asm.emit_op(op::CALLDATALOAD);
                }
            }
            crate::mir::Value::Inst(_) => {
                let slot = spill_slot.expect("computed stack argument has a validated spill slot");
                self.emit_spill_slot_addr(func, slot);
                self.asm.emit_op(op::MLOAD);
            }
            other => unreachable!("stack-arg mask admitted an unsupported value: {other:?}"),
        }
    }

    /// Stores the stack-passed arguments of `func_id` into their frame slots.
    /// Arguments were pushed in index order, so the highest index is on top;
    /// after the last store only the return address remains above the
    /// caller's drained stack.
    fn emit_stack_arg_prologue(&mut self, func_id: FunctionId, func: &Function) {
        if !self.runtime_stack_args {
            return;
        }
        if self.direct_stack_args.contains_key(&func_id)
            || self.lazy_stack_args.contains_key(&func_id)
        {
            return;
        }
        let Some(mask) = self.stack_arg_masks.get(&func_id).cloned() else { return };
        if mask.domain_size() != func.params.len() {
            return;
        }
        for i in (0..mask.domain_size()).rev() {
            if mask.contains(i) {
                let addr = self.static_frame_addr(
                    func_id,
                    EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                        + i as u64 * EvmMemoryLayout::WORD_SIZE,
                );
                self.asm.emit_push_deferred(addr);
                self.asm.emit_op(op::MSTORE);
            }
        }
    }

    /// Gives a repeated argument a valid frame home while retaining its first-use stack copy.
    fn materialize_lazy_stack_arg(&mut self, func_id: FunctionId, index: ArgIdx, value: ValueId) {
        if !self.scheduler.is_stack_only_value(value) {
            return;
        }
        let depth =
            self.scheduler.stack.find(value).unwrap_or_else(|| {
                panic!("lazy stack argument {value:?} was lost in its entry block")
            });
        assert!(depth < MAX_STACK_ACCESS, "lazy stack argument exceeded DUP16 reach");
        self.emit_stack_op(StackOp::Dup((depth + 1) as u8));

        let addr = self.static_frame_addr(
            func_id,
            EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                + (index.index() as u64) * EvmMemoryLayout::WORD_SIZE,
        );
        self.asm.emit_push_deferred(addr);
        self.scheduler.stack.push_unknown();
        self.asm.emit_op(op::MSTORE);
        self.scheduler.instruction_executed(2, None);
        self.scheduler.materialize_stack_only_value(value);
    }

    /// Materializes repeated arguments immediately before the entry block's first instruction.
    fn materialize_lazy_stack_args(
        &mut self,
        func_id: FunctionId,
        kind: &InstKind,
        block: BlockId,
        inst_idx: usize,
    ) {
        if block != BlockId::ENTRY || inst_idx != 0 {
            return;
        }
        let Some(plan) = self.lazy_stack_args.get(&func_id).cloned() else { return };
        let operands = kind.operands();
        for (index, value) in plan.args {
            debug_assert!(operands.contains(&value));
            if plan.frame_values.contains(value) {
                self.materialize_lazy_stack_arg(func_id, index, value);
            }
        }
    }

    /// Plans a bounded rotation that keeps computed arguments on the physical
    /// stack while the rest of the caller stack is drained. The resulting
    /// layout matches the existing stack-argument convention: selected
    /// arguments in descending index order above the return address.
    fn plan_retained_stack_args(
        &self,
        func: &Function,
        args: &[ValueId],
        mask: &DenseBitSet<usize>,
    ) -> Option<StackArgRetentionPlan> {
        let selected = mask.count();
        if mask.domain_size() != args.len()
            || selected == 0
            || selected > STACK_ARG_ROTATION_LIMIT
            || self.scheduler.stack.depth() > STACK_ARG_ROTATION_LIMIT + 1
        {
            return None;
        }

        // One physical word cannot fill two argument positions. Repeated
        // values keep the spill-reload path, which materializes each
        // occurrence independently.
        let mut selected_value_counts = FxHashMap::default();
        for (i, &arg) in args.iter().enumerate() {
            if mask.contains(i) && matches!(func.value(arg), crate::mir::Value::Inst(_)) {
                *selected_value_counts.entry(arg).or_insert(0usize) += 1;
            }
        }
        let candidates: Vec<_> = args
            .iter()
            .enumerate()
            .filter_map(|(i, &arg)| {
                (mask.contains(i)
                    && selected_value_counts.get(&arg) == Some(&1)
                    && self.scheduler.stack.contains(arg))
                .then_some(i)
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        self.build_stack_arg_retention_plan(args, mask, &candidates)
    }

    fn build_stack_arg_retention_plan(
        &self,
        args: &[ValueId],
        mask: &DenseBitSet<usize>,
        retained_indices: &[usize],
    ) -> Option<StackArgRetentionPlan> {
        let mut keep = FxHashMap::default();
        for &index in retained_indices {
            keep.insert(args[index], index);
        }

        let mut stack = self.scheduler.stack.as_slice().to_vec();
        let mut drain_ops = Vec::new();
        while stack.len() > keep.len() {
            let depth = stack.iter().position(|word| match word {
                Some(value) if keep.contains_key(value) => {
                    stack.iter().filter(|other| **other == *word).count() > 1
                }
                _ => true,
            })?;
            if depth > STACK_ARG_ROTATION_LIMIT {
                return None;
            }
            if depth != 0 {
                drain_ops.push(StackOp::Swap(depth as u8));
                stack.swap(0, depth);
            }
            drain_ops.push(StackOp::Pop);
            stack.remove(0);
        }

        let mut layout = Vec::with_capacity(mask.count() + 1);
        for word in stack {
            layout.push(StaticCallStackWord::Argument(*keep.get(&word?)?));
        }
        layout.insert(0, StaticCallStackWord::ReturnAddress);
        for i in 0..args.len() {
            if mask.contains(i) && !retained_indices.contains(&i) {
                layout.insert(0, StaticCallStackWord::Argument(i));
            }
        }

        let mut target: Vec<_> = (0..args.len())
            .filter(|&i| mask.contains(i))
            .map(StaticCallStackWord::Argument)
            .collect();
        target.reverse();
        target.push(StaticCallStackWord::ReturnAddress);
        if layout.len() != target.len() || layout.len() > STACK_ARG_ROTATION_LIMIT + 1 {
            return None;
        }

        let mut shuffle_ops = Vec::new();
        for target_depth in (1..layout.len()).rev() {
            if layout[target_depth] == target[target_depth] {
                continue;
            }
            let source_depth =
                layout[..=target_depth].iter().position(|&word| word == target[target_depth])?;
            if source_depth != 0 {
                shuffle_ops.push(StackOp::Swap(source_depth as u8));
                layout.swap(0, source_depth);
            }
            shuffle_ops.push(StackOp::Swap(target_depth as u8));
            layout.swap(0, target_depth);
        }
        debug_assert_eq!(layout, target);

        // Baseline drains every tracked word and reloads each computed stack
        // argument through at least PUSH1+MLOAD. A value without a stored slot
        // also pays at least DUP+PUSH1+MSTORE. Deferred addresses can only make
        // that baseline larger, so this is a conservative byte gate.
        let fresh = retained_indices
            .iter()
            .filter(|&&index| !self.scheduler.spills.is_stored(args[index]))
            .count();
        let baseline_cost = self.scheduler.stack.depth() + retained_indices.len() * 3 + fresh * 4;
        let planned_cost = drain_ops.len() + shuffle_ops.len();
        if planned_cost >= baseline_cost {
            return None;
        }

        let mut retained = DenseBitSet::new_empty(args.len());
        for &index in retained_indices {
            retained.insert(index);
        }
        Some(StackArgRetentionPlan { retained, drain_ops, shuffle_ops })
    }

    fn static_frame_addr(&mut self, func_id: FunctionId, offset: u64) -> DeferredConst {
        let offset = self.compact_static_frame_offset(func_id, offset);
        if let Some((id, references)) = self.static_frame_addr_consts.get_mut(&(func_id, offset)) {
            *references += 1;
            return *id;
        }
        let id = self.asm.new_deferred_const();
        self.static_frame_addr_consts.insert((func_id, offset), (id, 1));
        id
    }

    /// Total emitted frame size of `func_id`, including its exact spill area.
    fn emitted_frame_size(&self, module: &Module, func_id: FunctionId) -> u64 {
        let func = &module.functions[func_id];
        let header = if self.runtime_stack_args && self.static_frame_functions.contains(func_id) {
            0
        } else {
            EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
        };
        let size = header
            + ((func.params.len() + func.returns.len()) as u64) * EvmMemoryLayout::WORD_SIZE
            + func.internal_frame_size
            + self.function_spill_size(func_id);
        if self.stack_return_functions.contains(func_id) {
            size - EvmMemoryLayout::WORD_SIZE
        } else {
            size
        }
    }

    /// Places every referenced static frame and resolves the address and
    /// free-memory-pointer constants recorded during this pass.
    ///
    /// Placement is an overlay: `base(f) = region_start + depth(f)`, where
    /// `depth(f)` is the longest chain of static frames that can be live below
    /// an activation of `f`. Depth propagates along every call edge — a static
    /// caller contributes its frame size, a dynamic caller (recursive, or an
    /// external entry whose locals live below the region) only forwards its
    /// own depth, so a static function reached THROUGH a dynamic one is still
    /// placed above its static ancestors. Static functions are acyclic by
    /// construction, so every cycle in the graph is weight-zero and the
    /// relaxation converges. Functions that can never be simultaneously live
    /// end up sharing addresses; that is the point of the overlay.
    ///
    /// The heap floor moves up to `region_end`: each entry's free-pointer
    /// constant accounts for its exact spill area and every accepted static
    /// allocation, plus the overlaid helper region when one is referenced.
    fn resolve_static_frames(&mut self, module: &Module) {
        let uses_dynamic_internal_frames = !self.runtime_stack_args
            || module.functions.iter().any(|func| {
                func.instructions().any(|inst_id| {
                    matches!(
                        func.inst(inst_id).kind,
                        InstKind::InternalCall { function, .. }
                            if !self.static_frame_functions.contains(function)
                    )
                })
            });
        let low_memory_end = if uses_dynamic_internal_frames {
            EvmMemoryLayout::INTERNAL_FRAME_PTR_SLOT + EvmMemoryLayout::WORD_SIZE
        } else {
            EvmMemoryLayout::HEAP_START
        };
        let runtime_entries = std::mem::take(&mut self.runtime_entry_funcs);
        let entry_bases: FxHashMap<FunctionId, u64> = runtime_entries
            .iter()
            .copied()
            .map(|func_id| {
                (
                    func_id,
                    Self::external_spill_base(
                        &module.functions[func_id],
                        uses_dynamic_internal_frames,
                    ),
                )
            })
            .collect();
        let mut entry_ends: FxHashMap<FunctionId, u64> = runtime_entries
            .iter()
            .copied()
            .map(|func_id| (func_id, entry_bases[&func_id] + self.function_spill_size(func_id)))
            .collect();

        // Longest live-chain depth below each function, over all call edges.
        // Only emitted callers count: an unemitted function (an internal
        // `.body` clone nobody calls, unreachable dead code) stacks no real
        // frame below its callees.
        let mut edges = Vec::new();
        for (func_id, func) in module.functions.iter_enumerated() {
            if !self.function_labels.contains_key(&func_id) {
                continue;
            }
            for inst_id in func.instructions() {
                if let InstKind::InternalCall { function, .. } = func.inst(inst_id).kind {
                    edges.push((func_id, function));
                }
            }
            for block in func.blocks.iter() {
                if let Some(Terminator::TailCall { function, .. }) = &block.terminator {
                    edges.push((func_id, *function));
                }
            }
        }
        let mut depth: FxHashMap<FunctionId, u64> = FxHashMap::default();
        for _ in 0..=module.functions.len() {
            let mut changed = false;
            for &(caller, callee) in &edges {
                let mut contribution = depth.get(&caller).copied().unwrap_or(0);
                if self.static_frame_functions.contains(caller) {
                    contribution += self.emitted_frame_size(module, caller);
                }
                if contribution > depth.get(&callee).copied().unwrap_or(0) {
                    depth.insert(callee, contribution);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let placed: FxHashSet<FunctionId> =
            self.static_frame_addr_consts.keys().map(|&(func_id, _)| func_id).collect();
        let mut static_span = 0;
        for &func_id in &placed {
            let relative = depth.get(&func_id).copied().unwrap_or(0);
            static_span = static_span.max(relative + self.emitted_frame_size(module, func_id));
        }

        let layout = |max_entry_end: u64| {
            if placed.is_empty() {
                (max_entry_end, max_entry_end)
            } else {
                let start = max_entry_end.max(low_memory_end);
                (start, start + static_span)
            }
        };

        // Prefer eligible allocations before each entry's exact spill area,
        // then fall back to appending them after spills when only spill pushes
        // prevent the lower placement.
        // Entries overlay because only one runtime entry executes per call.
        // Reject any proposal that widens a shared heap/static-frame or
        // ranked-spill push.
        let mut static_alloc_sizes: FxHashMap<FunctionId, u64> = FxHashMap::default();
        let mut post_spill_entries = FxHashSet::default();
        for func_id in runtime_entries {
            let Some(allocations) = self.pending_static_allocs.remove(&func_id) else { continue };
            for (alloc, size) in allocations {
                let current_static_size = static_alloc_sizes.get(&func_id).copied().unwrap_or(0);
                let proposed_static_size = current_static_size + size;
                let current_end = entry_ends[&func_id];
                let proposed_end = current_end + size;
                let before_max = entry_ends.values().copied().max().unwrap_or(0);
                let after_max = entry_ends
                    .iter()
                    .map(|(&entry, &end)| if entry == func_id { proposed_end } else { end })
                    .max()
                    .unwrap_or(proposed_end);
                let (before_start, before_end) = layout(before_max);
                let (after_start, after_end) = layout(after_max);

                let mut addresses = Vec::with_capacity(self.static_frame_addr_consts.len() + 1);
                if self.runtime_free_memory_const.is_some() {
                    addresses.push(RelayoutAddress {
                        before: before_end,
                        after: after_end,
                        references: 1,
                    });
                }
                addresses.extend(self.static_frame_addr_consts.iter().map(
                    |(&(static_func, offset), &(_, references))| {
                        let relative = depth.get(&static_func).copied().unwrap_or(0) + offset;
                        RelayoutAddress {
                            before: before_start + relative,
                            after: after_start + relative,
                            references,
                        }
                    },
                ));
                let global_width_neutral = preserves_push_width(addresses.iter().copied());
                let spills_width_neutral =
                    self.external_spill_addr_consts.get(&func_id).is_none_or(|spills| {
                        let base = entry_bases[&func_id];
                        preserves_push_width(spills.iter().enumerate().map(
                            |(rank, &(_, references))| {
                                let offset = rank as u64 * 32;
                                RelayoutAddress {
                                    before: base + current_static_size + offset,
                                    after: base + proposed_static_size + offset,
                                    references,
                                }
                            },
                        ))
                    });

                if global_width_neutral
                    && spills_width_neutral
                    && !post_spill_entries.contains(&func_id)
                {
                    let static_address = entry_bases[&func_id] + current_static_size;
                    self.asm.set_deferred_alloc_static(alloc, U256::from(static_address));
                    entry_ends.insert(func_id, proposed_end);
                    static_alloc_sizes.insert(func_id, proposed_static_size);
                } else if global_width_neutral {
                    // If inserting before spills would widen one of their
                    // pushes, append after the exact spill area instead. Once
                    // an entry uses this suffix, later allocations must stay
                    // there so already-emitted static addresses never move.
                    self.asm.set_deferred_alloc_static(alloc, U256::from(current_end));
                    entry_ends.insert(func_id, proposed_end);
                    post_spill_entries.insert(func_id);
                } else {
                    self.asm.set_deferred_alloc_dynamic(alloc, U256::from(size));
                }
            }
        }

        // A retained candidate should always belong to an emitted external
        // entry. Lower defensively to the dynamic form if an unusual pipeline
        // shape leaves one behind.
        for (_, allocations) in self.pending_static_allocs.drain() {
            for (alloc, size) in allocations {
                self.asm.set_deferred_alloc_dynamic(alloc, U256::from(size));
            }
        }

        for (func_id, spills) in self.external_spill_addr_consts.drain() {
            let base =
                Self::external_spill_base(&module.functions[func_id], uses_dynamic_internal_frames)
                    + static_alloc_sizes.get(&func_id).copied().unwrap_or(0);
            for (rank, (id, _)) in spills.into_iter().enumerate() {
                self.asm.set_deferred_const(id, U256::from(base + rank as u64 * 32));
            }
        }

        let max_entry_end = entry_ends.values().copied().max().unwrap_or(0);
        let (region_start, region_end) = layout(max_entry_end);
        for (&(func_id, offset), &(id, _)) in &self.static_frame_addr_consts {
            let relative = depth.get(&func_id).copied().unwrap_or(0) + offset;
            self.asm.set_deferred_const(id, U256::from(region_start + relative));
        }
        if let Some(id) = self.runtime_free_memory_const.take() {
            self.asm.set_deferred_const(id, U256::from(region_end));
        }
    }

    fn external_spill_base(func: &Function, dynamic_frames_enabled: bool) -> u64 {
        let low_memory_start = if dynamic_frames_enabled && Self::uses_internal_frame_slot(func) {
            EvmMemoryLayout::INTERNAL_FRAME_PTR_SLOT + EvmMemoryLayout::WORD_SIZE
        } else {
            EvmMemoryLayout::HEAP_START
        };
        low_memory_start + func.internal_frame_size.max(func.external_static_return_size)
    }

    fn constructor_spill_base(&self, immutable_count: usize) -> u64 {
        immutable_staging_end(self.immutable_staging_base, immutable_count)
    }

    fn constructor_fixed_memory_end(&self, immutable_count: usize, spill_size: u64) -> u64 {
        self.constructor_spill_base(immutable_count)
            .checked_add(spill_size)
            .expect("constructor spill area overflow")
    }

    fn uses_internal_frame_slot(func: &Function) -> bool {
        func.instructions()
            .any(|inst_id| matches!(func.inst(inst_id).kind, InstKind::InternalCall { .. }))
    }

    fn emit_external_free_memory_start(&mut self) -> DeferredConst {
        let id = self.asm.new_deferred_const();
        self.asm.emit_push_deferred(id);
        self.asm.emit_push(U256::from(EvmMemoryLayout::FMP_SLOT));
        self.asm.emit_op(op::MSTORE);
        id
    }

    fn emit_spill_slot_addr(&mut self, func: &Function, slot: SpillSlot) {
        if self.in_internal_function {
            let spill_base = EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                + (func.params.len() as u64) * EvmMemoryLayout::WORD_SIZE
                + (func.returns.len() as u64) * EvmMemoryLayout::WORD_SIZE;
            self.emit_own_frame_addr(
                spill_base + func.internal_frame_size + u64::from(slot.offset) * 32,
            );
        } else if self.in_constructor {
            let spill_addr = self.constructor_spill_base(self.immutable_encodings.len())
                + u64::from(slot.offset) * EvmMemoryLayout::WORD_SIZE;
            self.asm.emit_push(U256::from(spill_addr));
        } else {
            // Route the address through a deferred constant and count the
            // reference; `assign_ranked_spill_addrs` renumbers the body's
            // slots hottest-first when it completes.
            let key = u64::from(slot.offset);
            let id = if let Some(entry) = self.spill_addr_consts.get_mut(&key) {
                entry.1 += 1;
                entry.0
            } else {
                let id = self.asm.new_deferred_const();
                self.spill_addr_consts.insert(key, (id, 1));
                id
            };
            self.asm.emit_push_deferred(id);
        }
    }

    /// Ranks the external body's spill slots by reference count, hottest
    /// first, so the most reloaded slots receive the shortest addresses after
    /// final layout. The ranking is a bijection over the same slot area —
    /// every site of a slot goes through one deferred constant — so sizes and
    /// disjointness are unchanged.
    fn assign_ranked_spill_addrs(&mut self, func_id: FunctionId) {
        if self.spill_addr_consts.is_empty() {
            return;
        }
        let mut slots: Vec<(u64, (DeferredConst, usize))> =
            self.spill_addr_consts.drain().collect();
        slots.sort_by(|a, b| b.1.1.cmp(&a.1.1).then(a.0.cmp(&b.0)));
        self.external_spill_addr_consts
            .insert(func_id, slots.into_iter().map(|(_, deferred)| deferred).collect());
    }

    fn emit_internal_arg_load(&mut self, index: ArgIdx) {
        self.emit_own_frame_addr(
            EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                + (index.index() as u64) * EvmMemoryLayout::WORD_SIZE,
        );
        self.asm.emit_op(op::MLOAD);
    }

    /// Returns the first internal-call result only when it is consumed. The call itself remains
    /// effectful, and additional returns are staged separately in the multi-return buffer.
    fn live_internal_call_result(
        result: Option<ValueId>,
        returns: usize,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) -> Option<ValueId> {
        result.filter(|&result| returns > 0 && !liveness.is_dead_after(result, block, inst_idx))
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_internal_call(
        &mut self,
        func_id: FunctionId,
        func: &Function,
        callee: FunctionId,
        args: &[ValueId],
        returns: usize,
        result: Option<ValueId>,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) {
        let Some(&callee_label) = self.function_labels.get(&callee) else {
            return;
        };
        let return_label = self.asm.new_label();

        // A static-frame callee needs none of the frame-pointer or
        // free-pointer bookkeeping below: its addresses are compile-time
        // constants.
        if self.static_frame_functions.contains(callee) {
            let preserved_words = self.emit_internal_call_static(
                func_id,
                func,
                callee,
                callee_label,
                return_label,
                args,
                returns,
                result,
                liveness,
                block,
                inst_idx,
            );
            self.internal_call_stack_edges.push(InternalCallStackEdge {
                caller: func_id,
                callee,
                preserved_words,
            });
            return;
        }

        self.internal_call_stack_edges.push(InternalCallStackEdge {
            caller: func_id,
            callee,
            preserved_words: 0,
        });

        // Frame layout: [reserved][saved frame ptr][args][returns][locals][spills].
        // The first slot is reserved (the return address used to live there;
        // it now travels on the EVM stack) so downstream offsets stay stable.
        // The spill suffix is only known after the callee body has emitted.
        let frame_size = self.asm.new_deferred_const();
        self.pending_frame_size_consts.push((frame_size, callee));

        // Spill values that are live after this call BEFORE consuming the
        // arguments. An argument that is also used later (e.g. a flag passed to
        // a helper and then stored, as in `tryAdd`) would otherwise be popped by
        // the arg-store loop below and then lost when the stack is cleared for
        // the call, leaving it unavailable at its later use.
        self.spill_live_stack_values(func, liveness, block, inst_idx);

        self.emit_new_internal_frame_base_tracked();

        // frame[32] = previous frame pointer
        self.asm.emit_push(U256::from(EvmMemoryLayout::INTERNAL_FRAME_PTR_SLOT));
        self.asm.emit_op(op::MLOAD);
        self.scheduler.stack.push_unknown();
        self.emit_internal_frame_store_from_top_preserving_base(32);

        for (i, &arg) in args.iter().enumerate() {
            self.emit_operand(func, arg);
            self.emit_internal_frame_store_from_top_preserving_base(
                EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                    + (i as u64) * EvmMemoryLayout::WORD_SIZE,
            );
        }

        // current_frame = frame
        self.emit_store_frame_base_to_current_frame_slot();

        // free_ptr += frame_size
        self.emit_store_new_free_pointer_from_frame_base(frame_size);

        self.pop_all_stack_values();
        self.scheduler.clear_stack();

        // The return address travels on the EVM stack, not in the frame: it is
        // pushed after the caller's stack is fully drained, so it is the only
        // physical value below the callee's execution. It is deliberately not
        // tracked by the scheduler — the model only describes the region above
        // it and every emitted DUP/SWAP/POP is model-relative, so nothing in
        // the callee can reach it. The callee's return consumes it with a bare
        // JUMP, and a tail call within the callee forwards it untouched.
        self.asm.emit_push_label(return_label);

        self.asm.emit_push_label(callee_label);
        self.asm.emit_op(op::JUMP);

        self.asm.define_label(return_label);
        self.scheduler.clear_stack();

        if let Some(result) =
            Self::live_internal_call_result(result, returns, liveness, block, inst_idx)
        {
            self.emit_current_internal_frame_addr(
                EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                    + (args.len() as u64) * EvmMemoryLayout::WORD_SIZE,
            );
            self.asm.emit_op(op::MLOAD);
            self.scheduler.stack.push(result);
            // Store the result to its reserved slot now, while it is on top.
            // Other value-producing instructions do this; internal calls did
            // not, so a reserved result (e.g. a recompute leaf of a live-out
            // cheap value) was never stored. No-op unless reserved and live.
            self.spill_top_value_if_live(func, liveness, block, inst_idx, result);
        }

        // Copy returns 2..N to an ephemeral buffer at the current free-memory
        // pointer. Keep the base below the loop and publish it through the
        // dedicated scratch word afterwards; the first return stays on the
        // stack. This happens before restoring the frame pointer while the
        // callee frame remains addressable.
        if returns > 1 {
            self.asm.emit_push(U256::from(EvmMemoryLayout::FMP_SLOT));
            self.asm.emit_op(op::MLOAD);
            self.asm.emit_push(U256::from(EvmMemoryLayout::MULTI_RETURN_BUFFER_PTR_SLOT));
            self.asm.emit_op(op::MSTORE);
            for i in 1..returns {
                self.emit_current_internal_frame_addr(
                    EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                        + (args.len() as u64) * EvmMemoryLayout::WORD_SIZE
                        + (i as u64) * EvmMemoryLayout::WORD_SIZE,
                );
                self.asm.emit_op(op::MLOAD);
                self.asm.emit_push(U256::from(EvmMemoryLayout::MULTI_RETURN_BUFFER_PTR_SLOT));
                self.asm.emit_op(op::MLOAD);
                self.asm.emit_push(U256::from((i as u64) * 32));
                self.asm.emit_op(op::ADD);
                self.asm.emit_op(op::MSTORE);
            }
        }

        // Deallocate the callee frame in strict LIFO order by restoring the
        // free memory pointer to the callee frame base. This must happen before
        // restoring the caller frame pointer because `emit_current_internal_frame_addr`
        // reads the internal-frame pointer slot. Do this only when the callee's declared
        // params/returns contain no memory pointer: memory pointer returns may
        // reference the callee's frame/heap region, and a memory pointer param lets
        // the callee install a fresh pointer into caller-visible memory. Solidity
        // allocation lowering zero-initializes new arrays/bytes/structs, so reclaimed
        // frame bytes need not be wiped.
        if self.restorable_internal_frames.contains(callee) {
            self.emit_current_internal_frame_addr(0);
            self.asm.emit_push(U256::from(EvmMemoryLayout::FMP_SLOT));
            self.asm.emit_op(op::MSTORE);
        }

        // Restore the caller frame pointer. If a result is on the stack, this leaves it there.
        self.emit_current_internal_frame_addr(32);
        self.asm.emit_op(op::MLOAD);
        self.asm.emit_push(U256::from(EvmMemoryLayout::INTERNAL_FRAME_PTR_SLOT));
        self.asm.emit_op(op::MSTORE);
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_static_call_stack(
        &self,
        func_id: FunctionId,
        func: &Function,
        callee: FunctionId,
        stack_mask: Option<&DenseBitSet<usize>>,
        returns: usize,
        result: Option<ValueId>,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) -> Option<StaticCallStackPlan> {
        let depth = self.scheduler.stack.depth();
        if !self.preserve_caller_stack
            || stack_mask.is_some()
            || !(1..MAX_STACK_ACCESS).contains(&depth)
            || self.recursive_stack_functions.contains(func_id)
            || self.recursion_reaching_functions.contains(callee)
        {
            return None;
        }

        let &next_inst = func.blocks[block].instructions.get(inst_idx + 1)?;
        let orders = Self::static_call_operand_orders(&func.inst(next_inst).kind);
        let needed = orders.first()?;
        if self.first_stack_value_not_needed_by(needed).is_some() {
            return None;
        }

        let live_result =
            Self::live_internal_call_result(result, returns, liveness, block, inst_idx);
        let mut post_call = self.scheduler.clone();
        if let Some(result) = live_result {
            post_call.stack.push(result);
        }

        let cost_model = self.operand_cost_model();
        let mut drained = self.scheduler.clone();
        let mut drain_cost = ScheduleCost::stack_drain_lower_bound(depth);
        let mut stored = FxHashSet::default();
        for value in self.scheduler.stack.iter().flatten() {
            if !liveness.is_dead_after(value, block, inst_idx)
                && Self::can_own_spill_slot(func, value)
                && !drained.spills.is_stored(value)
                && stored.insert(value)
            {
                drained.spills.allocate(value);
                drained.spills.mark_stored(value);
                drain_cost = drain_cost.plus(ScheduleCost::spill_store(cost_model));
            }
        }
        drained.clear_stack();
        if let Some(result) = live_result {
            drained.stack.push(result);
        }

        let next_idx = inst_idx + 1;
        let mut preserve_cost = None;
        let mut drained_next_cost = None;
        for operands in &orders {
            let preserved =
                self.preserved_operands_for(&post_call, func, operands, liveness, block, next_idx);
            let Some(plan) = post_call.plan_operands(
                operands,
                &preserved,
                func,
                self.gcx.sess.opts.optimization,
                self.gcx.sess.opts.evm_version,
                cost_model,
            ) else {
                continue;
            };
            let cost = plan.cost();
            if preserve_cost.is_none_or(|best: ScheduleCost| {
                cost.cmp_for(best, self.gcx.sess.opts.optimization).is_lt()
            }) {
                preserve_cost = Some(cost);
            }

            let preserved =
                self.preserved_operands_for(&drained, func, operands, liveness, block, next_idx);
            if let Some(plan) = drained.plan_operands(
                operands,
                &preserved,
                func,
                self.gcx.sess.opts.optimization,
                self.gcx.sess.opts.evm_version,
                cost_model,
            ) {
                let cost = plan.cost();
                if drained_next_cost.is_none_or(|best: ScheduleCost| {
                    cost.cmp_for(best, self.gcx.sess.opts.optimization).is_lt()
                }) {
                    drained_next_cost = Some(cost);
                }
            }
        }

        let drain_cost = drain_cost.plus(drained_next_cost?);
        preserve_cost
            .filter(|cost| cost.cmp_for(drain_cost, self.gcx.sess.opts.optimization).is_lt())
            .map(|_| StaticCallStackPlan { caller_stack: self.scheduler.stack.clone() })
    }

    fn static_call_operand_orders(kind: &InstKind) -> SmallVec<[SmallVec<[ValueId; 3]>; 2]> {
        let mut orders = SmallVec::new();
        let binary = match kind {
            InstKind::Add(a, b)
            | InstKind::Mul(a, b)
            | InstKind::And(a, b)
            | InstKind::Or(a, b)
            | InstKind::Xor(a, b)
            | InstKind::Eq(a, b)
            | InstKind::Lt(a, b)
            | InstKind::Gt(a, b)
            | InstKind::SLt(a, b)
            | InstKind::SGt(a, b) => Some((*a, *b, true)),
            InstKind::Sub(a, b)
            | InstKind::Div(a, b)
            | InstKind::SDiv(a, b)
            | InstKind::Mod(a, b)
            | InstKind::SMod(a, b)
            | InstKind::Exp(a, b)
            | InstKind::Shl(a, b)
            | InstKind::Shr(a, b)
            | InstKind::Sar(a, b)
            | InstKind::Byte(a, b)
            | InstKind::Keccak256(a, b)
            | InstKind::SignExtend(a, b) => Some((*a, *b, false)),
            _ => None,
        };
        if let Some((a, b, swappable)) = binary {
            orders.push(smallvec::smallvec![b, a]);
            if swappable && a != b {
                orders.push(smallvec::smallvec![a, b]);
            }
            return orders;
        }

        match kind {
            InstKind::Not(a)
            | InstKind::Clz(a)
            | InstKind::IsZero(a)
            | InstKind::MLoad(a)
            | InstKind::SLoad(a)
            | InstKind::TLoad(a)
            | InstKind::CalldataLoad(a)
            | InstKind::Balance(a)
            | InstKind::BlockHash(a)
            | InstKind::BlobHash(a)
            | InstKind::ExtCodeSize(a)
            | InstKind::ExtCodeHash(a) => orders.push(smallvec::smallvec![*a]),
            InstKind::AddMod(a, b, n) | InstKind::MulMod(a, b, n) => {
                orders.push(smallvec::smallvec![*n, *b, *a]);
            }
            InstKind::Select(condition, if_true, if_false) => {
                orders.push(smallvec::smallvec![*if_false, *if_true, *condition]);
            }
            _ => {}
        }
        orders
    }

    /// Call to a static-frame callee: arguments are stored at absolute
    /// addresses, the return address rides the EVM stack (same invariants as
    /// the dynamic path), and there is no frame-pointer save/update/restore
    /// and no free-pointer traffic — the callee's frame is a fixed region
    /// below the heap that its single live activation owns.
    #[allow(clippy::too_many_arguments)]
    fn emit_internal_call_static(
        &mut self,
        func_id: FunctionId,
        func: &Function,
        callee: FunctionId,
        callee_label: Label,
        return_label: Label,
        args: &[ValueId],
        returns: usize,
        result: Option<ValueId>,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) -> usize {
        let stack_mask =
            if self.runtime_stack_args { self.stack_arg_masks.get(&callee).cloned() } else { None };
        let caller_stack_plan = self.plan_static_call_stack(
            func_id,
            func,
            callee,
            stack_mask.as_ref(),
            returns,
            result,
            liveness,
            block,
            inst_idx,
        );
        if caller_stack_plan.is_none() {
            // The fallback drains the caller stack, so park every value needed after the call
            // before consuming arguments.
            self.spill_live_stack_values(func, liveness, block, inst_idx);
        }

        for (i, &arg) in args.iter().enumerate() {
            if stack_mask.as_ref().is_some_and(|mask| mask.contains(i)) {
                continue;
            }
            self.emit_operand(func, arg);
            let addr = self.static_frame_addr(
                callee,
                EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                    + (i as u64) * EvmMemoryLayout::WORD_SIZE,
            );
            self.asm.emit_push_deferred(addr);
            self.scheduler.stack.push_unknown();
            self.asm.emit_op(op::MSTORE);
            self.scheduler.instruction_executed(2, None);
        }

        let retention_plan =
            stack_mask.as_ref().and_then(|mask| self.plan_retained_stack_args(func, args, mask));

        // A computed argument not retained physically survives the drain in
        // its spill slot and is reloaded raw after it. Validate and retain the
        // exact slot before clearing the stack; failure is an invariant error
        // in every build instead of an unchecked MLOAD in release builds.
        let mut raw_spill_slots = vec![None; args.len()];
        if let Some(mask) = &stack_mask {
            for (i, &arg) in args.iter().enumerate() {
                if mask.contains(i)
                    && !retention_plan.as_ref().is_some_and(|plan| plan.retained.contains(i))
                    && matches!(func.value(arg), crate::mir::Value::Inst(_))
                {
                    let slot = if let Some(slot) = self.scheduler.reloadable_spill(arg) {
                        slot
                    } else {
                        self.emit_value(func, arg);
                        self.spill_value_if_needed(func, arg);
                        self.scheduler.reloadable_spill(arg).unwrap_or_else(|| {
                            panic!(
                                "computed stack argument {arg:?} is neither resident nor \
                                 runtime-reloadable in `{}`",
                                func.name
                            )
                        })
                    };
                    raw_spill_slots[i] = Some(slot);
                }
            }
        }

        let caller_stack = caller_stack_plan.map(|mut plan| {
            debug_assert_eq!(plan.caller_stack.as_slice(), self.scheduler.stack.as_slice());
            plan.caller_stack.inherit_max_depth(self.scheduler.stack.max_depth());
            plan.caller_stack
        });
        let preserved_words = caller_stack.as_ref().map_or(0, StackModel::depth);
        if let Some(plan) = &retention_plan {
            for &op in &plan.drain_ops {
                self.emit_stack_op(op);
            }
            debug_assert_eq!(self.scheduler.stack.depth(), plan.retained.count());
        } else if caller_stack.is_none() {
            self.pop_all_stack_values();
        }
        self.scheduler.clear_stack();

        self.asm.emit_push_label(return_label);
        // Stack-passed arguments ride above the return address, untracked by
        // the model like the return address itself; the callee prologue
        // stores them into its frame before its body runs.
        if let Some(mask) = &stack_mask {
            for (i, &arg) in args.iter().enumerate() {
                if mask.contains(i)
                    && !retention_plan.as_ref().is_some_and(|plan| plan.retained.contains(i))
                {
                    self.emit_raw_stack_arg(func, arg, raw_spill_slots[i]);
                }
            }
        }
        if let Some(plan) = &retention_plan {
            for &op in &plan.shuffle_ops {
                debug_assert!(matches!(op, StackOp::Swap(_)));
                self.asm.emit_op(op.opcode());
            }
        }
        self.asm.emit_push_label(callee_label);
        self.asm.emit_op(op::JUMP);

        self.asm.define_label(return_label);
        if let Some(caller_stack) = caller_stack {
            self.scheduler.stack = caller_stack;
        } else {
            self.scheduler.clear_stack();
        }

        if self.stack_return_functions.contains(callee) {
            debug_assert_eq!(returns, 1);
            if let Some(result) =
                Self::live_internal_call_result(result, returns, liveness, block, inst_idx)
            {
                // The callee left this word physically on the stack; adopt it into the caller's
                // model without emitting a load.
                self.scheduler.stack.push(result);
                self.spill_top_value_if_live(func, liveness, block, inst_idx, result);
            } else {
                self.asm.emit_op(op::POP);
            }
            return preserved_words;
        }

        if let Some(result) =
            Self::live_internal_call_result(result, returns, liveness, block, inst_idx)
        {
            let addr = self.static_frame_addr(
                callee,
                EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                    + (args.len() as u64) * EvmMemoryLayout::WORD_SIZE,
            );
            self.asm.emit_push_deferred(addr);
            self.asm.emit_op(op::MLOAD);
            self.scheduler.stack.push(result);
            self.spill_top_value_if_live(func, liveness, block, inst_idx, result);
        }

        // Copy return values 2..N into the same ephemeral buffer as the
        // dynamic-frame path.
        if returns > 1 {
            self.asm.emit_push(U256::from(EvmMemoryLayout::FMP_SLOT));
            self.asm.emit_op(op::MLOAD);
            self.asm.emit_push(U256::from(EvmMemoryLayout::MULTI_RETURN_BUFFER_PTR_SLOT));
            self.asm.emit_op(op::MSTORE);
            for i in 1..returns {
                let addr = self.static_frame_addr(
                    callee,
                    EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                        + ((args.len() + i) as u64) * EvmMemoryLayout::WORD_SIZE,
                );
                self.asm.emit_push_deferred(addr);
                self.asm.emit_op(op::MLOAD);
                self.asm.emit_push(U256::from(EvmMemoryLayout::MULTI_RETURN_BUFFER_PTR_SLOT));
                self.asm.emit_op(op::MLOAD);
                self.asm.emit_push(U256::from((i as u64) * 32));
                self.asm.emit_op(op::ADD);
                self.asm.emit_op(op::MSTORE);
            }
        }
        preserved_words
    }

    fn spill_live_stack_values(
        &mut self,
        func: &Function,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) {
        let stack_values: Vec<_> = self.scheduler.stack.iter().flatten().collect();
        for value in stack_values {
            if !liveness.is_dead_after(value, block, inst_idx) {
                self.spill_value_if_needed(func, value);
            }
        }
    }

    /// Emits a value to the stack.
    fn emit_value(&mut self, func: &Function, val: ValueId) {
        self.emit_value_impl(func, val, true);
    }

    /// Emits a consuming operand occurrence to the stack.
    fn emit_operand(&mut self, func: &Function, val: ValueId) {
        self.emit_value_impl(func, val, false);
    }

    /// Returns materialization costs for the active argument and spill addressing convention.
    fn operand_cost_model(&self) -> OperandCostModel {
        if self.in_internal_function
            && self
                .current_internal_function
                .is_none_or(|func_id| !self.static_frame_functions.contains(func_id))
        {
            OperandCostModel::DYNAMIC_FRAME
        } else {
            OperandCostModel::DIRECT
        }
    }

    /// Plans operand preparation for operations whose inputs remain valid while
    /// they are rearranged. Memory-mutating stores/copies and calls keep their
    /// freshness-aware emitters until the stack model represents value epochs.
    fn plan_operands(
        &self,
        func: &Function,
        operands: &[ValueId],
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) -> Option<OperandPlan> {
        let preserved =
            self.preserved_operands_for(&self.scheduler, func, operands, liveness, block, inst_idx);
        self.scheduler.plan_operands(
            operands,
            &preserved,
            func,
            self.gcx.sess.opts.optimization,
            self.gcx.sess.opts.evm_version,
            self.operand_cost_model(),
        )
    }

    fn preserved_operands_for(
        &self,
        scheduler: &StackScheduler,
        func: &Function,
        operands: &[ValueId],
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) -> SmallVec<[ValueId; 8]> {
        let mut preserved = SmallVec::<[ValueId; 8]>::new();
        for &value in operands {
            let alias_is_live = self
                .global_stack_aliases
                .get(&value)
                .is_some_and(|&alias| !liveness.is_dead_after(alias, block, inst_idx));
            let carried_arg_is_live = self.global_stack_active
                && matches!(func.value(value), crate::mir::Value::Arg(_))
                && !liveness.is_dead_after(value, block, inst_idx);
            if !preserved.contains(&value)
                && (!liveness.is_dead_after(value, block, inst_idx) || alias_is_live)
                && (!Self::is_rematerializable_value(func, value) || carried_arg_is_live)
                && (scheduler.reloadable_spill(value).is_none() || scheduler.stack.contains(value))
            {
                preserved.push(value);
            }
        }
        preserved
    }

    fn emit_operand_plan(&mut self, func: &Function, plan: OperandPlan) {
        let ops = self.scheduler.apply_operand_plan(plan);
        self.emit_scheduled_ops(func, ops);
    }

    fn emit_scheduled_ops(&mut self, func: &Function, ops: Vec<ScheduledOp>) {
        for op in ops {
            match op {
                ScheduledOp::Stack(stack_op) => {
                    self.asm.emit_op(stack_op.opcode());
                }
                ScheduledOp::PushImmediate(imm) => {
                    self.asm.emit_push(imm);
                }
                ScheduledOp::LoadSpill(slot) => {
                    // PUSH slot_offset, MLOAD
                    self.emit_spill_slot_addr(func, slot);
                    self.asm.emit_op(op::MLOAD);
                }
                ScheduledOp::LoadArg(index) => {
                    if self.in_internal_function {
                        self.emit_internal_arg_load(index);
                    } else if self.in_constructor {
                        self.emit_constructor_arg_load(index);
                    } else {
                        // Runtime function: load from calldata
                        // ABI encoding: selector (4 bytes) + args (32 bytes each)
                        // Offset = 4 + index * 32
                        let offset = 4 + (index.index() as u64) * 32;
                        self.asm.emit_push(U256::from(offset));
                        self.asm.emit_op(op::CALLDATALOAD);
                    }
                }
            }
        }
    }

    fn emit_value_impl(&mut self, func: &Function, val: ValueId, claim_top: bool) {
        if let Some(depth) = self.scheduler.stack.find(val)
            && depth >= MAX_STACK_ACCESS
            && self.scheduler.reloadable_spill(val).is_none()
            && !matches!(
                func.value(val),
                crate::mir::Value::Immediate(_) | crate::mir::Value::Arg(_)
            )
        {
            let slot = self.scheduler.spills.allocate(val);
            self.spill_deep_stack_value(func, val, slot, depth);
        }

        if self.scheduler.stack.find(val).is_none()
            && self.scheduler.should_recompute_unstored_spill(val)
        {
            self.emit_value_fresh(func, val);
            return;
        }

        let ops = if claim_top {
            self.scheduler.ensure_on_top(val, func)
        } else {
            self.scheduler.ensure_operand_on_top(val, func)
        }
        .to_vec();
        self.emit_scheduled_ops(func, ops);
    }

    /// Emits a value fresh, without trying to DUP from the stack.
    /// This is used for CALL operands where we need to guarantee correct values
    /// regardless of scheduler stack tracking state.
    fn emit_value_fresh(&mut self, func: &Function, val: ValueId) {
        match func.value(val) {
            crate::mir::Value::Immediate(imm) => {
                if let Some(u256) = imm.as_u256() {
                    self.asm.emit_push(u256);
                    self.scheduler.stack.push(val);
                }
            }
            crate::mir::Value::Arg(index) => {
                if let Some(depth) = self.scheduler.stack.find(val)
                    && depth < MAX_STACK_ACCESS
                {
                    self.emit_stack_op(StackOp::Dup(depth as u8 + 1));
                    return;
                }
                if self.in_internal_function {
                    self.emit_internal_arg_load(*index);
                } else if self.in_constructor {
                    self.emit_constructor_arg_load(*index);
                } else {
                    let offset = 4 + (index.index() as u64) * 32;
                    self.asm.emit_push(U256::from(offset));
                    self.asm.emit_op(op::CALLDATALOAD);
                }
                self.scheduler.stack.push(val);
            }
            crate::mir::Value::Inst(inst_id) => {
                // A value carried on the live stack is the current definition;
                // duplicate it instead of reloading or recomputing. A preserved
                // edge can carry a value that was never spilled, and
                // recomputing a definition such as an FMP load would observe
                // memory that changed since the definition executed.
                if let Some(depth) = self.scheduler.stack.find(val)
                    && depth < MAX_STACK_ACCESS
                {
                    self.emit_stack_op(StackOp::Dup(depth as u8 + 1));
                    return;
                }
                // For instruction results, we need to check if they're spilled
                // or if they're instruction results that produce fresh values (like GAS, MLOAD)
                if let Some(slot) = self.scheduler.spills.get(val)
                    && self.scheduler.spills.is_stored(val)
                {
                    // Load from spill slot. Reloadable covers slots whose
                    // defining block is emitted later: the definition still
                    // executes before any use at runtime.
                    self.emit_spill_slot_addr(func, slot);
                    self.asm.emit_op(op::MLOAD);
                    self.scheduler.stack.push(val);
                } else {
                    // Check if the instruction is one that we can "re-execute" to get a fresh value
                    // This handles GAS (which is always fresh) and MLOAD (which re-reads from
                    // memory)
                    let inst_kind = &func.inst(*inst_id).kind;
                    match inst_kind {
                        crate::mir::InstKind::Gas => {
                            self.asm.emit_op(op::GAS);
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::LoadImmutable(id) if !self.in_constructor => {
                            self.emit_load_immutable(*id);
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::CallValue => {
                            self.asm.emit_op(op::CALLVALUE);
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::Caller => {
                            self.asm.emit_op(op::CALLER);
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::Origin => {
                            self.asm.emit_op(op::ORIGIN);
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::CalldataSize => {
                            self.asm.emit_op(op::CALLDATASIZE);
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::InternalFrameAddr(offset) => {
                            self.emit_own_frame_addr(*offset);
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::ConstructorArgsBase => {
                            self.emit_constructor_args_base();
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::Timestamp => {
                            self.asm.emit_op(op::TIMESTAMP);
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::BlockNumber => {
                            self.asm.emit_op(op::NUMBER);
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::MLoad(offset) => {
                            // Re-reading a constant scratch location is safe, but the
                            // free-memory-pointer word moves: a pointer defined as
                            // `mload(0x40)` must reach this point through its spill
                            // slot. A slot that is reloadable but not yet stored
                            // belongs to a defining block emitted after this point
                            // that still executes first at runtime.
                            if func.value_u64(*offset) == Some(EvmMemoryLayout::FMP_SLOT) {
                                if let Some(slot) = self.scheduler.reloadable_spill(val) {
                                    self.emit_spill_slot_addr(func, slot);
                                    self.asm.emit_op(op::MLOAD);
                                    self.scheduler.stack.push(val);
                                    return;
                                }
                                panic!(
                                    "emit_value_fresh: rematerializing a stale \
                                     free-memory-pointer load: {val:?} in `{}`",
                                    func.name
                                );
                            }
                            self.emit_value_fresh(func, *offset);
                            self.asm.emit_op(op::MLOAD);
                            // Pop offset, push result
                            self.scheduler.stack.pop();
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::CalldataLoad(offset) => {
                            // Calldata is immutable, so re-reading it is
                            // always safe once the address rematerializes.
                            self.emit_value_fresh(func, *offset);
                            self.asm.emit_op(op::CALLDATALOAD);
                            // Pop offset, push result
                            self.scheduler.stack.pop();
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::Keccak256(offset, size) => {
                            // Re-emit KECCAK256 - memory content should still be valid.
                            // KECCAK256 reads s[0] = offset, s[1] = size, so emit the
                            // offset last so it ends up on top.
                            self.emit_value_fresh(func, *size);
                            self.emit_value_fresh(func, *offset);
                            self.asm.emit_op(op::KECCAK256);
                            // Pop offset and size, push result
                            self.scheduler.stack.pop();
                            self.scheduler.stack.pop();
                            self.scheduler.stack.push(val);
                        }
                        crate::mir::InstKind::Add(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::ADD, true);
                        }
                        crate::mir::InstKind::Sub(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::SUB, false);
                        }
                        crate::mir::InstKind::Mul(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::MUL, true);
                        }
                        crate::mir::InstKind::And(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::AND, true);
                        }
                        crate::mir::InstKind::Or(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::OR, true);
                        }
                        crate::mir::InstKind::Xor(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::XOR, true);
                        }
                        crate::mir::InstKind::Shl(shift, value) => {
                            self.emit_fresh_binary(func, val, *shift, *value, op::SHL, false);
                        }
                        crate::mir::InstKind::Shr(shift, value) => {
                            self.emit_fresh_binary(func, val, *shift, *value, op::SHR, false);
                        }
                        crate::mir::InstKind::Div(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::DIV, false);
                        }
                        crate::mir::InstKind::SDiv(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::SDIV, false);
                        }
                        crate::mir::InstKind::Mod(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::MOD, false);
                        }
                        crate::mir::InstKind::SMod(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::SMOD, false);
                        }
                        crate::mir::InstKind::Lt(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::LT, false);
                        }
                        crate::mir::InstKind::Gt(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::GT, false);
                        }
                        crate::mir::InstKind::SLt(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::SLT, false);
                        }
                        crate::mir::InstKind::SGt(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::SGT, false);
                        }
                        crate::mir::InstKind::Eq(a, b) => {
                            self.emit_fresh_binary(func, val, *a, *b, op::EQ, true);
                        }
                        crate::mir::InstKind::Sar(shift, value) => {
                            self.emit_fresh_binary(func, val, *shift, *value, op::SAR, false);
                        }
                        crate::mir::InstKind::SLoad(slot) => {
                            // Re-emit SLOAD. CALL operands are materialized in a
                            // tight sequence with no intervening store, so the
                            // storage slot reads the same value as the original
                            // load (same recompute contract as MLOAD above).
                            self.emit_value_fresh(func, *slot);
                            self.asm.emit_op(op::SLOAD);
                            self.scheduler.stack.pop();
                            self.scheduler.stack.push(val);
                        }
                        _ => {
                            // A value that cannot be re-executed (e.g. an
                            // internal-call result used to compute a CALL
                            // operand) is live on the stack: duplicate it rather
                            // than re-running it. If it is buried too deep to
                            // `DUP`, spill it to a reserved slot and reload.
                            if let Some(depth) = self.scheduler.stack.find(val) {
                                if depth < 16 {
                                    self.asm.emit_op(op::DUP1 + depth as u8);
                                    self.scheduler.stack.push(val);
                                } else {
                                    let slot = self.scheduler.spills.allocate(val);
                                    self.spill_deep_stack_value(func, val, slot, depth);
                                    self.emit_spill_slot_addr(func, slot);
                                    self.asm.emit_op(op::MLOAD);
                                    self.scheduler.stack.push(val);
                                }
                            } else if let Some(slot) = self.scheduler.reloadable_spill(val) {
                                // A defining block emitted later still stores
                                // this slot before the load executes at runtime.
                                self.emit_spill_slot_addr(func, slot);
                                self.asm.emit_op(op::MLOAD);
                                self.scheduler.stack.push(val);
                            } else {
                                panic!(
                                    "emit_value_fresh: value {val:?} ({:?}) is neither on the \
                                     stack, spilled, nor re-executable",
                                    func.inst(*inst_id).kind
                                );
                            }
                        }
                    }
                }
            }
            crate::mir::Value::Undef(_) => {
                // Undef values shouldn't appear in CALL operands
                panic!(
                    "emit_value_fresh: unexpected undef value {val:?}. \
                     CALL operands should be concrete values."
                );
            }
            crate::mir::Value::Error(_) => {
                // A lowering error fails compilation before codegen runs.
                panic!("emit_value_fresh: error sentinel {val:?} reached the backend");
            }
        }
    }

    fn emit_fresh_binary(
        &mut self,
        func: &Function,
        result: ValueId,
        a: ValueId,
        b: ValueId,
        opcode: u8,
        commutative: bool,
    ) {
        if commutative {
            self.emit_value_fresh(func, a);
            self.emit_value_fresh(func, b);
        } else {
            // EVM binary opcodes consume `a` from the top of stack and `b`
            // from the word below, matching the normal binary emitter.
            self.emit_value_fresh(func, b);
            self.emit_value_fresh(func, a);
        }
        self.asm.emit_op(opcode);
        self.scheduler.stack.pop();
        self.scheduler.stack.pop();
        self.scheduler.stack.push(result);
    }

    fn swapped_binary_opcode(opcode: u8) -> Option<u8> {
        Some(match opcode {
            op::ADD | op::MUL | op::AND | op::OR | op::XOR | op::EQ => opcode,
            op::LT => op::GT,
            op::GT => op::LT,
            op::SLT => op::SGT,
            op::SGT => op::SLT,
            _ => return None,
        })
    }

    /// Emits a binary operation with result tracking and liveness awareness.
    /// If an operand is still live after this instruction, we DUP it before it gets consumed.
    #[allow(clippy::too_many_arguments)]
    fn emit_binary_op_with_result(
        &mut self,
        func: &Function,
        a: ValueId,
        b: ValueId,
        opcode: u8,
        result: Option<ValueId>,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) {
        let mut selected =
            self.plan_operands(func, &[b, a], liveness, block, inst_idx).map(|plan| (opcode, plan));
        if a != b
            && selected.as_ref().is_none_or(|(_, plan)| !plan.is_free())
            && let Some(swapped_opcode) = Self::swapped_binary_opcode(opcode)
            && let Some(swapped) = self.plan_operands(func, &[a, b], liveness, block, inst_idx)
            && selected.as_ref().is_none_or(|(_, current)| {
                swapped.cost().cmp_for(current.cost(), self.gcx.sess.opts.optimization).is_lt()
            })
        {
            selected = Some((swapped_opcode, swapped));
        }
        if let Some((opcode, plan)) = selected {
            self.emit_operand_plan(func, plan);
            self.asm.emit_op(opcode);
            self.scheduler.instruction_executed(2, result);
            return;
        }

        // Check if operands are still live after this instruction.
        let a_is_live = !liveness.is_dead_after(a, block, inst_idx);

        // Special case: same operand used twice (e.g., a + a, a - a)
        if a == b {
            self.emit_value(func, a);
            if !self.block_local_copy_survives(liveness, block, a, 1) {
                self.spill_top_value_if_live(func, liveness, block, inst_idx, a);
            }
            // DUP for the second operand
            self.asm.emit_op(op::DUP1);
            self.scheduler.stack.dup(1);
            self.asm.emit_op(opcode);
            self.scheduler.instruction_executed(2, result);
            return;
        }

        // Operands that already sit on top of the tracked stack are consumed
        // in place when they are dead afterwards and own no reserved spill
        // slot, instead of being re-emitted and the stale copy nipped later
        // (`DUP2 <op> ... SWAP1 POP` becomes `<op>`).
        let a_dead_free =
            liveness.is_dead_after(a, block, inst_idx) && self.scheduler.spills.get(a).is_none();
        let b_dead_free =
            liveness.is_dead_after(b, block, inst_idx) && self.scheduler.spills.get(b).is_none();
        if self.scheduler.stack.top() == Some(a)
            && self.scheduler.stack.peek(1) == Some(b)
            && a_dead_free
            && b_dead_free
        {
            // The stack is already [b, a].
            self.asm.emit_op(opcode);
            self.scheduler.instruction_executed(2, result);
            return;
        }
        if self.scheduler.stack.top() == Some(b)
            && b_dead_free
            && self.scheduler.can_emit_value(a, func)
        {
            // b is in place below; put a above it.
            self.emit_value(func, a);
            if a_is_live
                && !Self::is_rematerializable_value(func, a)
                && !self.block_local_copy_survives(liveness, block, a, 1)
            {
                self.spill_value_if_needed(func, a);
            }
            self.asm.emit_op(opcode);
            self.scheduler.instruction_executed(2, result);
            return;
        }
        if self.scheduler.stack.top() == Some(a)
            && a_dead_free
            && self.scheduler.can_emit_value(b, func)
        {
            // a is in place; emit b above it and swap into [b, a].
            self.emit_value(func, b);
            if !self.block_local_copy_survives(liveness, block, b, 1) {
                self.spill_top_value_if_live(func, liveness, block, inst_idx, b);
            }
            self.asm.emit_op(op::SWAP1);
            self.scheduler.stack_swapped();
            self.asm.emit_op(opcode);
            self.scheduler.instruction_executed(2, result);
            return;
        }

        // Check if either operand is already on stack as an untracked value
        let a_can_emit = self.scheduler.can_emit_value(a, func);
        let b_can_emit = self.scheduler.can_emit_value(b, func);
        let has_untracked = self.scheduler.has_untracked_on_top();
        let has_untracked_at_1 = self.scheduler.has_untracked_at_depth(1);

        if !a_can_emit && b_can_emit && has_untracked {
            // a is an untracked value on top of stack, emit b, then SWAP
            self.emit_value(func, b);
            if !self.block_local_copy_survives(liveness, block, b, 1) {
                self.spill_top_value_if_live(func, liveness, block, inst_idx, b);
            }
            self.asm.emit_op(op::SWAP1);
            self.scheduler.stack_swapped();
        } else if a_can_emit && !b_can_emit && has_untracked {
            // b is an untracked value on top of stack, emit a on top
            self.emit_value(func, a);
            // Spill a if live-after (it's now at depth 0).
            if a_is_live
                && !Self::is_rematerializable_value(func, a)
                && !self.block_local_copy_survives(liveness, block, a, 1)
            {
                self.spill_value_if_needed(func, a);
            }
        } else if !a_can_emit && b_can_emit && has_untracked_at_1 {
            // a is an untracked value at depth 1, b is tracked on top
            // Stack is [b, a_untracked], need [a, b]
            self.asm.emit_op(op::SWAP1);
            self.scheduler.stack_swapped();
        } else {
            // Normal case: emit b first (bottom), then a (top)
            self.emit_value(func, b);
            if !self.block_local_copy_survives(liveness, block, b, 1) {
                self.spill_top_value_if_live(func, liveness, block, inst_idx, b);
            }
            self.emit_value(func, a);
            // Spill a if live-after (it's now at depth 0).
            if a_is_live
                && !Self::is_rematerializable_value(func, a)
                && !self.block_local_copy_survives(liveness, block, a, 1)
            {
                self.spill_value_if_needed(func, a);
            }
        }

        self.asm.emit_op(opcode);
        self.scheduler.instruction_executed(2, result);
    }

    /// Emits a unary operation with result tracking and liveness awareness.
    /// If the operand is still live after this instruction, we spill it after emitting.
    #[allow(clippy::too_many_arguments)]
    fn emit_unary_op_with_result(
        &mut self,
        func: &Function,
        a: ValueId,
        opcode: u8,
        result: Option<ValueId>,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) {
        if let Some(plan) = self.plan_operands(func, &[a], liveness, block, inst_idx) {
            self.emit_operand_plan(func, plan);
            self.asm.emit_op(opcode);
            self.scheduler.instruction_executed(1, result);
            return;
        }

        self.emit_value(func, a);
        if !self.block_local_copy_survives(liveness, block, a, 1) {
            self.spill_top_value_if_live(func, liveness, block, inst_idx, a);
        }

        self.asm.emit_op(opcode);
        self.scheduler.instruction_executed(1, result);
    }

    /// Emits a `LOG0`..=`LOG4` instruction. `operands` are given in stack order
    /// (deepest first, top last) and pushed in that order; the `LOG` then
    /// consumes all of them. Each operand still live after this instruction is
    /// spilled once it reaches the top, so a later use in the same block can
    /// reload it — the same operand-liveness handling as the arithmetic, store
    /// and copy paths. Without it, a topic value consumed by the `LOG` and used
    /// again later (e.g. an event that also stores its data word) would be lost.
    fn emit_log(
        &mut self,
        func: &Function,
        opcode: u8,
        operands: &[ValueId],
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) {
        if let Some(plan) = self.plan_operands(func, operands, liveness, block, inst_idx) {
            self.emit_operand_plan(func, plan);
            self.asm.emit_op(opcode);
            self.scheduler.instruction_executed(operands.len(), None);
            return;
        }

        for (i, &operand) in operands.iter().enumerate() {
            if i == 0 {
                self.emit_value(func, operand);
            } else {
                // Repeated operands (e.g. duplicate topics) need their own stack item.
                self.emit_operand(func, operand);
            }
            // Occurrences of `operand` emitted so far, this one included: the
            // instruction consumes that many copies net of the occurrences
            // still to be pushed.
            let seen = operands[..=i].iter().filter(|&&op| op == operand).count();
            if !self.block_local_copy_survives(liveness, block, operand, seen) {
                self.spill_top_value_if_live(func, liveness, block, inst_idx, operand);
            }
        }
        self.asm.emit_op(opcode);
        self.scheduler.instruction_executed(operands.len(), None);
    }

    /// Emits a store operation with liveness awareness.
    /// If the value operand is still live after this instruction, we spill it after emitting
    /// to preserve it for later use.
    #[allow(clippy::too_many_arguments)]
    fn emit_store_op_live_aware(
        &mut self,
        func: &Function,
        addr: ValueId,
        val: ValueId,
        opcode: u8,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) {
        // Check if addr is still live after this instruction.
        let addr_is_live = !liveness.is_dead_after(addr, block, inst_idx);

        // Operands already sitting on top of the tracked stack are consumed
        // in place when they are dead afterwards and own no reserved spill
        // slot, instead of being re-emitted and the stale copies popped later
        // (`DUP2 DUP2 MSTORE ... POP POP` becomes `MSTORE`). Mirrors the
        // binary-op fast paths.
        let addr_dead_free = !addr_is_live && self.scheduler.spills.get(addr).is_none();
        let val_dead_free = liveness.is_dead_after(val, block, inst_idx)
            && self.scheduler.spills.get(val).is_none();
        if addr_dead_free && val_dead_free && self.scheduler.stack.depth() >= 2 {
            if self.scheduler.stack.top() == Some(addr) && self.scheduler.stack.peek(1) == Some(val)
            {
                // The stack is already [addr, val].
                self.asm.emit_op(opcode);
                self.scheduler.instruction_executed(2, None);
                return;
            }
            if self.scheduler.stack.top() == Some(val) && self.scheduler.stack.peek(1) == Some(addr)
            {
                self.asm.emit_op(op::SWAP1);
                self.scheduler.stack_swapped();
                self.asm.emit_op(opcode);
                self.scheduler.instruction_executed(2, None);
                return;
            }
        }

        // Emit val
        self.emit_value(func, val);
        if !self.block_local_copy_survives(liveness, block, val, 1) {
            self.spill_top_value_if_live(func, liveness, block, inst_idx, val);
        }

        // Emit addr
        self.emit_operand(func, addr);
        // Spill addr if live-after (it's now at depth 0).
        let addr_consumed = if addr == val { 2 } else { 1 };
        if addr_is_live
            && !Self::is_rematerializable_value(func, addr)
            && !self.block_local_copy_survives(liveness, block, addr, addr_consumed)
        {
            self.spill_value_if_needed(func, addr);
        }

        self.asm.emit_op(opcode);
        self.scheduler.instruction_executed(2, None);
    }

    /// Emits a copy-style instruction (no result) with liveness awareness.
    /// `operands` are pushed in order, so the last one ends up on top of the
    /// stack; any operand still live after this instruction is spilled before
    /// the instruction consumes it, preserving it for later uses.
    fn emit_copy_op_live_aware(
        &mut self,
        func: &Function,
        operands: &[ValueId],
        opcode: u8,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) {
        for (i, &op) in operands.iter().enumerate() {
            if i == 0 {
                self.emit_value(func, op);
            } else {
                // Repeated operands need their own stack item each.
                self.emit_operand(func, op);
            }
            // See `emit_log`: copies consumed net of occurrences still to come.
            let seen = operands[..=i].iter().filter(|&&o| o == op).count();
            if !self.block_local_copy_survives(liveness, block, op, seen) {
                self.spill_top_value_if_live(func, liveness, block, inst_idx, op);
            }
        }

        self.asm.emit_op(opcode);
        self.scheduler.instruction_executed(operands.len(), None);
    }

    /// Emits an operation with liveness awareness.
    #[allow(clippy::too_many_arguments)]
    fn emit_nary_op(
        &mut self,
        func: &Function,
        operands: &[ValueId],
        opcode: u8,
        result: Option<ValueId>,
        liveness: &Liveness,
        block: BlockId,
        inst_idx: usize,
    ) {
        if let Some(plan) = self.plan_operands(func, operands, liveness, block, inst_idx) {
            self.emit_operand_plan(func, plan);
            self.asm.emit_op(opcode);
            self.scheduler.instruction_executed(operands.len(), result);
            return;
        }

        for (i, &operand) in operands.iter().enumerate() {
            if i == 0 {
                self.emit_value(func, operand);
            } else {
                self.emit_operand(func, operand);
            }
            let seen = operands[..=i].iter().filter(|&&op| op == operand).count();
            if !self.block_local_copy_survives(liveness, block, operand, seen) {
                self.spill_top_value_if_live(func, liveness, block, inst_idx, operand);
            }
        }
        self.asm.emit_op(opcode);
        self.scheduler.instruction_executed(operands.len(), result);
    }

    /// Generates a parallel copy.
    ///
    /// Phi copies move values from source to destination. The destination is typically
    /// a phi result that needs to be available in the successor block. We handle this
    /// by spilling the source value to the destination's spill slot.
    fn generate_copy(
        &mut self,
        func: &Function,
        copy: &ParallelCopy,
        temps: &mut FxHashMap<u32, ValueId>,
    ) {
        // Handle source: either a MIR value or a temporary
        match &copy.src {
            CopySource::Value(val) => {
                self.emit_operand(func, *val);
            }
            CopySource::Temp(temp_id) => {
                // Temporaries are tracked in our temps map with their ValueId
                if let Some(&temp_val) = temps.get(temp_id) {
                    // DUP the temp value to top of stack
                    if let Some(depth) = self.scheduler.stack.find(temp_val) {
                        let dup_n = (depth + 1) as u8;
                        self.asm.emit_op(op::dup(dup_n));
                        self.scheduler.stack.dup(dup_n);
                    }
                }
            }
        }

        // Handle destination: either a MIR value or a temporary
        match &copy.dst {
            CopyDest::Value(dst_val) => {
                // Spill the value on top of stack to the destination's spill slot
                // This allows the successor block to reload it
                let slot = self.scheduler.spills.reserve(*dst_val);
                self.emit_spill_slot_addr(func, slot);
                self.scheduler.stack.push_unknown();
                self.asm.emit_op(op::MSTORE);
                self.scheduler.stack.pop(); // pop the untracked offset
                self.scheduler.stack.pop(); // pop the value
                self.scheduler.spills.mark_stored(*dst_val);
            }
            CopyDest::Temp(temp_id) => {
                // Mark this temporary as defined - it's now on the stack
                // Get the ValueId of the value currently on top
                if let Some(val_on_top) = self.scheduler.stack.top() {
                    temps.insert(*temp_id, val_on_top);
                }
            }
        }
    }

    /// Pops all remaining values from the stack.
    /// This ensures the stack is empty before control flow transfer to another block.
    fn pop_all_stack_values(&mut self) {
        while self.scheduler.stack_depth() > 0 {
            self.asm.emit_op(op::POP);
            self.scheduler.stack.pop();
        }
    }

    fn emit_internal_return(&mut self, func: &Function, values: &[ValueId]) {
        if self
            .current_internal_function
            .is_some_and(|func_id| self.stack_return_functions.contains(func_id))
        {
            let [value] = values else {
                panic!("stack-return function `{}` did not return exactly one value", func.name)
            };
            self.pop_stack_values_not_needed_by(values);
            self.emit_value(func, *value);
            while self.scheduler.stack.depth() > 1 {
                self.emit_stack_op(StackOp::Swap(1));
                self.emit_stack_op(StackOp::Pop);
            }

            // The return address is the only untracked word immediately below the result. Put it
            // on top for JUMP and leave the result as the caller's physical stack input.
            self.asm.emit_op(op::SWAP1);
            self.asm.emit_op(op::JUMP);
            self.scheduler.clear_stack();
            return;
        }

        let return_base = EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
            + (func.params.len() as u64) * EvmMemoryLayout::WORD_SIZE;
        for (i, &value) in values.iter().enumerate() {
            self.emit_operand(func, value);
            self.emit_own_frame_addr(return_base + (i as u64) * 32);
            self.asm.emit_op(op::MSTORE);
            self.scheduler.stack.pop();
        }

        self.pop_all_stack_values();
        // The caller's return address is the untracked value at the bottom of
        // the stack; after popping every tracked value it is on top.
        self.asm.emit_op(op::JUMP);
    }

    fn emit_external_stop(&mut self) {
        if let Some(exit) = self.constructor_exit {
            self.asm.emit_push_label(exit);
            self.asm.emit_op(op::JUMP);
        } else {
            self.asm.emit_op(op::STOP);
        }
    }

    fn generate_terminator(
        &mut self,
        func: &Function,
        term: &Terminator,
        fallthrough: Option<BlockId>,
        preserve_stack: bool,
    ) {
        match term {
            Terminator::TailCall { function, args } => {
                // Control transfers to the target and never returns: store the
                // arguments at the callee's compile-time frame addresses and
                // jump. No return address is pushed and the caller's tracked
                // stack is not drained — whatever stays below the callee's
                // model (including the caller's own inherited return address)
                // is unreachable by model-relative operations, and the callee
                // never executes a `ret` that would consume it.
                if !args.is_empty() {
                    // `lower-evm-shaped` only forms argument-carrying tail
                    // calls to callees the backend statically frames.
                    assert!(
                        self.static_frame_functions.contains(*function),
                        "argument-carrying tail call to a non-static-frame callee"
                    );
                    for (i, &arg) in args.iter().enumerate() {
                        self.emit_operand(func, arg);
                        let addr = self.static_frame_addr(
                            *function,
                            EvmMemoryLayout::INTERNAL_FRAME_HEADER_SIZE
                                + (i as u64) * EvmMemoryLayout::WORD_SIZE,
                        );
                        self.asm.emit_push_deferred(addr);
                        self.scheduler.stack.push_unknown();
                        self.asm.emit_op(op::MSTORE);
                        self.scheduler.instruction_executed(2, None);
                    }
                }
                let label = self.function_labels[function];
                self.asm.emit_push_label(label);
                self.asm.emit_op(op::JUMP);
            }
            Terminator::Jump(target) => {
                // Pop any remaining values from the stack before jumping.
                // Each block normally starts with an empty stack, so we must
                // clean the stack before jumping — unless this edge preserves
                // its live stack into a single-predecessor target.
                if Some(*target) == fallthrough {
                    if !preserve_stack {
                        self.pop_all_stack_values();
                    }
                    return;
                }
                if !preserve_stack {
                    self.pop_all_stack_values();
                }
                self.asm.emit_push_label(self.block_labels[target]);
                self.asm.emit_op(op::JUMP);
            }

            Terminator::Branch { condition, then_block, else_block } => {
                if preserve_stack {
                    self.emit_value(func, *condition);
                } else {
                    // Retain a resident condition while draining the rest. Materializing it first
                    // can duplicate an accessible copy only to swap and pop the original.
                    self.pop_stack_values_not_needed_by(&[*condition]);
                    self.emit_value(func, *condition);
                }

                match fallthrough {
                    Some(next) if *else_block == next => {
                        // JUMPI consumes the condition; false falls through to `else_block`.
                        self.asm.emit_push_label(self.block_labels[then_block]);
                        self.asm.emit_op(op::JUMPI);
                        self.scheduler.stack.pop(); // condition consumed by JUMPI
                    }
                    Some(next) if *then_block == next => {
                        // Invert the condition so true falls through to `then_block`.
                        self.asm.emit_op(op::ISZERO);
                        self.scheduler.instruction_executed_untracked(1);
                        self.asm.emit_push_label(self.block_labels[else_block]);
                        self.asm.emit_op(op::JUMPI);
                        self.scheduler.stack.pop(); // inverted condition consumed by JUMPI
                    }
                    _ => {
                        // Neither target falls through. Route the likely-hot
                        // edge through JUMPI (16 gas) and leave the cold
                        // revert path on the trailing unconditional jump,
                        // instead of paying JUMPI + JUMP (24 gas) on the hot
                        // path.
                        if self.block_is_cold(*then_block) && !self.block_is_cold(*else_block) {
                            self.asm.emit_op(op::ISZERO);
                            self.scheduler.instruction_executed_untracked(1);
                            self.asm.emit_push_label(self.block_labels[else_block]);
                            self.asm.emit_op(op::JUMPI);
                            self.scheduler.stack.pop(); // inverted condition consumed by JUMPI

                            self.asm.emit_push_label(self.block_labels[then_block]);
                            self.asm.emit_op(op::JUMP);
                        } else {
                            // JUMPI consumes the condition
                            self.asm.emit_push_label(self.block_labels[then_block]);
                            self.asm.emit_op(op::JUMPI);
                            self.scheduler.stack.pop(); // condition consumed by JUMPI

                            self.asm.emit_push_label(self.block_labels[else_block]);
                            self.asm.emit_op(op::JUMP);
                        }
                    }
                }
            }

            Terminator::Switch { value, default, cases } => {
                self.emit_switch_terminator(func, *value, *default, cases, fallthrough);
            }

            Terminator::Return { values } => {
                if self.in_internal_function {
                    self.emit_internal_return(func, values);
                    return;
                }

                assert!(values.is_empty(), "external ABI returns with values must use ReturnData");
                self.emit_external_stop();
            }

            Terminator::Revert { offset, size } => {
                self.emit_value(func, *size);
                self.emit_operand(func, *offset);
                self.asm.emit_op(op::REVERT);
            }

            Terminator::ReturnData { offset, size } => {
                // Valid in internal functions too: a fused external body called
                // through an ABI wrapper returns straight to the external
                // caller, abandoning the internal frame.
                self.emit_value(func, *size);
                self.emit_operand(func, *offset);
                self.asm.emit_op(op::RETURN);
            }

            Terminator::Stop => {
                if self.in_internal_function {
                    self.emit_internal_return(func, &[]);
                } else {
                    self.emit_external_stop();
                }
            }

            Terminator::SelfDestruct { recipient } => {
                self.emit_value(func, *recipient);
                self.asm.emit_op(op::SELFDESTRUCT);
            }

            Terminator::Invalid => {
                self.asm.emit_op(op::INVALID);
            }
        }
    }
}

/// The artifact produced by the EVM backend.
#[derive(Clone, Debug, Default)]
pub struct EvmArtifact {
    /// Deployment (init) bytecode that, when run, returns the runtime code.
    pub deployment: Vec<u8>,
    /// Runtime bytecode, i.e. the code stored on-chain.
    pub runtime: Vec<u8>,
    /// Immutable placeholders in the runtime bytecode.
    pub(crate) immutable_references: Vec<ImmutableRef>,
    /// Final deployment-prefix EVM IR immediately before byte emission.
    pub deployment_evm_ir: Option<ir::Module>,
    /// Final runtime EVM IR immediately before byte emission.
    pub runtime_evm_ir: Option<ir::Module>,
}

impl crate::backend::Backend for EvmCodegen<'_> {
    type Output = EvmArtifact;

    fn lower_module(&mut self, module: &mut Module) -> EvmArtifact {
        self.generate_deployment_artifact(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{FunctionBuilder, Immediate, Instruction, MirType, TypeSize, Value};
    use solar_config::CompileOpts;
    use solar_interface::{Ident, Session, sym};
    use solar_sema::{Compiler, hir::Visibility};

    #[test]
    fn constructor_memory_regions_do_not_overlap() {
        let mut module = Module::new(Ident::with_dummy_span(sym::Test));
        let mut constructor = Function::new(Ident::with_dummy_span(sym::Test));
        constructor.attributes.is_constructor = true;
        constructor.internal_frame_size = 0x3000;
        module.add_function(constructor);
        let id = module.add_immutable(
            Ident::with_dummy_span(sym::x),
            MirType::UInt(TypeSize::new_int_bits(8)),
            None,
        );
        let staging_base = immutable_staging_base(&module);
        assert_eq!(staging_base, 0x3080);
        let runtime_len = staging_base as usize;

        let full_word = ImmutableRef {
            id,
            code_offset: runtime_len - 33,
            type_size: TypeSize::new_int_bits(256),
        };
        assert_eq!(EvmCodegen::runtime_copy_base(&module, runtime_len, &[full_word]), 0);

        let short =
            ImmutableRef { id, code_offset: runtime_len - 2, type_size: TypeSize::new_int_bits(8) };
        assert_eq!(EvmCodegen::runtime_copy_base(&module, runtime_len, &[short]), 0);

        let short = ImmutableRef {
            id,
            code_offset: runtime_len - 3,
            type_size: TypeSize::new_int_bits(16),
        };
        assert_eq!(
            EvmCodegen::runtime_copy_base(&module, runtime_len, &[short]),
            immutable_staging_end(staging_base, 1)
        );

        with_codegen(CompileOpts::default(), |mut codegen| {
            codegen.immutable_staging_base = staging_base;
            assert_eq!(codegen.constructor_spill_base(0), staging_base);
            assert_eq!(codegen.constructor_spill_base(1), immutable_staging_end(staging_base, 1));
            assert_eq!(
                codegen.constructor_fixed_memory_end(257, 0),
                immutable_staging_end(staging_base, 257)
            );
            assert_eq!(
                codegen.constructor_fixed_memory_end(1, 0x2000),
                immutable_staging_end(staging_base, 1) + 0x2000
            );
        });
    }

    fn with_codegen<T: Send>(opts: CompileOpts, f: impl FnOnce(EvmCodegen<'_>) -> T + Send) -> T {
        let compiler = Compiler::new(Session::builder().opts(opts).build());
        compiler.enter(|c| f(EvmCodegen::new(c.gcx())))
    }

    #[test]
    fn caller_stack_prefix_validation_rejects_overflow() {
        with_codegen(CompileOpts::default(), |mut codegen| {
            let mut module = Module::new(Ident::DUMMY);
            let mut entry = Function::new(Ident::with_dummy_span(sym::entry));
            entry.attributes.is_dispatch_entry = true;
            let entry = module.add_function(entry);
            let callee = module.add_function(Function::new(Ident::with_dummy_span(sym::Test)));

            codegen.recursive_stack_functions = DenseBitSet::new_empty(module.functions.len());
            codegen.function_stack_peaks.insert(entry, 1);
            codegen.function_stack_peaks.insert(callee, MAX_STACK_DEPTH - 1);
            codegen.internal_call_stack_edges.push(InternalCallStackEdge {
                caller: entry,
                callee,
                preserved_words: 1,
            });
            assert!(!codegen.caller_stack_prefixes_fit(&module));

            codegen.function_stack_peaks.insert(callee, MAX_STACK_DEPTH - 2);
            assert!(codegen.caller_stack_prefixes_fit(&module));
        });
    }

    #[test]
    fn empty_external_return_falls_off_end() {
        with_codegen(CompileOpts::default(), |mut codegen| {
            let mut function = Function::new(Ident::with_dummy_span(sym::Test));
            function.attributes.visibility = Visibility::External;
            FunctionBuilder::new(&mut function).ret(Vec::new());
            codegen.generate_function_body(FunctionId::from_usize(0), &function);

            assert!(codegen.asm.assemble().bytecode.is_empty());
        });
    }

    #[test]
    fn unreachable_phi_copies_do_not_leak_between_functions() {
        with_codegen(CompileOpts::default(), |mut codegen| {
            let mut first = Function::new(Ident::with_dummy_span(sym::Test));
            let mut builder = FunctionBuilder::new(&mut first);
            let unreachable_pred = builder.create_block();
            let unreachable_merge = builder.create_block();
            builder.stop();
            builder.switch_to_block(unreachable_pred);
            let value = builder.imm_u64(1);
            builder.jump(unreachable_merge);
            builder.switch_to_block(unreachable_merge);
            let value = builder.phi(vec![(unreachable_pred, value)]);
            builder.ret([value]);

            codegen.generate_function_body(FunctionId::from_usize(0), &first);
            assert!(codegen.block_copies.contains_key(&unreachable_pred));

            let mut second = Function::new(Ident::with_dummy_span(sym::Test));
            FunctionBuilder::new(&mut second).stop();
            codegen.generate_function_body(FunctionId::from_usize(1), &second);

            assert!(codegen.block_copies.is_empty());
        });
    }

    #[test]
    fn cross_block_recomputation_requires_stable_leaves() {
        let mut function = Function::new(Ident::DUMMY);
        let argument = function.alloc_param(MirType::uint256());
        let immediate = function.alloc_value(Value::Immediate(Immediate::uint256(U256::from(1))));
        let (safe_inst, safe) = function.alloc_value_inst(Instruction::new(
            InstKind::Add(argument, immediate),
            Some(MirType::uint256()),
        ));
        let (nested_safe_inst, nested_safe) = function.alloc_value_inst(Instruction::new(
            InstKind::Mul(safe, argument),
            Some(MirType::uint256()),
        ));
        let (calldata_inst, calldata) = function.alloc_value_inst(Instruction::new(
            InstKind::CalldataLoad(safe),
            Some(MirType::uint256()),
        ));
        let (calldata_safe_inst, calldata_safe) = function.alloc_value_inst(Instruction::new(
            InstKind::Add(calldata, immediate),
            Some(MirType::uint256()),
        ));
        let (context_inst, context) = function
            .alloc_value_inst(Instruction::new(InstKind::CallValue, Some(MirType::uint256())));
        let (immutable_inst, immutable) = function.alloc_value_inst(Instruction::new(
            InstKind::LoadImmutable(ImmutableId::from_usize(0)),
            Some(MirType::uint256()),
        ));
        let (mutable_inst, mutable) = function.alloc_value_inst(Instruction::new(
            InstKind::SLoad(immediate),
            Some(MirType::uint256()),
        ));
        let (unsafe_inst, unsafe_value) = function.alloc_value_inst(Instruction::new(
            InstKind::Add(mutable, immediate),
            Some(MirType::uint256()),
        ));
        function.blocks[BlockId::ENTRY].instructions.extend([
            safe_inst,
            nested_safe_inst,
            calldata_inst,
            calldata_safe_inst,
            context_inst,
            immutable_inst,
            mutable_inst,
            unsafe_inst,
        ]);
        let recomputable = EvmCodegen::cross_block_recomputable_values(&function);

        assert!(recomputable.contains(safe));
        assert!(recomputable.contains(nested_safe));
        assert!(recomputable.contains(calldata));
        assert!(recomputable.contains(calldata_safe));
        assert!(recomputable.contains(context));
        assert!(!recomputable.contains(immutable));
        assert!(!recomputable.contains(mutable));
        assert!(!recomputable.contains(unsafe_value));
    }

    #[test]
    fn cross_block_reload_excludes_phi_edge_uses() {
        let mut function = Function::new(Ident::DUMMY);
        let immediate = function.alloc_value(Value::Immediate(Immediate::uint256(U256::from(1))));
        let (edge_inst, edge_value) = function.alloc_value_inst(Instruction::new(
            InstKind::Add(immediate, immediate),
            Some(MirType::uint256()),
        ));
        let (direct_inst, direct_value) = function.alloc_value_inst(Instruction::new(
            InstKind::Mul(immediate, immediate),
            Some(MirType::uint256()),
        ));
        function.blocks[BlockId::ENTRY].instructions.extend([edge_inst, direct_inst]);

        let phi_block = function.alloc_block();
        let (phi_inst, _) = function.alloc_value_inst(Instruction::new(
            InstKind::Phi(vec![(BlockId::ENTRY, edge_value)]),
            Some(MirType::uint256()),
        ));
        function.blocks[phi_block].instructions.push(phi_inst);

        let direct_block = function.alloc_block();
        let (use_inst, _) = function.alloc_value_inst(Instruction::new(
            InstKind::Add(direct_value, immediate),
            Some(MirType::uint256()),
        ));
        function.blocks[direct_block].instructions.push(use_inst);

        let reloaded = EvmCodegen::cross_block_reload_values(&function);
        assert!(!reloaded.contains(edge_value));
        assert!(reloaded.contains(direct_value));
    }

    #[test]
    fn spill_color_accepts_only_disjoint_ranges() {
        let block0 = BlockId::from_usize(0);
        let block1 = BlockId::from_usize(1);
        let mut color = SpillColor::default();
        color.insert(&FxHashMap::from_iter([(block0, SpillLiveRange { start: 2, end: 4 })]));

        assert!(
            color.accepts(&FxHashMap::from_iter([(block0, SpillLiveRange { start: 5, end: 7 })]))
        );
        assert!(
            !color.accepts(&FxHashMap::from_iter([(block0, SpillLiveRange { start: 4, end: 7 })]))
        );
        assert!(
            color.accepts(&FxHashMap::from_iter([(block1, SpillLiveRange { start: 2, end: 4 })]))
        );
    }
}
