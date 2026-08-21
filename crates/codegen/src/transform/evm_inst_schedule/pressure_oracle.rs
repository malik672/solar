//! Exact semantic-pressure oracle for late MIR scheduling.

use crate::{
    analysis::Liveness,
    mir::{BlockId, Function, InstId, Instruction, Value, ValueId},
};
use solar_data_structures::{bit_set::GrowableBitSet, map::FxHashMap};
use std::ops::Range;

const TARGET: &str = "solar_codegen::evm_inst_schedule::pressure_oracle";
const MAX_INSTRUCTIONS: usize = 16;
const MAX_PHYSICAL_CANDIDATES_PER_REGION: usize = 6;
const MAX_EXHAUSTIVE_INSTRUCTIONS: usize = 8;
const MAX_TOPOLOGICAL_ORDERS: usize = 10_000;
type LiveSet = GrowableBitSet<ValueId>;

#[derive(Clone, Copy, Debug)]
pub(crate) enum PhysicalCandidateKind {
    Exhaustive,
    ExactMinimum,
    NearMinimum,
}

/// One legal semantic-pressure alternative for physical backend replay.
#[derive(Clone, Debug)]
pub(crate) struct PhysicalScheduleCandidate {
    pub(crate) kind: PhysicalCandidateKind,
    pub(crate) block: BlockId,
    pub(crate) range: Range<usize>,
    pub(crate) order: Vec<InstId>,
    pub(crate) current_peak: usize,
    pub(crate) current_area: u64,
    pub(crate) candidate_peak: usize,
    pub(crate) candidate_area: u64,
    pub(crate) exhaustive: bool,
    pub(crate) region_orders: usize,
}

/// Returns bounded semantic-pressure alternatives for analysis by the physical backend.
///
/// The input function has already received the ordinary DFS schedule. Each returned candidate
/// changes one barrier-delimited region while preserving its dependency order and every barrier.
pub(crate) fn physical_schedule_candidates(
    func: &Function,
    is_movable: impl Fn(&Instruction) -> bool,
) -> Vec<PhysicalScheduleCandidate> {
    physical_schedule_candidates_with_mode(func, is_movable, true)
}

/// Returns a small pressure-guided schedule set without enumerating every topological order.
pub(crate) fn bounded_physical_schedule_candidates(
    func: &Function,
    is_movable: impl Fn(&Instruction) -> bool,
) -> Vec<PhysicalScheduleCandidate> {
    physical_schedule_candidates_with_mode(func, is_movable, false)
}

fn physical_schedule_candidates_with_mode(
    func: &Function,
    is_movable: impl Fn(&Instruction) -> bool,
    exhaustive_small_regions: bool,
) -> Vec<PhysicalScheduleCandidate> {
    let liveness = Liveness::compute(func);
    let mut candidates = Vec::new();

    for (block_id, block) in func.blocks.iter_enumerated() {
        let boundaries = live_boundaries(&liveness, func, block_id, &block.instructions);
        let mut start = 0;
        for end in (0..=block.instructions.len()).filter(|&index| {
            index == block.instructions.len() || !is_movable(func.inst(block.instructions[index]))
        }) {
            let instructions = &block.instructions[start..end];
            if (2..=MAX_INSTRUCTIONS).contains(&instructions.len()) {
                let problem = PressureProblem::from_mir(
                    func,
                    instructions,
                    &boundaries[start],
                    &boundaries[end],
                );
                if let Some(optimal) = problem.solve_exact()
                    && let Some(current) = problem.score_order(instructions)
                {
                    let (orders, exhaustive) = if exhaustive_small_regions
                        && instructions.len() <= MAX_EXHAUSTIVE_INSTRUCTIONS
                    {
                        let (orders, complete) =
                            problem.all_topological_orders(MAX_TOPOLOGICAL_ORDERS);
                        if complete {
                            (orders, true)
                        } else {
                            (
                                problem.near_pressure_orders(
                                    &optimal,
                                    MAX_PHYSICAL_CANDIDATES_PER_REGION,
                                ),
                                false,
                            )
                        }
                    } else {
                        (
                            problem
                                .near_pressure_orders(&optimal, MAX_PHYSICAL_CANDIDATES_PER_REGION),
                            false,
                        )
                    };
                    let region_orders = orders.len();
                    for (candidate_index, compact_order) in orders.into_iter().enumerate() {
                        let cost = problem
                            .score_compact_order(&compact_order)
                            .expect("candidate must be a complete topological order");
                        let order = compact_order
                            .iter()
                            .map(|&index| instructions[index])
                            .collect::<Vec<_>>();
                        if order == instructions {
                            continue;
                        }
                        candidates.push(PhysicalScheduleCandidate {
                            kind: if exhaustive {
                                PhysicalCandidateKind::Exhaustive
                            } else if candidate_index == 0 {
                                PhysicalCandidateKind::ExactMinimum
                            } else {
                                PhysicalCandidateKind::NearMinimum
                            },
                            block: block_id,
                            range: start..end,
                            order,
                            current_peak: current.peak,
                            current_area: current.area,
                            candidate_peak: cost.peak,
                            candidate_area: cost.area,
                            exhaustive,
                            region_orders,
                        });
                    }
                }
            }
            start = end + usize::from(end < block.instructions.len());
        }
    }

    candidates
}

fn live_boundaries(
    liveness: &Liveness,
    func: &Function,
    block_id: BlockId,
    instructions: &[InstId],
) -> Vec<LiveSet> {
    let mut live = liveness.live_out(block_id).clone();
    if let Some(terminator) = &func.blocks[block_id].terminator {
        for operand in terminator.operands() {
            live.insert(operand);
        }
    }

    let mut boundaries = vec![LiveSet::with_capacity(func.num_values()); instructions.len() + 1];
    boundaries[instructions.len()] = live.clone();
    for (index, &inst_id) in instructions.iter().enumerate().rev() {
        if let Some(result) = func.inst_result_value(inst_id) {
            live.remove(result);
        }
        for operand in func.inst(inst_id).kind.operands() {
            live.insert(operand);
        }
        boundaries[index] = live.clone();
    }
    boundaries
}

pub(super) struct PressureOracle {
    liveness: Liveness,
    emit_exact: bool,
    report: bool,
    stats: PressureOracleStats,
}

impl PressureOracle {
    pub(super) fn new_if_enabled(func: &Function, emit_exact: bool) -> Option<Self> {
        let report = tracing::enabled!(target: TARGET, tracing::Level::DEBUG);
        (emit_exact || report).then(|| Self {
            liveness: Liveness::compute(func),
            emit_exact,
            report,
            stats: PressureOracleStats::default(),
        })
    }

    pub(super) fn schedule_block(
        &mut self,
        func: &Function,
        block_id: BlockId,
        original: &[InstId],
        ordered: &[InstId],
        is_movable: impl Fn(&Instruction) -> bool,
    ) -> Option<Vec<InstId>> {
        let live_boundaries = self.live_boundaries(func, block_id, original);
        let mut exact_order = self.emit_exact.then(|| Vec::with_capacity(original.len()));
        let mut segment_start = 0;
        for (index, &inst_id) in original.iter().enumerate() {
            if is_movable(func.inst(inst_id)) {
                continue;
            }
            let exact_segment = self.audit_segment(
                func,
                block_id,
                &original[segment_start..index],
                &ordered[segment_start..index],
                &live_boundaries[segment_start],
                &live_boundaries[index],
            );
            if let Some(output) = &mut exact_order {
                output.extend(
                    exact_segment.unwrap_or_else(|| ordered[segment_start..index].to_vec()),
                );
                output.push(inst_id);
            }
            segment_start = index + 1;
        }
        let exact_segment = self.audit_segment(
            func,
            block_id,
            &original[segment_start..],
            &ordered[segment_start..],
            &live_boundaries[segment_start],
            &live_boundaries[original.len()],
        );
        if let Some(output) = &mut exact_order {
            output.extend(exact_segment.unwrap_or_else(|| ordered[segment_start..].to_vec()));
        }
        exact_order
    }

    pub(super) fn report(&self, func: &Function) {
        if !self.report || self.stats.regions == 0 {
            return;
        }
        tracing::debug!(
            target: TARGET,
            function = %func.name,
            regions = self.stats.regions,
            exact_regions = self.stats.exact_regions,
            oversized_regions = self.stats.oversized_regions,
            current_peak_optimal = self.stats.current_peak_optimal,
            current_lex_optimal = self.stats.current_lex_optimal,
            peak_gap_sum = self.stats.peak_gap_sum,
            peak_gap_max = self.stats.peak_gap_max,
            area_gap_sum = self.stats.area_gap_sum,
            area_gap_max = self.stats.area_gap_max,
            "MIR scheduling pressure oracle census"
        );
    }

    fn live_boundaries(
        &self,
        func: &Function,
        block_id: BlockId,
        instructions: &[InstId],
    ) -> Vec<LiveSet> {
        live_boundaries(&self.liveness, func, block_id, instructions)
    }

    fn audit_segment(
        &mut self,
        func: &Function,
        block_id: BlockId,
        original: &[InstId],
        ordered: &[InstId],
        live_in: &LiveSet,
        live_out: &LiveSet,
    ) -> Option<Vec<InstId>> {
        if original.len() < 2 {
            return self.emit_exact.then(|| ordered.to_vec());
        }
        self.stats.regions += 1;
        if original.len() > MAX_INSTRUCTIONS {
            self.stats.oversized_regions += 1;
            return None;
        }

        let problem = PressureProblem::from_mir(func, original, live_in, live_out);
        let optimal = problem.solve_exact()?;
        let Some(current) = problem.score_order(ordered) else {
            tracing::debug!(
                target: TARGET,
                function = %func.name,
                block = block_id.index(),
                instructions = original.len(),
                "current MIR schedule is not a topological order of its pressure region"
            );
            return None;
        };
        self.stats.exact_regions += 1;
        self.stats.current_peak_optimal += usize::from(current.peak == optimal.cost.peak);
        self.stats.current_lex_optimal += usize::from(current == optimal.cost);
        let peak_gap = current.peak.saturating_sub(optimal.cost.peak);
        self.stats.peak_gap_sum += peak_gap;
        self.stats.peak_gap_max = self.stats.peak_gap_max.max(peak_gap);
        if current.peak == optimal.cost.peak {
            let area_gap = current.area.saturating_sub(optimal.cost.area);
            self.stats.area_gap_sum += area_gap;
            self.stats.area_gap_max = self.stats.area_gap_max.max(area_gap);
        }

        if current != optimal.cost {
            tracing::trace!(
                target: TARGET,
                function = %func.name,
                block = block_id.index(),
                instructions = original.len(),
                current_peak = current.peak,
                optimal_peak = optimal.cost.peak,
                current_area = current.area,
                optimal_area = optimal.cost.area,
                current_order = ?ordered,
                optimal_order = ?optimal.order.iter().map(|&index| original[index]).collect::<Vec<_>>(),
                current_kinds = ?ordered.iter().map(|&inst_id| &func.inst(inst_id).kind).collect::<Vec<_>>(),
                optimal_kinds = ?optimal.order.iter().map(|&index| &func.inst(original[index]).kind).collect::<Vec<_>>(),
                "current MIR schedule misses the exact pressure optimum"
            );
        }

        (self.emit_exact && optimal.cost.peak < current.peak)
            .then(|| optimal.order.iter().map(|&index| original[index]).collect())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PressureCost {
    peak: usize,
    area: u64,
}

#[derive(Debug)]
struct ExactPressureSchedule {
    cost: PressureCost,
    order: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct PressureValue {
    producer: u64,
    consumers: u64,
    live_in: bool,
    live_out: bool,
}

#[derive(Debug)]
struct PressureProblem {
    instructions: Vec<InstId>,
    predecessors: Vec<u64>,
    values: Vec<PressureValue>,
}

impl PressureProblem {
    fn from_mir(
        func: &Function,
        instructions: &[InstId],
        live_in: &LiveSet,
        live_out: &LiveSet,
    ) -> Self {
        debug_assert!(instructions.len() <= MAX_INSTRUCTIONS);
        let positions = instructions
            .iter()
            .enumerate()
            .map(|(index, &inst_id)| (inst_id, index))
            .collect::<FxHashMap<_, _>>();
        let mut predecessors = vec![0; instructions.len()];
        let mut values = FxHashMap::<ValueId, PressureValue>::default();

        for value in live_in.iter().filter(|&value| Self::counts_towards_pressure(func, value)) {
            values.entry(value).or_default().live_in = true;
        }
        for value in live_out.iter().filter(|&value| Self::counts_towards_pressure(func, value)) {
            values.entry(value).or_default().live_out = true;
        }
        for (index, &inst_id) in instructions.iter().enumerate() {
            let bit = 1 << index;
            for operand in func.inst(inst_id).kind.operands() {
                if let Value::Inst(dependency) = func.value(operand)
                    && let Some(&dependency_index) = positions.get(dependency)
                {
                    predecessors[index] |= 1 << dependency_index;
                }
                if Self::counts_towards_pressure(func, operand) {
                    values.entry(operand).or_default().consumers |= bit;
                }
            }
            if let Some(result) = func.inst_result_value(inst_id)
                && Self::counts_towards_pressure(func, result)
            {
                values.entry(result).or_default().producer = bit;
            }
        }

        Self {
            instructions: instructions.to_vec(),
            predecessors,
            values: values.into_values().collect(),
        }
    }

    fn counts_towards_pressure(func: &Function, value: ValueId) -> bool {
        matches!(func.value(value), Value::Inst(_) | Value::Arg(_) | Value::Error(_))
    }

    fn full_mask(&self) -> u64 {
        (1 << self.predecessors.len()) - 1
    }

    fn pressure(&self, scheduled: u64) -> usize {
        self.values
            .iter()
            .filter(|value| {
                let available = value.live_in || value.producer & scheduled != 0;
                let demanded = value.live_out || value.consumers & !scheduled != 0;
                available && demanded
            })
            .count()
    }

    fn is_ready(&self, instruction: usize, scheduled: u64) -> bool {
        let bit = 1 << instruction;
        scheduled & bit == 0 && self.predecessors[instruction] & !scheduled == 0
    }

    fn score_order(&self, instructions: &[InstId]) -> Option<PressureCost> {
        if instructions.len() != self.predecessors.len() {
            return None;
        }
        let positions = self
            .instructions
            .iter()
            .enumerate()
            .map(|(index, &inst_id)| (inst_id, index))
            .collect::<FxHashMap<_, _>>();
        let compact = instructions
            .iter()
            .map(|inst_id| positions.get(inst_id).copied())
            .collect::<Option<Vec<_>>>()?;
        self.score_compact_order(&compact)
    }

    fn score_compact_order(&self, order: &[usize]) -> Option<PressureCost> {
        if order.len() != self.predecessors.len() {
            return None;
        }
        let mut scheduled = 0;
        let initial = self.pressure(scheduled);
        let mut cost = PressureCost { peak: initial, area: initial as u64 };
        for &instruction in order {
            if instruction >= self.predecessors.len() || !self.is_ready(instruction, scheduled) {
                return None;
            }
            scheduled |= 1 << instruction;
            let pressure = self.pressure(scheduled);
            cost.peak = cost.peak.max(pressure);
            cost.area += pressure as u64;
        }
        (scheduled == self.full_mask()).then_some(cost)
    }

    fn solve_exact(&self) -> Option<ExactPressureSchedule> {
        let state_count = 1usize << self.predecessors.len();
        let full = self.full_mask() as usize;
        let pressures =
            (0..state_count).map(|scheduled| self.pressure(scheduled as u64)).collect::<Vec<_>>();
        let mut minimum_peak = vec![usize::MAX; state_count];
        minimum_peak[0] = pressures[0];

        for scheduled in 0..state_count {
            if minimum_peak[scheduled] == usize::MAX {
                continue;
            }
            for instruction in 0..self.predecessors.len() {
                if !self.is_ready(instruction, scheduled as u64) {
                    continue;
                }
                let next = scheduled | 1 << instruction;
                let peak = minimum_peak[scheduled].max(pressures[next]);
                minimum_peak[next] = minimum_peak[next].min(peak);
            }
        }
        let optimal_peak = minimum_peak[full];
        if optimal_peak == usize::MAX {
            return None;
        }

        // Minimize area only among schedules attaining the globally minimum peak. Keeping just
        // one lexicographically best prefix is unsound: a later unavoidable pressure spike can
        // equalize two different prefix peaks and make the discarded lower-area prefix optimal.
        let mut minimum_area = vec![u64::MAX; state_count];
        let mut parents = vec![None; state_count];
        minimum_area[0] = pressures[0] as u64;
        for scheduled in 0..state_count {
            if minimum_area[scheduled] == u64::MAX {
                continue;
            }
            for instruction in 0..self.predecessors.len() {
                if !self.is_ready(instruction, scheduled as u64) {
                    continue;
                }
                let next = scheduled | 1 << instruction;
                let pressure = pressures[next];
                if pressure > optimal_peak {
                    continue;
                }
                let area = minimum_area[scheduled] + pressure as u64;
                if area < minimum_area[next] {
                    minimum_area[next] = area;
                    parents[next] = Some((scheduled, instruction));
                }
            }
        }

        let mut order = Vec::with_capacity(self.predecessors.len());
        let mut scheduled = full;
        while scheduled != 0 {
            let (previous, instruction) = parents[scheduled]?;
            order.push(instruction);
            scheduled = previous;
        }
        order.reverse();
        Some(ExactPressureSchedule {
            cost: PressureCost { peak: optimal_peak, area: minimum_area[full] },
            order,
        })
    }

    /// Produces a small deterministic family whose peak is at most one above the exact minimum.
    /// Different circular ready-node priorities expose physically distinct orders without
    /// enumerating the potentially enormous set of all topological schedules.
    fn near_pressure_orders(
        &self,
        optimal: &ExactPressureSchedule,
        limit: usize,
    ) -> Vec<Vec<usize>> {
        let mut orders = vec![optimal.order.clone()];
        let instruction_count = self.predecessors.len();
        let peak_limit = optimal.cost.peak.saturating_add(1);

        for priority in 0..instruction_count.saturating_mul(2) {
            if orders.len() >= limit {
                break;
            }
            let reverse = priority >= instruction_count;
            let pivot = priority % instruction_count;
            let mut scheduled = 0;
            let mut order = Vec::with_capacity(instruction_count);
            while scheduled != self.full_mask() {
                let next = (0..instruction_count)
                    .filter(|&instruction| self.is_ready(instruction, scheduled))
                    .filter_map(|instruction| {
                        let next_scheduled = scheduled | 1 << instruction;
                        let pressure = self.pressure(next_scheduled);
                        (pressure <= peak_limit).then_some((
                            pressure,
                            if reverse {
                                (pivot + instruction_count - instruction) % instruction_count
                            } else {
                                (instruction + instruction_count - pivot) % instruction_count
                            },
                            instruction,
                        ))
                    })
                    .min()
                    .map(|(_, _, instruction)| instruction);
                let Some(instruction) = next else { break };
                order.push(instruction);
                scheduled |= 1 << instruction;
            }
            if scheduled == self.full_mask()
                && self.score_compact_order(&order).is_some_and(|cost| cost.peak <= peak_limit)
                && !orders.contains(&order)
            {
                orders.push(order);
            }
        }
        orders
    }

    /// Enumerates every topological order unless the hard order budget is exceeded.
    fn all_topological_orders(&self, limit: usize) -> (Vec<Vec<usize>>, bool) {
        fn visit(
            problem: &PressureProblem,
            scheduled: u64,
            order: &mut Vec<usize>,
            orders: &mut Vec<Vec<usize>>,
            limit: usize,
        ) -> bool {
            if orders.len() > limit {
                return false;
            }
            if scheduled == problem.full_mask() {
                orders.push(order.clone());
                return orders.len() <= limit;
            }
            for instruction in 0..problem.predecessors.len() {
                if !problem.is_ready(instruction, scheduled) {
                    continue;
                }
                order.push(instruction);
                if !visit(problem, scheduled | 1 << instruction, order, orders, limit) {
                    return false;
                }
                order.pop();
            }
            true
        }

        let mut orders = Vec::new();
        let complete = visit(self, 0, &mut Vec::new(), &mut orders, limit);
        if !complete {
            orders.clear();
        }
        (orders, complete)
    }
}

#[derive(Debug, Default)]
struct PressureOracleStats {
    regions: usize,
    exact_regions: usize,
    oversized_regions: usize,
    current_peak_optimal: usize,
    current_lex_optimal: usize,
    peak_gap_sum: usize,
    peak_gap_max: usize,
    area_gap_sum: u64,
    area_gap_max: u64,
}

#[cfg(test)]
mod tests {
    use super::{PressureCost, PressureProblem, PressureValue};

    #[test]
    fn exact_pressure_solver_closes_short_live_ranges() {
        // Two independent producer/consumer pairs. Interleaving both producers creates pressure
        // two, while completing either pair before starting the other keeps pressure at one.
        let problem = PressureProblem {
            instructions: Vec::new(),
            predecessors: vec![0, 0, 1 << 1, 1 << 0],
            values: vec![
                PressureValue { producer: 1 << 0, consumers: 1 << 3, ..Default::default() },
                PressureValue { producer: 1 << 1, consumers: 1 << 2, ..Default::default() },
            ],
        };

        assert_eq!(
            problem.score_compact_order(&[0, 1, 2, 3]),
            Some(PressureCost { peak: 2, area: 4 })
        );
        let exact = problem.solve_exact().unwrap();
        assert_eq!(exact.cost, PressureCost { peak: 1, area: 2 });
        assert_eq!(problem.score_compact_order(&exact.order), Some(exact.cost));

        let candidates = problem.near_pressure_orders(&exact, 6);
        assert!(candidates.len() > 1);
        assert!(candidates.iter().all(|order| {
            problem.score_compact_order(order).is_some_and(|cost| cost.peak <= exact.cost.peak + 1)
        }));

        let (all_orders, complete) = problem.all_topological_orders(10);
        assert!(complete);
        assert_eq!(all_orders.len(), 6);
        let (orders, complete) = problem.all_topological_orders(5);
        assert!(!complete);
        assert!(orders.is_empty());
    }

    #[test]
    fn pressure_includes_region_boundary_obligations() {
        let problem = PressureProblem {
            instructions: Vec::new(),
            predecessors: vec![0, 1 << 0],
            values: vec![
                PressureValue { live_in: true, live_out: true, ..Default::default() },
                PressureValue { producer: 1 << 0, consumers: 1 << 1, ..Default::default() },
            ],
        };

        let exact = problem.solve_exact().unwrap();
        assert_eq!(exact.cost, PressureCost { peak: 2, area: 4 });
    }
}
