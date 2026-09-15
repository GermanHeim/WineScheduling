"""Regression checks for the exact WineScheduling model-size reductions."""

from itertools import combinations, product
from pathlib import Path
import re
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyomo.environ as pyo
import WineSchedulingEconomic_GDP as wine


def _named_vessel_feasible(intervals, demands, capacity, reuse_window):
    order = sorted(range(len(intervals)), key=lambda k: (intervals[k][0], k))

    def assign(pos, last_finish):
        if pos == len(order):
            return True
        k = order[pos]
        start, finish = intervals[k]
        for labels in combinations(range(capacity), demands[k]):
            if all(
                last_finish[label] is None
                or (
                    last_finish[label] <= start
                    and start <= last_finish[label] + reuse_window
                )
                for label in labels
            ):
                updated = list(last_finish)
                for label in labels:
                    updated[label] = finish
                if assign(pos + 1, tuple(updated)):
                    return True
        return False

    return assign(0, (None,) * capacity)


def _aggregate_flow_feasible(intervals, demands, capacity, reuse_window):
    order = sorted(range(len(intervals)), key=lambda k: (intervals[k][0], k))

    def allocate_sources(eligible, need, remaining, q=0):
        if q == len(eligible):
            yield need, remaining
            return
        source = eligible[q]
        for amount in range(min(need, remaining[source]) + 1):
            updated = list(remaining)
            updated[source] -= amount
            yield from allocate_sources(eligible, need - amount, tuple(updated), q + 1)

    def route(pos, remaining, fresh_used):
        if pos == len(order):
            return True
        destination = order[pos]
        start, _ = intervals[destination]
        eligible = [
            source
            for source in order[:pos]
            if intervals[source][1] <= start
            and start <= intervals[source][1] + reuse_window
        ]
        for fresh, updated in allocate_sources(
            eligible, demands[destination], remaining
        ):
            if fresh_used + fresh > capacity:
                continue
            updated = list(updated)
            updated[destination] = demands[destination]
            if route(pos + 1, tuple(updated), fresh_used + fresh):
                return True
        return False

    return route(0, (0,) * len(intervals), 0)


def _cumulative_capacity_feasible(intervals, demands, capacity):
    return all(
        sum(
            demands[k]
            for k, (start, finish) in enumerate(intervals)
            if start <= checkpoint < finish
        )
        <= capacity
        for checkpoint, _ in intervals
    )


class PerformanceReductionTests(unittest.TestCase):
    def test_two_bit_relation_has_exactly_the_four_one_hot_states(self):
        expected = {
            (0, 0): "sequence_forward",
            (0, 1): "sequence_reverse",
            (1, 0): "overlap_forward",
            (1, 1): "overlap_reverse",
        }
        distances = {
            (o, r): (o + r, o + 1 - r, 1 - o + r, 1 - o + 1 - r)
            for o, r in product((0, 1), repeat=2)
        }
        branch_names = tuple(expected.values())
        for code, distance in distances.items():
            self.assertEqual(sum(d == 0 for d in distance), 1)
            self.assertEqual(branch_names[distance.index(0)], expected[code])

    def test_aggregate_barrique_flow_matches_named_vessels_exhaustively(self):
        interval_choices = [(s, s + d) for s in range(4) for d in (1, 2)]
        checked = 0
        for intervals in product(interval_choices, repeat=3):
            for demands in product((0, 1, 2), repeat=3):
                named = _named_vessel_feasible(intervals, demands, 2, 2)
                aggregate = _aggregate_flow_feasible(intervals, demands, 2, 2)
                self.assertEqual(named, aggregate, (intervals, demands))
                checked += 1
        self.assertEqual(checked, len(interval_choices) ** 3 * 3**3)

    def test_cumulative_jar_profile_matches_named_vessels_exhaustively(self):
        interval_choices = [(s, s + d) for s in range(4) for d in (1, 2)]
        checked = 0
        for intervals in product(interval_choices, repeat=3):
            for demands in product((0, 1, 2), repeat=3):
                named = _named_vessel_feasible(
                    intervals, demands, capacity=2, reuse_window=10**9
                )
                cumulative = _cumulative_capacity_feasible(intervals, demands, 2)
                self.assertEqual(named, cumulative, (intervals, demands))
                checked += 1
        self.assertEqual(checked, len(interval_choices) ** 3 * 3**3)

    def test_cumulative_jar_profile_handles_ties_and_half_open_boundaries(self):
        intervals = ((0, 2), (0, 1), (1, 2))
        demands = (1, 1, 1)
        self.assertTrue(_cumulative_capacity_feasible(intervals, demands, 2))
        self.assertTrue(
            _named_vessel_feasible(intervals, demands, capacity=2, reuse_window=10**9)
        )

        overloaded = (2, 1, 1)
        self.assertFalse(_cumulative_capacity_feasible(intervals, overloaded, 2))
        self.assertFalse(
            _named_vessel_feasible(
                intervals, overloaded, capacity=2, reuse_window=10**9
            )
        )

    def test_barrique_paths_accept_boundary_reuse_but_not_reentry_as_fresh(self):
        timely = ((0, 1), (1, 2), (3, 4))
        demands = (1, 1, 1)
        self.assertTrue(_named_vessel_feasible(timely, demands, 1, 1))
        self.assertTrue(_aggregate_flow_feasible(timely, demands, 1, 1))

        retired = ((0, 1), (3, 4))
        self.assertFalse(_named_vessel_feasible(retired, (1, 1), 1, 1))
        self.assertFalse(_aggregate_flow_feasible(retired, (1, 1), 1, 1))

    def test_reuse_window_pruning_condition_is_complete_on_integer_grid(self):
        window = 2
        for source_earliest_finish in range(5):
            for source_latest_finish in range(source_earliest_finish, 5):
                for destination_earliest_start in range(5):
                    for destination_latest_start in range(
                        destination_earliest_start, 5
                    ):
                        retained = max(
                            destination_earliest_start, source_earliest_finish
                        ) <= min(
                            destination_latest_start,
                            source_latest_finish + window,
                        )
                        feasible = any(
                            source_finish <= destination_start
                            <= source_finish + window
                            for source_finish in range(
                                source_earliest_finish,
                                source_latest_finish + 1,
                            )
                            for destination_start in range(
                                destination_earliest_start,
                                destination_latest_start + 1,
                            )
                        )
                        self.assertEqual(retained, feasible)

    def test_task_local_event_capacity_row_would_cut_valid_schedule(self):
        # Two full-pool tasks with the same local event number are feasible when
        # seq, although their event-index count sum is twice capacity.
        intervals = ((0, 1), (1, 2))
        demands = (2, 2)
        self.assertTrue(_named_vessel_feasible(intervals, demands, 2, 2))
        self.assertGreater(sum(demands), 2)

    def test_s1_optimized_build_uses_only_proven_reductions(self):
        source = (ROOT / "scenarios" / "parameters_s1.toml").read_text(
            encoding="utf-8"
        )
        source, replacements = re.subn(
            r"^(\s*n_max\s*=\s*)\d+",
            r"\g<1>10",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        self.assertEqual(replacements, 1)
        with tempfile.TemporaryDirectory() as temporary:
            parameters = Path(temporary) / "parameters_s1_n10.toml"
            parameters.write_text(source, encoding="utf-8")
            model = wine.create_wine_scheduling_model(parameters)
        self.assertFalse(hasattr(model, "BarrStartRank"))
        self.assertFalse(hasattr(model, "BarrCapacity"))
        self.assertFalse(hasattr(model, "JarCapacity"))
        self.assertFalse(hasattr(model, "JarRelOverlap"))
        self.assertTrue(hasattr(model, "JarSeqFwd"))
        self.assertTrue(hasattr(model, "JarRelationChoice"))
        # White1 is a five-campaign, non-outsourced line.
        self.assertEqual(model._batch_ub["Cs8"], 10000.0)
        self.assertEqual(list(model.CampaignRequiredActivations), ["8"])
        self.assertEqual(pyo.value(model.A06_min["Fa8"].lower), 5.0)
        self.assertEqual(pyo.value(model.A06_min["Cs8"].lower), 5.0)
        self.assertEqual(len(model.CampaignBranchRow), 16)
        self.assertEqual(len(model.CampaignBranchColumn), 10)
        self.assertEqual(len(model.CampaignBatchConservation), 16)
        self.assertEqual(len(model.CampaignPresenceOutsource), 10)
        for arc in model.BarrReuseArc:
            i1, n1, i2, n2 = arc
            self.assertEqual(
                model.BarrReuse[arc].ub,
                min(model.NumBarr[i1, n1].ub, model.NumBarr[i2, n2].ub),
            )
        variables = list(
            model.component_data_objects(pyo.Var, active=True, descend_into=True)
        )
        constraints = list(
            model.component_data_objects(
                pyo.Constraint, active=True, descend_into=True
            )
        )
        self.assertEqual(len(variables), 24545)
        self.assertEqual(sum(v.is_binary() for v in variables), 14343)
        self.assertEqual(len(constraints), 77406)

    def test_s2_keeps_outsourcing_as_campaign_presence_alternative(self):
        model = wine.create_wine_scheduling_model(
            ROOT / "scenarios" / "parameters_s2.toml"
        )
        self.assertEqual(model._batch_ub["Cs8"], 10000.0)
        self.assertEqual(list(model.CampaignRequiredActivations), [])
        self.assertEqual(len(model.CampaignPresenceOutsource), 10)
        for k in range(1, 6):
            self.assertEqual(model.CampaignOutsource["White1", k].ub, 5000.0)

    def test_zero_inventory_states_keep_discard_available_for_balance_feasibility(self):
        model = wine.create_wine_scheduling_model(
            ROOT / "scenarios" / "parameters_s1.toml"
        )
        zero_inventory_states = set(model.SZW) | set(model.SNIS)
        self.assertEqual(set(model.SD), set(model.SI) - zero_inventory_states)
        self.assertEqual(model._nondiscardable_states, frozenset())
        for state in zero_inventory_states:
            for event in model.n:
                discard = model.Discard[state, event]
                self.assertFalse(discard.fixed)

        # The terminal zero-wait balance must not reuse Discard[s, N]
        for state in model.SZW:
            terminal_expression = str(model.h16[state].body)
            self.assertNotIn("Discard", terminal_expression)

    def test_exact_named_vessel_comparator_replaces_only_pool_representation(self):
        source = (ROOT / "scenarios" / "parameters_s1.toml").read_text(
            encoding="utf-8"
        )
        source, replacements = re.subn(
            r"^(\s*n_max\s*=\s*)\d+",
            r"\g<1>5",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        self.assertEqual(replacements, 1)
        with tempfile.TemporaryDirectory() as temporary:
            parameters = Path(temporary) / "parameters_s1_n5.toml"
            parameters.write_text(source, encoding="utf-8")
            model = wine.create_wine_scheduling_model(
                parameters,
                {"individual_barriques": True, "individual_jars": True},
            )

        self.assertTrue(hasattr(model, "BarrVesselReuse"))
        self.assertTrue(hasattr(model, "JarVesselUse"))
        self.assertTrue(hasattr(model, "JarVesselOverlapConflict"))
        self.assertFalse(hasattr(model, "BarrReuse"))
        self.assertFalse(hasattr(model, "JarUseAtStart"))
        self.assertFalse(hasattr(model, "JarCapacityAtStart"))
        self.assertEqual(len(model.BarrVesselCount), len(model._barr_pool_te))
        self.assertEqual(len(model.JarVesselCount), len(model._jar_pool_te))

    def test_s3_demand_surge_requires_all_white1_campaign_completions(self):
        model = wine.create_wine_scheduling_model(
            ROOT / "scenarios" / "parameters_s3.toml"
        )
        self.assertEqual(model._batch_ub["Cs8"], 10000.0)
        self.assertEqual(list(model.CampaignRequiredActivations), ["8"])
        self.assertEqual(pyo.value(model.A06_min["Fa8"].lower), 5.0)
        self.assertEqual(pyo.value(model.A06_min["Cs8"].lower), 5.0)
        self.assertEqual(len(model.CampaignPresenceOutsource), 10)


if __name__ == "__main__":
    unittest.main()
