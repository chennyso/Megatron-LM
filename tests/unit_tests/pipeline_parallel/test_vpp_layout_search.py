import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.vpp_layout_search import bounded_compositions, layout_string, simulate_schedule


class TestVPPLayoutSearch(unittest.TestCase):
    def test_bounded_compositions_preserve_model_depth(self):
        values = list(bounded_compositions(total=36, parts=12, minimum=2, maximum=4))
        self.assertTrue(values)
        self.assertTrue(all(len(v) == 12 and sum(v) == 36 for v in values))
        self.assertTrue(all(2 <= x <= 4 for v in values for x in v))

    def test_layout_preserves_embedding_and_loss_order(self):
        parts = layout_string((2, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 2)).split("|")
        self.assertEqual(len(parts), 12)
        self.assertTrue(parts[0].startswith("E"))
        self.assertTrue(parts[-1].endswith("L"))
        self.assertNotIn("E", "|".join(parts[1:]))
        self.assertNotIn("L", "|".join(parts[:-1]))

    def test_grouped_schedule_is_acyclic(self):
        makespan = simulate_schedule(
            stage_costs=(32.0, 44.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0),
            pp=4,
            microbatches=16,
            group_size=4,
            comm_ms=4.84,
        )
        self.assertGreater(makespan, 0.0)


if __name__ == "__main__":
    unittest.main()
