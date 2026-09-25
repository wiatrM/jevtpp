import unittest

from game_session import GameSession, playable_lives


class FullGameLifecycleTests(unittest.TestCase):
    def test_three_playable_lives_and_game_over_marker(self):
        self.assertEqual([playable_lives(raw) for raw in (2, 1, 0, 255)], [3, 2, 1, 0])

    def test_flag_and_stage_change_continue_same_run_and_keep_memory_generation(self):
        run = GameSession()
        run.start({"world": 1, "stage": 1, "life": 2})
        self.assertIsNone(run.advance({"world": 1, "stage": 1, "life": 2, "flag_get": True},
                                      {"world": 1, "stage": 2, "life": 2}))
        self.assertEqual((run.world, run.stage, run.stages_cleared, run.stage_frame), (1, 2, 1, 0))
        self.assertEqual(run.memory_generation, 1)
        self.assertTrue(run.snapshot(2)["stage_transition"])

    def test_losing_last_extra_life_is_not_game_over(self):
        run = GameSession()
        run.start({"world": 2, "stage": 3, "life": 1})
        self.assertIsNone(run.advance({"world": 2, "stage": 3, "life": 1},
                                      {"world": 2, "stage": 3, "life": 0}))
        self.assertTrue(run.life_lost)
        self.assertFalse(run.game_over)
        self.assertEqual(run.snapshot(3)["lives_remaining"], 1)

    def test_game_over_requires_255_then_new_run_starts_empty(self):
        run = GameSession()
        run.start({"world": 1, "stage": 2, "life": 0})
        self.assertEqual(run.advance({"world": 1, "stage": 2, "life": 255},
                                     {"world": 1, "stage": 2, "life": 255}, terminated=True), "game_over")
        self.assertEqual(run.snapshot(4)["lives_remaining"], 0)
        run.mark_memory_reset()
        run.start({"world": 1, "stage": 1, "life": 2}, new_run=True)
        self.assertEqual((run.run_id, run.memory_generation, run.session_frame, run.stages_cleared), (2, 2, 0, 0))

    def test_final_8_4_clear_is_completion_even_without_env_done(self):
        run = GameSession()
        run.start({"world": 8, "stage": 4, "life": 1})
        self.assertEqual(run.advance({"world": 8, "stage": 4, "life": 1, "flag_get": True},
                                     {"world": 8, "stage": 4, "life": 1}), "game_complete")
        self.assertEqual(run.stages_cleared, 1)
        self.assertFalse(run.game_over)

    def test_warp_does_not_invent_clears_of_skipped_stages(self):
        run = GameSession()
        run.start({"world": 1, "stage": 2, "life": 2})
        self.assertIsNone(run.advance({"world": 1, "stage": 2, "life": 2, "flag_get": True},
                                      {"world": 4, "stage": 1, "life": 2}))
        self.assertEqual((run.world, run.stage, run.stages_cleared), (4, 1, 1))


if __name__ == "__main__":
    unittest.main()
