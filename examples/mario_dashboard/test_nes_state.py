"""Regression tests for observed lethal RAM-decoding failures."""
import unittest
from nes_state import StateTracker, collision_preview, episode_lifecycle


def ram_fixture():
    ram = [0] * 0x800
    ram[0x0e], ram[0xb5], ram[0xce], ram[0x86] = 8, 1, 176, 160
    for page in range(2):
        for column in range(16):
            for row in (11, 12):
                ram[0x500 + page * 208 + row * 16 + column] = 0x54
    return ram


def parse(tracker, ram, x=160):
    return tracker.parse({"x_pos":x,"y_pos":79}, ram, previous_action="right_run",
                         previous_reward=0, response_delay_frames=0)[0]


class RamRegressionTest(unittest.TestCase):
    def test_fourth_hostile_remains_visible_to_controller(self):
        ram=ram_fixture()
        for slot,(kind,dx,y) in enumerate(((6,99,184),(6,123,184),
                                            (6,147,184),(13,177,149))):
            x=160+dx
            ram[0x0f+slot]=1
            ram[0x16+slot]=kind
            ram[0x6e+slot]=x//256
            ram[0x87+slot]=x%256
            ram[0xcf+slot]=y
        upcoming=parse(StateTracker(1),ram)["hazard"]["upcoming_enemies"]
        self.assertEqual(len(upcoming),4)
        self.assertEqual(upcoming[3]["kind"],"piranha_plant")
        self.assertFalse(upcoming[3]["stompable"])

    def test_large_platform_ram_slot_is_moving_support_not_enemy(self):
        ram=ram_fixture()
        ram[0x0f],ram[0x16],ram[0x87],ram[0xcf]=1,0x28,200,128
        tracker=StateTracker(1)
        first=parse(tracker,ram)
        self.assertFalse(first["hazard"]["enemy_ahead"])
        support=first["collision"]["moving_surfaces"][0]
        self.assertEqual((support["left_x"],support["right_x"],support["y"]),
                         (200,232,128))
        self.assertIsNone(support["vx"])
        ram[0x87],ram[0xcf]=199,129
        second=parse(tracker,ram)
        self.assertEqual(second["collision"]["moving_surfaces"][0]["vx"],-1)
        self.assertEqual(second["collision"]["moving_surfaces"][0]["vy"],1)
        self.assertEqual(second["detectors"]["hostiles_visible"],0)

    def test_star_timer_is_observed_separately_from_size(self):
        ram=ram_fixture();ram[0x079f]=120
        state=parse(StateTracker(1),ram)
        self.assertTrue(state["player"]["star_active"])
        self.assertEqual(state["player"]["star_timer_ram_0x079f"],120)
        ram[0x079f]=0
        state=parse(StateTracker(1),ram)
        self.assertFalse(state["player"]["star_active"])

    def test_platform_visibility_does_not_depend_on_player_height(self):
        ram=ram_fixture()
        for row in range(13): ram[0x500+row*16+12]=0
        ram[0x500+2*16+12]=0x17
        for player_y in (16,176,250):
            surfaces=collision_preview(ram,160,player_y)["landing_surfaces"]
            self.assertTrue(any(s["left_x"]==192 and s["y"]==64 for s in surfaces))

    def test_offscreen_nametable_is_not_an_observed_landing(self):
        ram=ram_fixture()
        for row in range(13): ram[0x500+208+row*16+3]=0
        ram[0x500+208+2*16+3]=0x17
        self.assertFalse(collision_preview(ram,160,176)["landing_surfaces"])

    def test_respawn_y_wrap_is_not_a_251_pixel_jump(self):
        tracker=StateTracker(1);ram=ram_fixture()
        tracker.parse({"x_pos":160,"y_pos":2},ram,previous_action="right",
                      previous_reward=0,response_delay_frames=0)
        s,_=tracker.parse({"x_pos":160,"y_pos":253},ram,previous_action="right",
                         previous_reward=0,response_delay_frames=0)
        self.assertFalse(s["player"]["velocity_valid"])
        self.assertFalse(s["player"]["grounded"])
        self.assertEqual(s["player"]["vy"],0)

    def test_isolated_platform_is_landing_target_even_above_current_feet(self):
        ram=ram_fixture()
        for column in range(12,16):
            ram[0x500+10*16+column]=0x17
            ram[0x500+11*16+column]=ram[0x500+12*16+column]=0
        preview=collision_preview(ram,160,176)
        target=next(item for item in preview["landing_surfaces"] if item["y"]==192)
        self.assertEqual((target["left_x"],target["right_x"]),(192,256))
        self.assertIsNone(preview["columns"][2]["floor_y"])
        self.assertFalse(collision_preview(ram_fixture(),160,176)["landing_surfaces"])

    def test_side_pipe_requires_both_lip_tiles_and_observed_approach_floor(self):
        ram=ram_fixture()
        ram[0x500+6*16+11]=0x1c
        ram[0x500+7*16+11]=0x1f
        self.assertIsNone(collision_preview(ram,160,96)["side_pipe"])
        ram[0x500+8*16+10]=0x52
        route=collision_preview(ram,160,96)["side_pipe"]
        self.assertEqual(route["mouth_x"],176)
        self.assertEqual(route["floor_y"],160)
        ram[0x500+7*16+11]=0x14
        self.assertIsNone(collision_preview(ram,160,96)["side_pipe"])

    def test_persistent_falling_state_can_confirm_exact_floor_contact(self):
        ram=ram_fixture(); ram[0x1d]=2
        tracker=StateTracker(1)
        for _ in range(4): state=parse(tracker,ram)
        self.assertTrue(state["player"]["grounded"])
        self.assertEqual(state["player"]["ground_source"],"ram_support_contact")
        ram[0x500+11*16+10]=ram[0x500+12*16+10]=0
        self.assertFalse(parse(tracker,ram)["player"]["grounded"])

    def test_elevated_wall_exposes_verified_lower_corridor(self):
        ram=ram_fixture()
        ram[0x500+7*16+10]=0x51
        for row in range(2,8): ram[0x500+row*16+11]=0x51
        preview=collision_preview(ram,160,112)
        self.assertEqual(preview["lower_passage"]["floor_y"],208)
        self.assertEqual(preview["lower_passage"]["entrance_x"],144)
        self.assertEqual(preview["lower_passage"]["clearance_pixels"],48)

    def test_reserved_slot_five_powerup_is_visible(self):
        ram=ram_fixture()
        ram[0x14],ram[0x1b],ram[0x8c],ram[0xd4]=1,0x2e,180,128
        state=parse(StateTracker(1),ram)
        self.assertEqual(state["detectors"]["powerups_visible"],1)
        self.assertEqual(state["detectors"]["powerup_distance"],20)
        self.assertFalse(state["hazard"]["enemy_ahead"])

    def test_enemy_zero_is_live_green_koopa(self):
        ram = ram_fixture()
        ram[0x0f], ram[0x16], ram[0x87], ram[0xcf] = 1, 0, 200, 184
        state = parse(StateTracker(1),ram)
        self.assertTrue(state["hazard"]["enemy_ahead"])
        self.assertEqual(state["hazard"]["nearest_enemy_kind"], "green_koopa")
        self.assertEqual(state["hazard"]["enemy_distance"],40)

    def test_piranha_and_spiny_have_distinct_verified_ram_ids(self):
        for kind,name in ((0x0d,"piranha_plant"),(0x12,"spiny")):
            ram=ram_fixture()
            ram[0x0f],ram[0x16],ram[0x87],ram[0xcf]=1,kind,200,184
            state=parse(StateTracker(1),ram)
            self.assertEqual(state["hazard"]["nearest_enemy_kind"],name)
            self.assertFalse(state["hazard"]["nearest_enemy_stompable"])

    def test_stable_jump_apex_is_not_ground(self):
        ram = ram_fixture()
        ram[0x1d] = 1
        tracker = StateTracker(1)
        for _ in range(8):
            state = parse(tracker,ram)
            self.assertFalse(state["player"]["grounded"])
        self.assertEqual(state["player"]["ground_source"],"ram_0x1d")

    def test_pipe_edge_with_lower_floor_is_not_bottomless(self):
        preview = collision_preview(ram_fixture(),160,112)
        self.assertIsNone(preview["gap_distance_pixels"])
        self.assertEqual(preview["columns"][0]["floor_y"],208)

    def test_coins_do_not_provide_floor_support(self):
        ram = ram_fixture()
        # Remove floor from column directly ahead; a coin above it is not solid.
        for row in (11,12):
            ram[0x500 + row*16 + 11] = 0
        ram[0x500 + 7*16 + 11] = 0xc2
        preview = collision_preview(ram,160,176)
        self.assertEqual(preview["gap_distance_pixels"],4)

    def test_recovery_requires_support_behind_player(self):
        ram = ram_fixture()
        self.assertTrue(collision_preview(ram,160,176)["behind_floor_safe"])
        for row in (11,12):
            ram[0x500+row*16+8] = 0
        self.assertFalse(collision_preview(ram,160,176)["behind_floor_safe"])

    def test_overhead_brick_does_not_hide_pit_beneath_player(self):
        ram = ram_fixture()
        for row in (11,12):
            ram[0x500+row*16+11] = 0
        ram[0x500+7*16+11] = 0x51
        preview = collision_preview(ram,160,176)
        self.assertEqual(preview["gap_distance_pixels"],4)
        self.assertIsNone(preview["columns"][1]["floor_y"])

    def test_dead_enemy_is_filtered_but_shell_stays_hostile(self):
        ram = ram_fixture()
        ram[0x0f], ram[0x16], ram[0x87], ram[0xcf] = 1, 6, 200, 184
        ram[0x1e] = 4
        self.assertFalse(parse(StateTracker(1),ram)["hazard"]["enemy_ahead"])
        ram[0x16] = 0
        self.assertTrue(parse(StateTracker(1),ram)["hazard"]["enemy_ahead"])

    def test_dead_ram_is_reported_without_info_death_field(self):
        ram = ram_fixture()
        ram[0x0e] = 11
        self.assertTrue(parse(StateTracker(1),ram)["episode"]["dead"])


class CombatTelemetryTest(unittest.TestCase):
    def enemy_fixture(self, kind=6, x=180, state=0):
        ram = ram_fixture()
        ram[0x0f], ram[0x16], ram[0x1e], ram[0xcf] = 1, kind, state, 184
        ram[0x6e], ram[0x87] = divmod(x, 256)
        return ram

    def test_goomba_stomp_confirms_one_kill_and_one_stomp(self):
        tracker, ram = StateTracker(1), self.enemy_fixture()
        self.assertEqual(parse(tracker, ram)["detectors"]["hostiles_seen"], 1)
        ram[0x1e] = 4
        for _ in range(3):
            state = parse(tracker, ram)
            self.assertFalse(state["hazard"]["enemy_ahead"])
            self.assertEqual(state["detectors"]["eliminations_confirmed"], 1)
            self.assertEqual(state["detectors"]["eliminations_detected"], 1)
            self.assertEqual(state["detectors"]["stomps_confirmed"], 1)
        ram[0x1e] = 0x20
        self.assertEqual(parse(tracker, ram)["detectors"]["eliminations_confirmed"], 1)

    def test_defeated_bit_confirms_kill_without_inventing_a_stomp(self):
        for kind in (0, 1, 2, 5, 6):
            with self.subTest(kind=kind):
                tracker, ram = StateTracker(1), self.enemy_fixture(kind=kind)
                parse(tracker, ram)
                ram[0x1e] = 0x21
                result = parse(tracker, ram)["detectors"]
                self.assertEqual(result["eliminations_confirmed"], 1)
                self.assertEqual(result["stomps_confirmed"], 0)

    def test_shell_entry_is_a_stomp_but_stays_alive(self):
        for kind in (0, 1, 2):
            with self.subTest(kind=kind):
                tracker, ram = StateTracker(1), self.enemy_fixture(kind=kind)
                parse(tracker, ram)
                ram[0x1e] = 4
                for _ in range(3):
                    state = parse(tracker, ram)
                    self.assertTrue(state["hazard"]["enemy_ahead"])
                    self.assertEqual(state["detectors"]["stomps_confirmed"], 1)
                    self.assertEqual(state["detectors"]["eliminations_confirmed"], 0)

    def test_vanished_enemy_is_never_a_confirmed_kill(self):
        tracker, ram = StateTracker(1), self.enemy_fixture()
        parse(tracker, ram)
        ram[0x0f] = 0
        self.assertEqual(parse(tracker, ram)["detectors"]["eliminations_confirmed"], 0)
        ram[0x0f] = 1
        self.assertEqual(parse(tracker, ram)["detectors"]["hostiles_seen"], 2)

    def test_initial_corpse_is_not_an_observed_elimination(self):
        for kind, state in ((6, 4), (0, 0x20)):
            with self.subTest(kind=kind):
                tracker, ram = StateTracker(1), self.enemy_fixture(kind=kind, state=state)
                for _ in range(2):
                    counters = parse(tracker, ram)["detectors"]
                    self.assertEqual(counters["eliminations_confirmed"], 0)
                    self.assertEqual(counters["stomps_confirmed"], 0)

    def test_live_pass_is_deduplicated_and_not_counted_as_kill(self):
        tracker, ram = StateTracker(1), self.enemy_fixture()
        parse(tracker, ram, x=160)
        for mario_x in (181, 160, 182, 200):
            counters = parse(tracker, ram, x=mario_x)["detectors"]
            self.assertEqual(counters["hostiles_seen"], 1)
            self.assertEqual(counters["hostiles_passed_alive"], 1)
            self.assertEqual(counters["eliminations_confirmed"], 0)

    def test_nearby_enemies_preserves_live_hostile_behind_player(self):
        state = parse(StateTracker(1), self.enemy_fixture(x=150))
        self.assertEqual(state["hazard"]["upcoming_enemies"], [])
        nearby = state["hazard"]["nearby_enemies"]
        self.assertEqual(len(nearby), 1)
        self.assertEqual(nearby[0]["relative_x_pixels"], -10)
        self.assertEqual(nearby[0]["kind_id"], 6)

    def test_defeated_or_dying_crossing_is_not_a_live_pass(self):
        for death in (False, True):
            with self.subTest(player_dead=death):
                tracker, ram = StateTracker(1), self.enemy_fixture()
                parse(tracker, ram)
                if death:
                    ram[0x0e] = 11
                else:
                    ram[0x1e] = 4
                self.assertEqual(parse(tracker, ram, x=181)["detectors"]["hostiles_passed_alive"], 0)

    def test_slot_kind_or_large_position_change_starts_new_life(self):
        for replacement in ("kind", "position", "inactive"):
            with self.subTest(replacement=replacement):
                tracker, ram = StateTracker(1), self.enemy_fixture()
                parse(tracker, ram)
                ram[0x1e] = 4
                parse(tracker, ram)
                ram[0x1e] = 0
                if replacement == "kind":
                    ram[0x16] = 0
                elif replacement == "position":
                    ram[0x6e], ram[0x87] = divmod(300, 256)
                else:
                    ram[0x0f] = 0
                    parse(tracker, ram)
                    ram[0x0f] = 1
                self.assertEqual(parse(tracker, ram)["detectors"]["hostiles_seen"], 2)
                ram[0x1e] = 0x20
                self.assertEqual(parse(tracker, ram)["detectors"]["eliminations_confirmed"], 2)

    def test_collectibles_and_level_objects_never_enter_combat_counts(self):
        for kind in (0x2e, 0x2f, 0x30, 0x31):
            with self.subTest(kind=kind):
                tracker, ram = StateTracker(1), self.enemy_fixture(kind=kind)
                parse(tracker, ram)
                ram[0x1e] = 0x20
                counters = parse(tracker, ram, x=181)["detectors"]
                self.assertEqual(counters["hostiles_seen"], 0)
                self.assertEqual(counters["eliminations_confirmed"], 0)
                self.assertEqual(counters["hostiles_passed_alive"], 0)

    def test_reset_clears_combat_counts_and_slot_lives(self):
        tracker, ram = StateTracker(1), self.enemy_fixture()
        parse(tracker, ram)
        ram[0x1e] = 4
        parse(tracker, ram)
        tracker.reset()
        ram[0x1e] = 0
        counters = parse(tracker, ram)["detectors"]
        self.assertEqual(counters["hostiles_seen"], 1)
        self.assertEqual(counters["eliminations_confirmed"], 0)
        self.assertEqual(counters["stomps_confirmed"], 0)

    def test_velocity_needs_consecutive_same_kind_observations(self):
        tracker, ram = StateTracker(1), self.enemy_fixture(x=200)
        enemy = parse(tracker, ram)["hazard"]["upcoming_enemies"][0]
        self.assertFalse(enemy["velocity_observed"])
        self.assertEqual(enemy["relative_velocity_x"], 0)
        ram[0x87] = 199
        enemy = parse(tracker, ram)["hazard"]["upcoming_enemies"][0]
        self.assertTrue(enemy["velocity_observed"])
        self.assertEqual(enemy["relative_velocity_x"], -1)
        ram[0x16], ram[0x87] = 0, 198
        enemy = parse(tracker, ram)["hazard"]["upcoming_enemies"][0]
        self.assertFalse(enemy["velocity_observed"])
        self.assertEqual(enemy["relative_velocity_x"], 0)


class EpisodeLifecycleTest(unittest.TestCase):
    def test_each_ordinary_death_preserves_playable_lives(self):
        for raw_lives in (2, 1, 0):
            with self.subTest(raw_lives=raw_lives):
                ram = ram_fixture()
                ram[0x075a], ram[0x000e] = raw_lives, 11
                episode = episode_lifecycle({}, ram)
                self.assertTrue(episode["dead"])
                self.assertFalse(episode["game_over"])
                self.assertEqual(episode["lives"], raw_lives)
                self.assertEqual(episode["lives_remaining"], raw_lives + 1)
                self.assertEqual(episode["lifecycle_status"], "death")

    def test_life_underflow_is_explicit_game_over(self):
        ram = ram_fixture()
        ram[0x075a] = 255
        episode = episode_lifecycle({}, ram)
        self.assertTrue(episode["game_over"])
        self.assertEqual(episode["lives_remaining"], 0)
        self.assertEqual(episode["lifecycle_status"], "game_over")
        self.assertTrue(episode["lifecycle_evidence"]["game_over"]["ram_0x075a_equals_255"])

    def test_environment_life_counter_uses_same_sentinel(self):
        self.assertFalse(episode_lifecycle({"life": 0, "death": True}, None)["game_over"])
        self.assertTrue(episode_lifecycle({"life": 255}, None)["game_over"])
        self.assertTrue(episode_lifecycle({"game_over": True}, None)["game_over"])

    def test_generic_terminal_and_timeout_do_not_prove_game_over(self):
        episode = episode_lifecycle({"terminated": True, "truncated": True, "done": True,
                                     "time": 0, "death": True}, None)
        self.assertTrue(episode["dead"])
        self.assertIsNone(episode["game_over"])
        self.assertIsNone(episode["full_game_completed"])

    def test_ram_world_stage_and_decimal_time_are_reported(self):
        ram = ram_fixture()
        ram[0x075f], ram[0x075c] = 5, 2
        ram[0x07f8:0x07fb] = [3, 0, 7]
        episode = episode_lifecycle({}, ram)
        self.assertEqual((episode["world"], episode["stage"], episode["time_left"]), (6, 3, 307))
        self.assertEqual(episode["lifecycle_evidence"]["world"], "ram_0x075f")
        self.assertEqual(parse(StateTracker(1), ram)["level"], {"world": 6, "stage": 3})

    def test_invalid_clock_digits_remain_unknown(self):
        ram = ram_fixture()
        ram[0x07f8] = 255
        self.assertIsNone(episode_lifecycle({}, ram)["time_left"])

    def test_full_game_completed_requires_final_world_and_stage_clear(self):
        for world, stage, clear, completed in ((1, 1, True, False), (8, 3, True, False),
                                                (7, 4, True, False), (8, 4, False, False),
                                                (8, 4, True, True)):
            with self.subTest(world=world, stage=stage, clear=clear):
                episode = episode_lifecycle({"world": world, "stage": stage, "flag_get": clear}, None)
                self.assertEqual(episode["full_game_completed"], completed)
                self.assertEqual(episode["stage_clear"], clear)

    def test_ram_final_castle_clear_proves_full_game_completion(self):
        ram = ram_fixture()
        ram[0x075f], ram[0x075c], ram[0x0770] = 7, 3, 2
        episode = parse(StateTracker(1), ram)["episode"]
        self.assertTrue(episode["stage_clear"])
        self.assertTrue(episode["full_game_completed"])
        self.assertFalse(episode["game_over"])
        self.assertEqual(episode["lifecycle_status"], "full_game_completed")

    def test_flagpole_state_clears_stage_but_vine_does_not(self):
        ram = ram_fixture()
        ram[0x001d], ram[0x0016] = 3, 0x2f
        self.assertFalse(episode_lifecycle({}, ram)["stage_clear"])
        ram[0x0016] = 0x31
        episode = episode_lifecycle({}, ram)
        self.assertTrue(episode["stage_clear"])
        self.assertFalse(episode["full_game_completed"])

    def test_missing_evidence_is_unknown_instead_of_defaulted(self):
        for ram in (None, []):
            episode = episode_lifecycle({}, ram)
            for field in ("world", "stage", "lives", "lives_remaining", "time_left", "dead",
                          "stage_clear", "game_over", "full_game_completed"):
                self.assertIsNone(episode[field], field)
            self.assertEqual(episode["lifecycle_status"], "unknown")

    def test_ambiguous_lives_alias_cannot_trigger_game_over(self):
        episode = episode_lifecycle({"lives": 0}, None)
        self.assertEqual(episode["lives"], 0)
        self.assertIsNone(episode["lives_remaining"])
        self.assertIsNone(episode["game_over"])


if __name__ == "__main__":
    unittest.main()
