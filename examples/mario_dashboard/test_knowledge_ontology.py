import unittest
from knowledge_ontology import KnowledgeOntology


def state(*,size="small",collected=0,lives=3,game_over=False):
    return {"session":{"lives_remaining":lives,"game_over":game_over},
            "level":{"world":1,"stage":1},
            "player":{"x":128,"powerup_status":size},
            "detectors":{"powerups_collected":collected}}


def control(flight=None):
    return {"plan_intent":{"accepted":True},
            "graph":{"nodes":[{"id":"action_dynamics","output":{"last_flight":flight or {}}},
                              {"id":"world_belief","output":{"entities":[
                                  {"id":9,"kind_id":6,"status":"observed","x":220,"vx":-1,"visible":True}]}}]}}


class OntologyTest(unittest.TestCase):
    def test_post_step_life_loss_becomes_cross_run_actionable_evidence(self):
        ontology=KnowledgeOntology(retain_across_runs=True)
        before=state()
        before["player"]["x"]=469
        ontology.observe(before,control(),{}, {},frame=20,run_id=1,memory_generation=1)
        outcome=ontology.observe_outcome(state=before,control={
            "action":"noop","source":"jevtpp_target_skill","reason":"hunter_track_intercept",
            "persistent_tasks":{"target":{"visible":True,
                "x":479,"kind_id":6}}},session={"life_lost":True},
            frame=20,run_id=1,recent_trace=[{"reason":"hunter_track_intercept",
                "action":"noop","task":{"target":{"kind_id":6}}} for _ in range(12)])
        self.assertIn("observed_life_loss_near",{e["relation"] for e in outcome["edges"]})
        self.assertEqual(ontology.affordances()["death_zones"][0]["events"],1)
        self.assertEqual(ontology.affordances()["death_zones"][0]["unsafe_intercepts"],1)
        self.assertIn("observed_failure_context",{e["relation"] for e in outcome["edges"]})
        ontology.observe(state(),control(),{}, {},frame=0,run_id=2,memory_generation=1)
        language=ontology.model_context(run_id=2,stage="1-1",mode="hunter")
        self.assertIn("observed_life_loss_near",{s["relation"] for s in language["statements"]})
        self.assertEqual(ontology.affordances()["death_zones"][0]["runs"],1)
        ontology.observe_outcome(state=before,control={"action":"right_run",
            "persistent_tasks":{}},session={"life_lost":True},frame=30,run_id=2)
        self.assertEqual(ontology.affordances()["death_zones"][0]["events"],2)
        self.assertEqual(ontology.affordances()["death_zones"][0]["runs"],2)

    def test_outcome_requires_confirmed_life_loss(self):
        ontology=KnowledgeOntology(retain_across_runs=True)
        before=state()
        ontology.observe(before,control(),{}, {},frame=1,run_id=1,memory_generation=1)
        ontology.observe_outcome(state=before,control={"action":"noop"},
                                 session={"life_lost":False},frame=1,run_id=1)
        self.assertEqual(ontology.affordances()["death_zones"],[])

    def test_collision_frame_can_hide_target_without_erasing_precursor(self):
        ontology=KnowledgeOntology(retain_across_runs=True)
        before=state();before["player"]["x"]=469
        ontology.observe(before,control(),{}, {},frame=10,run_id=1,memory_generation=1)
        trace=[{"reason":"hunter_track_intercept","action":"noop",
                "task":{"target":{"kind_id":6,"x":479,"visible":True}}}
               for _ in range(24)]
        trace.append({"reason":"obstacle","action":"right_run",
                      "task":{"target":{"kind_id":6,"x":479,"visible":False}}})
        ontology.observe_outcome(state=before,control={"action":"right_run",
            "source":"jevtpp_reactive_controller","reason":"obstacle",
            "persistent_tasks":{"target":{"kind_id":6,"x":479,"visible":False}}},
            session={"life_lost":True},frame=35,run_id=1,recent_trace=trace)
        zone=ontology.affordances()["death_zones"][0]
        self.assertEqual(zone["unsafe_intercepts"],1)
        self.assertEqual(zone["target_kinds"],[6])

    def test_repeated_wait_below_target_is_distinct_failure_context(self):
        ontology=KnowledgeOntology(retain_across_runs=True)
        before=state();before["player"]["x"]=1117
        ontology.observe(before,control(),{}, {},frame=10,run_id=1,memory_generation=1)
        trace=[{"reason":"hunter_wait_for_reachable_target","action":"noop",
                "task":{"target":{"kind_id":6,"x":1120,"visible":True}}}
               for _ in range(18)]
        view=ontology.observe_outcome(state=before,control={"action":"right_jump",
            "persistent_tasks":{"target":{"kind_id":6,"x":1120,"visible":False}}},
            session={"life_lost":True},frame=29,run_id=1,recent_trace=trace)
        zone=ontology.affordances()["death_zones"][0]
        self.assertEqual(zone["unsafe_waits"],1)
        self.assertEqual(zone["unsafe_intercepts"],0)
        self.assertTrue(any(e["from"]=="skill:wait_for_reachable_target" and
                            e["relation"]=="observed_failure_context" for e in view["edges"]))

    def test_evidence_links_and_model_attention_are_distinct(self):
        ontology=KnowledgeOntology()
        flight={"start_frame":10,"end_frame":46,"start_x":100,"landed_x":180,
                "held_frames":16,"result":"landing_confirmed","signature":"right_jump",
                "takeoff_confirmed":True}
        knowledge={"context":{"jump_samples":3,"jump_duration_mean_frames":36,
                              "jump_range_mean_pixels":80,
                              "enemy_speeds":{"6":{"mean":-1.0,"samples":9}}}}
        model={"judgment_kind":"plan","selected_slot":0,"model_plan":"stomp:right_jump"}
        first=ontology.observe(state(),control(flight),knowledge,model,
                               frame=47,run_id=1,memory_generation=1)
        relations={e["relation"]:e for e in first["edges"]}
        self.assertEqual(relations["observed_flight"]["events"],1)
        self.assertEqual(relations["observed_horizontal_motion"]["attributes"]["samples"],9)
        self.assertTrue(any(n["model_focus"] for n in first["nodes"]))
        self.assertTrue(all(e["source"]!="model" for e in first["edges"]))
        second=ontology.observe(state(),control(flight),knowledge,model,
                                frame=48,run_id=1,memory_generation=1)
        self.assertEqual(next(e for e in second["edges"] if e["relation"]=="observed_flight")["events"],1)

    def test_growth_and_size_buffer_require_observed_transitions(self):
        ontology=KnowledgeOntology()
        empty={"context":{}}
        ontology.observe(state(),control(),empty,{},frame=1,run_id=1,memory_generation=1)
        bigger=ontology.observe(state(size="big",collected=1),control(),empty,{},
                                frame=2,run_id=1,memory_generation=1)
        self.assertIn("observed_growth",{e["relation"] for e in bigger["edges"]})
        smaller=ontology.observe(state(size="small",collected=1),control(),empty,{},
                                 frame=3,run_id=1,memory_generation=1)
        self.assertIn("observed_size_loss_without_life_loss",
                      {e["relation"] for e in smaller["edges"]})
        ontology.observe(state(size="big",collected=1),control(),empty,{},
                         frame=4,run_id=1,memory_generation=1)
        dead=ontology.observe(state(size="small",collected=1,lives=2),control(),empty,{},
                              frame=5,run_id=1,memory_generation=1)
        edge=next(e for e in dead["edges"] if e["relation"]=="observed_size_loss_without_life_loss")
        self.assertEqual(edge["events"],1)

    def test_reset_at_game_over_and_new_run(self):
        ontology=KnowledgeOntology()
        ontology.observe(state(),control(),{"context":{"jump_samples":2}}, {},
                         frame=1,run_id=1,memory_generation=1)
        cleared=ontology.observe(state(game_over=True),control(),{}, {},
                                 frame=2,run_id=1,memory_generation=1)
        self.assertEqual(cleared["edges"],[])
        new=ontology.observe(state(),control(),{}, {},frame=0,run_id=2,memory_generation=2)
        self.assertFalse(any(n["id"]=="jump:range" for n in new["nodes"]))

    def test_retained_scope_names_enemies_and_injects_relations_across_runs(self):
        ontology=KnowledgeOntology(retain_across_runs=True)
        first=ontology.observe(state(size="tall",collected=1),control(),{"context":{}},{},
                               frame=3,run_id=1,memory_generation=1)
        self.assertTrue(any(n["label"].startswith("GOOMBA #9") for n in first["nodes"]))
        terminal=ontology.observe(state(game_over=True),control(),{}, {},
                                  frame=4,run_id=1,memory_generation=1)
        self.assertGreater(len(terminal["edges"]),0)
        second=ontology.observe(state(),control(),{}, {},frame=0,run_id=2,memory_generation=1)
        self.assertEqual(second["scope"],"cross_run_session")
        self.assertEqual(second["runs_observed"],2)
        self.assertTrue(any(n["id"]=="enemy:1:9" for n in second["nodes"]))
        self.assertTrue(any(n["id"]=="enemy:2:9" for n in second["nodes"]))
        language=ontology.model_context(run_id=2,stage="1-1",mode="hunter")
        self.assertTrue(any(s["subject"].startswith("GOOMBA #9 · R2") for s in language["statements"]))
        self.assertFalse(any(s["subject"].endswith("R1") for s in language["statements"]))
        self.assertLessEqual(len(language["statements"]),12)
        ontology.set_retention(True)
        self.assertEqual(ontology.observe(state(),control(),{}, {},frame=1,run_id=2,
                         memory_generation=1)["runs_observed"],2)

    def test_tall_transition_star_timer_and_verified_apex(self):
        ontology=KnowledgeOntology()
        initial=state();initial["player"]["star_timer_ram_0x079f"]=0
        ontology.observe(initial,control(),{}, {},frame=1,run_id=1,memory_generation=1)
        growth=state(size="tall",collected=1);growth["player"]["star_timer_ram_0x079f"]=0
        grown=ontology.observe(growth,control(),{}, {},
                               frame=2,run_id=1,memory_generation=1)
        self.assertIn("observed_growth",{e["relation"] for e in grown["edges"]})
        flight={"start_frame":2,"end_frame":24,"start_x":100,"landed_x":150,
                "start_feet_y":200,"apex_feet_y":130,"held_frames":16,
                "result":"landing_confirmed","signature":"right_jump",
                "takeoff_confirmed":True,"contact_suspected":False,
                "target":{"y":170}}
        star=state(size="tall",collected=1)
        star["player"]["star_timer_ram_0x079f"]=200
        view=ontology.observe(star,control(flight),{}, {},frame=24,run_id=1,memory_generation=1)
        relations={e["relation"] for e in view["edges"]}
        self.assertIn("observed_apex_height",relations)
        self.assertIn("reached_higher_support",relations)
        self.assertIn("activates_invincibility_timer",relations)
        self.assertEqual(next(e for e in view["edges"] if e["relation"]=="observed_apex_height")
                         ["attributes"]["last_height_px"],70)
        language=ontology.model_context(run_id=1,stage="1-1",mode="collector")
        self.assertIn("activates_invincibility_timer",{s["relation"] for s in language["statements"]})
        self.assertNotIn("active_star_timer",{s["relation"] for s in language["statements"]})
        active=ontology.model_context(run_id=1,stage="1-1",mode="collector",star_active=True)
        self.assertIn("active_star_timer",{s["relation"] for s in active["statements"]})

    def test_star_contact_elimination_needs_spatial_defeat_and_no_stomp(self):
        ontology=KnowledgeOntology(retain_across_runs=True)
        previous=state();previous["player"].update(screen_y=176,star_timer_ram_0x079f=120)
        previous["detectors"]["stomps_confirmed"]=0
        ontology.observe(previous,control(),{}, {},frame=10,run_id=1,memory_generation=1)
        observed=state();observed["player"].update(x=128,screen_y=176,star_timer_ram_0x079f=119)
        observed["detectors"]["stomps_confirmed"]=0
        observed["hazard"]={"elimination_events":[{"source":"RAM_alive_to_defeated",
            "kind_id":6,"x":134,"screen_y":190}]}
        result=ontology.observe(observed,control(),{}, {},frame=11,run_id=1,memory_generation=1)
        relation="contact_elimination_without_life_loss_while_star_active"
        self.assertIn(relation,{e["relation"] for e in result["edges"]})
        self.assertEqual(ontology.affordances()["star_contact_kinds"],[])
        observed["hazard"]["elimination_events"][0]["x"]=130
        ontology.observe(observed,control(),{}, {},frame=12,run_id=1,memory_generation=1)
        self.assertEqual(ontology.affordances()["star_contact_kinds"],[6])
        self.assertIn(relation,{e["relation"] for e in ontology.model_context(
            run_id=1,stage="1-1",mode="hunter")["statements"]})
        stomp=state();stomp["player"].update(x=128,screen_y=176,star_timer_ram_0x079f=118)
        stomp["detectors"]["stomps_confirmed"]=1
        stomp["hazard"]=observed["hazard"]
        ontology.observe(stomp,control(),{}, {},frame=13,run_id=1,memory_generation=1)
        edge=next(e for e in ontology.edges.values() if e["relation"]==relation)
        self.assertEqual(edge["events"],2)
        far=state();far["player"].update(x=128,screen_y=176,star_timer_ram_0x079f=117)
        far["detectors"]["stomps_confirmed"]=1
        far["hazard"]={"elimination_events":[{"source":"RAM_alive_to_defeated",
            "kind_id":6,"x":260,"screen_y":190}]}
        ontology.observe(far,control(),{}, {},frame=14,run_id=1,memory_generation=1)
        self.assertEqual(edge["events"],2)


if __name__=="__main__":
    unittest.main()
