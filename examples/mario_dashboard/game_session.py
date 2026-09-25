"""Full-game lifecycle derived from post-step NES state and transition events.

The gym SuperMarioBros-v0 environment ends on game over, not on a flag. Its
returned info can precede the internal transition skip, so the caller supplies
both the event info and a fresh post-step RAM/info snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _number(info: dict[str, Any], key: str, default: int) -> int:
    value = info.get(key, default)
    return int(value) if isinstance(value, (int, float)) else default


def playable_lives(raw_life: int) -> int:
    # This ROM stores extra lives: 2, 1, 0 are three playable lives. 0xff is
    # the true game-over marker.
    return 0 if raw_life == 0xff else max(0, raw_life + 1)


@dataclass
class GameSession:
    run_id: int = 1
    memory_generation: int = 1
    session_frame: int = 0
    stage_frame: int = 0
    stages_cleared: int = 0
    deaths: int = 0
    world: int = 1
    stage: int = 1
    raw_life: int = 2
    stage_transition: bool = False
    stage_clear: bool = False
    life_lost: bool = False
    game_over: bool = False
    game_complete: bool = False

    def start(self, info: dict[str, Any], *, new_run: bool = False) -> None:
        if new_run:
            self.run_id += 1
        self.session_frame = self.stage_frame = self.stages_cleared = self.deaths = 0
        self.world = _number(info, "world", 1)
        self.stage = _number(info, "stage", 1)
        self.raw_life = _number(info, "life", 2)
        self.stage_transition = self.stage_clear = self.life_lost = False
        self.game_over = self.game_complete = False

    def advance(self, event_info: dict[str, Any], current_info: dict[str, Any],
                *, terminated: bool = False, truncated: bool = False) -> str | None:
        previous_world, previous_stage, previous_life = self.world, self.stage, self.raw_life
        self.world = _number(current_info, "world", _number(event_info, "world", self.world))
        self.stage = _number(current_info, "stage", _number(event_info, "stage", self.stage))
        self.raw_life = _number(current_info, "life", _number(event_info, "life", self.raw_life))
        event_life = _number(event_info, "life", self.raw_life)
        self.stage_transition = (self.world, self.stage) != (previous_world, previous_stage)
        self.stage_clear = bool(event_info.get("flag_get"))
        self.game_complete = self.stage_clear and (previous_world, previous_stage) == (8, 4)
        self.game_over = self.raw_life == 0xff or event_life == 0xff or bool(terminated and not self.game_complete)
        self.life_lost = (self.raw_life < previous_life and self.raw_life != 0xff) or (
            self.game_over and previous_life != 0xff)
        if self.life_lost:
            self.deaths += 1
        old_index = (previous_world - 1) * 4 + previous_stage
        new_index = (self.world - 1) * 4 + self.stage
        if self.stage_transition and new_index > old_index:
            # A warp may skip several numbered stages; it is one completed
            # transition, not proof that every intervening stage was cleared.
            self.stages_cleared += 1
        if self.game_complete and not self.stage_transition:
            self.stages_cleared += 1
        self.session_frame += 1
        self.stage_frame = 0 if self.stage_transition else self.stage_frame + 1
        if self.game_complete:
            return "game_complete"
        if self.game_over:
            return "game_over"
        if truncated:
            return "truncated"
        return None

    def mark_memory_reset(self) -> None:
        self.memory_generation += 1

    def snapshot(self, epoch: int) -> dict[str, Any]:
        return {"run_id": self.run_id, "epoch": epoch,
                "session_frame": self.session_frame, "stage_frame": self.stage_frame,
                "world": self.world, "stage": self.stage,
                "lives_remaining": playable_lives(self.raw_life),
                "life_counter_raw": self.raw_life,
                "stages_cleared": self.stages_cleared, "deaths": self.deaths,
                "stage_transition": self.stage_transition, "stage_clear": self.stage_clear,
                "life_lost": self.life_lost, "game_over": self.game_over,
                "game_complete": self.game_complete,
                "memory_generation": self.memory_generation,
                "source": "post_step_ram"}
