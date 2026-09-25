"""Structured NES telemetry for the JevT++ Mario decision service.

The model receives object-centric state, never framebuffer pixels.  The RAM
layout is the public Super Mario Bros. memory layout used by NES tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


ENEMY_NAMES = {
    0x00: "green_koopa",
    0x01: "koopa_variant_01",
    0x02: "buzzy_beetle",
    0x03: "red_koopa",
    0x05: "hammer_bro",
    0x06: "goomba",
    0x07: "bloober",
    0x08: "bullet_bill",
    0x0A: "grey_cheep_cheep",
    0x0B: "red_cheep_cheep",
    0x0C: "podoboo",
    0x0D: "piranha_plant",
    0x0E: "jumping_green_paratroopa",
    0x11: "lakitu",
    0x12: "spiny",
    0x14: "flying_cheep_cheep",
    0x2D: "bowser",
    0x2E: "powerup",
    0x2F: "vine",
    0x30: "flagpole",
    0x31: "flagpole",
    0x33: "cannon_bullet_bill",
}
# The SMB object dispatch table uses 0x24..0x2A for large platforms,
# including the horizontally moving 0x28 object. They occupy enemy slots in
# RAM but are navigable supports, never creatures to kill.
MOVING_PLATFORM_IDS = set(range(0x24, 0x2B))
NON_HOSTILE_IDS = {0x2E, 0x2F, 0x30, 0x31} | MOVING_PLATFORM_IDS


@dataclass
class HostileLife:
    kind_id: int
    x: int
    state: int
    relative_x: int
    eliminated: bool = False
    passed_alive: bool = False


def _defeated(kind_id: int, state: int) -> bool:
    return bool(state & 0x20 or (kind_id == 0x06 and state == 4))


def _byte(ram: Sequence[int] | None, address: int) -> int:
    if ram is None or not 0 <= address < len(ram):
        return 0
    return int(ram[address])


def episode_lifecycle(info: dict[str, Any], ram: Sequence[int] | None) -> dict[str, Any]:
    """Decode SMB lifecycle without treating every death/terminal as GAME OVER.

    These addresses and the 255 life sentinel match gym_super_mario_bros.smb_env.
    Its raw life counter is extra lives: 2, 1, 0 are all playable; 255 is exhausted.
    Missing observations stay None so callers cannot mistake defaults for proof.
    """
    evidence: dict[str, Any] = {}

    def observed_byte(address: int) -> int | None:
        return _byte(ram, address) if ram is not None and address < len(ram) else None

    def number(name: str, address: int, *, offset: int = 0) -> int | None:
        if info.get(name) is not None:
            evidence[name] = f"info.{name}"
            return int(info[name])
        raw = observed_byte(address)
        evidence[name] = f"ram_0x{address:04x}" if raw is not None else "unavailable"
        return raw + offset if raw is not None else None

    def flag(name: str) -> bool | None:
        value = info.get(name)
        return bool(value) if value is not None and value in (False, True) else None

    world = number("world", 0x075F, offset=1)
    stage = number("stage", 0x075C, offset=1)
    raw_lives = number("life", 0x075A)
    lives = raw_lives
    if lives is None and info.get("lives") is not None:
        lives = int(info["lives"])
        evidence["life"] = "info.lives_encoding_unspecified"
    lives_remaining = (0 if raw_lives == 255 else raw_lives + 1) if raw_lives is not None else None
    if info.get("time") is not None:
        time_left = int(info["time"])
        evidence["time_left"] = "info.time"
    else:
        digits = [observed_byte(address) for address in range(0x07F8, 0x07FB)]
        valid_digits = all(digit is not None and 0 <= digit <= 9 for digit in digits)
        time_left = sum(int(digit) * place for digit, place in zip(digits, (100, 10, 1))) if valid_digits else None
        evidence["time_left"] = "ram_0x07f8_0x07fa_decimal" if valid_digits else "unavailable"

    player_state, viewport = observed_byte(0x000E), observed_byte(0x00B5)
    death_observations = {name: flag(name) for name in ("death", "dead") if flag(name) is not None}
    if player_state is not None:
        death_observations["ram_0x000e_dead_or_dying"] = player_state in (6, 11)
    if viewport is not None:
        death_observations["ram_0x00b5_below_viewport"] = viewport > 1
    dead = any(death_observations.values()) if death_observations else None
    evidence["dead"] = death_observations

    clear_observations = {name: flag(name) for name in ("flag_get", "clear", "stage_clear")
                          if flag(name) is not None}
    game_mode, movement_state = observed_byte(0x0770), observed_byte(0x001D)
    if game_mode is not None:
        clear_observations["ram_0x0770_end_of_world"] = game_mode == 2
    stage_enemies = [observed_byte(address) for address in range(0x0016, 0x001B)]
    if movement_state is not None and all(kind is not None for kind in stage_enemies):
        clear_observations["ram_0x001d_flag_or_bowser"] = (
            movement_state == 3 and any(kind in (0x2D, 0x31) for kind in stage_enemies))
    stage_clear = any(clear_observations.values()) if clear_observations else None
    evidence["stage_clear"] = clear_observations

    game_over_observations = {name: flag(name) for name in ("game_over", "is_game_over")
                              if flag(name) is not None}
    if raw_lives is not None:
        game_over_observations[f"{evidence['life']}_equals_255"] = raw_lives == 255
    game_over = any(game_over_observations.values()) if game_over_observations else None
    evidence["game_over"] = game_over_observations
    evidence["raw_life_counter"] = raw_lives
    evidence["lives_remaining"] = "SMB raw life + 1; 255 means exhausted" if raw_lives is not None else "unavailable"

    # Neither a generic done/truncated flag nor clearing 1-1 proves completion.
    if stage_clear is False or (world is not None and world != 8) or (stage is not None and stage != 4):
        full_game_completed = False
    elif world == 8 and stage == 4 and stage_clear is True:
        full_game_completed = True
    else:
        full_game_completed = None
    evidence["full_game_completed"] = {"world": world, "stage": stage, "stage_clear": stage_clear,
                                       "rule": "world == 8 and stage == 4 and stage_clear"}
    status = ("game_over" if game_over is True else "full_game_completed" if full_game_completed is True else
              "death" if dead is True else "stage_clear" if stage_clear is True else
              "active" if dead is False and game_over is False else "unknown")
    return {"world": world, "stage": stage, "lives": lives, "lives_remaining": lives_remaining,
            "time_left": time_left, "dead": dead, "stage_clear": stage_clear,
            "game_over": game_over, "full_game_completed": full_game_completed,
            "lifecycle_status": status, "lifecycle_evidence": evidence}


def _solid(tile: int) -> bool:
    # Coins occur in block RAM but do not support the player.
    return tile not in (0, 0xC2, 0xC3)


def _tile(ram: Sequence[int] | None, x: int, y: int) -> int:
    row = (y - 32) // 16
    if not 0 <= row < 13 or x < 0:
        return 0
    return _byte(ram, 0x500 + ((x // 256) % 2) * 208 + row * 16 + (x % 256) // 16)


def collision_preview(ram: Sequence[int] | None, x: int, top_y: int) -> dict[str, Any]:
    """Scan full-height block RAM, using pixel edges rather than tile offsets.

    The player origin retains a 32px sprite box even for small Mario. The old
    nine-row window lost the floor while airborne and treated pipe edges as pits.
    """
    feet = top_y + 32
    front = x + 12
    gap_distance = obstacle_distance = None
    gap_width = obstacle_height = 0
    columns = []
    overhead_blocks = []
    landing_surfaces = []
    lower_passage = None
    side_pipe = None
    for tx in range((x + 8) // 16, (x + 8) // 16 + 13):
        world_x = tx * 16
        surfaces = [32 + row * 16 for row in range(13)
                    if _solid(_tile(ram, world_x, 32 + row * 16))]
        floor = min((sy for sy in surfaces if sy >= feet - 4), default=None)
        body = [sy for sy in surfaces if top_y + 16 <= sy < feet - 4]
        columns.append({"x": world_x, "relative_x": world_x - front,
                        "floor_y": floor, "solid_y": surfaces,
                        "tile_ids": [_tile(ram,world_x,32+row*16) for row in range(13)]})
        for surface in surfaces:
            if top_y - 96 <= surface <= top_y - 24 and not _solid(_tile(ram,world_x,surface+16)):
                overhead_blocks.append({"x":world_x,"y":surface,"relative_x":world_x+8-x,
                                        "tile_id":_tile(ram,world_x,surface),"source":"observed_ram_solid"})
        if world_x + 16 < front:
            continue
        # An overhead brick is not a floor beneath Mario. Counting any solid
        # row hid the first 32px of the second pit and made takeoff depend on a
        # one-pixel alignment difference.
        if floor is None and gap_distance is None:
            gap_distance = max(0, world_x - front)
        if gap_distance is not None and floor is None:
            gap_width += 16
        if body and obstacle_distance is None:
            obstacle_distance = max(0, world_x - front)
            wall_top = min(body)
            while wall_top > 32 and _solid(_tile(ram, world_x, wall_top - 16)):
                wall_top -= 16
            obstacle_height = feet - wall_top
        if body and lower_passage is None and 0 <= world_x-front <= 48:
            for lower_floor in (sy for sy in surfaces if sy >= feet+24):
                roof=max((sy+16 for sy in surfaces if sy < lower_floor),default=32)
                if lower_floor-roof < 40:
                    continue
                for offset in range(16,97,16):
                    entrance=((x-offset)//16)*16
                    if (_solid(_tile(ram,entrance+8,lower_floor)) and
                        all(not _solid(_tile(ram,entrance+8,sy))
                            for sy in range((feet//16)*16,lower_floor,16))):
                        lower_passage={"floor_y":lower_floor,"roof_y":roof,"clearance_pixels":lower_floor-roof,
                                       "entrance_x":entrance,"wall_x":world_x,"source":"observed_ram_lower_corridor"}
                        break
                if lower_passage is not None:
                    break
    behind_floor_safe = all(any(_solid(_tile(ram,x-offset,32+row*16))
                                for row in range(13) if 32+row*16 >= feet-4)
                            for offset in (16,32,48)) if ram is not None else False
    # SMB RenderSidewaysPipe uses metatiles 1c/1f at the open left lip.
    # Only expose a route when the lip and its adjacent standing floor are
    # observed together. This is an interaction affordance, not an empty tile.
    for tx in range(max(0,(x-64)//16),(x+97)//16):
        mouth_x=tx*16
        for pipe_top in range(32,192,16):
            if (_tile(ram,mouth_x,pipe_top)==0x1c and
                _tile(ram,mouth_x,pipe_top+16)==0x1f and
                _solid(_tile(ram,mouth_x-16,pipe_top+32)) and
                not _solid(_tile(ram,mouth_x-16,pipe_top+16))):
                side_pipe={"mouth_x":mouth_x,"top_y":pipe_top,"floor_y":pipe_top+32,
                           "approach_x":mouth_x-24,"direction":"right",
                           "source":"observed_ram_side_pipe_1c_1f"}
                break
        if side_pipe is not None: break
    # Resource targets may remain just behind Mario after an enemy jump.
    # This short rear window is still visible; it is not a remembered level map.
    for tx in range((x-48)//16,(x+8)//16):
        world_x = tx*16
        for surface in range(32,208,16):
            tile = _tile(ram,world_x,surface)
            if tile == 0xC1 and top_y-96 <= surface <= top_y-24:
                overhead_blocks.append({"x":world_x,"y":surface,"relative_x":world_x+8-x,
                                        "tile_id":tile,"source":"observed_ram_solid"})
    # The planner may brake/return. Expose only the currently visible RAM
    # interval; off-screen nametable entries are not trustworthy observations.
    camera_x = _byte(ram, 0x071A) * 256 + _byte(ram, 0x071C)
    planning_columns = [{"x": tx * 16,
                         "solid_y": [sy for sy in range(32, 240, 16)
                                     if _solid(_tile(ram, tx * 16, sy))]}
                        for tx in range((camera_x + 15) // 16, (camera_x + 256) // 16)] if ram is not None else []
    # Rolling nametables outside the viewport may contain a previous screen.
    # Keep isolated supports stable vertically, but never invent offscreen ones.
    landing_surfaces = []
    for column in planning_columns:
        surfaces = column["solid_y"]
        for surface in surfaces:
            if surface-16 in surfaces or any(other>surface for other in surfaces):
                continue
            world_x = column["x"]
            previous=next((item for item in landing_surfaces
                           if item["y"]==surface and item["right_x"]==world_x),None)
            if previous is not None: previous["right_x"]=world_x+16
            else: landing_surfaces.append({"left_x":world_x,"right_x":world_x+16,"y":surface,
                                          "source":"observed_ram_isolated_support"})
    return {"available": ram is not None, "player_feet_y": feet,
            "planning_columns": planning_columns,
            "behind_floor_safe":behind_floor_safe,
            "gap_distance_pixels": gap_distance, "gap_width_pixels": gap_width,
            "obstacle_distance_pixels": obstacle_distance,
            "obstacle_height_pixels": obstacle_height, "columns": columns,
            "overhead_blocks":overhead_blocks,"lower_passage":lower_passage,"side_pipe":side_pipe,
            "landing_surfaces":landing_surfaces}


def unwrap_ram(environment: Any) -> Sequence[int] | None:
    current = environment
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        ram = getattr(current, "ram", None)
        if ram is not None:
            return ram
        current = getattr(current, "env", None)
        if current is None:
            break
    return None


@dataclass
class StateTracker:
    horizon: int = 8
    previous_x: int | None = None
    previous_y: int | None = None
    best_x: int = 0
    stalled_frames: int = 0
    enemy_dx: dict[int, int] = field(default_factory=dict)
    enemy_samples: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    airborne_frames: int = 0
    stable_y_frames: int = 0
    takeoff_x: int = 0
    last_grounded_gap: int | None = None
    last_grounded_gap_width: int = 0
    last_grounded_obstacle: int | None = None
    last_grounded_obstacle_height: int = 0
    last_trusted_grid: list[str] = field(default_factory=list)
    tracked_action: str | None = None
    action_start_x: int = 0
    action_frames: int = 0
    frames_seen: int = 0
    start_x: int | None = None
    previous_status: str | None = None
    previous_coins: int | None = None
    previous_score: int | None = None
    eliminations_confirmed: int = 0
    stomps_confirmed: int = 0
    hostiles_seen: int = 0
    hostiles_passed_alive: int = 0
    hostiles_passed_then_eliminated: int = 0
    powerups_collected: int = 0
    hostile_lives: dict[int, HostileLife] = field(default_factory=dict)

    @property
    def eliminations_detected(self) -> int:
        """Compatibility alias: eliminations now require observed RAM evidence."""
        return self.eliminations_confirmed

    def reset(self) -> None:
        self.__dict__.update(StateTracker(self.horizon).__dict__)

    def _track_combat(self, ram: Sequence[int] | None, mario_x: int, *, dead: bool) -> None:
        self.elimination_events = []
        # Read raw slots before the visible-enemy filter discards corpses.
        # Disappearance alone never establishes how an enemy was removed.
        for slot in range(5):
            kind_id = _byte(ram, 0x0016 + slot)
            if not _byte(ram, 0x000F + slot) or kind_id in NON_HOSTILE_IDS:
                self.hostile_lives.pop(slot, None)
                continue
            enemy_x = _byte(ram, 0x006E + slot) * 256 + _byte(ram, 0x0087 + slot)
            state = _byte(ram, 0x001E + slot)
            relative_x = enemy_x - mario_x
            defeated = _defeated(kind_id, state)
            life = self.hostile_lives.get(slot)
            if life is None or life.kind_id != kind_id or abs(enemy_x - life.x) > 64:
                # A slot is storage, not an enemy identity. A first observation
                # in a defeated state is not evidence of a new elimination.
                self.hostile_lives[slot] = HostileLife(
                    kind_id, enemy_x, state, relative_x, eliminated=defeated)
                if not defeated:
                    self.hostiles_seen += 1
                continue
            previously_defeated = _defeated(kind_id, life.state)
            if not life.eliminated and not previously_defeated:
                if state == 4 and life.state != 4 and kind_id in {0, 1, 2, 6}:
                    self.stomps_confirmed += 1
                if defeated:
                    life.eliminated = True
                    self.eliminations_confirmed += 1
                    self.elimination_events.append({"slot": slot, "kind_id": kind_id,
                        "x": enemy_x, "screen_y": _byte(ram, 0x00CF + slot),
                        "previous_state": life.state, "state": state,
                        "source": "RAM_alive_to_defeated"})
                    if life.passed_alive:
                        self.hostiles_passed_then_eliminated += 1
                elif not dead and not life.passed_alive and life.relative_x >= 0 > relative_x:
                    life.passed_alive = True
                    self.hostiles_passed_alive += 1
            life.x, life.state, life.relative_x = enemy_x, state, relative_x

    def _enemies(self, ram: Sequence[int] | None, mario_x: int, mario_y: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        # SMB reserves slot 5 for the emerging/moving power-up. Scanning only
        # the five ordinary enemy slots made every mushroom invisible.
        for slot in range(6):
            if not _byte(ram, 0x000F + slot):
                self.enemy_dx.pop(slot, None)
                self.enemy_samples.pop(slot, None)
                continue
            kind_id = _byte(ram, 0x0016 + slot)
            enemy_state = _byte(ram, 0x001E + slot)
            if _defeated(kind_id, enemy_state):
                self.enemy_dx.pop(slot, None)
                self.enemy_samples.pop(slot, None)
                continue
            enemy_x = _byte(ram, 0x006E + slot) * 256 + _byte(ram, 0x0087 + slot)
            dx = enemy_x - mario_x
            if not -192 <= dx <= 320:
                self.enemy_dx.pop(slot, None)
                self.enemy_samples.pop(slot, None)
                continue
            last_dx = self.enemy_dx.get(slot, dx)
            previous = self.enemy_samples.get(slot)
            entity_y = _byte(ram, 0x00CF + slot)
            velocity_observed = bool(previous and previous[0] == kind_id and
                                     previous[2] == self.frames_seen - 1 and abs(enemy_x - previous[1]) <= 16)
            self.enemy_samples[slot] = (kind_id, enemy_x, self.frames_seen, entity_y)
            self.enemy_dx[slot] = dx
            result.append(
                {
                    "slot": slot,
                    "kind": ("moving_platform" if kind_id in MOVING_PLATFORM_IDS else
                             ENEMY_NAMES.get(kind_id, f"enemy_0x{kind_id:02x}")),
                    "kind_id": kind_id,
                    "stompable": kind_id in {0,1,2,6},
                    "state": enemy_state,
                    "x": enemy_x,
                    "screen_y": entity_y,
                    "vx": enemy_x - previous[1] if velocity_observed else None,
                    "vy": entity_y - previous[3] if velocity_observed else None,
                    "category": "collectible" if kind_id == 0x2E else
                                "moving_support" if kind_id in MOVING_PLATFORM_IDS else
                                "level_object" if kind_id in {0x2F, 0x30, 0x31} else "hostile",
                    "relative_x_pixels": dx,
                    "relative_y_pixels": _byte(ram, 0x00CF + slot) - mario_y,
                    "relative_velocity_x": dx - last_dx if velocity_observed else 0,
                    "velocity_observed": velocity_observed,
                }
            )
        return sorted(result, key=lambda item: item["relative_x_pixels"])

    @staticmethod
    def _grid(ram: Sequence[int] | None, mario_x: int, mario_y: int,
              enemies: list[dict[str, Any]]) -> list[str]:
        if ram is None:
            return []
        width, height = 11, 9
        cells = [["." for _ in range(width)] for _ in range(height)]
        for row, dy_tiles in enumerate(range(-4, 5)):
            for column, dx_tiles in enumerate(range(-2, 9)):
                sample_x = mario_x + dx_tiles * 16
                sample_y = mario_y + dy_tiles * 16
                page = (sample_x // 256) % 2
                tile_x = (sample_x % 256) // 16
                tile_y = (sample_y - 32) // 16
                if 0 <= tile_y < 13:
                    address = 0x0500 + page * 208 + tile_y * 16 + tile_x
                    if _solid(_byte(ram, address)):
                        cells[row][column] = "#"
        cells[4][2] = "M"
        for enemy in enemies:
            column = 2 + round(enemy["relative_x_pixels"] / 16)
            row = 4 + round(enemy["relative_y_pixels"] / 16)
            if 0 <= row < height and 0 <= column < width and cells[row][column] != "M":
                cells[row][column] = "P" if enemy.get("category") == "collectible" else "E"
        return ["".join(row) for row in cells]

    @staticmethod
    def _terrain(grid: list[str]) -> dict[str, Any]:
        if not grid or not any("M" in row for row in grid):
            return {"geometry_available": False, "gap_distance_tiles": None,
                    "obstacle_distance_tiles": None, "clear_forward_tiles": 0}
        mario_row = next(index for index, row in enumerate(grid) if "M" in row)
        mario_col = grid[mario_row].index("M")
        ground_row = next((row for row in range(mario_row + 1, len(grid))
                           if grid[row][mario_col] == "#"), None)
        gap = obstacle = None
        gap_width = obstacle_height = 0
        clear = 0
        if ground_row is not None:
            for column in range(mario_col + 1, len(grid[0])):
                distance = column - mario_col
                # The route can step down from a pipe or block onto lower
                # ground. Requiring support on exactly Mario's current row
                # turns every ledge into a fake bottomless gap.
                supported = any(grid[row][column] == "#"
                                for row in range(ground_row, len(grid)))
                blocked = ground_row > 0 and grid[ground_row - 1][column] == "#"
                if not supported and gap is None:
                    gap = distance
                if gap is not None and not supported:
                    gap_width += 1
                if blocked and obstacle is None:
                    obstacle = distance
                    probe = ground_row - 1
                    while probe >= 0 and grid[probe][column] == "#":
                        obstacle_height += 1
                        probe -= 1
                if gap is None and obstacle is None:
                    clear = distance
        return {
            "geometry_available": ground_row is not None,
            "gap_distance_tiles": gap,
            "obstacle_distance_tiles": obstacle,
            "gap_ahead": gap is not None and gap <= 3,
            "obstacle_ahead": obstacle is not None and obstacle <= 3,
            "gap_width_tiles_visible": gap_width,
            "obstacle_height_tiles": obstacle_height,
            "clear_forward_tiles": clear,
        }

    def parse(self, info: dict[str, Any], ram: Sequence[int] | None, *,
              previous_action: str | None, previous_reward: float,
              response_delay_frames: int, objective_mode: str = "speedrun") -> tuple[dict[str, Any], dict[str, Any]]:
        x = int(info.get("x_pos", info.get("progress", _byte(ram, 0x6D) * 256 + _byte(ram, 0x86))) or 0)
        screen_y = _byte(ram, 0xCE) if ram is not None else int(info.get("y_pixel", 0) or 0)
        y = int(info.get("y_pos", 255 - screen_y) or 0)
        dx = 0 if self.previous_x is None else x - self.previous_x
        dy = 0 if self.previous_y is None else y - self.previous_y
        # Respawn/viewport wraps are discontinuities, not jump impulses.
        velocity_valid = abs(dx) <= 16 and abs(dy) <= 16
        if not velocity_valid:
            dx = dy = 0
        enemies = self._enemies(ram, x, screen_y)
        grid = self._grid(ram, x, screen_y, enemies)
        terrain = self._terrain(grid)
        self.stable_y_frames = self.stable_y_frames + 1 if abs(dy) <= 1 else 0
        grid_support = bool(grid and any(row[2] == "#" for row in grid[5:7]))
        # 0x1D is the game's movement state: 0 grounded, 1 jumping,
        # 2 falling, 3 flag/vine. Stable Y is not proof of a ground contact.
        movement_state = _byte(ram, 0x1D)
        grounded = movement_state == 0 if ram is not None else (abs(dy) <= 1 and grid_support)
        feet=screen_y+32
        support_contact=(ram is not None and movement_state==2 and self.stable_y_frames>=3 and dy==0 and
                         feet%16==0 and _solid(_tile(ram,x+8,feet+1)) and
                         not _solid(_tile(ram,x+8,feet-1)) and _byte(ram,0xb5)==1)
        grounded=grounded or support_contact
        if not velocity_valid:
            grounded = False
        preview = collision_preview(ram, x, screen_y)
        moving_surfaces = [{"left_x":entity["x"],"right_x":entity["x"]+32,
                            "y":entity["screen_y"],"vx":entity["vx"],"vy":entity["vy"],
                            "velocity_observed":entity["velocity_observed"],
                            "slot":entity["slot"],"kind_id":entity["kind_id"],
                            "source":"observed_ram_moving_support"}
                           for entity in enemies if entity["category"]=="moving_support"]
        preview["moving_surfaces"] = moving_surfaces
        terrain.update({
            "geometry_available": preview["available"],
            "gap_distance_tiles": None if preview["gap_distance_pixels"] is None else (preview["gap_distance_pixels"] + 15) // 16,
            "obstacle_distance_tiles": None if preview["obstacle_distance_pixels"] is None else (preview["obstacle_distance_pixels"] + 15) // 16,
            "gap_ahead": preview["gap_distance_pixels"] is not None and preview["gap_distance_pixels"] <= 48,
            "obstacle_ahead": preview["obstacle_distance_pixels"] is not None and preview["obstacle_distance_pixels"] <= 48,
            "gap_width_tiles_visible": preview["gap_width_pixels"] // 16,
            "obstacle_height_tiles": (preview["obstacle_height_pixels"] + 15) // 16,
        })
        terrain["drop_ahead"] = False
        terrain["drop_distance_tiles"] = None
        # On top of a pipe/block, absence of support at the same elevation is
        # usually a safe step down to the main floor, not a bottomless pit.
        # Treating it as a gap made the controller hold a full jump across the
        # following safe ground and overshoot into the next real pit.
        if ram is None and grounded and y > 96 and terrain.get("gap_distance_tiles") is not None:
            terrain["drop_ahead"] = True
            terrain["drop_distance_tiles"] = terrain["gap_distance_tiles"]
            terrain["gap_distance_tiles"] = None
            terrain["gap_width_tiles_visible"] = 0
            terrain["gap_ahead"] = False
        self.airborne_frames = 0 if grounded else self.airborne_frames + 1
        if grounded:
            self.takeoff_x = x
            self.last_grounded_gap = terrain["gap_distance_tiles"]
            self.last_grounded_gap_width = terrain["gap_width_tiles_visible"]
            self.last_grounded_obstacle = terrain["obstacle_distance_tiles"]
            self.last_grounded_obstacle_height = terrain["obstacle_height_tiles"]
            if terrain["geometry_available"]:
                self.last_trusted_grid = list(grid)
        self.stalled_frames = self.stalled_frames + 1 if self.previous_x is not None and x <= self.previous_x else 0
        self.best_x = max(self.best_x, x, int(info.get("progress_max", 0) or 0))
        committed_gap_pixels = max(80, self.last_grounded_gap_width * 16 + 32)
        crossing_gap = (not grounded and self.last_grounded_gap is not None and
                        self.last_grounded_gap <= 3 and
                        max(0, x - self.takeoff_x) <= committed_gap_pixels)
        crossing_obstacle = (not grounded and self.last_grounded_obstacle is not None and
                             self.last_grounded_obstacle <= 3 and self.airborne_frames <= 30)

        hostile_entities = [entity for entity in enemies if entity["kind_id"] not in NON_HOSTILE_IDS]
        ahead = [enemy for enemy in hostile_entities if enemy["relative_x_pixels"] >= 0]
        powerups = [entity for entity in enemies if entity["kind_id"] == 0x2E]
        powerups_ahead = [entity for entity in powerups if entity["relative_x_pixels"] >= -96]
        lifecycle = episode_lifecycle(info, ram)
        self._track_combat(ram, x, dead=lifecycle["dead"] is True)
        nearest = ahead[0] if ahead else None
        enemy_distance = nearest["relative_x_pixels"] if nearest else 999
        relative_velocity = nearest["relative_velocity_x"] if nearest else 0
        closing_speed = max(0, -relative_velocity)
        contact_frames = round(enemy_distance / closing_speed) if nearest and closing_speed else None
        reaction_horizon = self.horizon + max(0, response_delay_frames)
        jump_deadline = contact_frames - response_delay_frames - 8 if contact_frames is not None else None
        landing_frames = None if grounded else max(0, 42 - self.airborne_frames)
        hazard = {
            "enemy_behind_distance": min((-enemy["relative_x_pixels"] for enemy in hostile_entities
                                           if -96 <= enemy["relative_x_pixels"] < 0), default=999),
            "enemy_ahead": bool(nearest),
            "enemy_distance": enemy_distance,
            "nearest_enemy_kind": nearest["kind"] if nearest else None,
            "nearest_enemy_stompable": bool(nearest and nearest["stompable"]),
            "relative_velocity_x": relative_velocity,
            "estimated_contact_frames": contact_frames,
            "projected_distance_after_reaction_pixels": max(0, enemy_distance + relative_velocity * reaction_horizon),
            "contact_within_reaction_horizon": contact_frames is not None and contact_frames <= reaction_horizon,
            "jump_must_start_this_decision": grounded and jump_deadline is not None and 0 <= jump_deadline <= self.horizon,
            "takeoff_deadline_frames": jump_deadline,
            "takeoff_window_already_missed": jump_deadline is not None and jump_deadline < 0,
            "estimated_landing_frames": landing_frames,
            "will_land_before_contact": landing_frames is not None and contact_frames is not None and landing_frames < contact_frames,
            "spacing_to_second_enemy_pixels": ahead[1]["relative_x_pixels"] - enemy_distance if len(ahead) > 1 else None,
            # NES exposes five hostile slots. Truncating to three hid the
            # piranha immediately behind a three-Goomba wave in 1-2, making
            # the controller plan a stomp onto a lethal pipe.
            "upcoming_enemies": ahead[:5],
            "nearby_enemies": sorted(hostile_entities, key=lambda enemy: abs(enemy["relative_x_pixels"]))[:5],
            "elimination_events": self.elimination_events,
        }
        engaging_enemy = (not grounded and nearest is not None and enemy_distance <= 160 and
                           self.airborne_frames <= 36)
        gap_pixels = 999 if terrain["gap_distance_tiles"] is None else terrain["gap_distance_tiles"] * 16
        obstacle_pixels = 999 if terrain["obstacle_distance_tiles"] is None else terrain["obstacle_distance_tiles"] * 16
        terrain.update({
            "gap_distance": gap_pixels,
            "obstacle_distance": obstacle_pixels,
            "observation_reliability": "high_ram" if ram is not None else "unavailable",
            "last_grounded_preview": {
                "gap_distance_tiles": self.last_grounded_gap,
                "gap_width_tiles_visible": self.last_grounded_gap_width,
                "obstacle_distance_tiles": self.last_grounded_obstacle,
                "obstacle_height_tiles": self.last_grounded_obstacle_height,
            },
        })
        if previous_action != self.tracked_action:
            self.tracked_action = previous_action
            self.action_start_x = x
            self.action_frames = 1 if previous_action else 0
        elif previous_action:
            self.action_frames += 1
        action_progress = x - self.action_start_x
        control_outcome = ("blocked" if self.stalled_frames >= 4 else
                           "jump_in_progress" if not grounded else
                           "advanced" if action_progress > 0 else "not_enough_evidence")

        status = str(info.get("status", "small"))
        coins = int(info.get("coins", 0) or 0)
        score = int(info.get("score", 0) or 0)
        coin_delta = 0 if self.previous_coins is None else max(0, coins - self.previous_coins)
        score_delta = 0 if self.previous_score is None else max(0, score - self.previous_score)
        status_rank = {"small": 0, "tall": 1, "fireball": 2}
        if self.previous_status is not None and status_rank.get(status, 0) > status_rank.get(self.previous_status, 0):
            self.powerups_collected += 1
        self.previous_status, self.previous_coins, self.previous_score = status, coins, score
        self.frames_seen += 1
        if self.start_x is None:
            self.start_x = x
        elapsed_seconds = max(1 / 60, self.frames_seen / 60)
        pace = max(0.0, (self.best_x - self.start_x) / elapsed_seconds)
        mode_labels = {
            "speedrun": "Reach the flag as quickly as possible while preserving the life.",
            "hunter": "Eliminate every reachable hostile, then reach the flag.",
            "collector": "Collect visible powerups and coins, then reach the flag.",
            "score_attack": "Maximize score from enemies, blocks, coins, and powerups before finishing.",
        }
        target_powerup = min(powerups_ahead, key=lambda item: abs(item["relative_x_pixels"])) if powerups_ahead else None

        # These checks are deterministic evidence channels, not extra model
        # guesses.  This mirrors the useful part of the Doom pipeline: derive
        # small, testable facts from the simulator and let typed judgments vote
        # on top of them.  Keeping every check in the state also makes the
        # complete controller path inspectable in the live dashboard.
        checkers = [
            {"id": "sensor_trust", "label": "RAM / GRID TRUST",
             "status": "pass" if ram is not None else "warn",
             "signal": ram is None,
             "value": terrain["observation_reliability"],
             "reason": "authoritative movement RAM and full-height block sample" if ram is not None else "RAM unavailable"},
            {"id": "ground_contact", "label": "GROUND CONTACT",
             "status": "pass" if grounded else "active", "signal": not grounded,
             "value": "grounded" if grounded else f"airborne {self.airborne_frames}f",
             "reason": "jump may start" if grounded else "preserve committed trajectory"},
            {"id": "gap_risk", "label": "FORWARD SUPPORT",
             "status": "block" if terrain["gap_ahead"] else "pass",
             "signal": bool(terrain["gap_ahead"] or crossing_gap),
             "value": "crossing" if crossing_gap else
                      (f"gap {terrain['gap_distance_tiles']}t" if terrain["gap_ahead"] else "supported"),
             "reason": "hold jump across committed gap" if crossing_gap else
                       "takeoff required" if terrain["gap_ahead"] else "no bottomless gap in trusted preview"},
            {"id": "obstacle_risk", "label": "HEADROOM / WALL",
             "status": "block" if terrain["obstacle_ahead"] else "pass",
             "signal": bool(terrain["obstacle_ahead"] or crossing_obstacle),
             "value": "crossing" if crossing_obstacle else
                      (f"wall {terrain['obstacle_distance_tiles']}t" if terrain["obstacle_ahead"] else "clear"),
             "reason": "sustain obstacle jump" if crossing_obstacle else
                       "vertical clearance required" if terrain["obstacle_ahead"] else "forward corridor clear"},
            {"id": "enemy_contact", "label": "ENEMY CONTACT",
             "status": "block" if hazard["contact_within_reaction_horizon"] else
                       "warn" if nearest and enemy_distance <= 160 else "pass",
             "signal": bool(nearest and enemy_distance <= 176),
             "value": "none" if nearest is None else f"{nearest['kind']} {enemy_distance}px",
             "reason": "contact inside controller horizon" if hazard["contact_within_reaction_horizon"] else
                       "enemy approaching" if nearest and enemy_distance <= 160 else "no immediate hostile"},
            {"id": "takeoff_window", "label": "TAKEOFF WINDOW",
             "status": "block" if hazard["takeoff_window_already_missed"] else
                       "active" if hazard["jump_must_start_this_decision"] else "pass",
             "signal": bool(hazard["jump_must_start_this_decision"] or hazard["takeoff_window_already_missed"]),
             "value": "open now" if hazard["jump_must_start_this_decision"] else
                      "missed" if hazard["takeoff_window_already_missed"] else
                      (f"{jump_deadline}f" if jump_deadline is not None else "not armed"),
             "reason": "act in this decision batch" if hazard["jump_must_start_this_decision"] else
                       "reaction began too late" if hazard["takeoff_window_already_missed"] else "outside takeoff window"},
            {"id": "trajectory_commit", "label": "TRAJECTORY COMMIT",
             "status": "active" if crossing_gap or crossing_obstacle or engaging_enemy else "pass",
             "signal": bool(crossing_gap or crossing_obstacle or engaging_enemy),
             "value": "gap" if crossing_gap else "obstacle" if crossing_obstacle else
                      "enemy" if engaging_enemy else "none",
             "reason": "do not release jump mid-hazard" if crossing_gap or crossing_obstacle or engaging_enemy else
                       "controller may select a new macro"},
            {"id": "landing_safety", "label": "LANDING SAFETY",
             "status": "warn" if not grounded and nearest and not hazard["will_land_before_contact"] else "pass",
             "signal": bool(not grounded),
             "value": "grounded" if grounded else
                      (f"eta {landing_frames}f" if landing_frames is not None else "unknown"),
             "reason": "enemy may reach landing zone first" if not grounded and nearest and not hazard["will_land_before_contact"] else
                       "landing window currently viable"},
            {"id": "progress_watchdog", "label": "PROGRESS WATCHDOG",
             "status": "block" if self.stalled_frames >= 4 else "pass",
             "signal": self.stalled_frames >= 4,
             "value": f"stalled {self.stalled_frames}f" if self.stalled_frames else f"+{max(0, action_progress)}px",
             "reason": "escape macro required" if self.stalled_frames >= 4 else "forward progress observed"},
            {"id": "objective_target", "label": "OBJECTIVE TARGET",
             "status": "active", "signal": True,
             "value": ("hostile" if objective_mode == "hunter" and nearest else
                       "power-up" if objective_mode == "collector" and target_powerup else
                       "score event" if objective_mode == "score_attack" and (nearest or target_powerup) else "finish"),
             "reason": mode_labels.get(objective_mode, mode_labels["speedrun"])},
            {"id": "latency_budget", "label": "REACTION BUDGET",
             "status": "warn" if response_delay_frames > self.horizon else "pass",
             "signal": response_delay_frames > 0,
             "value": f"{reaction_horizon}f horizon",
             "reason": f"model delay {response_delay_frames}f + action window {self.horizon}f"},
        ]
        checker_summary = {
            "blocking": sum(check["status"] == "block" for check in checkers),
            "warnings": sum(check["status"] == "warn" for check in checkers),
            "active": sum(bool(check["signal"]) for check in checkers),
            "total": len(checkers),
        }

        model_state = {
            "objective": mode_labels.get(objective_mode, mode_labels["speedrun"]),
            "strategy": {"mode": objective_mode,
                         "mode_speedrun": objective_mode == "speedrun",
                         "mode_hunter": objective_mode == "hunter",
                         "mode_collector": objective_mode == "collector",
                         "mode_score_attack": objective_mode == "score_attack"},
            "level": {"world": lifecycle["world"], "stage": lifecycle["stage"]},
            "player": {"x": x, "y": y, "vx": dx, "vy": dy, "velocity_valid": velocity_valid, "grounded": grounded,
                       "screen_y": screen_y, "feet_y": screen_y + 32,
                       "movement_state": movement_state,
                       "physics_vx": (_byte(ram, 0x57) if _byte(ram, 0x57) < 128 else _byte(ram, 0x57) - 256) / 16,
                       "ground_source": "ram_support_contact" if support_contact else "ram_0x1d" if ram is not None else "unavailable",
                       "jump_phase": "grounded" if grounded else "rising" if dy > 0 else "falling",
                       "powerup_status": status,
                       "star_timer_ram_0x079f": _byte(ram, 0x079F) if ram is not None else None,
                       "star_active": bool(_byte(ram, 0x079F)) if ram is not None else None},
            "trajectory": {"airborne_frames": self.airborne_frames,
                           "horizontal_distance_since_takeoff_pixels": max(0, x - self.takeoff_x),
                           "crossing_gap": crossing_gap,
                           "crossing_obstacle": crossing_obstacle,
                           "engaging_enemy": engaging_enemy,
                           "committed_geometry": ("gap" if crossing_gap else
                                                  "obstacle" if crossing_obstacle else "none")},
            "hazard": hazard,
            "terrain": terrain,
            "collision": preview,
            "reaction_timing": {"action_horizon_frames": self.horizon,
                                "last_inference_delay_frames": response_delay_frames,
                                "total_reaction_horizon_frames": reaction_horizon},
            "recent_control": {"action": previous_action, "reward": previous_reward,
                               "frames_observed": self.action_frames,
                               "progress_gained_pixels": action_progress,
                               "stalled": self.stalled_frames, "outcome": control_outcome},
            "detectors": {"hostiles_visible": len(ahead),
                          "powerups_visible": len(powerups_ahead),
                          "powerup_distance": target_powerup["relative_x_pixels"] if target_powerup else 999,
                          "powerup_relative_y": target_powerup["relative_y_pixels"] if target_powerup else None,
                          "coins": coins, "coins_gained_this_frame": coin_delta,
                          "eliminations_detected": self.eliminations_detected,
                          "eliminations_confirmed": self.eliminations_confirmed,
                          "stomps_confirmed": self.stomps_confirmed,
                          "hostiles_seen": self.hostiles_seen,
                          "hostiles_passed_alive": self.hostiles_passed_alive,
                          "hostiles_passed_not_eliminated": self.hostiles_passed_alive - self.hostiles_passed_then_eliminated,
                          "powerups_collected": self.powerups_collected},
            "optimization": {"score": score, "score_delta_this_frame": score_delta,
                             "progress_pixels_per_second": round(pace, 2),
                             "route_progress": self.best_x,
                             "time_budget_seconds": int(info.get("time", 0) or 0)},
            "checker_bus": {"summary": checker_summary, "checks": checkers},
            "episode": {**lifecycle, "progress": x, "best_progress": self.best_x},
        }
        debug = {
            "visible_enemies": enemies,
            "local_grid": grid,
            "trusted_grid": self.last_trusted_grid,
            "grid_source": "live_ram_full_height" if ram is not None else "unavailable",
        }
        self.previous_x, self.previous_y = x, y
        return model_state, debug
