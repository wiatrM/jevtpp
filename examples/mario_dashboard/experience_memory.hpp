#pragma once

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mario {

// Observational memory, deliberately separate from policy and model weights.
// Unseen cells are unknown; recorded values never change control thresholds.
class KnowledgeMemory {
    using Json = nlohmann::json;
    static constexpr std::size_t max_columns = 512;
    static constexpr std::size_t max_enemy_types = 64;

    struct RunningStats {
        std::uint64_t count = 0;
        double mean = 0, m2 = 0, minimum = 0, maximum = 0;
        void add(double sample) {
            if (!std::isfinite(sample)) return;
            if (count++ == 0) minimum = maximum = sample;
            minimum = std::min(minimum, sample);
            maximum = std::max(maximum, sample);
            const double delta = sample - mean;
            mean += delta / static_cast<double>(count);
            m2 += delta * (sample - mean);
        }
        Json json() const {
            return {{"count", count}, {"mean", count ? Json(mean) : Json(nullptr)},
                    {"min", count ? Json(minimum) : Json(nullptr)},
                    {"max", count ? Json(maximum) : Json(nullptr)},
                    {"stddev", count > 1 ? Json(std::sqrt(std::max(0., m2 / (count-1)))) : Json(nullptr)},
                    {"m2", m2}};
        }
        static RunningStats read(const Json& value) {
            RunningStats stats;
            stats.count = value.at("count").get<std::uint64_t>();
            if (stats.count > 1000000000ULL) throw std::invalid_argument("statistics count exceeds limit");
            if (!stats.count) return stats;
            stats.mean = value.at("mean").get<double>();
            stats.minimum = value.at("min").get<double>();
            stats.maximum = value.at("max").get<double>();
            stats.m2 = value.at("m2").get<double>();
            if (!std::isfinite(stats.mean) || !std::isfinite(stats.minimum) ||
                !std::isfinite(stats.maximum) || !std::isfinite(stats.m2) || stats.m2 < 0 ||
                stats.minimum > stats.mean || stats.maximum < stats.mean)
                throw std::invalid_argument("invalid running statistics");
            return stats;
        }
    };
    struct Column {
        int world = 1, stage = 1, x = 0;
        std::vector<int> solid_y;
        std::uint64_t observations = 0, changes = 0, seen_order = 0;
        int last_epoch = -1, last_frame = -1;
        bool imported = false;
        Json json() const {
            return {{"world",world}, {"stage",stage}, {"x",x}, {"solid_y",solid_y},
                    {"observations",observations}, {"changes",changes}, {"last_epoch",last_epoch},
                    {"last_frame",last_frame}, {"seen_order",seen_order},
                    {"imported",imported},
                    {"confidence",std::min(.95, .5 + .05 * static_cast<double>(observations))},
                    {"confidence_kind","observation-support heuristic, not a calibrated probability"},
                    {"source","visible_block_ram"}};
        }
    };
    struct EnemySample { int kind = -1, frame = 0; double x = 0; };
    struct EnemyKnowledge {
        std::string name;
        RunningStats velocity;
        std::uint64_t observations = 0;
        Json json() const {
            return {{"name",name},{"observations",observations},{"velocity_x_px_per_frame",velocity.json()}};
        }
    };
    struct Jump {
        bool active = false, bounced = false;
        int start_frame = 0, held_frames = 0;
        double start_x = 0, start_feet = 0, max_height = 0, previous_dy = 0;
    };
    struct LearningDelta {
        bool observation_accepted = false;
        int columns_observed = 0, new_columns = 0, changed_columns = 0;
        int new_enemy_types = 0, enemy_velocity_samples = 0, jump_samples = 0, bounce_flights = 0;
        Json json() const {
            return {{"source","current_observation_delta"},{"observation_accepted",observation_accepted},
                {"columns_observed",columns_observed},{"new_columns",new_columns},{"changed_columns",changed_columns},
                {"new_enemy_types",new_enemy_types},{"enemy_velocity_samples",enemy_velocity_samples},
                {"jump_samples",jump_samples},{"bounce_flights",bounce_flights}};
        }
    };

public:
    explicit KnowledgeMemory(std::string environment = "SuperMarioBros-1-1-v0")
        : environment_(std::move(environment)) {}

    void reset_episode(int epoch) {
        epoch_ = epoch;
        frame_ = -1;
        previous_valid_ = false;
        terminal_recorded_ = false;
        enemies_previous_.clear();
        jump_ = {};
        learning_delta_ = {};
    }

    Json learning_delta() const { return learning_delta_.json(); }

    void observe(const Json& state, int epoch, int frame) {
        const auto session = state.value("session",Json::object());
        const bool physical_session = session.value("run_id",Json(nullptr)).is_number_integer() &&
            session.value("deaths",Json(nullptr)).is_number_integer() &&
            session.value("world",Json(nullptr)).is_number_integer() &&
            session.value("stage",Json(nullptr)).is_number_integer();
        if (physical_session) {
            const auto identity = std::to_string(session.at("run_id").get<int>()) + ":" +
                std::to_string(session.at("deaths").get<int>()) + ":" +
                std::to_string(session.at("world").get<int>()) + ":" + std::to_string(session.at("stage").get<int>());
            if (identity != physical_episode_id_) reset_episode(epoch);
            else epoch_ = epoch; // model invalidation is not a physical reset
            physical_episode_id_ = identity;
        } else {
            physical_episode_id_.clear();
            if (epoch != epoch_) reset_episode(epoch);
        }
        learning_delta_ = {};
        if (frame < 0 || frame <= frame_) return; // duplicate/out-of-order snapshot
        const auto player = state.value("player",Json::object());
        const auto episode = state.value("episode",Json::object());
        const auto level = state.value("level",Json::object());
        const auto collision = state.value("collision",Json::object());
        const auto world = level.value("world",Json(nullptr)), stage = level.value("stage",Json(nullptr));
        if (!world.is_number_integer() || !stage.is_number_integer() || world.get<int>() < 1 || stage.get<int>() < 1) {
            // Do not assign unknown RAM to a fictitious 1-1 map or train across
            // a gap in world identity. Aggregate evidence remains available.
            world_ = stage_ = 0;
            previous_valid_ = false;
            enemies_previous_.clear();
            jump_ = {};
            frame_ = frame;
            return;
        }
        learning_delta_.observation_accepted = true;
        world_ = world.get<int>();
        stage_ = stage.get<int>();
        current_x_ = player.value("x",0.);
        const double feet = player.value("feet_y",player.value("screen_y",176.) + 32.);
        const bool grounded = player.value("grounded",false);
        const bool session_dead = session.value("life_lost",Json(nullptr)) == true;
        const bool session_clear = session.value("stage_transition",Json(nullptr)) == true || session.value("game_complete",Json(nullptr)) == true;
        const bool dead = episode.value("dead",Json(nullptr)) == true || session_dead;
        const bool clear = episode.value("stage_clear",Json(nullptr)) == true || session_clear;
        const bool session_lifecycle = session.value("run_id",Json(nullptr)).is_number_integer() &&
            session.value("session_frame",Json(nullptr)).is_number_integer();
        // In full-game mode the pre-skip RAM flag and post-skip transition are
        // two observations of one event. Count only the authoritative session
        // lifecycle; raw episode terminals remain the stage-demo fallback.
        const bool outcome_dead = session_lifecycle ? session_dead : dead;
        const bool outcome_clear = session_lifecycle ? session_clear : clear;
        std::string session_event;
        if ((session_dead || session_clear) && session.value("run_id",Json(nullptr)).is_number_integer() &&
            session.value("session_frame",Json(nullptr)).is_number_integer())
            session_event = std::to_string(session.at("run_id").get<int>()) + ":" +
                std::to_string(session.at("session_frame").get<int>()) + (session_clear ? ":clear" : ":death");
        ++observed_frames_;
        if (collision.value("available",false) && !dead) {
            // Buffer pages ahead of the viewport can still hold stale tiles.
            // Explicit viewport bounds are preferred; the conservative fallback
            // records only the nearby 80px corridor, not all buffered columns.
            const int left = collision.value("visible_left_world_x",static_cast<int>(current_x_));
            const int right = collision.value("visible_right_world_x",static_cast<int>(current_x_)+80);
            for (const auto& candidate : collision.value("columns",Json::array())) {
                if (!candidate.is_object() || !candidate.contains("x") || !candidate.contains("solid_y")) continue;
                const int x = candidate.at("x").get<int>();
                if (x < left || x + 16 > right || x < 0 || x % 16 != 0) continue;
                auto ys = valid_rows(candidate.at("solid_y"));
                const auto key = column_key(world_,stage_,x);
                auto& column = columns_[key];
                ++learning_delta_.columns_observed;
                if (!column.observations) ++learning_delta_.new_columns;
                if (column.observations && column.solid_y != ys) {
                    ++column.changes;
                    ++learning_delta_.changed_columns;
                }
                column.world = world_; column.stage = stage_; column.x = x;
                column.solid_y = std::move(ys);
                ++column.observations;
                column.last_epoch = epoch;
                column.last_frame = frame;
                column.imported = false;
                column.seen_order = ++seen_order_;
            }
            while (columns_.size() > max_columns) {
                const auto oldest = std::min_element(columns_.begin(),columns_.end(),[](const auto& a,const auto& b) {
                    return a.second.seen_order < b.second.seen_order;
                });
                columns_.erase(oldest);
            }
        }

        std::map<int,EnemySample> next_enemies;
        const auto hazard = state.value("hazard",Json::object());
        for (const auto& enemy : hazard.value("upcoming_enemies",Json::array())) {
            if (!enemy.is_object() || enemy.value("category","") != "hostile") continue;
            const int kind = enemy.value("kind_id",-1), slot = enemy.value("slot",-1);
            if (kind < 0 || kind > 255 || slot < 0 || slot > 5) continue;
            const double x = enemy.value("x",current_x_ + enemy.value("relative_x_pixels",0.));
            auto found = enemy_types_.find(kind);
            if (found == enemy_types_.end() && enemy_types_.size() >= max_enemy_types) continue;
            if (found == enemy_types_.end()) ++learning_delta_.new_enemy_types;
            auto& knowledge = enemy_types_[kind];
            knowledge.name = enemy.value("kind",std::string("enemy_")+std::to_string(kind));
            ++knowledge.observations;
            const auto previous = enemies_previous_.find(slot);
            if (previous != enemies_previous_.end() && previous->second.kind == kind) {
                const int elapsed = frame - previous->second.frame;
                const double delta = x - previous->second.x;
                if (elapsed > 0 && elapsed <= 4 && std::abs(delta) <= 16*elapsed) {
                    knowledge.velocity.add(delta/elapsed);
                    ++learning_delta_.enemy_velocity_samples;
                }
            }
            next_enemies[slot] = {kind,frame,x};
        }
        enemies_previous_ = std::move(next_enemies);

        if (previous_valid_ && !dead && !clear) {
            const bool consecutive = frame - frame_ <= 4;
            if (consecutive && previous_grounded_ && !grounded && player.value("movement_state",1) == 1) {
                jump_ = {true,false,frame_,0,previous_x_,previous_feet_,0,0};
            }
            if (jump_.active) {
                if (!consecutive) jump_.active = false; // do not learn from missing trajectory samples
                else {
                    const double dy = feet - previous_feet_;
                    jump_.max_height = std::max(jump_.max_height,jump_.start_feet - feet);
                    if (jump_.previous_dy > 1 && dy < -1) jump_.bounced = true;
                    if (std::abs(dy) > 1) jump_.previous_dy = dy;
                    const auto recent = state.value("recent_control",Json::object());
                    const auto action_field = recent.find("action");
                    const std::string action = action_field != recent.end() && action_field->is_string()
                        ? action_field->get<std::string>() : std::string();
                    if (action.find("jump") != std::string::npos) ++jump_.held_frames;
                    if (grounded) {
                        ++completed_flights_;
                        if (jump_.bounced) { ++bounce_flights_; ++learning_delta_.bounce_flights; }
                        else {
                            ++learning_delta_.jump_samples;
                            jump_duration_.add(frame-jump_.start_frame);
                            jump_range_.add(current_x_-jump_.start_x);
                            jump_height_.add(jump_.max_height);
                            jump_hold_.add(jump_.held_frames);
                        }
                        jump_.active = false;
                    }
                }
            }
        }
        if (outcome_dead || outcome_clear) {
            const bool unseen = session_lifecycle ? !session_event.empty() && session_event != last_session_terminal_event_
                                                  : !terminal_recorded_;
            if (unseen) {
                if (outcome_clear) ++successes_; else ++failures_;
            }
            if (!session_event.empty()) last_session_terminal_event_ = session_event;
            terminal_recorded_ = true;
        }
        if (dead || clear) jump_.active = false;
        previous_x_ = current_x_;
        previous_feet_ = feet;
        previous_grounded_ = grounded;
        previous_valid_ = true;
        frame_ = frame;
    }

    Json context() const {
        int known_ahead = 0, empty_ahead = 0;
        for (const auto& [key,column] : columns_) {
            if (column.world == world_ && column.stage == stage_ && column.x >= current_x_ && column.x <= current_x_+160) {
                ++known_ahead;
                if (column.solid_y.empty()) ++empty_ahead;
            }
        }
        Json speeds = Json::object();
        for (const auto& [id,enemy] : enemy_types_) {
            if (enemy.velocity.count >= 2)
                speeds[std::to_string(id)] = {{"mean",enemy.velocity.mean},{"samples",enemy.velocity.count}};
            if (speeds.size() >= 8) break;
        }
        return {{"source","empirical_telemetry_memory"},{"known_columns_ahead",known_ahead},
                {"remembered_empty_columns_ahead",empty_ahead},{"jump_samples",jump_duration_.count},
                {"jump_duration_mean_frames",jump_duration_.count ? Json(jump_duration_.mean) : Json(nullptr)},
                {"jump_range_mean_pixels",jump_range_.count ? Json(jump_range_.mean) : Json(nullptr)},
                {"enemy_speeds",speeds},{"stage_successes",successes_},{"deaths",failures_},
                {"control_adaptation",false}};
    }

    Json snapshot() const {
        auto result = export_json();
        result["epoch"] = epoch_;
        result["frame"] = frame_;
        result["context"] = context();
        Json nodes = Json::array({
            {{"id","experience"},{"label","Persistent experience"},{"source","telemetry"}},
            {{"id","observed_map"},{"label","Observed columns"},{"count",columns_.size()}},
            {{"id","jump_estimate"},{"label","Completed jump samples"},{"count",jump_duration_.count}},
            {{"id","outcomes"},{"label","Episode outcomes"},{"successes",successes_},{"failures",failures_}},
            {{"id","memory_context"},{"label","Empirical context"},{"control_adaptation",false}}
        });
        Json edges = Json::array();
        for (const auto& id : {"observed_map","jump_estimate","outcomes"}) {
            edges.push_back({{"from","experience"},{"to",id},{"kind","observed"}});
            edges.push_back({{"from",id},{"to","memory_context"},{"kind","summarized"}});
        }
        for (const auto& [kind,enemy] : enemy_types_) {
            const auto id = std::string("enemy_type_")+std::to_string(kind);
            nodes.push_back({{"id",id},{"label",enemy.name},{"samples",enemy.velocity.count}});
            edges.push_back({{"from","experience"},{"to",id},{"kind","observed"}});
            edges.push_back({{"from",id},{"to","memory_context"},{"kind","estimated_velocity"}});
        }
        result["graph"] = {{"nodes",nodes},{"edges",edges}};
        return result;
    }

    Json export_json() const {
        Json columns = Json::array(), enemies = Json::array();
        for (const auto& [key,column] : columns_) columns.push_back(column.json());
        for (const auto& [kind,enemy] : enemy_types_) {
            auto item = enemy.json(); item["kind_id"] = kind; enemies.push_back(std::move(item));
        }
        return {{"schema","jevtpp_mario_experience"},{"version",1},{"environment",environment_},
                {"columns",columns},{"enemy_types",enemies},{"observed_frames",observed_frames_},
                {"jumps",{{"duration_frames",jump_duration_.json()},{"range_pixels",jump_range_.json()},
                          {"height_pixels",jump_height_.json()},{"hold_frames",jump_hold_.json()},
                          {"completed_flights",completed_flights_},{"bounce_flights_excluded",bounce_flights_}}},
                {"outcomes",{{"successes",successes_},{"failures",failures_}}},
                {"scope","Observed RAM only. No unseen topology or automatic policy learning."}};
    }

    bool import_json(const Json& value, std::string* error = nullptr) {
        try {
            if (value.at("schema") != "jevtpp_mario_experience" || value.at("version") != 1 ||
                value.at("environment") != environment_)
                throw std::invalid_argument("experience schema/version/environment mismatch");
            KnowledgeMemory loaded(environment_);
            const auto& columns = value.at("columns");
            const auto& enemies = value.at("enemy_types");
            if (!columns.is_array() || columns.size() > max_columns || !enemies.is_array() || enemies.size() > max_enemy_types)
                throw std::invalid_argument("experience storage bounds exceeded");
            for (const auto& item : columns) {
                Column column;
                column.world = item.at("world").get<int>(); column.stage = item.at("stage").get<int>();
                column.x = item.at("x").get<int>();
                if (column.world < 1 || column.world > 99 || column.stage < 1 || column.stage > 99 ||
                    column.x < 0 || column.x > 1000000 || column.x % 16)
                    throw std::invalid_argument("invalid column coordinate");
                column.solid_y = valid_rows(item.at("solid_y"));
                column.observations = item.at("observations").get<std::uint64_t>();
                column.changes = item.at("changes").get<std::uint64_t>();
                column.seen_order = item.at("seen_order").get<std::uint64_t>();
                column.last_epoch = item.at("last_epoch").get<int>();
                column.last_frame = item.at("last_frame").get<int>();
                // Epochs are local process counters and can repeat after a
                // restart. Keep their original provenance, but never label a
                // loaded observation as current until it is observed again.
                column.imported = true;
                if (!column.observations || column.observations > 1000000000ULL || column.changes > column.observations)
                    throw std::invalid_argument("invalid column observation count");
                loaded.seen_order_ = std::max(loaded.seen_order_,column.seen_order);
                loaded.columns_[column_key(column.world,column.stage,column.x)] = std::move(column);
            }
            for (const auto& item : enemies) {
                const int kind = item.at("kind_id").get<int>();
                if (kind < 0 || kind > 255) throw std::invalid_argument("invalid enemy type");
                EnemyKnowledge enemy;
                enemy.name = item.at("name").get<std::string>();
                if (enemy.name.size() > 128) throw std::invalid_argument("enemy name too long");
                enemy.observations = item.at("observations").get<std::uint64_t>();
                if (enemy.observations > 1000000000ULL) throw std::invalid_argument("enemy observation count exceeds limit");
                enemy.velocity = RunningStats::read(item.at("velocity_x_px_per_frame"));
                loaded.enemy_types_[kind] = std::move(enemy);
            }
            const auto& jumps = value.at("jumps");
            loaded.jump_duration_ = RunningStats::read(jumps.at("duration_frames"));
            loaded.jump_range_ = RunningStats::read(jumps.at("range_pixels"));
            loaded.jump_height_ = RunningStats::read(jumps.at("height_pixels"));
            loaded.jump_hold_ = RunningStats::read(jumps.at("hold_frames"));
            if (loaded.jump_duration_.count != loaded.jump_range_.count || loaded.jump_duration_.count != loaded.jump_height_.count ||
                loaded.jump_duration_.count != loaded.jump_hold_.count)
                throw std::invalid_argument("inconsistent jump statistics");
            loaded.completed_flights_ = jumps.at("completed_flights").get<std::uint64_t>();
            loaded.bounce_flights_ = jumps.at("bounce_flights_excluded").get<std::uint64_t>();
            if (loaded.completed_flights_ > 1000000000ULL || loaded.bounce_flights_ > loaded.completed_flights_ ||
                loaded.jump_duration_.count != loaded.completed_flights_ - loaded.bounce_flights_)
                throw std::invalid_argument("inconsistent completed flight count");
            loaded.successes_ = value.at("outcomes").at("successes").get<std::uint64_t>();
            loaded.failures_ = value.at("outcomes").at("failures").get<std::uint64_t>();
            loaded.observed_frames_ = value.at("observed_frames").get<std::uint64_t>();
            if (loaded.observed_frames_ > 1000000000ULL || loaded.successes_ > loaded.observed_frames_ ||
                loaded.failures_ > loaded.observed_frames_)
                throw std::invalid_argument("invalid outcome/frame counts");
            *this = std::move(loaded); // commit only after every field validates
            if (error) error->clear();
            return true;
        } catch (const std::exception& failure) {
            if (error) *error = failure.what();
            return false;
        }
    }

private:
    static std::string column_key(int world,int stage,int x) {
        return std::to_string(world)+":"+std::to_string(stage)+":"+std::to_string(x);
    }
    static std::vector<int> valid_rows(const Json& rows) {
        if (!rows.is_array() || rows.size() > 13) throw std::invalid_argument("invalid collision rows");
        std::set<int> unique;
        for (const auto& row : rows) {
            const int y = row.get<int>();
            if (y < 32 || y > 224 || (y-32)%16) throw std::invalid_argument("invalid collision row coordinate");
            unique.insert(y);
        }
        return {unique.begin(),unique.end()};
    }
    std::string environment_;
    std::map<std::string,Column> columns_;
    std::map<int,EnemyKnowledge> enemy_types_;
    std::map<int,EnemySample> enemies_previous_;
    RunningStats jump_duration_, jump_range_, jump_height_, jump_hold_;
    Jump jump_;
    LearningDelta learning_delta_;
    std::uint64_t seen_order_ = 0, observed_frames_ = 0, successes_ = 0, failures_ = 0;
    std::uint64_t completed_flights_ = 0, bounce_flights_ = 0;
    int epoch_ = -1, frame_ = -1, world_ = 1, stage_ = 1;
    double current_x_ = 0, previous_x_ = 0, previous_feet_ = 0;
    bool previous_valid_ = false, previous_grounded_ = false, terminal_recorded_ = false;
    std::string last_session_terminal_event_;
    std::string physical_episode_id_;
};

} // namespace mario
