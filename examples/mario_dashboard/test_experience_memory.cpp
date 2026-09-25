#include "experience_memory.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using Json = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Json state(int x = 160, int feet = 208, bool grounded = true, int enemy_x = 220) {
    return {{"level",{{"world",1},{"stage",1}}},
            {"player",{{"x",x},{"feet_y",feet},{"grounded",grounded},{"movement_state",grounded ? 0 : 1}}},
            {"episode",{{"dead",false},{"stage_clear",false}}},
            {"recent_control",{{"action",grounded ? "right_run" : "right_run_jump"}}},
            {"collision",{{"available",true},{"columns",Json::array({
                {{"x",160},{"solid_y",Json::array({208,224})}},
                {{"x",176},{"solid_y",Json::array({208,224})}},
                {{"x",192},{"solid_y",Json::array()}},
                {{"x",208},{"solid_y",Json::array({208,224})}},
                {{"x",224},{"solid_y",Json::array({208,224})}},
                {{"x",448},{"solid_y",Json::array({208,224})}}})}}},
            {"hazard",{{"upcoming_enemies",Json::array({
                {{"kind_id",0},{"kind","green_koopa"},{"slot",0},{"category","hostile"},
                 {"x",enemy_x},{"relative_x_pixels",enemy_x-x}}})}}}};
}

int main() {
    try {
        mario::KnowledgeMemory memory;
        memory.observe(state(),1,0);
        auto snapshot = memory.snapshot();
        require(snapshot.at("columns").size() == 5,"must not remember non-visible buffered tiles");
        require(snapshot.at("context").at("remembered_empty_columns_ahead") == 1,"observed empty tile is remembered");
        memory.observe(state(),1,0);
        require(memory.export_json().at("observed_frames") == 1,"duplicate snapshot must not inflate evidence");
        memory.observe(state(162,202,false,218),1,1);
        memory.observe(state(166,198,false,216),1,2);
        memory.observe(state(169,200,false,214),1,3);
        memory.observe(state(173,208,true,212),1,4);
        snapshot = memory.export_json();
        require(snapshot.at("jumps").at("duration_frames").at("count") == 1,"one completed jump");
        require(snapshot.at("jumps").at("duration_frames").at("mean") == 4,"complete flight duration");
        require(snapshot.at("jumps").at("range_pixels").at("mean") == 13,"observed horizontal range");
        require(snapshot.at("jumps").at("height_pixels").at("mean") == 10,"observed peak height");
        require(snapshot.at("enemy_types").at(0).at("velocity_x_px_per_frame").at("mean") == -2,
                "enemy estimate must exclude Mario movement and keep type zero");
        memory.reset_episode(2);
        require(memory.context().at("jump_samples") == 1,"episode reset retains aggregate knowledge");
        auto terminal = state(); terminal["episode"]["dead"] = true;
        memory.observe(terminal,2,0); memory.observe(terminal,2,1);
        require(memory.context().at("deaths") == 1,"terminal death is counted once per episode");
        terminal["episode"]["dead"] = false; terminal["episode"]["stage_clear"] = true;
        memory.observe(terminal,3,0);
        require(memory.context().at("stage_successes") == 1,"terminal flag is remembered");

        mario::KnowledgeMemory restored;
        std::string error;
        require(restored.import_json(memory.export_json(),&error),"valid memory roundtrip");
        auto expected_import = memory.export_json();
        for (auto& column : expected_import["columns"]) column["imported"] = true;
        require(restored.export_json() == expected_import,"JSON persistence preserves evidence and marks prior observations");
        restored.observe(state(200),1,0);
        bool saw_refreshed = false, saw_prior = false;
        const auto refreshed = restored.export_json();
        for (const auto& column : refreshed.at("columns")) {
            if (column.at("x") == 208) {
                require(column.at("imported") == false,"fresh RAM refresh removes imported marker even if epoch repeats");
                require(column.at("last_epoch") == 1 && column.at("last_frame") == 0,"refresh records current provenance");
                saw_refreshed = true;
            }
            if (column.at("x") == 176) {
                require(column.at("imported") == true,"unrefreshed map remains historical when an epoch number repeats");
                saw_prior = true;
            }
        }
        require(saw_refreshed && saw_prior,"fixture includes both refreshed and historical observations");
        const auto before = restored.export_json();
        auto invalid = before; invalid["environment"] = "different-level-contract";
        require(!restored.import_json(invalid,&error),"cross-environment memory rejected");
        require(restored.export_json() == before,"failed load must not partially overwrite memory");
        invalid = before; invalid["jumps"]["duration_frames"]["m2"] = -10;
        require(!restored.import_json(invalid,&error),"invalid statistics rejected");

        mario::KnowledgeMemory bounded;
        for (int i = 0; i < 530; ++i) {
            auto observation = state(i*16);
            observation["collision"]["columns"] = Json::array({{{"x",i*16},{"solid_y",Json::array({208,224})}}});
            bounded.observe(observation,1,i);
        }
        require(bounded.snapshot().at("columns").size() == 512,"observed topology is bounded");
        mario::KnowledgeMemory discontinuous;
        discontinuous.observe(state(),1,0);
        discontinuous.observe(state(163,202,false),1,1);
        discontinuous.observe(state(180,208,true),1,20);
        require(discontinuous.context().at("jump_samples") == 0,"missing trajectory samples must not train jump estimate");
        mario::KnowledgeMemory bounced;
        bounced.observe(state(),1,0);
        bounced.observe(state(162,198,false),1,1);
        bounced.observe(state(164,200,false),1,2);
        bounced.observe(state(166,190,false),1,3);
        bounced.observe(state(170,208,true),1,4);
        require(bounced.export_json().at("jumps").at("bounce_flights_excluded") == 1,"bounce flight is labeled separately");
        require(bounced.context().at("jump_samples") == 0,"bounce must not pollute normal jump estimates");
        mario::KnowledgeMemory respawn;
        respawn.observe(state(),1,0);
        respawn.observe(state(163,208,true,500),1,1);
        require(respawn.export_json().at("enemy_types").at(0).at("velocity_x_px_per_frame").at("count") == 0,
                "teleported/reused enemy slots must not create impossible movement estimates");
        std::cout << "experience memory regressions passed\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
