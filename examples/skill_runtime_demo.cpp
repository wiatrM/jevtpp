#include <jevt/graph_json.hpp>
#include <jevt/skill.hpp>
#include <fstream>
#include <iostream>

// Local, non-actuating demonstration: identical compiled operators support
// warehouse and support-routing graphs loaded from runtime JSON.
namespace {
using json = nlohmann::json;
using namespace jevt::skills;
json read_json(const char* path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error(std::string("cannot open ") + path);
    return json::parse(input);
}
jevt::operator_registry registry() {
    jevt::operator_registry result;
    result.add("observe.v1", [](const jevt::state_value&) {
        return [](const jevt::graph_context& ctx) { return jevt::graph_node_result::success(ctx.snapshot().input); };
    });
    result.add("propose.v1", [](const jevt::state_value& params) {
        const auto config = json::parse(params.content);
        jevt::graph_json_detail::keys(config, {"action", "priority", "utility", "requirements", "emergency"});
        if (!config.at("action").is_string()) throw std::invalid_argument("action must be a string");
        (void)config.value("priority", 0);
        (void)config.value("utility", 0.0);
        if (!config.value("requirements", json::array()).is_array()) throw std::invalid_argument("requirements must be an array");
        for (const auto& requirement : config.value("requirements", json::array())) {
            jevt::graph_json_detail::keys(requirement, {"field", "equals"});
            (void)requirement.at("field").get<std::string>();
            (void)requirement.at("equals");
        }
        return [config](const jevt::graph_context& ctx) {
            if (ctx.dependencies().size() != 1) return jevt::graph_node_result::failure("proposal needs exactly one observation dependency");
            const auto observed = json::parse(ctx.dependencies().front()->output.content);
            json conditions = json::array();
            for (const auto& requirement : config.value("requirements", json::array())) {
                const auto field = requirement.at("field").get<std::string>();
                const auto value = !observed.contains(field) ? "unknown" :
                    (observed.at(field) == requirement.at("equals") ? "true" : "false");
                conditions.push_back({{"id", field}, {"value", value}, {"detail", "observed equality contract"}});
            }
            auto output = config;
            output["conditions"] = std::move(conditions);
            return jevt::graph_node_result::success(jevt::json_state(output.dump()));
        };
    });
    result.add("arbitrate.v1", [](const jevt::state_value&) {
        return [](const jevt::graph_context& ctx) {
            const snapshot stamp{ctx.snapshot().id, 0, 0, 0};
            std::vector<proposal<std::string>> candidates;
            for (const auto* dependency : ctx.dependencies()) {
                const auto value = json::parse(dependency->output.content);
                proposal<std::string> p{dependency->id, value.at("action"), "configured_operator",
                    value.at("action"), stamp, 0, value.value("priority", 0), value.value("utility", 0.0),
                    value.value("emergency", false), {}};
                for (const auto& condition : value.at("conditions")) {
                    const auto raw = condition.at("value").get<std::string>();
                    p.conditions.push_back({condition.at("id"), raw == "true" ? truth::true_value :
                        raw == "false" ? truth::false_value : truth::unknown, condition.at("detail")});
                }
                candidates.push_back(std::move(p));
            }
            const auto chosen = arbitrate<std::string>(stamp, candidates);
            json output{{"action", chosen.action ? json(*chosen.action) : json(nullptr)},
                        {"selected_id", chosen.selected_id}, {"emergency", chosen.emergency},
                        {"actuated", false}, {"trace", json::array()}};
            for (const auto& item : chosen.trace)
                output["trace"].push_back({{"id", item.id}, {"status", status_name(item.status)}, {"reason", item.reason}});
            return jevt::graph_node_result::success(jevt::json_state(output.dump()));
        };
    });
    return result;
}
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: jevt_skill_runtime_demo graph.json observation.json [expected-action]\n";
        return 2;
    }
    try {
        const auto graph = registry().compile(jevt::parse_graph_spec(read_json(argv[1])));
        const auto trace = graph.evaluate({"local-demo-run", jevt::json_state(read_json(argv[2]).dump())});
        if (trace.nodes.empty()) throw std::runtime_error("empty graph result");
        for (const auto& node : trace.nodes)
            if (node.status != jevt::graph_status::succeeded)
                throw std::runtime_error(node.id + ": " + node.message);
        const auto output = json::parse(trace.nodes.back().output.content);
        std::cout << output.dump(2) << '\n';
        if (argc == 4 && output.value("action", "") != argv[3]) return 1;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
