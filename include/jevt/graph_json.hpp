#pragma once

// Opt-in adapter. Applications including this header supply nlohmann_json;
// neither graph.hpp nor operator_registry.hpp depends on a JSON implementation.
#include "jevt/operator_registry.hpp"
#include <nlohmann/json.hpp>
#include <initializer_list>

namespace jevt {
namespace graph_json_detail {
inline void keys(const nlohmann::json& value, std::initializer_list<std::string_view> allowed) {
    if (!value.is_object()) throw std::invalid_argument("graph declaration must be an object");
    for (auto it = value.begin(); it != value.end(); ++it) {
        bool found = false;
        for (const auto key : allowed) if (it.key() == key) { found = true; break; }
        if (!found) throw std::invalid_argument("unknown graph declaration field: " + it.key());
    }
}
}

[[nodiscard]] inline graph_spec parse_graph_spec(const nlohmann::json& value) {
    using graph_json_detail::keys;
    keys(value, {"schema_version", "nodes"});
    graph_spec result;
    result.schema_version = value.at("schema_version").get<std::string>();
    if (result.schema_version != "jevt.graph.v1") throw std::invalid_argument("unsupported graph schema_version");
    const auto& nodes = value.at("nodes");
    if (!nodes.is_array() || nodes.empty() || nodes.size() > 1024)
        throw std::invalid_argument("graph requires an array of 1..1024 nodes");
    for (const auto& node : nodes) {
        keys(node, {"id", "operator", "depends_on", "params", "require_successful_dependencies"});
        operator_node_spec spec;
        spec.id = node.at("id").get<std::string>();
        spec.operator_id = node.at("operator").get<std::string>();
        spec.dependencies = node.value("depends_on", std::vector<std::string>{});
        spec.parameters = json_state(node.value("params", nlohmann::json::object()).dump());
        spec.require_successful_dependencies = node.value("require_successful_dependencies", true);
        result.nodes.push_back(std::move(spec));
    }
    return result;
}
} // namespace jevt
