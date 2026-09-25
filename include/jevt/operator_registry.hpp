#pragma once

#include "jevt/graph.hpp"
#include <map>

namespace jevt {

// Parser-independent IR: JSON/YAML/etc. are optional adapters, not core deps.
struct operator_node_spec {
    std::string id;
    std::string operator_id;
    std::vector<std::string> dependencies;
    state_value parameters;
    bool require_successful_dependencies = true;
};
struct graph_spec {
    std::string schema_version = "jevt.graph.v1";
    std::vector<operator_node_spec> nodes;
};

class operator_registry {
public:
    using evaluator = std::function<graph_node_result(const graph_context&)>;
    // Factory validates parameters at graph loading, captures them by value.
    using factory = std::function<evaluator(const state_value&)>;
    void add(std::string id, factory make) {
        if (id.empty() || id.size() > 128 || !make) throw std::invalid_argument("operator requires 1..128 byte ID and factory");
        if (!factories_.emplace(std::move(id), std::move(make)).second)
            throw std::invalid_argument("duplicate registered operator");
    }
    [[nodiscard]] decision_graph compile(const graph_spec& spec) const {
        if (spec.schema_version != "jevt.graph.v1") throw std::invalid_argument("unsupported graph schema_version");
        if (spec.nodes.empty() || spec.nodes.size() > 1024) throw std::invalid_argument("graph requires 1..1024 nodes");
        std::vector<graph_node> nodes;
        nodes.reserve(spec.nodes.size());
        std::size_t parameter_bytes = 0;
        // Validate resource bounds before invoking any application factory.
        for (const auto& item : spec.nodes) {
            if (item.id.empty() || item.id.size() > 128 || item.operator_id.empty() || item.operator_id.size() > 128)
                throw std::invalid_argument("node and operator IDs require 1..128 bytes");
            if (item.dependencies.size() > 1024) throw std::invalid_argument("too many declared dependencies");
            for (const auto& dependency : item.dependencies)
                if (dependency.empty() || dependency.size() > 128) throw std::invalid_argument("dependency IDs require 1..128 bytes");
            if (item.parameters.content.size() > 65536) throw std::invalid_argument("operator parameters exceed 64 KiB");
            parameter_bytes += item.parameters.content.size();
            if (parameter_bytes > 1048576) throw std::invalid_argument("graph parameters exceed 1 MiB");
        }
        for (const auto& item : spec.nodes) {
            const auto it = factories_.find(item.operator_id);
            if (it == factories_.end()) throw std::invalid_argument("unregistered operator: " + item.operator_id);
            nodes.push_back({item.id, item.dependencies, it->second(item.parameters), {}, item.require_successful_dependencies});
        }
        // Existing graph validates duplicate IDs, unknown dependencies and cycles.
        return decision_graph(std::move(nodes));
    }
private:
    std::map<std::string, factory, std::less<>> factories_;
};

} // namespace jevt
