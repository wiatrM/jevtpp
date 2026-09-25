#pragma once

#include "jevt/system_one.hpp"
#include <functional>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>

namespace jevt {

// DAG nodes compose evaluations; fields inside System One remain independent.
enum class graph_status { succeeded, skipped, error, stale };
[[nodiscard]] constexpr std::string_view graph_status_name(graph_status value) noexcept {
    switch (value) {
    case graph_status::succeeded: return "succeeded";
    case graph_status::skipped: return "skipped";
    case graph_status::error: return "error";
    case graph_status::stale: return "stale";
    }
    return "error";
}

struct graph_snapshot { std::string id; state_value input; };

struct graph_node_result {
    graph_status status = graph_status::succeeded;
    state_value output;
    std::string message;
    // Empty means computed from this evaluation. Cached/external producers must
    // supply the snapshot ID from which their result was actually computed.
    std::string snapshot_id;
    [[nodiscard]] static graph_node_result success(state_value output, std::string snapshot_id = {}) {
        return {graph_status::succeeded, std::move(output), {}, std::move(snapshot_id)};
    }
    [[nodiscard]] static graph_node_result skip(std::string reason) {
        return {graph_status::skipped, {}, std::move(reason), {}};
    }
    [[nodiscard]] static graph_node_result failure(std::string reason) {
        return {graph_status::error, {}, std::move(reason), {}};
    }
    [[nodiscard]] static graph_node_result stale(std::string reason, std::string snapshot_id = {}) {
        return {graph_status::stale, {}, std::move(reason), std::move(snapshot_id)};
    }
};

struct graph_node_trace : graph_node_result {
    std::string id;
    std::vector<std::string> dependencies;
    std::chrono::nanoseconds duration{};
};

// References last only for the callback. at() rejects undeclared predecessors.
class graph_context {
public:
    [[nodiscard]] const graph_snapshot& snapshot() const noexcept { return snapshot_; }
    [[nodiscard]] std::span<const graph_node_trace* const> dependencies() const noexcept { return dependencies_; }
    [[nodiscard]] const graph_node_trace* find(std::string_view id) const noexcept {
        for (const auto* dependency : dependencies_) if (dependency->id == id) return dependency;
        return nullptr;
    }
    [[nodiscard]] const graph_node_trace& at(std::string_view id) const {
        if (const auto* dependency = find(id)) return *dependency;
        throw std::out_of_range("undeclared graph dependency: " + std::string(id));
    }
private:
    friend class decision_graph;
    graph_context(const graph_snapshot& snapshot, std::span<const graph_node_trace* const> dependencies)
        : snapshot_(snapshot), dependencies_(dependencies) {}
    const graph_snapshot& snapshot_;
    std::span<const graph_node_trace* const> dependencies_;
};

struct graph_node {
    std::string id;
    std::vector<std::string> dependencies;
    std::function<graph_node_result(const graph_context&)> evaluate;
    std::function<bool(const graph_context&)> condition;
    // Explicit fallback nodes may inspect unavailable inputs. They must check
    // dependency.status before using its output.
    bool require_successful_dependencies = true;
};

struct graph_edge { std::string from; std::string to; };
struct graph_trace {
    std::string snapshot_id;
    std::vector<graph_node_trace> nodes;
    std::vector<graph_edge> edges;
    std::chrono::nanoseconds duration{};
    [[nodiscard]] const graph_node_trace& at(std::string_view id) const {
        for (const auto& node : nodes) if (node.id == id) return node;
        throw std::out_of_range("unknown graph node: " + std::string(id));
    }
};

struct graph_run_options {
    // Cooperative: callbacks are not interrupted. Late outputs are stale.
    std::optional<std::chrono::steady_clock::time_point> deadline;
};

class decision_graph {
public:
    explicit decision_graph(std::vector<graph_node> nodes) : nodes_(std::move(nodes)) {
        std::unordered_map<std::string, std::size_t> indices;
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            const auto& node = nodes_[i];
            if (node.id.empty()) throw std::invalid_argument("graph node ID must not be empty");
            if (!node.evaluate) throw std::invalid_argument("graph node has no evaluator: " + node.id);
            if (!indices.emplace(node.id, i).second)
                throw std::invalid_argument("duplicate graph node ID: " + node.id);
        }
        dependency_indices_.resize(nodes_.size());
        std::vector<std::vector<std::size_t>> consumers(nodes_.size());
        std::vector<std::size_t> remaining(nodes_.size());
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            std::unordered_set<std::string> seen;
            for (const auto& dependency : nodes_[i].dependencies) {
                const auto producer = indices.find(dependency);
                if (producer == indices.end())
                    throw std::invalid_argument("missing graph dependency: " + nodes_[i].id + " -> " + dependency);
                if (!seen.insert(dependency).second)
                    throw std::invalid_argument("duplicate graph dependency: " + nodes_[i].id + " -> " + dependency);
                dependency_indices_[i].push_back(producer->second);
                consumers[producer->second].push_back(i);
                ++remaining[i];
            }
        }
        for (std::size_t i = 0; i < nodes_.size(); ++i) if (remaining[i] == 0) order_.push_back(i);
        for (std::size_t cursor = 0; cursor < order_.size(); ++cursor)
            for (const auto consumer : consumers[order_[cursor]])
                if (--remaining[consumer] == 0) order_.push_back(consumer);
        if (order_.size() != nodes_.size()) throw std::invalid_argument("decision graph contains a dependency cycle");
    }

    [[nodiscard]] graph_trace evaluate(const graph_snapshot& snapshot, graph_run_options options = {}) const {
        if (snapshot.id.empty()) throw std::invalid_argument("graph snapshot ID must not be empty");
        const auto started = std::chrono::steady_clock::now();
        graph_trace trace;
        trace.snapshot_id = snapshot.id;
        trace.nodes.reserve(nodes_.size()); // dependency pointers remain stable
        std::vector<std::size_t> trace_indices(nodes_.size());
        for (const auto index : order_) {
            const auto& node = nodes_[index];
            std::vector<const graph_node_trace*> dependencies;
            dependencies.reserve(dependency_indices_[index].size());
            for (const auto predecessor : dependency_indices_[index]) {
                dependencies.push_back(&trace.nodes[trace_indices[predecessor]]);
                trace.edges.push_back({nodes_[predecessor].id, node.id});
            }
            const graph_context context{snapshot, dependencies};
            const auto node_started = std::chrono::steady_clock::now();
            graph_node_result result;
            const auto expired = [&] { return options.deadline && std::chrono::steady_clock::now() >= *options.deadline; };
            bool blocked = false;
            if (expired()) {
                result = graph_node_result::stale("evaluation deadline expired");
                blocked = true;
            } else if (node.require_successful_dependencies) {
                for (const auto* dependency : dependencies) {
                    if (dependency->status == graph_status::succeeded) continue;
                    result = dependency->status == graph_status::stale
                        ? graph_node_result::stale("stale dependency: " + dependency->id)
                        : graph_node_result::skip("unavailable dependency: " + dependency->id);
                    blocked = true;
                    break;
                }
            }
            if (!blocked) {
                try {
                    result = node.condition && !node.condition(context)
                        ? graph_node_result::skip("condition is false") : node.evaluate(context);
                } catch (const std::exception& error) {
                    result = graph_node_result::failure(error.what());
                } catch (...) {
                    result = graph_node_result::failure("non-standard exception in graph callback");
                }
                if (result.status == graph_status::succeeded && expired()) {
                    result.status = graph_status::stale;
                    result.message = "evaluation deadline expired during callback";
                }
                if (result.status == graph_status::succeeded && !result.snapshot_id.empty() && result.snapshot_id != snapshot.id) {
                    result.status = graph_status::stale;
                    result.message = "output snapshot does not match evaluation snapshot";
                }
            }
            if (result.snapshot_id.empty()) result.snapshot_id = snapshot.id;
            graph_node_trace item;
            static_cast<graph_node_result&>(item) = std::move(result);
            item.id = node.id;
            item.dependencies = node.dependencies;
            item.duration = std::chrono::steady_clock::now() - node_started;
            trace_indices[index] = trace.nodes.size();
            trace.nodes.push_back(std::move(item));
        }
        trace.duration = std::chrono::steady_clock::now() - started;
        return trace;
    }
private:
    std::vector<graph_node> nodes_;
    std::vector<std::vector<std::size_t>> dependency_indices_;
    std::vector<std::size_t> order_;
};

} // namespace jevt
