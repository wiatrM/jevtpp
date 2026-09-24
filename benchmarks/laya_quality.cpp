#include <jevt/laya.hpp>
#include <jevt/system_one.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
enum class category { billing, technical, sales, spam };
constexpr auto categories = jevt::schema<category, "quality.categories">(
    jevt::option<category::billing>("Invoices, payments, or refund requests."),
    jevt::option<category::technical>("Software bugs, crashes, or login failures."),
    jevt::option<category::sales>("Upgrades, enterprise pricing, or new purchases."),
    jevt::option<category::spam>("Unsolicited marketing or automated noise."));
constexpr auto model = jevt::decision_model<"quality.support">(
    "Evaluate the incoming customer support ticket.",
    jevt::choice<"category">("Which team should own this request?", categories),
    jevt::noul<"is_urgent">("Does this require attention within one hour?"));

struct example { std::string_view state; category expected_category; bool expected_urgent; };
constexpr std::array cases{
    example{"My card was charged twice. Please refund the duplicate payment.", category::billing, false},
    example{"Production login is down for every employee and there is no workaround.", category::technical, true},
    example{"Please quote the enterprise plan for 500 seats next quarter.", category::sales, false},
    example{"BUY FOLLOWERS NOW!!! Visit our promotion link for guaranteed growth.", category::spam, false},
    example{"The mobile app crashes whenever I upload an invoice.", category::technical, false},
    example{"Our annual invoice has the wrong tax number and is due tomorrow.", category::billing, true},
    example{"We want to upgrade today; procurement needs pricing within an hour.", category::sales, true},
    example{"Automated crypto investment offer. Limited time, click this link.", category::spam, false},
};

double percentile(std::vector<double> values, double quantile) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(std::ceil(quantile * values.size()) - 1.0);
    return values[std::min(index, values.size() - 1)];
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 2) {
        std::cerr << "usage: jevt_laya_quality MODEL_DIRECTORY\n";
        return 2;
    }
    auto backend = jevt::models::laya_multilingual(std::filesystem::path{argv[1]}, 2);
    auto brain = jevt::bind_system_one(model, backend);
    std::vector<double> latencies;
    double category_nll = 0.0, urgent_brier = 0.0;
    std::size_t category_correct = 0, urgent_correct = 0;
    for (const auto& item : cases) {
        const auto started = std::chrono::steady_clock::now();
        auto answer = brain.evaluate(item.state);
        latencies.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
        if (!answer) throw std::runtime_error(answer.error_value().message);
        const auto& choice = answer->get<"category">();
        const auto& urgent = answer->get<"is_urgent">();
        category_correct += choice.value() == item.expected_category;
        urgent_correct += urgent.value() == item.expected_urgent;
        const auto expected = static_cast<std::size_t>(item.expected_category);
        category_nll -= std::log(std::max(1e-12F, choice.probabilities()[expected]));
        const auto difference = urgent.probability_true() - static_cast<float>(item.expected_urgent);
        urgent_brier += difference * difference;
    }
    const auto count = static_cast<double>(cases.size());
    std::cout << std::fixed << std::setprecision(4)
              << "examples=" << cases.size() << '\n'
              << "category_accuracy=" << category_correct / count << '\n'
              << "category_nll=" << category_nll / count << '\n'
              << "urgent_accuracy=" << urgent_correct / count << '\n'
              << "urgent_brier=" << urgent_brier / count << '\n'
              << "latency_p50_ms=" << percentile(latencies, 0.50) << '\n'
              << "latency_p95_ms=" << percentile(latencies, 0.95) << '\n'
              << "latency_p99_ms=" << percentile(latencies, 0.99) << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "Laya quality benchmark failed: " << exception.what() << '\n';
    return 1;
}
