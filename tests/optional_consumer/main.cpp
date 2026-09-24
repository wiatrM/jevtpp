#include <jevt/asio.hpp>
#include <jevt/remote.hpp>
#include <jevt/testing.hpp>

class offline_transport final : public jevt::http_transport {
public:
    jevt::result<jevt::http_response> perform(const jevt::http_request&) override {
        return jevt::error{jevt::error_code::backend_failure, "offline consumer smoke test"};
    }
};

int main() {
    // Exercise installed public symbols without network access or credentials.
    jevt::remote_options options;
    options.api_key = "not-a-real-key";
    options.max_retries = 0;
    auto remote = std::make_shared<jevt::typesafe_backend>(options, std::make_shared<offline_transport>());
    if (remote->name().empty() || remote->list_models()) return 1;
    auto scripted = std::make_shared<jevt::testing::scripted_backend>();
    boost::asio::io_context executor;
    jevt::asio_backend async(executor.get_executor(), scripted);
    async.shutdown();
    return scripted->snapshot().consumed_steps == 0 ? 0 : 2;
}
