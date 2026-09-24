#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace jevt {

class Diagnostics;

struct HttpServerOptions {
  // Loopback by default so diagnostics are not exposed to the network by
  // accident. Set "0.0.0.0" explicitly to listen on every IPv4 interface.
  std::string bind_address = "127.0.0.1";
  std::uint16_t port = 0;  // 0 asks the operating system for a free port.
  int listen_backlog = 16;
  std::size_t max_request_bytes = 16 * 1024;
};

class HttpServer {
 public:
  explicit HttpServer(Diagnostics& diagnostics);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;
  HttpServer(HttpServer&&) = delete;
  HttpServer& operator=(HttpServer&&) = delete;

  // Starts one lightweight accept thread. Throws std::system_error or
  // std::runtime_error when binding/listening fails.
  void start(HttpServerOptions options = {});
  void stop() noexcept;

  [[nodiscard]] bool running() const noexcept;
  [[nodiscard]] std::uint16_t port() const noexcept;
  [[nodiscard]] std::string url() const;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace jevt
