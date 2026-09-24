#include "jevt/http_server.hpp"

#include "jevt/diagnostics.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")
#endif
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace jevt {
namespace {

#ifdef _WIN32
using Socket = SOCKET;
constexpr Socket kInvalidSocket = INVALID_SOCKET;
int socket_error() { return WSAGetLastError(); }
void close_socket(Socket socket) { closesocket(socket); }
class SocketRuntime {
 public:
  SocketRuntime() {
    WSADATA data{};
    const int status = WSAStartup(MAKEWORD(2, 2), &data);
    if (status != 0) throw std::system_error(status, std::system_category(), "WSAStartup");
  }
  ~SocketRuntime() { WSACleanup(); }
};
#else
using Socket = int;
constexpr Socket kInvalidSocket = -1;
int socket_error() { return errno; }
void close_socket(Socket socket) { ::close(socket); }
class SocketRuntime {};
#endif

constexpr std::string_view kDashboard = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>jevt diagnostics</title><style>
:root{color-scheme:dark;--bg:#0b1020;--card:#151c31;--muted:#96a0ba;--good:#50e3a4;--bad:#ff6685;--accent:#6ea8ff}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#eef2ff;font:14px system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:28px}
h1{font-size:24px;margin:0 0 4px}.sub{color:var(--muted);margin-bottom:24px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:var(--card);border:1px solid #26304e;border-radius:12px;padding:16px}.label{color:var(--muted);font-size:12px}.value{font-size:28px;font-weight:700;margin-top:5px}.wide{grid-column:span 2}canvas{width:100%;height:230px}table{width:100%;border-collapse:collapse;margin-top:8px}th,td{text-align:left;padding:8px;border-bottom:1px solid #26304e}th{color:var(--muted)}.ok{color:var(--good)}.error{color:var(--bad)}
@media(max-width:800px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:480px){.grid{display:block}.card{margin-bottom:12px}}
</style></head><body><main><h1>jevt diagnostics</h1><div class="sub" id="updated">Waiting for telemetry…</div>
<section class="grid"><div class="card"><div class="label">Calls</div><div class="value" id="calls">—</div></div>
<div class="card"><div class="label">Success rate</div><div class="value" id="rate">—</div></div>
<div class="card"><div class="label">p95 latency</div><div class="value" id="p95">—</div></div>
<div class="card"><div class="label">Errors / abstains</div><div class="value" id="errors">—</div></div>
<div class="card wide"><div class="label">Calls per decision</div><canvas id="chart"></canvas></div>
<div class="card wide"><div class="label">Recent calls (opaque tags only)</div><table><thead><tr><th>Decision</th><th>Outcome</th><th>Latency</th><th>Tag</th></tr></thead><tbody id="recent"></tbody></table></div></section>
</main><script>
const $=id=>document.getElementById(id); function esc(v){const e=document.createElement('span');e.textContent=v??'';return e.innerHTML}
function chart(items){const c=$('chart'),d=devicePixelRatio||1,w=c.clientWidth,h=230;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.scale(d,d);x.clearRect(0,0,w,h);const top=items.slice().sort((a,b)=>b.stats.calls-a.stats.calls).slice(0,10),max=Math.max(1,...top.map(v=>v.stats.calls));top.forEach((v,i)=>{const y=12+i*21,bw=(w-150)*v.stats.calls/max;x.fillStyle='#6ea8ff';x.fillRect(140,y,bw,13);x.fillStyle='#dce5ff';x.font='12px system-ui';x.textAlign='right';x.fillText(v.decision.slice(0,20),134,y+11);x.textAlign='left';x.fillText(v.stats.calls,145+bw,y+11)})}
async function update(){try{const r=await fetch('/api/stats',{cache:'no-store'}),s=await r.json(),t=s.total;$('calls').textContent=t.calls.toLocaleString();$('rate').textContent=(t.calls?100*t.successes/t.calls:0).toFixed(1)+'%';$('p95').textContent=t.latency_ms.p95.toFixed(2)+' ms';$('errors').textContent=t.errors+' / '+t.abstains;$('updated').textContent='Updated '+new Date(s.generated_at_unix_ms).toLocaleTimeString();chart(s.decisions);$('recent').innerHTML=s.recent_calls.slice(0,12).map(v=>`<tr><td>${esc(v.decision)}</td><td class="${v.outcome}">${esc(v.outcome)}</td><td>${v.latency_ms.toFixed(3)} ms</td><td>${esc(v.tag||'')}</td></tr>`).join('')}catch(e){$('updated').textContent='Telemetry unavailable: '+e.message}}
update();setInterval(update,2000);addEventListener('resize',update);
</script></body></html>)HTML";

using Clock = std::chrono::steady_clock;
constexpr auto kIdleTimeout = std::chrono::seconds{2};

bool nonblocking(Socket socket) {
#ifdef _WIN32
  u_long enabled = 1;
  return ::ioctlsocket(socket, FIONBIO, &enabled) == 0;
#else
  const int flags = ::fcntl(socket, F_GETFL, 0);
  return flags != -1 && ::fcntl(socket, F_SETFL, flags | O_NONBLOCK) != -1;
#endif
}

bool retry_socket_error() {
  const auto code = socket_error();
#ifdef _WIN32
  return code == WSAEWOULDBLOCK || code == WSAEINTR;
#else
  return code == EWOULDBLOCK || code == EAGAIN || code == EINTR;
#endif
}

// Poll in short slices: shutdown() need not interrupt a blocking Winsock recv.
// Nonblocking I/O also handles readiness becoming stale without losing the
// cancellation bound. poll avoids select's descriptor-number/FD_SETSIZE limit.
bool wait_ready(Socket socket, bool writing, Clock::time_point deadline,
                const std::atomic<bool>& running) {
  while (running.load()) {
    const auto remaining = deadline - Clock::now();
    if (remaining <= Clock::duration::zero()) return false;
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count();
    const int timeout = static_cast<int>(std::clamp<decltype(millis)>(millis, 1, 50));
#ifdef _WIN32
    WSAPOLLFD descriptor{socket, static_cast<SHORT>(writing ? POLLWRNORM : POLLRDNORM), 0};
    const int ready = ::WSAPoll(&descriptor, 1, timeout);
#else
    pollfd descriptor{socket, static_cast<short>(writing ? POLLOUT : POLLIN), 0};
    const int ready = ::poll(&descriptor, 1, timeout);
#endif
    if (!running.load()) return false;
    if (ready > 0) return true; // recv/send resolves EOF and socket errors.
    if (ready < 0 && !retry_socket_error()) return false;
  }
  return false;
}

bool send_all(Socket socket, std::string_view bytes, const std::atomic<bool>& running) {
  auto deadline = Clock::now() + kIdleTimeout;
  while (!bytes.empty()) {
    if (!wait_ready(socket, true, deadline, running)) return false;
#ifdef _WIN32
    const int sent = ::send(socket, bytes.data(), static_cast<int>(bytes.size()), 0);
#else
    const auto sent = ::send(socket, bytes.data(), bytes.size(), MSG_NOSIGNAL);
#endif
    if (sent < 0 && retry_socket_error()) continue;
    if (sent <= 0) return false;
    bytes.remove_prefix(static_cast<std::size_t>(sent));
    deadline = Clock::now() + kIdleTimeout;
  }
  return true;
}

void write_response(Socket socket, int status, std::string_view status_text,
                    std::string_view content_type, std::string_view body,
                    const std::atomic<bool>& running) {
  std::ostringstream head;
  head << "HTTP/1.1 " << status << ' ' << status_text << "\r\n"
       << "Content-Type: " << content_type << "\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Cache-Control: no-store\r\n"
       << "X-Content-Type-Options: nosniff\r\n"
       << "Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'; "
          "script-src 'unsafe-inline'; connect-src 'self'\r\n"
       << "Referrer-Policy: no-referrer\r\n"
       << "Connection: close\r\n";
  if (status == 405) head << "Allow: GET\r\n";
  head << "\r\n";
  const auto header = head.str();
  if (send_all(socket, header, running)) send_all(socket, body, running);
}

}  // namespace

struct HttpServer::Impl {
  explicit Impl(Diagnostics& value) : diagnostics(value) {}
  ~Impl() { stop(); }

  void start(HttpServerOptions requested) {
    std::lock_guard lock(lifecycle_mutex);
    if (is_running.load()) throw std::logic_error("HTTP diagnostics server is already running");
    if (requested.max_request_bytes < 64) {
      throw std::invalid_argument("max_request_bytes must be at least 64");
    }
    Socket created = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (created == kInvalidSocket) {
      throw std::system_error(socket_error(), std::system_category(), "socket");
    }
    int yes = 1;
#ifdef _WIN32
    ::setsockopt(created, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&yes), sizeof(yes));
#else
    ::setsockopt(created, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(requested.port);
    if (::inet_pton(AF_INET, requested.bind_address.c_str(), &address.sin_addr) != 1) {
      close_socket(created);
      throw std::invalid_argument("bind_address must be an IPv4 address");
    }
    if (::bind(created, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(created, requested.listen_backlog) != 0) {
      const int error = socket_error();
      close_socket(created);
      throw std::system_error(error, std::system_category(), "bind/listen");
    }
    sockaddr_in actual{};
#ifdef _WIN32
    int actual_length = sizeof(actual);
#else
    socklen_t actual_length = sizeof(actual);
#endif
    if (::getsockname(created, reinterpret_cast<sockaddr*>(&actual), &actual_length) != 0) {
      const int error = socket_error();
      close_socket(created);
      throw std::system_error(error, std::system_category(), "getsockname");
    }
    options = std::move(requested);
    bound_port.store(ntohs(actual.sin_port));
    listen_socket.store(created);
    is_running.store(true);
    worker = std::thread([this] { accept_loop(); });
  }

  void stop() noexcept {
    std::lock_guard lock(lifecycle_mutex);
    if (!is_running.exchange(false)) return;
    const Socket socket = listen_socket.exchange(kInvalidSocket);
    if (socket != kInvalidSocket) {
#ifdef _WIN32
      ::shutdown(socket, SD_BOTH);
#else
      ::shutdown(socket, SHUT_RDWR);
#endif
      close_socket(socket);
    }
    {
      std::lock_guard client_lock(client_mutex);
      if (active_client != kInvalidSocket) {
#ifdef _WIN32
        ::shutdown(active_client, SD_BOTH);
#else
        ::shutdown(active_client, SHUT_RDWR);
#endif
      }
    }
    if (worker.joinable()) worker.join();
    bound_port.store(0);
  }

  void accept_loop() noexcept {
    while (is_running.load()) {
      const Socket listening = listen_socket.load();
      if (listening == kInvalidSocket) break;
      const Socket client = ::accept(listening, nullptr, nullptr);
      if (client == kInvalidSocket) {
        if (!is_running.load()) break;
        continue;
      }
      {
        std::lock_guard client_lock(client_mutex);
        if (!is_running.load()) {
          close_socket(client);
          break;
        }
        active_client = client;
      }
      if (nonblocking(client)) handle(client);
      {
        std::lock_guard client_lock(client_mutex);
        active_client = kInvalidSocket;
        close_socket(client);
      }
    }
  }

  void send_response(Socket client, int status, std::string_view status_text,
                     std::string_view content_type, std::string_view body) {
    write_response(client, status, status_text, content_type, body, is_running);
  }

  void handle(Socket client) noexcept {
    try {
      std::string request;
      request.reserve(1024);
      char buffer[1024];
      auto deadline = Clock::now() + kIdleTimeout;
      while (request.find("\r\n\r\n") == std::string::npos &&
             request.size() < options.max_request_bytes) {
        if (!wait_ready(client, false, deadline, is_running)) break;
        const auto remaining = std::min(sizeof(buffer), options.max_request_bytes - request.size());
#ifdef _WIN32
        const int read = ::recv(client, buffer, static_cast<int>(remaining), 0);
#else
        const auto read = ::recv(client, buffer, remaining, 0);
#endif
        if (read < 0 && retry_socket_error()) continue;
        if (read <= 0) break;
        request.append(buffer, static_cast<std::size_t>(read));
        deadline = Clock::now() + kIdleTimeout;
      }
      if (!is_running.load()) return;
      if (request.find("\r\n\r\n") == std::string::npos) {
        if (request.size() == options.max_request_bytes)
          send_response(client, 431, "Request Header Fields Too Large", "text/plain; charset=utf-8", "headers too large\n");
        else
          send_response(client, 400, "Bad Request", "text/plain; charset=utf-8", "incomplete headers\n");
        return;
      }
      const auto line_end = request.find("\r\n");
      const std::string_view line(request.data(),
                                  line_end == std::string::npos ? request.size() : line_end);
      const auto first_space = line.find(' ');
      const auto second_space = first_space == std::string_view::npos
                                    ? std::string_view::npos
                                    : line.find(' ', first_space + 1);
      if (first_space == std::string_view::npos || first_space == 0 ||
          second_space == std::string_view::npos || second_space == first_space + 1 ||
          (line.substr(second_space + 1) != "HTTP/1.1" && line.substr(second_space + 1) != "HTTP/1.0")) {
        send_response(client, 400, "Bad Request", "text/plain; charset=utf-8", "bad request\n");
        return;
      }
      if (line.substr(0, first_space) != "GET") {
        send_response(client, 405, "Method Not Allowed", "text/plain; charset=utf-8", "method not allowed\n");
        return;
      }
      auto path = line.substr(first_space + 1, second_space - first_space - 1);
      if (const auto query = path.find('?'); query != std::string_view::npos) path = path.substr(0, query);
      if (path == "/healthz") {
        send_response(client, 200, "OK", "text/plain; charset=utf-8", "ok\n");
      } else if (path == "/api/stats") {
        const auto body = diagnostics.to_json();
        send_response(client, 200, "OK", "application/json; charset=utf-8", body);
      } else if (path == "/metrics") {
        const auto body = diagnostics.to_prometheus();
        send_response(client, 200, "OK", "text/plain; version=0.0.4; charset=utf-8", body);
      } else if (path == "/") {
        send_response(client, 200, "OK", "text/html; charset=utf-8", kDashboard);
      } else {
        send_response(client, 404, "Not Found", "text/plain; charset=utf-8", "not found\n");
      }
    } catch (...) {
      // Diagnostics must never terminate the hosting process.
    }
  }

  SocketRuntime runtime;
  Diagnostics& diagnostics;
  HttpServerOptions options;
  std::atomic<Socket> listen_socket{kInvalidSocket};
  std::atomic<bool> is_running{false};
  std::atomic<std::uint16_t> bound_port{0};
  std::thread worker;
  std::mutex lifecycle_mutex;
  std::mutex client_mutex;
  Socket active_client = kInvalidSocket;
};

HttpServer::HttpServer(Diagnostics& diagnostics) : impl_(new Impl(diagnostics)) {}
HttpServer::~HttpServer() { delete impl_; }
void HttpServer::start(HttpServerOptions options) { impl_->start(std::move(options)); }
void HttpServer::stop() noexcept { impl_->stop(); }
bool HttpServer::running() const noexcept { return impl_->is_running.load(); }
std::uint16_t HttpServer::port() const noexcept { return impl_->bound_port.load(); }
std::string HttpServer::url() const {
  const auto host = impl_->options.bind_address == "0.0.0.0"
                        ? std::string("127.0.0.1")
                        : impl_->options.bind_address;
  return "http://" + host + ':' + std::to_string(port()) + '/';
}

}  // namespace jevt
