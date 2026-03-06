
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "http_server_monitor.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace vsag::eval {

HttpServerMonitor::HttpServerMonitor(int port) : port_(port) {
    server_ = std::make_unique<httplib::Server>();
    start_time_ = std::chrono::steady_clock::now();
    setup_routes();
}

HttpServerMonitor::~HttpServerMonitor() {
    Stop();
}

void
HttpServerMonitor::Start() {
    if (running_) {
        return;
    }
    running_ = true;
    server_thread_ = std::thread(&HttpServerMonitor::run_server, this);
}

void
HttpServerMonitor::Stop() {
    if (!running_) {
        return;
    }
    running_ = false;
    server_->stop();
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void
HttpServerMonitor::run_server() {
    std::cout << "[HTTP Monitor] Starting server on port " << port_ << std::endl;
    if (!server_->listen("0.0.0.0", port_)) {
        std::cerr << "[HTTP Monitor] Failed to start server on port " << port_ << std::endl;
    }
}

void
HttpServerMonitor::setup_routes() {
    // API endpoints
    server_->Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(get_status_json().dump(), "application/json");
    });

    server_->Get("/api/metrics", [this](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        res.set_content(current_metrics_.dump(), "application/json");
    });

    server_->Get("/api/history", [this](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        JsonType history_json = JsonType::array();
        for (const auto& h : case_history_) {
            history_json.push_back(h);
        }
        res.set_content(history_json.dump(), "application/json");
    });

    // Main page
    server_->Get("/", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(get_index_html(), "text/html");
    });

    // Static files (if web_root is set)
    if (!web_root_.empty()) {
        server_->set_base_dir(web_root_);
    }

    // CORS support
    server_->set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
    });
}

JsonType
HttpServerMonitor::get_status_json() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    JsonType status;
    status["total_cases"] = total_cases_;
    status["completed_cases"] = completed_cases_;
    status["current_case"] = current_case_;
    status["case_status"] = case_status_;
    status["progress"] = current_progress_;
    status["running"] = running_.load();

    auto now = std::chrono::steady_clock::now();
    auto total_elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
    status["total_elapsed_seconds"] = total_elapsed;

    if (case_status_ == "running") {
        auto case_elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(now - case_start_time_).count();
        status["case_elapsed_seconds"] = case_elapsed;
    }

    return status;
}

void
HttpServerMonitor::SetTotalCases(int total) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    total_cases_ = total;
}

void
HttpServerMonitor::SetCurrentCase(const std::string& name) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    // Save current case to history if completed
    if (!current_case_.empty() && case_status_ == "completed") {
        JsonType history_entry;
        history_entry["name"] = current_case_;
        history_entry["metrics"] = current_metrics_;
        case_history_.push_back(history_entry);
    }
    current_case_ = name;
    case_status_ = "running";
    current_progress_ = 0.0F;
    case_start_time_ = std::chrono::steady_clock::now();
}

void
HttpServerMonitor::SetCaseStatus(const std::string& status) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    case_status_ = status;
}

void
HttpServerMonitor::MarkCaseCompleted() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    case_status_ = "completed";
    completed_cases_++;
    current_progress_ = 100.0F;

    // Save to history
    if (!current_case_.empty()) {
        JsonType history_entry;
        history_entry["name"] = current_case_;
        history_entry["metrics"] = current_metrics_;
        case_history_.push_back(history_entry);
    }
}

void
HttpServerMonitor::UpdateMetrics(const JsonType& metrics) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_metrics_ = metrics;
}

void
HttpServerMonitor::UpdateProgress(float percent_complete) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_progress_ = std::max(0.0F, std::min(100.0F, percent_complete));
}

void
HttpServerMonitor::SetWebRoot(const std::string& path) {
    web_root_ = path;
}

std::string
HttpServerMonitor::get_index_html() const {
    std::string html = R"(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VSAG Performance Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        header h1 { font-size: 28px; margin-bottom: 10px; }
        header p { opacity: 0.9; }
        .card {
            background: white;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .card h2 {
            font-size: 18px;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .progress-bar {
            background: #e0e0e0;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin: 15px 0;
        }
        .progress-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .stat-box label {
            display: block;
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .stat-box value {
            display: block;
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .status-running { color: #28a745; }
        .status-completed { color: #007bff; }
        .status-failed { color: #dc3545; }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .metrics-table th, .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .metrics-table th {
            background: #f8f9fa;
            font-weight: 600;
        }
        #metrics-content {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            white-space: pre-wrap;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
        }
        .refresh-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #28a745;
            margin-left: 10px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .case-item {
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 3px solid #ddd;
        }
        .case-item.completed { border-color: #28a745; }
        .case-item.running { border-color: #007bff; }
        .case-item.failed { border-color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>VSAG Performance Monitor</h1>
            <p>Real-time benchmark progress and metrics</p>
        </header>

        <div class="card">
            <h2>
                Overall Progress
                <span class="refresh-indicator"></span>
            </h2>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-bar" style="width: 0%">0%</div>
            </div>
            <div class="stats-grid">
                <div class="stat-box">
                    <label>Current Case</label>
                    <value id="current-case">-</value>
                </div>
                <div class="stat-box">
                    <label>Cases</label>
                    <value id="case-count">0/0</value>
                </div>
                <div class="stat-box">
                    <label>Status</label>
                    <value id="status">Idle</value>
                </div>
                <div class="stat-box">
                    <label>Elapsed Time</label>
                    <value id="elapsed">00:00</value>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Current Metrics</h2>
            <div id="metrics-content">No metrics available</div>
        </div>

        <div class="card">
            <h2>Completed Cases</h2>
            <div id="history-list">No completed cases yet</div>
        </div>
    </div>

    <script>
        async function fetchStatus() {
            try {
                const [statusRes, metricsRes] = await Promise.all([
                    fetch('/api/status'),
                    fetch('/api/metrics')
                ]);
                const status = await statusRes.json();
                const metrics = await metricsRes.json();
                updateUI(status, metrics);
            } catch (e) {
                console.error('Failed to fetch status:', e);
            }
        }

        function updateUI(status, metrics) {
            // Update progress bar
            const totalProgress = status.total_cases > 0
                ? ((status.completed_cases + status.progress/100) / status.total_cases * 100)
                : 0;
            const bar = document.getElementById('progress-bar');
            bar.style.width = totalProgress + '%';
            bar.textContent = Math.round(totalProgress) + '%';

            // Update stats
            document.getElementById('current-case').textContent = status.current_case || '-';
            document.getElementById('case-count').textContent =
                `${status.completed_cases}/${status.total_cases}`;

            const statusEl = document.getElementById('status');
            statusEl.textContent = status.case_status;
            statusEl.className = 'status-' + status.case_status;

            // Format elapsed time
            const minutes = Math.floor(status.total_elapsed_seconds / 60);
            const seconds = status.total_elapsed_seconds % 60;
            document.getElementById('elapsed').textContent =
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

            // Update metrics
            document.getElementById('metrics-content').textContent =
                JSON.stringify(metrics, null, 2);
        }

        async function fetchHistory() {
            try {
                const res = await fetch('/api/history');
                const history = await res.json();
                updateHistory(history);
            } catch (e) {
                console.error('Failed to fetch history:', e);
            }
        }

        function updateHistory(history) {
            const container = document.getElementById('history-list');
            if (history.length === 0) {
                container.innerHTML = 'No completed cases yet';
                return;
            }
            container.innerHTML = history.map(h =>
                `<div class="case-item completed">
                    <strong>${h.name}</strong>
                    <pre>${JSON.stringify(h.metrics, null, 2)}</pre>
                 </div>`
            ).join('');
        }

        // Poll every 1 second
        setInterval(() => {
            fetchStatus();
            fetchHistory();
        }, 1000);

        // Initial fetch
        fetchStatus();
        fetchHistory();
    </script>
</body>
</html>)";

    return html;
}

}  // namespace vsag::eval
