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

#include "html_report.h"

#include <set>
#include <sstream>
#include <vector>

#include "dataset_profiler.h"
#include "sha256.h"
#include "svg_chart.h"

namespace vsag::dataset_profile {

namespace {

constexpr const char* STYLE = R"CSS(
:root{--bg:#f6f7f9;--panel:#fff;--text:#1d232b;--muted:#5b6673;--line:#dde2e8;--accent:#2f6fdb;
--band:rgba(47,111,219,.16);--critical:#c62828;--warning:#b26a00;--notice:#1f6fb2;--info:#5b6673;
--c0:#0072b2;--c1:#e69f00;--c2:#009e73;--c3:#cc79a7;--c4:#56b4e9;--c5:#d55e00;--c6:#7a7a00;--c7:#555}
@media (prefers-color-scheme:dark){:root{--bg:#12161c;--panel:#1b2129;--text:#e6eaef;--muted:#9aa5b1;
--line:#2c3540;--accent:#6ea8ff;--band:rgba(110,168,255,.18);--critical:#ff6b6b;--warning:#ffb547;
--notice:#6ec1ff;--info:#9aa5b1;--c0:#56b4e9;--c1:#f0b74a;--c2:#3fcf9f;--c3:#e08fc0;--c4:#8fd0f5;
--c5:#ff8a4c;--c6:#c8c84a;--c7:#aaa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif}
header{padding:20px 28px 12px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{margin:0 0 6px;font-size:22px}h2{font-size:18px;margin:24px 0 10px}h3{margin:0;font-size:15px}
.meta{color:var(--muted);font-size:13px}.meta code{font-size:12px}
nav{position:sticky;top:0;z-index:2;display:flex;gap:4px;flex-wrap:wrap;padding:8px 24px;
background:var(--panel);border-bottom:1px solid var(--line)}
nav button{border:1px solid var(--line);background:transparent;color:var(--text);border-radius:6px;
padding:5px 12px;cursor:pointer;font:inherit}nav button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
main{padding:8px 24px 40px}main>section{max-width:1500px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px}
figure.card,div.card{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
figcaption{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.scope{font-size:11px;border-radius:999px;padding:2px 8px;white-space:nowrap}
.scope.full{background:rgba(0,158,115,.14);color:var(--c2)}.scope.sample{background:rgba(230,159,0,.16);color:var(--c1)}
.scope.mixed{background:rgba(85,85,85,.14);color:var(--muted)}
.definition{color:var(--muted);font-size:12.5px;margin:6px 0 2px}.key{font-size:11.5px;color:var(--muted);margin:0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:6px}
td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
td.num{text-align:right;font-variant-numeric:tabular-nums}th{color:var(--muted);font-weight:600}
.summary-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.stat .label{color:var(--muted);font-size:12px}.stat .value{font-size:19px;font-weight:600}
.sev{display:inline-block;font-size:11px;font-weight:700;border-radius:4px;padding:1px 6px;color:#fff;text-transform:uppercase}
.sev.critical{background:var(--critical)}.sev.warning{background:var(--warning)}.sev.notice{background:var(--notice)}.sev.info{background:var(--info)}
a.evidence{display:inline-block;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;
border:1px solid var(--line);border-radius:4px;padding:0 4px;margin:1px 2px;color:var(--accent);text-decoration:none;
max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:bottom}
.stat a.evidence{margin:4px 0 0}
.banner{border-left:4px solid var(--warning);background:var(--panel);padding:10px 14px;border-radius:6px;margin:10px 0}
.banner.ok{border-left-color:var(--c2)}.banner.bad{border-left-color:var(--critical)}
.hyp{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--panel);margin:10px 0}
.chip{display:inline-block;font-size:11px;border:1px solid var(--line);border-radius:999px;padding:0 8px;margin-right:4px;color:var(--muted)}
svg.chart{width:100%;height:auto;display:block}
svg .axis,svg .tick-mark{stroke:var(--muted);stroke-width:1}svg .grid{stroke:var(--line);stroke-width:1}
svg .tick,svg .legend-text,svg .value{fill:var(--muted);font-size:11.5px}svg .axis-title{fill:var(--muted);font-size:12px}
svg .note{fill:var(--warning);font-size:11px}
svg .bar{fill:var(--accent)}svg .band{fill:var(--band);stroke:none}svg .line{fill:none;stroke-width:1.6}
svg .dashed{stroke-dasharray:4 3}svg .overlay{fill:none;stroke:var(--text);stroke-width:1}
svg .s0,svg .c0{stroke:var(--c0);fill:var(--c0)}svg .s1,svg .c1{stroke:var(--c1);fill:var(--c1)}
svg .s2,svg .c2{stroke:var(--c2);fill:var(--c2)}svg .s3,svg .c3{stroke:var(--c3);fill:var(--c3)}
svg .s4,svg .c4{stroke:var(--c4);fill:var(--c4)}svg .s5,svg .c5{stroke:var(--c5);fill:var(--c5)}
svg .s6,svg .c6{stroke:var(--c6);fill:var(--c6)}svg .s7,svg .c7{stroke:var(--c7);fill:var(--c7)}
svg path.line{fill:none}svg .pt{stroke:none;opacity:.75}svg .log{display:none}
.log-mode svg .log{display:inline}.log-mode svg .lin{display:none}
button.scale{float:right;font-size:11px;border:1px solid var(--line);border-radius:4px;background:transparent;color:var(--muted);cursor:pointer}
.flash{outline:3px solid var(--accent);outline-offset:2px}
input#metric-filter{width:100%;max-width:420px;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--text)}
details summary{cursor:pointer;color:var(--muted)}
)CSS";

constexpr const char* SCRIPT = R"JS(
(function(){
const tabs=[...document.querySelectorAll('nav button[data-tab]')];
function show(id){if(!document.getElementById(id))return;document.querySelectorAll('main>section').forEach(s=>{s.hidden=s.id!==id;});
tabs.forEach(b=>b.classList.toggle('active',b.dataset.tab===id));history.replaceState(null,'','#'+id);}
tabs.forEach(b=>b.addEventListener('click',()=>show(b.dataset.tab)));
document.addEventListener('click',e=>{const a=e.target.closest('a[data-target]');if(!a)return;e.preventDefault();
const t=document.getElementById(a.dataset.target);if(!t)return;const s=t.closest('main>section');if(s)show(s.id);
t.scrollIntoView({behavior:'smooth',block:'center'});t.classList.add('flash');setTimeout(()=>t.classList.remove('flash'),1500);});
document.querySelectorAll('button.scale').forEach(b=>b.addEventListener('click',()=>{const f=b.closest('figure');
f.classList.toggle('log-mode');b.textContent=f.classList.contains('log-mode')?'log y':'linear y';}));
const filter=document.getElementById('metric-filter');if(filter){filter.addEventListener('input',()=>{
const q=filter.value.toLowerCase();document.querySelectorAll('#metric-index tbody tr').forEach(r=>{
r.hidden=q!==''&&!r.textContent.toLowerCase().includes(q);});});}
const dl=document.getElementById('download-json');if(dl){dl.addEventListener('click',()=>{
const data=document.getElementById('profile-data').textContent;const blob=new Blob([data],{type:'application/json'});
const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=dl.dataset.name;a.click();
setTimeout(()=>URL.revokeObjectURL(a.href),1000);});}
const initial=location.hash.slice(1);
if(initial&&document.getElementById(initial)&&document.getElementById(initial).matches('main>section')){show(initial);}
else if(tabs.length){show(tabs[0].dataset.tab);}
})();
)JS";

double
number_at(const JsonType& root, const std::string& path, double fallback = 0.0) {
    const JsonType* node = find_path(root, path);
    return node != nullptr && node->is_number() ? node->get<double>() : fallback;
}

std::string
text_value(const JsonType& value) {
    if (value.is_number()) {
        return format_number_html(value.get<double>());
    }
    if (value.is_string()) {
        return escape_html(value.get<std::string>());
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? "yes" : "no";
    }
    if (value.is_null()) {
        return "n/a";
    }
    return escape_html(value.dump());
}

std::vector<double>
doubles(const JsonType& array) {
    std::vector<double> values;
    if (!array.is_array()) {
        return values;
    }
    for (const auto& item : array) {
        values.push_back(item.is_number() ? item.get<double>()
                                          : std::numeric_limits<double>::quiet_NaN());
    }
    return values;
}

class Renderer {
public:
    explicit Renderer(const JsonType& report) : report_(report) {
        seed_ = report.at("config").value("seed", 0ULL);
    }

    std::string
    Render() {
        std::ostringstream body;
        std::vector<std::pair<std::string, std::string>> tabs;
        tabs.emplace_back("overview", "Overview");
        const JsonType& splits = report_.at("splits");
        for (const auto& item : splits.items()) {
            tabs.emplace_back("split-" + item.key(), "Split: " + item.key());
        }
        if (report_.contains("cross_split")) {
            tabs.emplace_back("cross", "Queries vs base");
        }
        if (report_.contains("ground_truth") || report_.contains("labels")) {
            tabs.emplace_back("truth", "Ground truth & labels");
        }
        tabs.emplace_back("ai", "AI analysis");
        tabs.emplace_back("config", "Configuration & metrics");

        // Split tabs register their cards first so overview evidence chips can resolve them.
        std::ostringstream split_sections;
        for (const auto& item : splits.items()) {
            split_sections << split_section(item.key(), item.value());
        }
        std::string cross = report_.contains("cross_split") ? cross_section() : "";
        std::string truth =
            (report_.contains("ground_truth") || report_.contains("labels")) ? truth_section() : "";

        body << "<main>" << overview_section() << split_sections.str() << cross << truth
             << ai_section() << config_section() << "</main>";

        std::ostringstream page;
        const JsonType& input = report_.at("input");
        const std::string file = input.value("file_name", "");
        page << R"(<!doctype html><html lang="en"><head><meta charset="utf-8">)"
             << R"(<meta http-equiv="Content-Security-Policy" content=")"
             << html_content_security_policy() << "\">"
             << R"(<meta name="viewport" content="width=device-width, initial-scale=1">)"
             << R"(<meta name="generator" content="VSAG profile_dataset">)"
             << "<title>Dataset profile: " << escape_html(file) << "</title><style>" << STYLE
             << "</style></head><body>";
        page << "<header><h1>VSAG dataset profile &middot; " << escape_html(file)
             << "</h1><div class=\"meta\">" << escape_html(input.value("vector_type", ""))
             << " vectors &middot; metric <b>"
             << escape_html(report_.at("config").value("metric", "")) << "</b> ("
             << escape_html(report_.at("config").value("metric_source", "")) << ") &middot; schema "
             << escape_html(report_.value("schema", "")) << " v"
             << escape_html(report_.value("schema_version", "")) << " &middot; seed " << seed_
             << " &middot; generated "
             << escape_html(report_.contains("run") ? report_["run"].value("started_at", "") : "")
             << "<br>metrics SHA-256 <code>"
             << escape_html(report_.contains("determinism")
                                ? report_["determinism"].value("metrics_sha256", "")
                                : "")
             << "</code> &middot; this page loads no external resources</div></header><nav>";
        for (const auto& [id, label] : tabs) {
            page << "<button data-tab=\"" << escape_html(id) << "\">" << escape_html(label)
                 << "</button>";
        }
        page << "</nav>" << body.str();
        // JSON may only contain <, > and & inside strings, where \u escapes are equivalent. Escaping
        // them keeps untrusted text (e.g. AI output) from closing the block or opening comments.
        const std::string data = report_.dump(-1, ' ', true);
        std::string safe_data;
        safe_data.reserve(data.size());
        for (const char character : data) {
            if (character == '<') {
                safe_data += "\\u003c";
            } else if (character == '>') {
                safe_data += "\\u003e";
            } else if (character == '&') {
                safe_data += "\\u0026";
            } else {
                safe_data += character;
            }
        }
        page << R"(<script type="application/json" id="profile-data">)" << safe_data << "</script>";
        page << "<script>" << SCRIPT << "</script></body></html>\n";
        return page.str();
    }

private:
    [[nodiscard]] std::string
    scope_badge(const JsonType& section, const JsonType& split) const {
        const std::string scope = section.value("scope", "full");
        std::ostringstream out;
        if (scope == "sample" && section.contains("sample")) {
            const JsonType& sample = section["sample"];
            out << R"(<span class="scope sample" title=")"
                << escape_html(sample.value("method", ""))
                << "\">Sample &middot; n = " << format_number_html(sample.value("used_rows", 0.0))
                << " of " << format_number_html(sample.value("population", 0.0))
                << " &middot; seed " << seed_ << "</span>";
        } else if (scope == "sample") {
            out << "<span class=\"scope sample\">Sample &middot; seed " << seed_ << "</span>";
        } else if (scope == "mixed") {
            out << "<span class=\"scope mixed\">Mixed: full-split and sampled fields</span>";
        } else {
            out << "<span class=\"scope full\">Full split &middot; N = "
                << format_number_html(split.value("rows", 0.0)) << "</span>";
        }
        return out.str();
    }

    std::string
    card(const std::string& key,
         const std::string& title,
         const JsonType& section,
         const JsonType& split,
         const std::string& content,
         bool log_toggle = false) {
        cards_.insert(key);
        std::ostringstream out;
        out << R"(<figure class="card" id="card-)" << escape_html(key) << "\"><figcaption><h3>"
            << escape_html(title) << "</h3>" << scope_badge(section, split) << "</figcaption>";
        if (log_toggle) {
            out << R"(<button class="scale" type="button">linear y</button>)";
        }
        out << content << "<p class=\"definition\">" << escape_html(section.value("definition", ""))
            << "</p>"
            << "<p class=\"key\"><code>" << escape_html(key) << "</code></p></figure>";
        return out.str();
    }

    static std::string
    table(const std::vector<std::pair<std::string, JsonType>>& rows) {
        std::ostringstream out;
        out << "<table>";
        for (const auto& [label, value] : rows) {
            out << "<tr><th>" << escape_html(label) << "</th><td class=\"num\">"
                << text_value(value) << "</td></tr>";
        }
        out << "</table>";
        return out.str();
    }

    static std::string
    summary_table(const JsonType& summary) {
        std::ostringstream out;
        out << "<table><tr>";
        for (const char* key : {"count", "min", "p05", "p50", "p95", "max", "mean", "std"}) {
            out << "<th>" << key << "</th>";
        }
        out << "</tr><tr>";
        for (const char* key : {"count", "min", "p05", "p50", "p95", "max", "mean", "std"}) {
            out << "<td class=\"num\">"
                << (summary.contains(key) ? text_value(summary[key]) : "n/a") << "</td>";
        }
        out << "</tr></table>";
        return out.str();
    }

    std::string
    histogram_card(const std::string& key,
                   const std::string& title,
                   const JsonType& section,
                   const JsonType& split,
                   const JsonType& histogram,
                   const JsonType& summary,
                   const std::string& x_label) {
        if (!histogram.is_object() || !histogram.contains("counts")) {
            return "";
        }
        ChartOptions options;
        options.x_label = x_label;
        options.y_label = "rows";
        std::string content = histogram_svg(histogram, options);
        if (summary.is_object()) {
            content += summary_table(summary);
        }
        return card(key, title, section, split, content, true);
    }

    [[nodiscard]] std::string
    evidence_chip(const std::string& key) const {
        // Link to the closest enclosing chart card, e.g. splits.train.norms for
        // splits.train.norms.summary.p50.
        std::string target = key;
        while (!target.empty() && cards_.count(target) == 0) {
            const auto dot = target.rfind('.');
            target = dot == std::string::npos ? "" : target.substr(0, dot);
        }
        const std::string anchor = target.empty() ? "metric-index" : "card-" + target;
        const JsonType* value = find_path(report_, key);
        std::string title =
            value != nullptr && !value->is_object() && !value->is_array() ? text_value(*value) : "";
        return R"(<a class="evidence" href="#)" + escape_html(anchor) + "\" data-target=\"" +
               escape_html(anchor) + "\" title=\"" + title + "\">" + escape_html(key) + "</a>";
    }

    [[nodiscard]] std::string
    signals_table() const {
        const JsonType& signals = report_.at("signals");
        std::ostringstream out;
        if (signals.empty()) {
            out << "<p class=\"meta\">No rule-based observations were triggered.</p>";
            return out.str();
        }
        out << "<table><tr><th>Severity</th><th>Observation</th><th>Evidence</th></tr>";
        for (const auto& signal : signals) {
            out << "<tr><td><span class=\"sev " << escape_html(signal.value("severity", "info"))
                << "\">" << escape_html(signal.value("severity", "")) << "</span></td><td>"
                << escape_html(signal.value("message", ""))
                << "<div class=\"key\">rule: " << escape_html(signal.value("rule", ""))
                << "</div></td><td>";
            for (const auto& key : signal.value("evidence", JsonType::array())) {
                out << evidence_chip(key.get<std::string>());
            }
            out << "</td></tr>";
        }
        out << "</table>";
        return out.str();
    }

    std::string
    overview_section() {
        std::ostringstream out;
        out << "<section id=\"overview\"><h2>Overview</h2>";
        for (const auto& item : report_.at("splits").items()) {
            const std::string p = "splits." + item.key() + ".";
            const JsonType& split = item.value();
            const bool sparse = split.value("kind", "") == "sparse";
            out << "<h3 style=\"margin:14px 0 8px\">Split <code>" << escape_html(item.key())
                << "</code></h3><div class=\"summary-cards\">";
            auto stat = [&](const std::string& label, const std::string& path, bool ratio = false) {
                const JsonType* node = find_path(report_, path);
                if (node == nullptr) {
                    return;
                }
                std::string value = text_value(*node);
                if (ratio && node->is_number()) {
                    char buffer[32];
                    std::snprintf(buffer, sizeof(buffer), "%.3g%%", node->get<double>() * 100.0);
                    value = buffer;
                }
                out << R"(<div class="stat"><div class="label">)" << escape_html(label)
                    << "</div><div class=\"value\">" << value << "</div><div class=\"key\">"
                    << evidence_chip(path) << "</div></div>";
            };
            stat("rows", p + "rows");
            if (!sparse) {
                stat("dimensions", p + "dim");
                stat("storage type", p + "storage_type");
            } else {
                stat("max non-zeros", p + "basic.max_nonzeros");
                stat("distinct terms", p + "terms.distinct_terms");
            }
            stat("invalid values", p + "invalid_values.total");
            stat("exact duplicate rows", p + "duplicates.duplicate_ratio", true);
            stat("unit-norm rows", p + "norms.unit_norm_ratio", true);
            stat("median LID", p + "intrinsic_dimensionality.lid_mle.summary.p50");
            stat("median relative contrast", p + "relative_contrast.summary.p50");
            stat("hubness skewness", p + "hubness.skewness");
            stat("PCs for 90% variance", p + "pca.components_for_variance.90");
            stat("SINDI term_id_limit", p + "terms.min_term_id_limit");
            out << "</div>";
        }
        out << "<h2>Rule-based observations</h2><p class=\"meta\">Deterministic rules with fixed "
               "thresholds; "
               "available offline and independent of any AI provider.</p>"
            << signals_table();
        out << "</section>";
        return out.str();
    }

    std::string
    split_section(const std::string& name, const JsonType& split) {
        std::ostringstream cards;
        const std::string p = "splits." + name;
        const bool sparse = split.value("kind", "") == "sparse";
        auto has = [&split](const char* key) { return split.contains(key); };
        if (has("invalid_values")) {
            const JsonType& s = split["invalid_values"];
            std::vector<std::pair<std::string, JsonType>> rows;
            for (const auto& item : s.items()) {
                if (item.key() != "scope" && item.key() != "definition") {
                    rows.emplace_back(item.key(), item.value());
                }
            }
            cards << card(p + ".invalid_values", "Invalid values", s, split, table(rows));
        }
        if (has("duplicates")) {
            const JsonType& s = split["duplicates"];
            std::string examples;
            for (const auto& group : s.value("examples", JsonType::array())) {
                examples += "<div class=\"key\">group of " + text_value(group["size"]) + ": rows " +
                            escape_html(group["rows"].dump()) + "</div>";
            }
            cards << card(p + ".duplicates",
                          "Exact duplicates",
                          s,
                          split,
                          table({{"duplicate rows", s.value("duplicate_rows", JsonType())},
                                 {"duplicate ratio", s.value("duplicate_ratio", JsonType())},
                                 {"groups", s.value("groups", JsonType())},
                                 {"largest group", s.value("largest_group", JsonType())}}) +
                              examples);
        }
        if (has("norms")) {
            const JsonType& s = split["norms"];
            cards << histogram_card(p + ".norms",
                                    "Row L2 norms",
                                    s,
                                    split,
                                    s.value("histogram", JsonType()),
                                    s.value("summary", JsonType()),
                                    "L2 norm");
        }
        if (has("values") && split["values"].contains("histogram")) {
            const JsonType& s = split["values"];
            std::vector<std::pair<std::string, JsonType>> rows = {
                {"mean", s.value("mean", JsonType())},
                {"std", s.value("std", JsonType())},
                {"skewness", s.value("skewness", JsonType())},
                {"excess kurtosis", s.value("excess_kurtosis", JsonType())}};
            if (s.contains("outliers_6_sigma")) {
                rows.emplace_back("beyond 6 sigma (ratio)",
                                  s["outliers_6_sigma"].value("ratio", JsonType()));
            }
            if (s.contains("negative_values")) {
                rows.emplace_back("negative values", s["negative_values"]);
            }
            ChartOptions options;
            options.x_label = sparse ? "stored value" : "coordinate value";
            options.y_label = "values";
            cards << card(p + ".values",
                          sparse ? "Stored values" : "Coordinate values",
                          s,
                          split,
                          histogram_svg(s["histogram"], options) + table(rows),
                          true);
        }
        if (has("dimensions")) {
            const JsonType& s = split["dimensions"];
            const JsonType& per = s["per_dimension"];
            std::vector<double> mean = doubles(per["mean"]);
            std::vector<double> std = doubles(per["std"]);
            std::vector<double> low(mean.size());
            std::vector<double> high(mean.size());
            for (uint64_t d = 0; d < mean.size(); ++d) {
                low[d] = mean[d] - std[d];
                high[d] = mean[d] + std[d];
            }
            ChartOptions options;
            options.x_label = "dimension";
            options.y_label = "value";
            std::vector<LineSeries> lines = {{"mean", mean, 0, false},
                                             {"min", doubles(per["min"]), 5, true},
                                             {"max", doubles(per["max"]), 5, true}};
            const JsonType& summary = s["summary"];
            cards << card(
                p + ".dimensions",
                "Per-dimension mean ± std, min and max",
                s,
                split,
                line_svg(lines, {}, low, high, options) +
                    table(
                        {{"constant dimensions", summary.value("constant_dimensions", JsonType())},
                         {"max / median std", summary.value("std_ratio_max_to_median", JsonType())},
                         {"max / median range",
                          summary.value("range_ratio_max_to_median", JsonType())},
                         {"variance share of top 10% dims",
                          summary.value("variance_share_top_10pct_dimensions", JsonType())},
                         {"mean-vector norm / mean norm",
                          summary.value("mean_vector_norm_ratio", JsonType())}}));
            ChartOptions shape;
            shape.x_label = "dimension";
            std::vector<LineSeries> shape_lines = {
                {"excess kurtosis", doubles(per["excess_kurtosis"]), 3, false},
                {"skewness", doubles(per["skewness"]), 2, false}};
            cards << card(p + ".dimensions.per_dimension",
                          "Per-dimension shape (skewness, excess kurtosis)",
                          s,
                          split,
                          line_svg(shape_lines, {}, {}, {}, shape));
            ChartOptions zero;
            zero.x_label = "dimension";
            zero.y_label = "ratio";
            std::vector<LineSeries> zero_lines = {
                {"zero ratio", doubles(per["zero_ratio"]), 4, false}};
            if (per.contains("outlier_ratio_6_sigma")) {
                zero_lines.push_back(
                    {"beyond 6 sigma", doubles(per["outlier_ratio_6_sigma"]), 5, false});
            }
            cards << card(p + ".dimensions.per_dimension.zero_ratio",
                          "Per-dimension zeros and 6-sigma outliers",
                          s,
                          split,
                          line_svg(zero_lines, {}, {}, {}, zero));
        }
        if (has("sparsity")) {
            const JsonType& s = split["sparsity"];
            cards << histogram_card(
                p + ".sparsity",
                sparse ? "Non-zeros per record" : "Non-zero coordinates per row",
                s,
                split,
                s["nonzeros_per_row"].value("histogram", JsonType()),
                s["nonzeros_per_row"].value("summary", JsonType()),
                "non-zeros");
        }
        if (has("terms")) {
            const JsonType& s = split["terms"];
            cards << histogram_card(p + ".terms",
                                    "Document frequency (log10)",
                                    s,
                                    split,
                                    s["document_frequency"].value("log10_histogram", JsonType()),
                                    s["document_frequency"].value("summary", JsonType()),
                                    "log10(document frequency)");
            std::vector<std::string> labels;
            std::vector<double> values;
            for (const auto& term : s.value("top_terms", JsonType::array())) {
                labels.push_back("term " + term["term"].dump());
                values.push_back(term["document_frequency"].get<double>());
            }
            ChartOptions options;
            cards << card(
                p + ".terms.top_terms",
                "Most frequent terms (longest posting lists)",
                s,
                split,
                bar_svg(labels, values, options) +
                    table({{"distinct terms", s.value("distinct_terms", JsonType())},
                           {"min term_id_limit", s.value("min_term_id_limit", JsonType())},
                           {"term id density", s.value("term_id_density", JsonType())},
                           {"Zipf exponent", s.value("zipf_exponent", JsonType())},
                           {"Gini", s.value("gini", JsonType())},
                           {"top 1% terms posting share",
                            s.value("top_1pct_terms_posting_share", JsonType())}}));
            const JsonType& retention = s["doc_prune_retained_term_fraction"];
            std::vector<double> x;
            std::vector<double> kept;
            for (const auto& item : retention["by_ratio"].items()) {
                x.push_back(std::stod(item.key()) / 100.0);
                kept.push_back(item.value().get<double>());
            }
            ChartOptions prune;
            prune.x_label = "SINDI doc_prune_ratio";
            prune.y_label = "terms kept";
            JsonType prune_section = s;
            prune_section["definition"] = retention.value("definition", "");
            cards << card(
                p + ".terms.doc_prune_retained_term_fraction",
                "Terms kept by SINDI doc pruning",
                prune_section,
                split,
                line_svg({{"mean fraction of terms kept", kept, 2, false}}, x, {}, {}, prune));
        }
        if (has("pairwise_distances")) {
            const JsonType& s = split["pairwise_distances"];
            cards << histogram_card(p + ".pairwise_distances",
                                    "Pairwise distances (random sampled pairs)",
                                    s,
                                    split,
                                    s["metric_distance"].value("histogram", JsonType()),
                                    s["metric_distance"].value("summary", JsonType()),
                                    "metric distance");
            cards << histogram_card(p + ".pairwise_distances.cosine_similarity",
                                    "Pairwise cosine similarity",
                                    s,
                                    split,
                                    s["cosine_similarity"].value("histogram", JsonType()),
                                    s["cosine_similarity"].value("summary", JsonType()),
                                    "cosine similarity");
        }
        if (has("nearest_neighbors")) {
            const JsonType& s = split["nearest_neighbors"];
            cards << histogram_card(p + ".nearest_neighbors",
                                    "First nearest-neighbor distance",
                                    s,
                                    split,
                                    s["first_neighbor_distance"].value("histogram", JsonType()),
                                    s["first_neighbor_distance"].value("summary", JsonType()),
                                    "metric distance");
            cards << histogram_card(
                p + ".nearest_neighbors.kth_neighbor_distance",
                "k-th nearest-neighbor distance (k = " + text_value(s.value("k", JsonType())) + ")",
                s,
                split,
                s["kth_neighbor_distance"].value("histogram", JsonType()),
                s["kth_neighbor_distance"].value("summary", JsonType()),
                "metric distance");
        }
        if (has("intrinsic_dimensionality")) {
            const JsonType& s = split["intrinsic_dimensionality"];
            std::string extra =
                table({{"TwoNN estimate", s.value("two_nn", JsonType())},
                       {"median LID / dim", s.value("lid_to_dim_ratio", JsonType())}});
            ChartOptions options;
            options.x_label = "local intrinsic dimensionality (MLE)";
            options.y_label = "queries";
            cards << card(p + ".intrinsic_dimensionality",
                          "Local intrinsic dimensionality",
                          s,
                          split,
                          histogram_svg(s["lid_mle"]["histogram"], options) +
                              summary_table(s["lid_mle"]["summary"]) + extra,
                          true);
        }
        if (has("relative_contrast")) {
            const JsonType& s = split["relative_contrast"];
            cards << histogram_card(p + ".relative_contrast",
                                    "Relative contrast",
                                    s,
                                    split,
                                    s.value("histogram", JsonType()),
                                    s.value("summary", JsonType()),
                                    "mean distance / nearest distance");
        }
        if (has("hubness")) {
            const JsonType& s = split["hubness"];
            ChartOptions options;
            options.x_label = "k-occurrence N_k";
            options.y_label = "rows";
            cards << card(p + ".hubness",
                          "Hubness (k-occurrence)",
                          s,
                          split,
                          histogram_svg(s["histogram"], options) +
                              table({{"skewness", s.value("skewness", JsonType())},
                                     {"anti-hub ratio", s.value("anti_hub_ratio", JsonType())},
                                     {"max occurrence", s.value("max_occurrence", JsonType())}}),
                          true);
        }
        if (has("pca") && !split["pca"].value("skipped", false)) {
            const JsonType& s = split["pca"];
            ChartOptions options;
            options.x_label = "principal component";
            options.y_label = "variance share";
            std::vector<double> x;
            x.reserve(s["explained_variance_ratio"].size());
            for (uint64_t i = 0; i < s["explained_variance_ratio"].size(); ++i) {
                x.push_back(static_cast<double>(i + 1));
            }
            std::vector<std::pair<std::string, JsonType>> rows;
            for (const auto& item : s["components_for_variance"].items()) {
                rows.emplace_back("components for " + item.key() + "% variance", item.value());
            }
            rows.emplace_back("effective rank", s.value("effective_rank", JsonType()));
            rows.emplace_back("participation ratio", s.value("participation_ratio", JsonType()));
            cards << card(
                p + ".pca",
                "PCA spectrum (" + s.value("input", std::string()) + ")",
                s,
                split,
                line_svg({{"cumulative", doubles(s["cumulative_explained_variance"]), 0, false},
                          {"per component", doubles(s["explained_variance_ratio"]), 1, false}},
                         x,
                         {},
                         {},
                         options) +
                    table(rows));
        } else if (has("pca")) {
            cards << card(p + ".pca",
                          "PCA spectrum",
                          split["pca"],
                          split,
                          "<p class=\"meta\">Skipped: " +
                              escape_html(split["pca"].value("reason", "")) + "</p>");
        }
        if (has("projection")) {
            const JsonType& s = split["projection"];
            JsonType overlay = JsonType::array();
            std::string note;
            if (name == "train" && report_.contains("cross_split") &&
                report_["cross_split"].contains("query_projection")) {
                overlay = report_["cross_split"]["query_projection"]["points"];
                note =
                    "<p class=\"meta\">Hollow markers: sampled test queries on the same axes.</p>";
            }
            ChartOptions options;
            options.height = 330;
            options.x_label = "PC1";
            options.y_label = "PC2";
            cards << card(p + ".projection",
                          "2-D PCA projection (colored by k-means cluster)",
                          s,
                          split,
                          scatter_svg(s["points"], overlay, options) + note);
        }
        if (has("clustering")) {
            const JsonType& s = split["clustering"];
            std::vector<std::string> labels;
            std::vector<double> sizes = doubles(s["sizes"]);
            labels.reserve(sizes.size());
            for (uint64_t i = 0; i < sizes.size(); ++i) {
                labels.push_back("cluster " + std::to_string(i + 1));
            }
            cards << card(
                p + ".clustering",
                "k-means cluster sizes (sorted)",
                s,
                split,
                bar_svg(labels, sizes, ChartOptions{}) +
                    table({{"silhouette", s.value("silhouette", JsonType())},
                           {"size CV", s.value("size_cv", JsonType())},
                           {"max / mean size", s.value("max_to_mean_size", JsonType())},
                           {"normalized entropy", s.value("normalized_entropy", JsonType())},
                           {"converged", s.value("converged", JsonType())}}));
        }
        if (has("quantization")) {
            const JsonType& s = split["quantization"];
            std::vector<std::string> labels;
            std::vector<double> recall;
            std::ostringstream rows;
            rows << "<table><tr><th>scheme</th><th>recall@k</th><th>rel. "
                    "error</th><th>bytes/vector</th><th>query encoded</th></tr>";
            for (const auto& item : s["schemes"].items()) {
                labels.push_back(item.key());
                recall.push_back(item.value().value("recall_at_k", 0.0));
                rows << "<tr><td>" << escape_html(item.key()) << "</td><td class=\"num\">"
                     << text_value(item.value()["recall_at_k"]) << "</td><td class=\"num\">"
                     << text_value(item.value()["mean_relative_reconstruction_error"])
                     << "</td><td class=\"num\">"
                     << text_value(item.value()["code_bytes_per_vector"]) << "</td><td>"
                     << text_value(item.value()["encodes_query"]) << "</td></tr>";
            }
            rows << "</table><p class=\"meta\">k = " << text_value(s.value("k", JsonType()))
                 << ", base rows " << text_value(s.value("base_rows", JsonType())) << ", queries "
                 << text_value(s.value("query_rows", JsonType())) << " ("
                 << escape_html(s.value("query_source", "")) << ")</p>";
            cards << card(p + ".quantization",
                          "Simulated quantization recall@k",
                          s,
                          split,
                          bar_svg(labels, recall, ChartOptions{}) + rows.str());
        }
        std::ostringstream out;
        out << "<section id=\"split-" << escape_html(name) << "\"><h2>Split <code>"
            << escape_html(name) << "</code> &middot; "
            << format_number_html(split.value("rows", 0.0)) << " rows";
        if (!sparse) {
            out << " &times; " << format_number_html(split.value("dim", 0.0)) << " dims ("
                << escape_html(split.value("storage_type", "")) << ")";
        }
        out << "</h2><div class=\"grid\">" << cards.str() << "</div></section>";
        return out.str();
    }

    std::string
    cross_section() {
        const JsonType& c = report_["cross_split"];
        const JsonType split = JsonType{{"rows", number_at(report_, "splits.test.rows")}};
        std::ostringstream cards;
        std::vector<std::pair<std::string, JsonType>> rows;
        for (const auto& item : c.items()) {
            if (item.value().is_object() && item.value().contains("value")) {
                rows.emplace_back(item.key() + " (" + item.value().value("scope", "") + ")",
                                  item.value()["value"]);
            }
        }
        cards << card("cross_split", "Query versus base shift", c, split, table(rows));
        if (c.contains("standardized_mean_shift")) {
            cards << card("cross_split.standardized_mean_shift",
                          "Per-dimension standardized mean shift",
                          JsonType{{"scope", "full"},
                                   {"definition",
                                    "|mean_test - mean_train| / std_train for each dimension."}},
                          split,
                          summary_table(c["standardized_mean_shift"]["summary"]));
        }
        for (const char* key : {"query_first_neighbor_distance", "postings_per_query"}) {
            if (c.contains(key)) {
                cards << card(std::string("cross_split.") + key,
                              key,
                              c[key],
                              split,
                              summary_table(c[key]["summary"]));
            }
        }
        if (c.contains("out_of_distribution_ratio")) {
            cards << card(
                "cross_split.out_of_distribution_ratio",
                "Out-of-distribution queries",
                c["out_of_distribution_ratio"],
                split,
                table({{"ratio", c["out_of_distribution_ratio"].value("value", JsonType())},
                       {"threshold",
                        c["out_of_distribution_ratio"].value("threshold", JsonType())}}));
        }
        return "<section id=\"cross\"><h2>Queries (test) versus base (train)</h2><div "
               "class=\"grid\">" +
               cards.str() + "</div></section>";
    }

    std::string
    truth_section() {
        std::ostringstream cards;
        const JsonType split = JsonType{{"rows", number_at(report_, "splits.test.rows")}};
        if (report_.contains("ground_truth")) {
            const JsonType& g = report_["ground_truth"];
            std::vector<std::pair<std::string, JsonType>> rows;
            for (const auto& item : g.items()) {
                if (!item.value().is_object() && item.key() != "scope" &&
                    item.key() != "definition") {
                    rows.emplace_back(item.key(), item.value());
                }
            }
            cards << card("ground_truth", "Ground-truth consistency", g, split, table(rows));
        }
        if (report_.contains("labels")) {
            const JsonType& l = report_["labels"];
            std::ostringstream content;
            for (const char* name : {"train", "test"}) {
                if (l.contains(name)) {
                    std::vector<std::pair<std::string, JsonType>> rows;
                    for (const auto& item : l[name].items()) {
                        rows.emplace_back(std::string(name) + " " + item.key(), item.value());
                    }
                    content << table(rows);
                }
            }
            cards << card("labels",
                          "Labels",
                          l,
                          JsonType{{"rows", number_at(report_, "splits.train.rows")}},
                          content.str());
            if (l.contains("query_selectivity")) {
                ChartOptions options;
                options.x_label = "share of base rows with the query label";
                options.y_label = "queries";
                cards << card("labels.query_selectivity",
                              "Filtered-search selectivity",
                              l,
                              split,
                              histogram_svg(l["query_selectivity"]["histogram"], options) +
                                  summary_table(l["query_selectivity"]["summary"]),
                              true);
            }
        }
        return R"(<section id="truth"><h2>Ground truth and labels</h2><div class="grid">)" +
               cards.str() + "</div></section>";
    }

    std::string
    ai_section() {
        std::ostringstream out;
        out << "<section id=\"ai\"><h2>AI-assisted interpretation</h2>";
        if (!report_.contains("ai_analysis")) {
            out << "<div class=\"banner\">AI analysis was not enabled. The deterministic metrics "
                   "and rule-based "
                   "observations above are complete without it. Enable it with "
                   "<code>--ai-provider</code> (see the "
                   "documentation); only aggregate metrics are sent.</div></section>";
            return out.str();
        }
        const JsonType& ai = report_["ai_analysis"];
        const std::string status = ai.value("status", "");
        out << "<div class=\"banner " << (status == "ok" ? "ok" : "bad")
            << "\"><b>Status: " << escape_html(status) << "</b> &middot; provider "
            << escape_html(ai.value("provider", "")) << " &middot; model "
            << escape_html(ai.value("model", "")) << " &middot; request SHA-256 <code>"
            << escape_html(ai.value("request_sha256", ""))
            << "</code> &middot; evidence keys sent: "
            << text_value(ai.value("evidence_key_count", JsonType()))
            << " &middot; raw vectors sent: no";
        if (ai.contains("message")) {
            out << "<br>" << escape_html(ai.value("message", ""));
        }
        out << "</div>";
        if (!ai.contains("analysis")) {
            out << "</section>";
            return out.str();
        }
        const JsonType& analysis = ai["analysis"];
        out << "<div class=\"banner\">Generated by a language model from the aggregate metrics "
               "only. Every item "
               "below is a <b>hypothesis</b> tied to the cited metrics, not a guaranteed optimal "
               "configuration; "
               "validate it with reproducible VSAG benchmarks (eval_performance) before changing "
               "an index.</div>";
        out << "<div class=\"card\"><h3>Summary</h3><p>"
            << escape_html(analysis.value("summary", "")) << "</p></div>";
        out << "<h3 "
               "style=\"margin-top:16px\">Findings</h3><table><tr><th>Severity</th><th>Finding</"
               "th><th>Evidence</th></tr>";
        for (const auto& finding : analysis.value("findings", JsonType::array())) {
            out << "<tr><td><span class=\"sev " << escape_html(finding.value("severity", "info"))
                << "\">" << escape_html(finding.value("severity", "")) << "</span></td><td><b>"
                << escape_html(finding.value("title", "")) << "</b><br>"
                << escape_html(finding.value("detail", "")) << "</td><td>";
            for (const auto& key : finding.value("evidence", JsonType::array())) {
                out << evidence_chip(key.get<std::string>());
            }
            out << "</td></tr>";
        }
        out << "</table><h3 style=\"margin-top:16px\">Recommendations to validate</h3>";
        for (const auto& item : analysis.value("recommendations", JsonType::array())) {
            out << R"(<div class="hyp"><span class="chip">)"
                << escape_html(item.value("category", ""))
                << "</span><span class=\"chip\">confidence: "
                << escape_html(item.value("confidence", ""))
                << "</span><h3 style=\"margin-top:6px\">" << escape_html(item.value("title", ""))
                << "</h3><p><b>Hypothesis:</b> " << escape_html(item.value("hypothesis", ""))
                << "</p><p><b>Rationale:</b> " << escape_html(item.value("rationale", ""))
                << "</p><p><b>How to validate:</b> " << escape_html(item.value("validation", ""))
                << "</p><div>";
            for (const auto& key : item.value("evidence", JsonType::array())) {
                out << evidence_chip(key.get<std::string>());
            }
            out << "</div></div>";
        }
        for (const auto& [key, title] : std::vector<std::pair<std::string, std::string>>{
                 {"uncertainties", "Uncertainty and limitations"},
                 {"additional_measurements", "Measurements needed before deciding"}}) {
            out << "<h3 style=\"margin-top:16px\">" << title << "</h3><ul>";
            for (const auto& entry : analysis.value(key, JsonType::array())) {
                out << "<li>"
                    << escape_html(entry.is_string() ? entry.get<std::string>() : entry.dump())
                    << "</li>";
            }
            out << "</ul>";
        }
        if (ai.contains("rejected_items") && !ai["rejected_items"].empty()) {
            out << "<details><summary>" << ai["rejected_items"].size()
                << " AI items rejected by validation (missing or unknown evidence, overclaiming, "
                   "malformed)</summary><table>";
            for (const auto& item : ai["rejected_items"]) {
                out << "<tr><td>" << escape_html(item.value("reason", "")) << "</td><td><code>"
                    << escape_html(item.value("item", JsonType()).dump()) << "</code></td></tr>";
            }
            out << "</table></details>";
        }
        out << "</section>";
        return out.str();
    }

    void
    collect_metrics(const JsonType& node, const std::string& path, std::ostringstream& rows) const {
        if (node.is_object()) {
            for (const auto& item : node.items()) {
                const std::string& key = item.key();
                if (key == "definition" || key == "per_dimension" || key == "histogram" ||
                    key == "log10_histogram" || key == "points" || key == "examples" ||
                    key == "top_terms") {
                    continue;
                }
                std::string child = path;
                if (!child.empty()) {
                    child += '.';
                }
                child += key;
                collect_metrics(item.value(), child, rows);
            }
        } else if (node.is_array()) {
            if (node.size() <= 8) {
                for (uint64_t i = 0; i < node.size(); ++i) {
                    collect_metrics(node[i], path + "." + std::to_string(i), rows);
                }
            }
        } else {
            rows << "<tr><td><code>" << escape_html(path) << "</code></td><td class=\"num\">"
                 << text_value(node) << "</td></tr>";
        }
    }

    [[nodiscard]] std::string
    config_section() const {
        std::ostringstream out;
        out << "<section id=\"config\"><h2>Configuration and sampling</h2><div class=\"grid\"><div "
               "class=\"card\"><h3>Options that affect metrics</h3>";
        std::vector<std::pair<std::string, JsonType>> rows;
        for (const auto& item : report_.at("config").items()) {
            rows.emplace_back(
                item.key(), item.value().is_array() ? JsonType(item.value().dump()) : item.value());
        }
        out << table(rows) << "</div><div class=\"card\"><h3>Input and reproducibility</h3>";
        std::vector<std::pair<std::string, JsonType>> input_rows;
        for (const auto& item : report_.at("input").items()) {
            input_rows.emplace_back(item.key(),
                                    item.value().is_array() || item.value().is_object()
                                        ? JsonType(item.value().dump())
                                        : item.value());
        }
        if (report_.contains("determinism")) {
            input_rows.emplace_back("metrics SHA-256",
                                    report_["determinism"].value("metrics_sha256", ""));
            input_rows.emplace_back("excluded sections",
                                    report_["determinism"]["excluded_sections"].dump());
        }
        if (report_.contains("run")) {
            for (const auto& item : report_["run"].items()) {
                input_rows.emplace_back("run." + item.key(), item.value());
            }
        }
        out << table(input_rows)
            << "<p class=\"meta\">Sampling: every sample-based metric uses rows drawn uniformly "
               "without replacement "
               "(Floyd's algorithm) from an xoshiro256** stream derived from the seed and the "
               "split name, then a "
               "seeded shuffle; each analysis uses a prefix of that order. Full-split metrics are "
               "merged in fixed "
               "4096-row blocks, so results do not depend on threads or chunk size.</p>"
            << R"(<p><button id="download-json" type="button" data-name=")"
            << escape_html(report_.at("input").value("file_name", "dataset") + ".profile.json")
            << "\">Download embedded JSON report</button></p></div></div>";
        out << "<h2>Metric index</h2><input id=\"metric-filter\" placeholder=\"Filter metric keys, "
               "e.g. lid or "
               "duplicate\"><table "
               "id=\"metric-index\"><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>";
        std::ostringstream metric_rows;
        for (const char* section : {"splits", "cross_split", "ground_truth", "labels"}) {
            if (report_.contains(section)) {
                collect_metrics(report_[section], section, metric_rows);
            }
        }
        out << metric_rows.str() << "</tbody></table></section>";
        return out.str();
    }

    const JsonType& report_;
    uint64_t seed_{0};
    std::set<std::string> cards_;
};

}  // namespace

std::string
html_content_security_policy() {
    return std::string("default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-") +
           sha256_base64(SCRIPT) +
           "'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'";
}

std::string
render_html_report(const JsonType& report) {
    return Renderer(report).Render();
}

}  // namespace vsag::dataset_profile
