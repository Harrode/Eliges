/* Clinical Trial Eligibility Screening System — Vue 3 frontend
 * Same styling (style.css) and backend API as the vanilla version;
 * state and rendering are handled reactively by Vue. */

const { createApp, nextTick } = Vue;

function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatReport(text) {
  var lines = text.split("\n");
  var html = "";
  var inList = false;
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i].trim();
    if (line.match(/^#{1,3}\s+/)) {
      if (inList) { html += "</ul>"; inList = false; }
      html += "<h2>" + esc(line.replace(/^#{1,3}\s+/, "")) + "</h2>";
      continue;
    }
    if (line.match(/^[\-\*]\s+/)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + esc(line.replace(/^[\-\*]\s+/, "")) + "</li>";
      continue;
    }
    if (line.match(/^\d+\.\s+/)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + esc(line.replace(/^\d+\.\s+/, "")) + "</li>";
      continue;
    }
    if (line === "") {
      if (inList) { html += "</ul>"; inList = false; }
      continue;
    }
    if (inList) { html += "</ul>"; inList = false; }
    html += "<p>" + esc(line) + "</p>";
  }
  if (inList) html += "</ul>";
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return html;
}

function formatMarkdown(text) {
  if (!text) return "";
  var html = esc(text);
  html = html.replace(/^### (.+)$/gm, '<h4 style="margin:8px 0 4px;color:#374151">$1</h4>');
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/^[\-\*] (.+)$/gm, '<li style="margin:2px 0">$1</li>');
  html = html.replace(/^\d+\. (.+)$/gm, '<li style="margin:2px 0">$1</li>');
  html = html.replace(/((<li.*<\/li>\s*)+)/g, '<ul style="margin:4px 0;padding-left:20px">$1</ul>');
  html = html.replace(/\n\n/g, "<br>");
  html = html.replace(/\n/g, "<br>");
  return html;
}

function renderPatientDetail(p) {
  var html = "";
  html += "<h3>Basic Info</h3>";
  html += "<div class='patient-info'>";
  html += "<div class='info-item'><div class='info-label'>Name</div><div class='info-value'>" + esc(p.name) + "</div></div>";
  html += "<div class='info-item'><div class='info-label'>Age</div><div class='info-value'>" + p.age + " y/o</div></div>";
  html += "<div class='info-item'><div class='info-label'>Gender</div><div class='info-value'>" + esc(p.gender) + "</div></div>";
  html += "<div class='info-item'><div class='info-label'>Department</div><div class='info-value'>" + esc(p.department) + "</div></div>";
  html += "<div class='info-item'><div class='info-label'>Admission date</div><div class='info-value'>" + esc(p.admission_date || "-") + "</div></div>";
  html += "<div class='info-item'><div class='info-label'>Discharge date</div><div class='info-value'>" + esc(p.discharge_date || "-") + "</div></div>";
  html += "<div class='info-item'><div class='info-label'>Record no.</div><div class='info-value'>" + esc(p.patient_id) + "</div></div>";
  html += "<div class='info-item'><div class='info-label'>Encounter type</div><div class='info-value'>" + esc(p.encounter_type || "-") + "</div></div>";
  html += "</div>";

  html += "<h3>Diagnosis</h3>";
  html += "<div style='margin-bottom:16px'>";
  html += "<div style='font-size:13px;margin-bottom:8px'><b>Primary diagnosis: </b>" + esc(p.diagnosis) + "</div>";
  html += "<div style='font-size:13px;margin-bottom:8px'><b>Chief complaint: </b>" + esc(p.chief_complaint) + "</div>";
  html += "<div style='font-size:13px;margin-bottom:8px'><b>History of present illness: </b>" + esc(p.history_present) + "</div>";
  if (p.past_history) html += "<div style='font-size:13px;margin-bottom:8px'><b>Past history: </b>" + esc(p.past_history) + "</div>";
  html += "</div>";

  if (p.lab_results && p.lab_results.length > 0) {
    html += "<h3>Lab Results</h3>";
    html += "<table class='lab-table'>";
    html += "<thead><tr><th>Test</th><th>Result</th><th>Unit</th></tr></thead><tbody>";
    for (var i = 0; i < p.lab_results.length; i++) {
      var lab = p.lab_results[i];
      html += "<tr><td>" + esc(lab.name) + "</td><td>" + lab.value + "</td><td>" + esc(lab.unit) + "</td></tr>";
    }
    html += "</tbody></table>";
  }

  if (p.vital_signs) {
    html += "<h3>Vital Signs</h3>";
    html += "<div class='vital-signs'>";
    html += "<div class='vital-item'><div class='vital-label'>Systolic BP</div><div class='vital-value'>" + (p.vital_signs.bp_systolic || "-") + "</div></div>";
    html += "<div class='vital-item'><div class='vital-label'>Diastolic BP</div><div class='vital-value'>" + (p.vital_signs.bp_diastolic || "-") + "</div></div>";
    html += "<div class='vital-item'><div class='vital-label'>Heart rate</div><div class='vital-value'>" + (p.vital_signs.heart_rate || "-") + "</div></div>";
    html += "<div class='vital-item'><div class='vital-label'>Temperature</div><div class='vital-value'>" + (p.vital_signs.temperature || "-") + "\u00b0C</div></div>";
    html += "</div>";
  }

  if (p.medications) {
    html += "<h3>Medications</h3>";
    html += "<div style='font-size:13px;margin-bottom:16px;background:var(--panel);padding:12px;border-radius:8px'>" + esc(p.medications) + "</div>";
  }
  if (p.procedure_notes) {
    html += "<h3>Procedure / Surgery Notes</h3>";
    html += "<div style='font-size:13px;margin-bottom:16px;background:var(--panel);padding:12px;border-radius:8px'>" + esc(p.procedure_notes) + "</div>";
  }
  if (p.discharge_summary) {
    html += "<h3>Discharge Summary</h3>";
    html += "<div style='font-size:13px;margin-bottom:16px;background:var(--panel);padding:12px;border-radius:8px'>" + esc(p.discharge_summary) + "</div>";
  }
  return html;
}

createApp({
  data() {
    return {
      query: "",
      searching: false,
      analyzing: false,
      errorMsg: "",
      timingInfo: "",
      currentData: null,
      currentLayer: null,
      versions: [],           // [{id, ...search response}]
      selectedVersions: [],
      compareView: null,
      chatContext: "",
      chatMessages: [{ role: "ai", html: "Hi! I answer questions based on the retrieved results." }],
      chatInput: "",
      chatLoading: false,
      analysisHtml: "",
      editedConditions: "",
      condError: "",
      rerunning: false,
      layerDetailHtml: "<div class='layer-panel'><h3>Click an architecture layer above to view data</h3><div class='desc'>Click Layer 1-5 to view that layer's input, output, method, and intermediate data.</div></div>",
      modalOpen: false,
      modalName: "Patient details",
      modalHtml: "",
      showViz: false,
      model: "deepseek-v4-flash",
      retrieval: "auto",
      index: "emr_cardiometabolic",
      esOk: false,
      esText: "ES: checking...",
      profileName: "\u2014",
      profileId: "generic",
      chartInstances: {},
      archLayers: [
        { id: "nlu", n: "Layer 1", t: "LLM NLU", s: "Natural Language Understanding" },
        { id: "query", n: "Layer 2", t: "Query Building", s: "Conditions to Query" },
        { id: "search", n: "Layer 3", t: "Full-text Retrieval", s: "ES Search Engine" },
        { id: "filter", n: "Layer 4", t: "Rule Filter", s: "Hard-constraint Check" },
        { id: "llm", n: "Layer 5", t: "LLM Interpretation", s: "Analysis & Q&A" }
      ],
      chips: [
        { label: "Temporal: diabetes, last 1mo", q: "Diabetic patients admitted within the last 1 month" },
        { label: "Temporal: 2025-2026 discharge", q: "Hypertension patients discharged between January 2025 and March 2026" },
        { label: "Temporal: MI, last 6mo", q: "Patients with myocardial infarction in the last 6 months" },
        { label: "Negation: diabetes, exclude liver", q: "Include type 2 diabetes patients, exclude those with a history of liver disease" },
        { label: "Negation: CHD, exclude anticoagulant", q: "Include coronary heart disease patients, exclude those on anticoagulants" },
        { label: "Negation: breast cancer, exclude pregnancy", q: "Include breast cancer patients, exclude pregnant or lactating women" },
        { label: "Text: chronic kidney disease", q: "Find all patients diagnosed with chronic kidney disease" },
        { label: "Text: chest pain, cardiology", q: "Find cardiology patients whose chief complaint includes chest pain" },
        { label: "Compound: diabetes + exclude cancer", q: "Include male type 2 diabetes patients aged 40 to 70, exclude those with a malignancy" },
        { label: "Compound: hepatitis + eGFR, exclude immunosuppressant", q: "Include hepatitis patients with eGFR below 60, exclude those on immunosuppressants" }
      ]
    };
  },

  mounted() {
    this.loadConfig();
    this.loadProfiles();
  },

  methods: {
    fmtCount(n) {
      if (n === null || n === undefined) return "\u2014";
      return String(n);
    },
    shortQuery(q) {
      var s = String(q || "");
      return s.length > 12 ? s.substring(0, 12) + "..." : s;
    },
    diagHtml(p) {
      var h = p.highlighted || {};
      return h.diagnosis ? h.diagnosis.join(" ... ") : esc(p.diagnosis);
    },
    chiefHtml(p) {
      var h = p.highlighted || {};
      return h.chief_complaint ? h.chief_complaint.join(" ... ") : esc(p.chief_complaint);
    },
    histHtml(p) {
      var h = p.highlighted || {};
      return h.history_present ? h.history_present.join(" ... ") : "";
    },

    async loadConfig() {
      try {
        var res = await fetch("/api/config");
        if (!res.ok) return;
        var cfg = await res.json();
        this.model = cfg.model;
        this.retrieval = cfg.retrieval;
        this.index = cfg.es_index;
        this.esOk = !!cfg.es_available;
        this.esText = cfg.es_available ? "ES: connected" : "ES: disconnected";
      } catch (e) { /* backend unreachable; status stays "checking" */ }
    },

    async loadProfiles() {
      try {
        var res = await fetch("/api/profiles");
        if (!res.ok) return;
        var data = await res.json();
    var name = data.active_profile || "generic";
        this.profileId = name;
    for (var i = 0; i < (data.profiles || []).length; i++) {
      if (data.profiles[i].profile_id === name) {
        name = data.profiles[i].display_name || name;
        break;
      }
    }
        this.profileName = name;
      } catch (e) { /* ignore */ }
    },

    async changeIndex() {
      var res = await fetch("/api/set-index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index: this.index })
      });
      var cfg = await res.json();
      this.loadProfiles();
    alert("Switched dataset to: " + cfg.es_index + "\nProfile: " + (cfg.cohort_profile || "generic"));
    },

    changeModel() {
      fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: this.model })
      });
    },

    changeRetrieval() {
      fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ retrieval: this.retrieval })
      });
    },

    buildTimingSummary(data) {
  var t = "NLU " + data.timing.nlu_ms + "ms / L3 " + data.timing.search_ms + "ms | " + data.search_backend;
  var pipe = data.retrieval_pipeline;
  if (pipe) {
        t += " | N=" + this.fmtCount(pipe.index_total);
        if (pipe.stage1_candidates != null) t += " \u2192 K1=" + pipe.stage1_candidates;
        t += " \u2192 K2=" + this.fmtCount(pipe.stage2_hits);
        t += " \u2192 L4=" + this.fmtCount(pipe.after_l4_filter);
  }
  if (data.cohort_profile) t += " | profile:" + data.cohort_profile;
  if (data.loop_count) t += " | loop:" + data.loop_count;
  return t;
    },

    buildChatContext(query, data) {
      var ctx = "Query: " + query + "\nHits: " + data.total + "  cases\n";
      for (var i = 0; i < data.results.length; i++) {
        var p = data.results[i];
        ctx += "Patient" + (i + 1) + "\uff1a" + p.name + "\uff0c" + p.age + " y/o\uff0c" + p.gender +
          "\uff0cDepartment\uff1a" + (p.department || "-") + "\uff0cDiagnosis: " + p.diagnosis +
          "\uff0cAdmission: " + (p.admission_date || "-") + "\uff0cDischarge: " + (p.discharge_date || "-");
        if (p.lab_results && p.lab_results.length > 0) {
          ctx += "\uff0cLab Results\uff1a";
          for (var j = 0; j < p.lab_results.length; j++) {
            var lab = p.lab_results[j];
            ctx += lab.name + "=" + lab.value + lab.unit;
            if (j < p.lab_results.length - 1) ctx += "\u3001";
          }
        }
        ctx += "\n";
      }
      return ctx;
    },

    async applySearchData(data, query) {
      this.currentData = data;
      var entry = Object.assign({ id: data.version_id }, data);
      var existing = this.versions.findIndex(function(x) { return x.id === data.version_id; });
      if (existing >= 0) this.versions.splice(existing, 1, entry);
      else this.versions.push(entry);
      this.chatContext = this.buildChatContext(query, data);
      this.timingInfo = this.buildTimingSummary(data);
      await this.renderCharts(data);
    },

    async doSearch() {
      var query = this.query.trim();
      if (!query || this.searching) return;
      this.searching = true;
      this.errorMsg = "";
      this.compareView = null;
      try {
        var res = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: query, series_names: [] })
        });
        if (!res.ok) throw new Error("HTTP " + res.status);
        var data = await res.json();
        await this.applySearchData(data, query);
        this.currentLayer = "search";
        this.showLayer("search");
      } catch (e) {
        this.errorMsg = "Request failed";
        this.currentData = null;
      } finally {
        this.searching = false;
      }
    },

    async rerunWithConditions() {
      if (!this.currentData || this.rerunning) return;
      this.condError = "";
      var conds;
      try {
        conds = JSON.parse(this.editedConditions);
        if (typeof conds !== "object" || conds === null || Array.isArray(conds)) {
          throw new Error("must be a JSON object");
        }
      } catch (e) {
        this.condError = "Invalid JSON: " + e.message;
        return;
      }
      this.rerunning = true;
      this.errorMsg = "";
      this.compareView = null;
      var query = this.currentData.query;
      try {
        var res = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: query, series_names: [], conditions: conds })
        });
        if (!res.ok) throw new Error("HTTP " + res.status);
        var data = await res.json();
        await this.applySearchData(data, query);
        // Stay on the L1 panel so the visitor sees the edited conditions echoed back
        this.showLayer("nlu");
      } catch (e) {
        this.condError = "Request failed";
      } finally {
        this.rerunning = false;
      }
    },

    toggleVersion(vid) {
      var idx = this.selectedVersions.indexOf(vid);
      if (idx >= 0) {
        this.selectedVersions.splice(idx, 1);
      } else {
        if (this.selectedVersions.length >= 2) this.selectedVersions.shift();
        this.selectedVersions.push(vid);
      }
      if (this.selectedVersions.length === 1) {
        var self = this;
        var v = this.versions.find(function(x) { return x.id === self.selectedVersions[0]; });
        if (v) {
          this.compareView = null;
          this.currentData = v;
          this.timingInfo = this.buildTimingSummary(v);
          if (this.currentLayer) this.showLayer(this.currentLayer);
          this.renderCharts(v);
        }
      }
    },

    compareVersions() {
      if (this.selectedVersions.length < 2) { alert("Please select two versions first"); return; }
      var self = this;
      var v1 = this.versions.find(function(x) { return x.id === self.selectedVersions[0]; });
      var v2 = this.versions.find(function(x) { return x.id === self.selectedVersions[1]; });
  var html = "<div class='panel'><h2>Version comparison: V1 vs V2</h2>";
  html += "<div class='kv'>";
  html += "<div class='k'>V1 query</div><div class='v'>" + esc(v1.query) + "</div>";
  html += "<div class='k'>V1 hits</div><div class='v'>" + v1.total + " </div>";
  html += "<div class='k'>V2 query</div><div class='v'>" + esc(v2.query) + "</div>";
  html += "<div class='k'>V2 hits</div><div class='v'>" + v2.total + " </div></div>";
  var v1ids = {}, v2ids = {}, i;
  for (i = 0; i < v1.results.length; i++) v1ids[v1.results[i].id] = true;
  for (i = 0; i < v2.results.length; i++) v2ids[v2.results[i].id] = true;
  var onlyV1 = [], onlyV2 = [], both = [];
  for (i = 0; i < v1.results.length; i++) {
    if (!v2ids[v1.results[i].id]) onlyV1.push(v1.results[i].name);
    else both.push(v1.results[i].name);
  }
  for (i = 0; i < v2.results.length; i++) {
    if (!v1ids[v2.results[i].id]) onlyV2.push(v2.results[i].name);
  }
  html += "<div style='margin-top:12px'>";
  html += "<div class='section-title'>Difference analysis</div>";
      html += "<div class='diff-remove'>V1-only: " + (onlyV1.length ? onlyV1.map(esc).join(", ") : "none") + "</div>";
      html += "<div class='diff-add'>V2-only: " + (onlyV2.length ? onlyV2.map(esc).join(", ") : "none") + "</div>";
      html += "<div style='margin-top:4px;font-size:12px;color:#6b7280'>Common to both: " + (both.length ? both.map(esc).join(", ") : "none") + "</div></div></div>";
      this.compareView = html;
    },

    showLayer(layer) {
      this.currentLayer = layer;
      if (!this.currentData || !this.currentData.layers || !this.currentData.layers[layer]) {
        this.layerDetailHtml = "<div class='layer-panel'><h3>No data</h3><div class='desc'>Please run a search first.</div></div>";
    return;
  }
      var ld = this.currentData.layers[layer];
      if (layer === "nlu") {
        var condsSrc = this.currentData.conditions || ld.output || {};
        this.editedConditions = JSON.stringify(condsSrc, null, 2);
        this.condError = "";
      }
  var html = "<div class='layer-panel'>";
  html += "<h3>" + esc(ld.layer || "Layer") + "</h3>";
  html += "<div class='desc'>" + esc(ld.description || "") + "</div>";

  if (layer === "search" && ld.funnel && ld.funnel.length) {
    html += "<div class='pipeline-funnel' style='margin:10px 0'>";
    for (var fi = 0; fi < ld.funnel.length; fi++) {
      var st = ld.funnel[fi];
          if (fi > 0) html += "<div class='pipeline-arrow'>\u2192</div>";
      html += "<div class='pipeline-step" + (st.id === "K2" ? " highlight" : "") + "'>";
      html += "<div class='step-id'>" + esc(st.id) + "</div>";
          html += "<div class='step-count'>" + this.fmtCount(st.count) + "</div>";
      html += "<div class='step-label'>" + esc(st.label) + "</div>";
      html += "<div class='step-detail'>" + esc(st.detail || "") + "</div>";
      html += "</div>";
    }
    html += "</div>";
  }

  html += "<div class='kv'>";
  var skipKeys = {
    layer: true, description: true, raw_dsl: true, raw_dsl_available: true,
    funnel: true, top_candidates: true
  };
  var lkeys = Object.keys(ld);
  for (var ki = 0; ki < lkeys.length; ki++) {
    var k = lkeys[ki];
    if (skipKeys[k]) continue;
    var v = ld[k];
    if (typeof v === "object") v = JSON.stringify(v, null, 2);
    else v = String(v);
    html += "<div class='k'>" + esc(k) + "</div>";
    if (v.length > 200) html += "<div class='v'><pre class='mono' style='max-height:150px'>" + esc(v) + "</pre></div>";
        else html += "<div class='v'>" + esc(v) + "</div>";
  }
  html += "</div>";

  if (layer === "search" && ld.top_candidates && ld.top_candidates.length) {
    html += "<div class='section-title' style='margin-top:12px'>Stage2 top candidates: " + ld.top_candidates.length + "</div>";
    html += "<div class='kv'>";
    for (var ci = 0; ci < ld.top_candidates.length; ci++) {
      var c = ld.top_candidates[ci];
      html += "<div class='k'>" + esc(c.name || ("#" + (ci + 1))) + "</div>";
          html += "<div class='v'>score " + c.score + " \u00b7 " + esc(c.diagnosis || "") + "</div>";
    }
    html += "</div>";
  }

      var rawDsl = layer === "query" && this.currentData.raw_debug ? this.currentData.raw_debug.query_dsl : null;
  if (rawDsl) {
    var rawText = typeof rawDsl === "object" ? JSON.stringify(rawDsl, null, 2) : String(rawDsl);
    html += "<details style='margin-top:12px'><summary style='cursor:pointer;color:#2563eb;font-size:13px'>View full ES DSL / debug details</summary><pre class='mono' style='max-height:260px;margin-top:8px'>" + esc(rawText) + "</pre></details>";
  }
  html += "</div>";
      this.layerDetailHtml = html;
    },

    async renderCharts(data) {
  if (typeof Chart === "undefined") return;
  if (!data.results || data.results.length === 0) {
        this.showViz = false;
    return;
  }
      this.showViz = true;
      await nextTick();
  var results = data.results;
      var refs = { chartAge: this.$refs.chartAge, chartGender: this.$refs.chartGender, chartDept: this.$refs.chartDept, chartScore: this.$refs.chartScore };
      for (var key in refs) {
        if (this.chartInstances[key]) { this.chartInstances[key].destroy(); this.chartInstances[key] = null; }
      }

  Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans SC', sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.color = "#6b7280";

  var commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
        animation: { duration: 500, easing: "easeOutQuart" }
  };

      // Age distribution (gradient bars)
  var ageBuckets = { "<30": 0, "30-40": 0, "40-50": 0, "50-60": 0, "60-70": 0, "70+": 0 };
  for (var i = 0; i < results.length; i++) {
    var age = results[i].age || 0;
    if (age < 30) ageBuckets["<30"]++;
    else if (age < 40) ageBuckets["30-40"]++;
    else if (age < 50) ageBuckets["40-50"]++;
    else if (age < 60) ageBuckets["50-60"]++;
    else if (age < 70) ageBuckets["60-70"]++;
    else ageBuckets["70+"]++;
  }
      var ageCanvas = refs.chartAge;
  var ageCtx = ageCanvas.getContext("2d");
  var ageGradient = ageCtx.createLinearGradient(0, 0, 0, 180);
  ageGradient.addColorStop(0, "#3b82f6");
  ageGradient.addColorStop(1, "#93c5fd");
      this.chartInstances.chartAge = new Chart(ageCanvas, {
    type: "bar",
    data: {
      labels: Object.keys(ageBuckets),
      datasets: [{
        data: Object.values(ageBuckets),
        backgroundColor: ageGradient,
        borderRadius: 6,
        borderSkipped: false,
        barThickness: 28
      }]
    },
    options: {
      ...commonOptions,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1f2937",
          titleFont: { size: 12 },
          bodyFont: { size: 11 },
          padding: 8,
          cornerRadius: 6,
          callbacks: { label: function(c) { return c.parsed.y + " patients"; } }
        }
      },
      scales: {
        y: { beginAtZero: true, ticks: { stepSize: 1, font: { size: 10 } }, grid: { color: "#f3f4f6", drawBorder: false } },
        x: { grid: { display: false }, ticks: { font: { size: 10 } } }
      }
    }
  });

      // Gender distribution (doughnut)
  var genderCount = {};
      for (var gi = 0; gi < results.length; gi++) { var g = results[gi].gender || "Unknown"; genderCount[g] = (genderCount[g] || 0) + 1; }
      this.chartInstances.chartGender = new Chart(refs.chartGender, {
    type: "doughnut",
    data: {
      labels: Object.keys(genderCount),
      datasets: [{
        data: Object.values(genderCount),
        backgroundColor: ["#3b82f6", "#f43f5e", "#9ca3af"],
        borderWidth: 0,
        hoverOffset: 8
      }]
    },
    options: {
      ...commonOptions,
      cutout: "65%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { padding: 16, usePointStyle: true, pointStyle: "circle", font: { size: 11 } }
        },
        tooltip: {
          backgroundColor: "#1f2937",
          padding: 8,
          cornerRadius: 6,
          callbacks: { label: function(c) { return c.label + ": " + c.parsed + " patients"; } }
        }
      }
    }
  });

      // Department distribution (horizontal bars)
  var deptCount = {};
      for (var di = 0; di < results.length; di++) { var d = results[di].department || "Unknown"; deptCount[d] = (deptCount[d] || 0) + 1; }
  var deptLabels = Object.keys(deptCount);
  var deptColors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"];
      this.chartInstances.chartDept = new Chart(refs.chartDept, {
    type: "bar",
    data: {
      labels: deptLabels,
      datasets: [{
        data: Object.values(deptCount),
        backgroundColor: deptColors.slice(0, deptLabels.length).map(function(c) { return c + "cc"; }),
        borderRadius: 4,
        borderSkipped: false,
        barThickness: 18
      }]
    },
    options: {
      ...commonOptions,
      indexAxis: "y",
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1f2937",
          padding: 8,
          cornerRadius: 6,
          callbacks: { label: function(c) { return c.parsed.x + " patients"; } }
        }
      },
      scales: {
        x: { beginAtZero: true, ticks: { stepSize: 1, font: { size: 10 } }, grid: { color: "#f3f4f6", drawBorder: false } },
        y: { grid: { display: false }, ticks: { font: { size: 11 } } }
      }
    }
  });

      // Match score (bars)
  var scoreLabels = [], scoreData = [], scoreColors = [];
      for (var si = 0; si < results.length; si++) {
        scoreLabels.push(results[si].name || ("P" + (si + 1)));
        scoreData.push(results[si].score);
        var s = results[si].score;
    scoreColors.push(s > 10 ? "#10b981" : s > 5 ? "#f59e0b" : "#9ca3af");
  }
      this.chartInstances.chartScore = new Chart(refs.chartScore, {
    type: "bar",
    data: {
      labels: scoreLabels,
      datasets: [{
        data: scoreData,
        backgroundColor: scoreColors.map(function(c) { return c + "cc"; }),
        borderRadius: 6,
        borderSkipped: false,
        barThickness: 28
      }]
    },
    options: {
      ...commonOptions,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1f2937",
          padding: 8,
          cornerRadius: 6,
          callbacks: { label: function(c) { return "Match score: " + c.parsed.y; } }
        }
      },
      scales: {
        y: { beginAtZero: true, grid: { color: "#f3f4f6", drawBorder: false }, ticks: { font: { size: 10 } } },
        x: { grid: { display: false }, ticks: { font: { size: 10 } } }
      }
    }
  });
    },

    async showPatientDetail(patientId) {
      this.modalName = "Patient details";
      this.modalHtml = "<div style='text-align:center;padding:40px'><span class='spinner'></span> Loading...</div>";
      this.modalOpen = true;
      try {
        var res = await fetch("/api/patient/" + patientId);
        if (!res.ok) throw new Error("HTTP " + res.status);
        var data = await res.json();
        var p = data.patient;
        this.modalName = p.name + " 's clinical record";
        this.modalHtml = renderPatientDetail(p);
      } catch (e) {
        this.modalHtml = "<div class='bad'>Load failed</div>";
      }
    },

    async doAnalyze() {
      if (!this.currentData || this.analyzing) return;
      this.analyzing = true;
      this.analysisHtml = "";
      try {
        var res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: this.currentData.query,
            patient_count: this.currentData.total,
            sample_patients: this.currentData.results.slice(0, 5)
          })
        });
        if (!res.ok) throw new Error("HTTP " + res.status);
        var data = await res.json();
        this.analysisHtml = formatReport(data.analysis || "No report generated");
      } catch (e) {
        this.analysisHtml = "<div class='bad'>Request failed</div>";
      } finally {
        this.analyzing = false;
      }
    },

    async doChat() {
      var q = this.chatInput.trim();
      if (!q || this.chatLoading) return;
      this.chatInput = "";
      this.chatMessages.push({ role: "user", raw: true, text: q });
      this.chatLoading = true;
      this.scrollChat();
      try {
        var res = await fetch("/api/qa-chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, context: this.chatContext || "No retrieval-result context yet." })
        });
        if (!res.ok) throw new Error("HTTP " + res.status);
        var data = await res.json();
        this.chatMessages.push({ role: "ai", html: formatMarkdown(data.answer || "No answer returned") });
      } catch (e) {
        this.chatMessages.push({ role: "ai bad", html: "Request failed" });
      } finally {
        this.chatLoading = false;
        this.scrollChat();
      }
    },

    async scrollChat() {
      await nextTick();
      var box = this.$refs.chatMessages;
      if (box) box.scrollTop = box.scrollHeight;
    }
  }
}).mount("#app");
