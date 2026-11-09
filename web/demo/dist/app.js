"use strict";

const byId = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("ko-KR");
const distance = new Intl.NumberFormat("ko-KR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const latency = new Intl.NumberFormat("ko-KR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const labels = { full: "동네 전체", faiss: "FAISS 200" };
const svgNS = "http://www.w3.org/2000/svg";

function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (className) node.className = className;
  return node;
}

function shape(tag, attributes, content) {
  const node = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (content !== undefined) node.textContent = content;
  return node;
}

function init(data) {
  const state = { key: data.profiles[0].key, radius: 3, policy: "full", ref: null };
  const profileSelect = byId("profile");
  profileSelect.replaceChildren(...data.profiles.map((p) => new Option(p.label, p.key)));
  profileSelect.disabled = false;
  profileSelect.addEventListener("change", () => change("key", profileSelect.value));

  function change(key, value) {
    state[key] = value;
    state.ref = null;
    render();
  }

  function segments(id, key, choices) {
    const buttons = choices.map(([value, label]) => {
      const button = element("button", label);
      button.type = "button";
      button.addEventListener("click", () => change(key, value));
      return button;
    });
    byId(id).replaceChildren(...buttons);
    return () => buttons.forEach((button, i) => button.setAttribute("aria-pressed", choices[i][0] === state[key]));
  }

  const updateRadius = segments("radii", "radius", data.radii.map((km) => [km, `${km} km`]));
  const updatePolicy = segments("policies", "policy", Object.entries(labels));

  function pick(ref) {
    state.ref = ref;
    for (const node of document.querySelectorAll("[data-ref]")) {
      const selected = Number(node.dataset.ref) === ref;
      node.classList.toggle("is-selected", selected);
      node.setAttribute("aria-pressed", selected);
    }
  }

  function renderRadar(items) {
    const svg = byId("radar");
    const center = 200;
    const scale = 31;
    svg.replaceChildren();
    const range = state.radius * scale;
    svg.append(shape("rect", { x: 30, y: 30, width: 340, height: 340, rx: 9, class: "map-grid" }));
    for (let p = 30; p <= 370; p += 34) {
      svg.append(shape("path", { d: `M${p} 30V370 M30 ${p}H370`, class: "map-grid" }));
    }
    svg.append(shape("circle", { cx: center, cy: center, r: range, class: "map-range" }));
    for (let km = 1; km <= 5; km += 1) {
      const r = km * scale;
      if (km !== state.radius) svg.append(shape("circle", { cx: center, cy: center, r, class: "map-ring" }));
      svg.append(shape("text", { x: center + 6, y: center - r - 4, class: "map-tick" }, `${km} km`));
    }
    svg.append(shape("path", { d: "M345 65V47m-4 6 4-6 4 6", fill: "none", stroke: "#d8e6ce" }));
    svg.append(shape("text", { x: 345, y: 80, class: "map-tick", "text-anchor": "middle" }, "N"));
    svg.append(shape("circle", { cx: center, cy: center, r: 4.5, fill: "#f2ad7a" }));

    // Paint rank 1 last so it stays visible when nearby points overlap.
    items.map((item, i) => [item, i + 1]).reverse().forEach(([item, rank]) => {
      if (!item.xy) return;
      const [east, north] = item.xy;
      const x = center + east * scale;
      const y = center - north * scale;
      const dot = shape("g", { class: "map-dot", tabindex: 0, role: "button", "data-ref": item.ref,
        "aria-label": `${rank}위 ${item.category}`, "aria-pressed": "false" });
      dot.append(shape("title", {}, `${rank}위 · ${item.category} · ${distance.format(item.km)} km`));
      dot.append(shape("circle", { cx: x, cy: y, r: 10.5 }));
      dot.append(shape("text", { x, y: y + 3.5, "text-anchor": "middle" }, rank));
      dot.addEventListener("click", () => pick(item.ref));
      dot.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          pick(item.ref);
        }
      });
      svg.append(dot);
    });
  }

  function render() {
    const profile = data.profiles.find((p) => p.key === state.key);
    const results = profile.runs[state.radius];
    const active = results[state.policy];
    const other = results[state.policy === "full" ? "faiss" : "full"];
    const shared = active.items.filter((item) => other.items.some((match) => match.ref === item.ref)).length;
    updateRadius();
    updatePolicy();
    byId("area").textContent = profile.label;
    byId("borough").textContent = `${profile.borough} / NYC`;
    byId("radius-badge").textContent = `${state.radius} km`;
    byId("history").textContent = profile.history.join(" → ");
    byId("candidate-count").replaceChildren(
      `반경 내 ${number.format(active.nearby)}곳 중 `,
      element("strong", `${number.format(active.candidates)}곳`),
      " 재랭킹",
    );
    byId("overlap").textContent = `두 정책의 상위 10곳 중 ${shared}곳이 같습니다.`;
    byId("ranking").replaceChildren(...active.items.map((item, i) => {
      const li = element("li");
      const button = element("button", undefined, "rank-row");
      button.type = "button";
      button.dataset.ref = item.ref;
      button.append(element("span", String(i + 1).padStart(2, "0"), "rank-num"),
        element("span", item.category, "rank-name"), element("span", `${distance.format(item.km)} km`, "rank-distance"));
      button.addEventListener("click", () => pick(item.ref));
      li.append(button);
      return li;
    }));
    renderRadar(active.items);
    pick(state.ref ?? active.items[0]?.ref ?? null);
  }

  const bench = data.benchmark;
  byId("bench-source").textContent = `${bench.cpu} · ${number.format(bench.requests)}건`;
  byId("bench-rows").replaceChildren(...bench.conditions.map((row) => {
    const tr = element("tr");
    for (const value of [row.concurrency, labels[row.policy], `${latency.format(row.p99)} ms`, distance.format(row.rps)]) {
      tr.append(element("td", value));
    }
    return tr;
  }));
  render();
}

fetch("./demo-data.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then(init)
  .catch(() => { byId("candidate-count").textContent = "데모 데이터를 불러오지 못했습니다. 페이지를 새로고침해 주세요."; });
