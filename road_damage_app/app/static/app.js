// Road damage dashboard: sends images / video frames to the FastAPI backend and draws the boxes.
const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"];
const MAX_FRAME_WIDTH = 960;

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const video = $("video");

const state = {
  mode: "image",
  classes: [],
  colors: {},
  models: {},
  imageFile: null,
  imageEl: null,
  lastResult: null,
  streaming: false,
  stream: null,
};

// ---------- API ----------
async function loadModels() {
  const badge = $("modelBadge");
  try {
    const res = await fetch("/api/models");
    if (!res.ok) throw new Error(res.statusText);
    const { default: def, models } = await res.json();
    state.models = Object.fromEntries(models.map((m) => [m.id, m]));
    const select = $("modelSelect");
    select.innerHTML = models.map((m) => `<option value="${m.id}">${m.label}</option>`).join("");
    select.value = def;
    select.disabled = models.length < 2;
    // all models share the same classes; colours follow the class, not the model
    state.classes = models[0].classes;
    state.classes.forEach((c, i) => (state.colors[c] = PALETTE[i % PALETTE.length]));
    badge.classList.add("ok");
    updateBadge();
    renderLegend();
    renderClassCounts({});
  } catch (err) {
    $("modelText").textContent = "API offline";
    badge.classList.add("err");
  }
}

function updateBadge() {
  const m = state.models[$("modelSelect").value];
  if (m) $("modelText").textContent = `${m.label} · ${m.device_name}${m.half ? " (FP16)" : ""}`;
}

async function detect(blob) {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");
  form.append("model", $("modelSelect").value);
  form.append("conf", $("conf").value);
  form.append("iou", $("iou").value);
  let res;
  try {
    res = await fetch("/api/detect", { method: "POST", body: form });
  } catch {
    throw new Error("Server not reachable — is run.py still running?");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

// ---------- Drawing ----------
function drawDetections(result) {
  const showLabels = $("showLabels").checked;
  const line = Math.max(2, Math.round(canvas.width / 320));
  const fontSize = Math.max(12, Math.round(canvas.width / 55));
  ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
  ctx.textBaseline = "top";

  for (const d of result.detections) {
    const [x1, y1, x2, y2] = d.box;
    const color = state.colors[d.class_name] || PALETTE[0];
    ctx.strokeStyle = color;
    ctx.lineWidth = line;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    if (showLabels) {
      const text = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`;
      const pad = Math.round(fontSize * 0.3);
      const w = ctx.measureText(text).width + pad * 2;
      const h = fontSize + pad * 2;
      const ty = y1 - h >= 0 ? y1 - h : y1;
      ctx.fillStyle = color;
      ctx.fillRect(x1 - line / 2, ty, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, x1 - line / 2 + pad, ty + pad);
    }
  }
}

function showCanvas() {
  $("emptyState").hidden = true;
  canvas.hidden = false;
  $("downloadBtn").disabled = false;
  $("resetBtn").disabled = false;
}

function resetAll() {
  stopStream();
  state.imageFile = null;
  state.imageEl = null;
  state.lastResult = null;
  $("imageInput").value = "";
  $("videoInput").value = "";
  video.removeAttribute("src");
  video.load();
  $("videoName").textContent = "Frames are sent to the API while it plays";
  $("videoToggle").disabled = true;

  canvas.hidden = true;
  $("emptyState").hidden = false;
  $("downloadBtn").disabled = true;
  $("resetBtn").disabled = true;
  $("totalCount").textContent = "–";
  $("inferMs").textContent = "–";
  $("fps").textContent = "–";
  renderClassCounts({});
  $("detTable").innerHTML = `<tr><td colspan="3" class="muted">No detections yet</td></tr>`;
  setStatus("");
}

function redrawImage() {
  if (!state.imageEl) return;
  canvas.width = state.imageEl.naturalWidth;
  canvas.height = state.imageEl.naturalHeight;
  ctx.drawImage(state.imageEl, 0, 0);
  if (state.lastResult) drawDetections(state.lastResult);
}

// ---------- Results panel ----------
function renderLegend() {
  $("legend").innerHTML = state.classes
    .map((c) => `<span><i class="swatch" style="background:${state.colors[c]}"></i>${c}</span>`)
    .join("");
}

function renderClassCounts(counts) {
  const max = Math.max(1, ...Object.values(counts));
  $("classList").innerHTML = state.classes
    .map((c) => {
      const n = counts[c] || 0;
      return `<li>
        <span class="name"><i class="swatch" style="background:${state.colors[c]}"></i>${c}</span>
        <span class="count">${n}</span>
        <span class="bar"><i style="width:${(n / max) * 100}%;background:${state.colors[c]}"></i></span>
      </li>`;
    })
    .join("");
}

function renderResults(result, fps) {
  $("totalCount").textContent = result.detections.length;
  $("inferMs").textContent = `${result.timing_ms.inference.toFixed(0)} ms`;
  $("fps").textContent = fps ? fps.toFixed(1) : "–";
  renderClassCounts(result.counts);

  const rows = result.detections.map((d) => {
    const [x1, y1, x2, y2] = d.box;
    return `<tr>
      <td><span class="name"><i class="swatch" style="background:${state.colors[d.class_name]}"></i> ${d.class_name}</span></td>
      <td>${(d.confidence * 100).toFixed(1)}%</td>
      <td>${Math.round(x2 - x1)} × ${Math.round(y2 - y1)}</td>
    </tr>`;
  });
  $("detTable").innerHTML = rows.length ? rows.join("") : `<tr><td colspan="3" class="muted">No damage detected</td></tr>`;
}

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

// ---------- Image mode ----------
async function runImage() {
  if (!state.imageFile) return;
  setStatus("Detecting…");
  try {
    const t0 = performance.now();
    state.lastResult = await detect(state.imageFile);
    redrawImage();
    renderResults(state.lastResult);
    setStatus(`${state.lastResult.model_label}: done in ${(performance.now() - t0).toFixed(0)} ms (round trip)`);
  } catch (err) {
    setStatus(err.message, true);
  }
}

function loadImage(file) {
  if (!file || !file.type.startsWith("image/")) return setStatus("Please choose an image file", true);
  stopStream();
  state.imageFile = file;
  state.lastResult = null;
  const img = new Image();
  img.onload = () => {
    state.imageEl = img;
    redrawImage();
    showCanvas();
    runImage();
  };
  img.src = URL.createObjectURL(file);
}

// ---------- Video / webcam mode ----------
function frameBlob() {
  const scale = Math.min(1, MAX_FRAME_WIDTH / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
}

async function streamLoop() {
  let last = performance.now();
  while (state.streaming && !video.paused && !video.ended) {
    try {
      const blob = await frameBlob();
      const result = await detect(blob);
      if (!state.streaming) break;
      drawDetections(result); // canvas still holds the frame that was sent
      const now = performance.now();
      renderResults(result, 1000 / (now - last));
      last = now;
      setStatus(`Live · ${result.model_label}`);
    } catch (err) {
      setStatus(err.message, true);
      break;
    }
  }
  if (video.ended) setStatus("Video finished");
  stopStream(false);
}

async function startStream(source) {
  stopStream();
  state.imageEl = null;
  if (source === "webcam") {
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    } catch (err) {
      return setStatus("Camera not available: " + err.message, true);
    }
    video.srcObject = state.stream;
  }
  await video.play();
  showCanvas();
  state.streaming = true;
  updateStreamButtons();
  streamLoop();
}

function stopStream(pause = true) {
  state.streaming = false;
  if (pause && !video.paused) video.pause();
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    video.srcObject = null;
  }
  updateStreamButtons();
}

function updateStreamButtons() {
  $("videoToggle").textContent = state.streaming && state.mode === "video" ? "Stop" : "Start";
  $("webcamToggle").textContent = state.streaming && state.mode === "webcam" ? "Stop camera" : "Start camera";
}

// ---------- Wiring ----------
document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    stopStream();
    state.mode = tab.dataset.mode;
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab);
    });
    document.querySelectorAll("[data-panel]").forEach((p) => (p.hidden = p.dataset.panel !== state.mode));
    setStatus("");
  })
);

$("imageInput").addEventListener("change", (e) => {
  loadImage(e.target.files[0]);
  e.target.value = ""; // allow choosing the same file again
});
$("resetBtn").addEventListener("click", resetAll);
const dz = $("dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => loadImage(e.dataTransfer.files[0]));

$("videoInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  stopStream();
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  $("videoName").textContent = file.name;
  $("videoToggle").disabled = false;
});
$("videoToggle").addEventListener("click", () => (state.streaming ? stopStream() : startStream("video")));
$("webcamToggle").addEventListener("click", () => (state.streaming ? stopStream() : startStream("webcam")));

let debounce;
for (const id of ["conf", "iou"]) {
  $(id).addEventListener("input", () => {
    $(id + "Out").textContent = Number($(id).value).toFixed(2);
    if (state.mode === "image") {
      clearTimeout(debounce);
      debounce = setTimeout(runImage, 250);
    }
  });
}
$("showLabels").addEventListener("change", () => state.mode === "image" && redrawImage());
$("modelSelect").addEventListener("change", () => {
  updateBadge();
  if (state.mode === "image") runImage(); // re-detect the same image with the new model
});

$("downloadBtn").addEventListener("click", () => {
  const a = document.createElement("a");
  a.download = "road_damage_result.png";
  a.href = canvas.toDataURL("image/png");
  a.click();
});

loadModels();
