console.log("🔥 JS IS RELOADED", Date.now());


const BASE_WIDTH = 832;
const BASE_HEIGHT = 480;
const DISPLAY_RESOLUTIONS = {
    "832x480": { width: 832, height: 480 },
    "640x640": { width: 640, height: 640 },
};

const DEFAULT_MODEL_LIMIT = 16;
const DEFAULT_USER_LIMIT = 20;
const DEFAULT_ALIGNMENT = 12;
const DEFAULT_OVERLAP = 33;

const state = {
    sessionId: null,
    modelLimit: DEFAULT_MODEL_LIMIT,
    userLimit: DEFAULT_USER_LIMIT,
    frameAlignment: DEFAULT_ALIGNMENT,
    overlapFrames: DEFAULT_OVERLAP,
    fps: 15,
    recording: false,
    frames: [],
    captureTimer: null,
    brushSize: 6,
    brushColor: "#0f172a",
    tool: "brush",
    eraserSize: 24,
    drawing: false,
    playbackActive: false,
    playbackRaf: null,
    lastFrameUrl: null,
    lastPrompt: null,
    lastRawPrompt: null,
    lastRefinedPrompt: null,
    lastPoint: null,
};

const pageDataset = document.body?.dataset || {};
const refineOnRun = pageDataset.refineOnRun === "true";
const keepRawPrompt = pageDataset.keepRawPrompt === "true" || refineOnRun;

const promptInput = document.getElementById("promptInput");

const runBtn = document.getElementById("runBtn");
const refinePromptBtn = document.getElementById("refinePromptBtn");
const clearCanvasBtn = document.getElementById("clearCanvasBtn");
const statusBox = document.getElementById("statusBox");
const modelFramesInput = document.getElementById("modelFrames");
const resolutionSelect = document.getElementById("resolutionSelect");
const seedInput = document.getElementById("seedInput");
const overlapFramesInput = document.getElementById("overlapFrames");
const modelFramesValue = document.getElementById("modelFramesValue");
const overlapFramesValue = document.getElementById("overlapFramesValue");
const brushSizeInput = document.getElementById("brushSize");
const brushColorInput = document.getElementById("brushColor");
const brushSizeValue = document.getElementById("brushSizeValue");
const toolBrushBtn = document.getElementById("toolBrushBtn");
const toolEraserBtn = document.getElementById("toolEraserBtn");
const eraserSizeInput = document.getElementById("eraserSize");
const eraserSizeValue = document.getElementById("eraserSizeValue");
function updateBrushSizeDisplay(value) {
    if (brushSizeValue) {
        brushSizeValue.textContent = `${value} px`;
    }
}
function updateEraserSizeDisplay(value) {
    if (eraserSizeValue) {
        eraserSizeValue.textContent = `${value} px`;
    }
}
function updateModelFramesDisplay(value) {
    if (modelFramesValue) {
        const parsed = parseInt(value, 10);
        modelFramesValue.textContent = `${Number.isFinite(parsed) ? parsed : state.modelLimit}`;
    }
}
function updateOverlapFramesDisplay(value) {
    if (overlapFramesValue) {
        const parsed = parseInt(value, 10);
        overlapFramesValue.textContent = `${Number.isFinite(parsed) ? parsed : state.overlapFrames}`;
    }
}

const canvas = document.getElementById("sketchCanvas");
const ctx = canvas.getContext("2d");
const FRAME_MIME = "image/webp";
const FRAME_QUALITY = 0.82;
const previewVideo = document.createElement("video");
previewVideo.muted = true;
previewVideo.playsInline = true;
previewVideo.crossOrigin = "anonymous";
previewVideo.style.display = "none";
document.body.appendChild(previewVideo);
previewVideo.addEventListener("ended", () => {
    stopPlayback(true);
    if (state.sessionId) {
        fetch(`/api/session/${state.sessionId}`)
            .then((res) => (res.ok ? res.json() : null))
            .then((info) => {
                if (info?.last_frame_url) {
                    applyLastFrame(info.last_frame_url);
                }
                if (info?.frame_alignment) {
                    state.frameAlignment = info.frame_alignment;
                }
                if (info?.overlap_frames) {
                    state.overlapFrames = info.overlap_frames;
                }
            });
    }
});
previewVideo.addEventListener("error", () => {
    stopPlayback(true);
    setStatus("Preview playback failed");
});

const backgroundImage = new Image();
const backgroundCanvas = document.createElement("canvas");
const backgroundCtx = backgroundCanvas.getContext("2d");
backgroundImage.onload = () => {
    renderBackground();
};

function captureFrameDataURL() {
    return canvas.toDataURL(FRAME_MIME, FRAME_QUALITY);
}

function setStatus(text) {
    statusBox.textContent = text;
}

function resizeCanvasForDisplay() {
    stopPlayback(false);
    const key = resolutionSelect.value;
    const display = DISPLAY_RESOLUTIONS[key];
    canvas.width = display.width;
    canvas.height = display.height;
    backgroundCanvas.width = display.width;
    backgroundCanvas.height = display.height;
    canvas.style.width = `${display.width}px`;
    canvas.style.height = `${display.height}px`;
    ctx.imageSmoothingEnabled = false;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    // ctx.fillStyle = "#ffffff";
    // ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawBackgroundTexture();
    if (state.lastFrameUrl) {
        applyLastFrame(state.lastFrameUrl);
    }
}

function renderBackground() {
    if (!backgroundImage.complete || !backgroundImage.naturalWidth) {
        return;
    }
    backgroundCtx.clearRect(0, 0, backgroundCanvas.width, backgroundCanvas.height);
    backgroundCtx.drawImage(backgroundImage, 0, 0, backgroundCanvas.width, backgroundCanvas.height);
    ctx.drawImage(backgroundCanvas, 0, 0, canvas.width, canvas.height);
}

function drawBackgroundTexture() {
    if (!backgroundImage.src) {
        backgroundImage.src = "/static/blank_canvas_white.jpg";
        return;
    }
    renderBackground();
}

function stopPlayback(applyFrame = false) {
    if (state.playbackRaf) {
        cancelAnimationFrame(state.playbackRaf);
        state.playbackRaf = null;
    }
    if (state.playbackActive) {
        previewVideo.pause();
    }
    state.playbackActive = false;
    canvas.classList.remove("is-playing");
    if (applyFrame && state.lastFrameUrl) {
        applyLastFrame(state.lastFrameUrl);
    }
    updateCursorStyle();
}

function buildCursorDataUrl(size, color) {
    const clamped = Math.max(6, Math.min(96, Math.round(size)));
    const radius = Math.max(2, Math.floor(clamped / 2) - 1);
    const center = clamped / 2;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${clamped}" height="${clamped}" viewBox="0 0 ${clamped} ${clamped}"><circle cx="${center}" cy="${center}" r="${radius}" fill="none" stroke="${color}" stroke-width="2"/></svg>`;
    const encoded = encodeURIComponent(svg);
    const hot = Math.floor(clamped / 2);
    return `url("data:image/svg+xml;utf8,${encoded}") ${hot} ${hot}, auto`;
}

function updateCursorStyle() {
    if (state.playbackActive) {
        canvas.style.cursor = "not-allowed";
        return;
    }
    if (state.tool === "eraser") {
        canvas.style.cursor = buildCursorDataUrl(state.eraserSize, "#0f172a");
        return;
    }
    canvas.style.cursor = buildCursorDataUrl(state.brushSize, state.brushColor);
}

function setTool(tool) {
    state.tool = tool;
    if (toolBrushBtn) {
        toolBrushBtn.classList.toggle("is-active", tool === "brush");
    }
    if (toolEraserBtn) {
        toolEraserBtn.classList.toggle("is-active", tool === "eraser");
    }
    updateCursorStyle();
}

function eraseAt(x, y) {
    if (!backgroundCanvas.width || !backgroundCanvas.height) {
        return;
    }
    const radius = Math.max(1, state.eraserSize / 2);
    const size = radius * 2;
    const sx = Math.max(0, x - radius);
    const sy = Math.max(0, y - radius);
    const sw = Math.min(size, backgroundCanvas.width - sx);
    const sh = Math.min(size, backgroundCanvas.height - sy);
    const dx = sx;
    const dy = sy;
    ctx.drawImage(backgroundCanvas, sx, sy, sw, sh, dx, dy, sw, sh);
}

function eraseLine(from, to) {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const distance = Math.hypot(dx, dy);
    const step = Math.max(1, state.eraserSize / 2);
    const steps = Math.max(1, Math.floor(distance / step));
    for (let i = 0; i <= steps; i += 1) {
        const t = i / steps;
        eraseAt(from.x + dx * t, from.y + dy * t);
    }
}

function drawVideoFrame() {
    if (!state.playbackActive) {
        return;
    }
    ctx.drawImage(previewVideo, 0, 0, canvas.width, canvas.height);
    state.playbackRaf = requestAnimationFrame(drawVideoFrame);
}

function startPlayback(url) {
    if (!url) {
        return;
    }
    stopPlayback(false);
    const source = `${url}?t=${Date.now()}`;
    previewVideo.onloadeddata = () => {
        previewVideo.onloadeddata = null;
        previewVideo
            .play()
            .then(() => {
                state.playbackActive = true;
                canvas.classList.add("is-playing");
                updateCursorStyle();
                drawVideoFrame();
            })
            .catch(() => {
                stopPlayback(true);
                setStatus("Preview playback failed");
            });
    };
    previewVideo.src = source;
    previewVideo.load();
}

function getCanvasPos(evt) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: ((evt.clientX || evt.touches?.[0]?.clientX) - rect.left) * (canvas.width / rect.width),
        y: ((evt.clientY || evt.touches?.[0]?.clientY) - rect.top) * (canvas.height / rect.height),
    };
}

function startRecording() {
    if (state.recording) {
        return;
    }
    state.recording = true;
    state.frames.push(captureFrameDataURL())
    const interval = Math.max(40, Math.floor(1000 / state.fps));
    state.captureTimer = setInterval(() => {
        // if (state.frames.length >= state.userLimit) {
        //     stopRecording();
        //     return;
        // }
        state.frames.push(captureFrameDataURL());
    }, interval);
}

function stopRecording() {
    if (!state.recording) {
        return;
    }
    clearInterval(state.captureTimer);
    state.captureTimer = null;
    state.recording = false;
}

function limitFrames(frames, limit) {
    if (frames.length <= limit) {
        return frames;
    }
    const result = [];
    const step = frames.length / limit;
    for (let i = 0; i < limit; i += 1) {
        result.push(frames[frames.length - 1 - Math.floor(i * step)]);
    }
    return result.reverse();
}

function enforceFrameAlignment(frames) {
    const alignment = Math.max(1, state.frameAlignment || 1);
    if (alignment === 1 || frames.length === 0) {
        return frames.slice();
    }
    const aligned = frames.slice();
    const remainder = aligned.length % alignment;
    if (remainder === 0) {
        return aligned;
    }
    // Pad to next alignment boundary by repeating the last frame
    const fill = aligned[aligned.length - 1];
    const needed = alignment - remainder;
    for (let i = 0; i < needed; i++) {
        aligned.push(fill);
    }
    return aligned;
}

function prepareFramesForUpload(frames) {
    const alignment = Math.max(1, state.frameAlignment || 1);
    const limit = Math.max(state.userLimit || alignment, alignment);
    const maxAlignedLimit = Math.max(alignment, Math.floor(limit / alignment) * alignment);
    const capped = limitFrames(frames, maxAlignedLimit);
    return enforceFrameAlignment(capped);
}

function normalizeOverlap(value) {
    const parsed = parseInt(value, 10);
    const base = Number.isFinite(parsed) ? parsed : state.overlapFrames || DEFAULT_OVERLAP;
    const clamped = Math.max(1, base);
    return 4 * Math.floor((clamped - 1) / 4) + 1;
}

async function updateOverlapFrames(value) {
    const normalized = normalizeOverlap(value);
    overlapFramesInput.value = normalized;
    state.overlapFrames = normalized;
    updateOverlapFramesDisplay(normalized);
    setStatus(`Updating overlap to ${normalized}`);
    const res = await fetch("/api/overlap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ overlap_frames: normalized }),
    });
    if (!res.ok) {
        setStatus("Overlap update failed");
        return;
    }
    const data = await res.json();
    if (data?.overlap_frames) {
        state.overlapFrames = data.overlap_frames;
    }
    setStatus(`Overlap frames: ${state.overlapFrames}`);
}

async function createSession(prompt) {
    const payload = {
        prompt,
        width: BASE_WIDTH,
        height: BASE_HEIGHT,
        fps: state.fps,
    };
    const res = await fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    if (!res.ok) {
        throw new Error("Failed to create session");
    }
    const data = await res.json();
    state.sessionId = data.session_id;
    state.modelLimit = data.model_frame_limit;
    state.userLimit = data.user_frame_limit;
    state.frameAlignment = data.frame_alignment || 1;
    // state.overlapFrames = data.overlap_frames || state.overlapFrames || DEFAULT_OVERLAP;
    setStatus("Session ready");
}

async function ensureSession(prompt) {
    if (state.sessionId) {
        return;
    }
    await createSession(prompt);
}

async function ensureSessionAndPrompt(prompt) {
    // Make sure we have a session first
    await ensureSession(prompt);
    if (!state.sessionId) {
        setStatus("!state.sessionId");
        return;
    }

    // If the prompt changed since last time, update it on the server
    if (state.lastPrompt !== prompt) {
        setStatus("Updating prompt");
        const res = await fetch(`/api/session/${state.sessionId}/prompt`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt }),
        });
        if (!res.ok) {
            setStatus("Failed to update prompt");
            return;
        }
        state.lastPrompt = prompt;
    }
}

// test

function updateVideoSource(url) {
    startPlayback(url);
}

function applyLastFrame(url) {
    if (!url) {
        return;
    }
    state.lastFrameUrl = url;
    if (state.playbackActive) {
        return;
    }
    const img = new Image();
    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = `${url}?t=${Date.now()}`;
}

function resetSessionState() {
    state.sessionId = null;
    state.modelLimit = DEFAULT_MODEL_LIMIT;
    state.userLimit = DEFAULT_USER_LIMIT;
    state.frameAlignment = DEFAULT_ALIGNMENT;
    state.overlapFrames = DEFAULT_OVERLAP;
    state.tool = "brush";
    state.eraserSize = 24;
    state.lastFrameUrl = null;
    // modelFramesInput.max = state.modelLimit;
    // modelFramesInput.value = Math.min(
    //     parseInt(modelFramesInput.value || state.modelLimit, 10),
    //     state.modelLimit,
    // );
    // updateModelFramesDisplay(modelFramesInput.value);
    // overlapFramesInput.value = state.overlapFrames;
    // updateOverlapFramesDisplay(state.overlapFrames);
    // if (eraserSizeInput) {
    //     eraserSizeInput.value = state.eraserSize;
    // }
    updateEraserSizeDisplay(state.eraserSize);
    setTool(state.tool);
}

async function resetServerSession() {
    if (!state.sessionId) {
        resetSessionState();
        return true;
    }
    const res = await fetch(`/api/session/${state.sessionId}`, { method: "DELETE" });
    if (!res.ok) {
        return false;
    }
    resetSessionState();
    return true;
}

async function appendSketchIfNeeded() {
    if (!state.sessionId) {
        return true;
    }
    if (state.recording) {
        stopRecording();
        throw new Error("Recording is active");
    }
    if (state.frames.length === 0) {
        return true;
    }
    // state.frames.push(captureFrameDataURL());
    const frames = prepareFramesForUpload(state.frames);
    setStatus("Appending sketch...");
    const res = await fetch(`/api/session/${state.sessionId}/sketch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames, overlap_frames: state.overlapFrames }),
    });
    if (!res.ok) {
        setStatus("Sketch upload failed");
        return false;
    }
    const data = await res.json();
    state.lastFrameUrl = data.last_frame_url;
    // updateVideoSource(data.segment_url);
    setStatus(`Sketch appended (${data.frames}), total ${data.total_frames}`);
    state.frames = [];
    return true;
}

async function refinePrompt() {
    const raw = promptInput.value.trim();
    if (!raw) {
        setStatus("Prompt is required");
        return;
    }
    if (refinePromptBtn) {
        refinePromptBtn.disabled = true;
    }
    if (runBtn) {
        runBtn.disabled = true;
    }
    let refined = null;
    if (state.lastRawPrompt === raw && state.lastRefinedPrompt) {
        refined = state.lastRefinedPrompt;
    } else {
        setStatus("Refining prompt...");
        refined = await requestRefinedPrompt(raw);
        if (refined) {
            state.lastRawPrompt = raw;
            state.lastRefinedPrompt = refined;
        }
    }
    if (!refined) {
        if (refinePromptBtn) {
            refinePromptBtn.disabled = false;
        }
        if (runBtn) {
            runBtn.disabled = false;
        }
        return;
    }
    if (!keepRawPrompt) {
        promptInput.value = refined;
        state.lastPrompt = null;
    }
    if (refinePromptBtn) {
        refinePromptBtn.disabled = false;
    }
    if (runBtn) {
        runBtn.disabled = false;
    }
    setStatus("Prompt refined");
}

async function requestRefinedPrompt(raw) {
    const res = await fetch("/api/prompt/refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: raw }),
    });
    if (!res.ok) {
        setStatus("Refine failed");
        return null;
    }
    const data = await res.json();
    return data?.prompt || null;
}

async function runModel() {
    const rawPrompt = promptInput.value.trim();
    if (!rawPrompt) {
        setStatus("Prompt is required");
        return;
    }
    let prompt = rawPrompt;
    if (refineOnRun) {
        let refined = null;
        if (state.lastRawPrompt === rawPrompt && state.lastRefinedPrompt) {
            refined = state.lastRefinedPrompt;
        } else {
            setStatus("Refining prompt...");
            refined = await requestRefinedPrompt(rawPrompt);
            if (refined) {
                state.lastRawPrompt = rawPrompt;
                state.lastRefinedPrompt = refined;
            }
        }
        if (!refined) {
            return;
        }
        prompt = refined;
    }
    await ensureSessionAndPrompt(prompt);
    const appended = await appendSketchIfNeeded();
    if (!appended) {
        return;
    }
    const framesRequested = Math.max(1, Math.min(
        parseInt(modelFramesInput.value || state.modelLimit, 10),
        state.modelLimit,
    ));
    const payload = { frames: framesRequested, overlap_frames: state.overlapFrames };
    if (seedInput) {
        const seedValue = seedInput.value.trim();
        if (seedValue !== "") {
            const parsedSeed = parseInt(seedValue, 10);
            if (!Number.isNaN(parsedSeed)) {
                payload.seed = parsedSeed;
            }
        }
    }
    setStatus("Generating...");
    const res = await fetch(`/api/session/${state.sessionId}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    if (!res.ok) {
        setStatus("Generation failed");
        return;
    }
    const data = await res.json();
    state.lastFrameUrl = data.last_frame_url;
    updateVideoSource(data.segment_url);
    setStatus(`Generated ${data.frames} frames, total ${data.total_frames}`);
}

async function clearCanvas() {
    // Stop any ongoing recording or playback
    stopRecording();
    stopPlayback(false);

    // Clear the drawing
    // ctx.fillStyle = "#ffffff";
    // ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawBackgroundTexture(); 
    state.frames = [];
    state.lastFrameUrl = null;

    const hasSession = Boolean(state.sessionId);
    
    if (!hasSession) {
        // No server session yet → just a local clear
        setStatus("Canvas cleared");
        return;
    }
    // If we *do* have a server session, delete it completely
    setStatus("Resetting session...");
    const ok = await resetServerSession();
    if (!ok) {
        setStatus("Session reset failed");
        return;
    }

    // Now there is no sessionId on the client, so the next Run will create a NEW session
    setStatus("Canvas cleared. New session will start on next Run.");
    // setStatus("Clearing session cache...");
    // const res = await fetch(`/api/session/${state.sessionId}/clear`, { method: "POST" });
    // if (!res.ok) {
    //     setStatus("Session clear failed");
    //     return;
    // }
    // setStatus("Canvas cleared (session cache reset)");
}

function onPointerDown(evt) {
    evt.preventDefault();
    if (state.playbackActive) {
        stopPlayback(false);
    }
    state.drawing = true;
    const { x, y } = getCanvasPos(evt);
    state.lastPoint = { x, y };
    if (state.tool === "brush") {
        ctx.strokeStyle = state.brushColor;
        ctx.lineWidth = state.brushSize;
        ctx.beginPath();
        ctx.moveTo(x, y);
    } else {
        eraseAt(x, y);
    }
    startRecording();
}

function onPointerMove(evt) {
    if (!state.drawing) {
        return;
    }
    evt.preventDefault();
    const { x, y } = getCanvasPos(evt);
    if (state.tool === "brush") {
        ctx.lineTo(x, y);
        ctx.stroke();
    } else if (state.lastPoint) {
        eraseLine(state.lastPoint, { x, y });
    }
    state.lastPoint = { x, y };
}

function finishStroke(evt) {
    if (!state.drawing) {
        return;
    }
    evt?.preventDefault();
    if (state.tool === "brush") {
        ctx.closePath();
    }
    state.drawing = false;
    state.lastPoint = null;
    stopRecording();
}

function bindCanvasEvents() {
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", finishStroke);
    canvas.addEventListener("pointerleave", finishStroke);
    canvas.addEventListener("pointercancel", finishStroke);
}

function initControls() {
    runBtn.addEventListener("click", runModel);
    if (refinePromptBtn) {
        refinePromptBtn.addEventListener("click", refinePrompt);
    }
    clearCanvasBtn.addEventListener("click", clearCanvas);
    resolutionSelect.addEventListener("change", resizeCanvasForDisplay);
    brushSizeInput.addEventListener("input", () => {
        state.brushSize = parseInt(brushSizeInput.value, 10);
        updateBrushSizeDisplay(state.brushSize);
        updateCursorStyle();
    });
    brushColorInput.addEventListener("input", () => {
        state.brushColor = brushColorInput.value;
        updateCursorStyle();
    });
    if (toolBrushBtn) {
        toolBrushBtn.addEventListener("click", () => setTool("brush"));
    }
    if (toolEraserBtn) {
        toolEraserBtn.addEventListener("click", () => setTool("eraser"));
    }
    if (eraserSizeInput) {
        eraserSizeInput.addEventListener("input", () => {
            state.eraserSize = parseInt(eraserSizeInput.value, 10);
            updateEraserSizeDisplay(state.eraserSize);
            updateCursorStyle();
        });
    }
    modelFramesInput.addEventListener("input", () => {
        const value = parseInt(modelFramesInput.value || state.modelLimit, 10);
        updateModelFramesDisplay(value);
    });
    overlapFramesInput.addEventListener("input", () => {
        const value = parseInt(overlapFramesInput.value || state.overlapFrames, 10);
        updateOverlapFramesDisplay(value);
    });
    overlapFramesInput.addEventListener("change", () => {
        updateOverlapFrames(overlapFramesInput.value);
    });
    updateBrushSizeDisplay(parseInt(brushSizeInput.value, 10));
    updateEraserSizeDisplay(parseInt(eraserSizeInput.value, 10));
    setTool(state.tool);
}

function bootstrap() {
    resizeCanvasForDisplay();
    drawBackgroundTexture(); 
    bindCanvasEvents();
    initControls();
    overlapFramesInput.value = state.overlapFrames;
    updateModelFramesDisplay(modelFramesInput.value || state.modelLimit);
    updateOverlapFramesDisplay(state.overlapFrames);
    setStatus("Waiting for prompt");
}

bootstrap();

