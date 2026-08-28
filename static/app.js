/**
 * RT-STAMP-SLR Client App Logic
 * Real-time Webcam loop, Canvas Skeleton overlay, Motion Energy Graph,
 * API requests to FastAPI backend, Language Toggle, and TTS Audio output.
 */

let isWebcamRunning = false;
let mediaStream = null;
let currentLanguage = 'english';
let motionHistory = new Array(50).fill(0);
let thresholdHistory = new Array(50).fill(0.015);
let frameId = 0;
let lastTranslationText = "";

// DOM Elements
const videoEl = document.getElementById('webcamVideo');
const canvasEl = document.getElementById('landmarkCanvas');
const ctx = canvasEl.getContext('2d');
const motionGraphCanvas = document.getElementById('motionGraphCanvas');
const graphCtx = motionGraphCanvas.getContext('2d');

const btnToggleWebcam = document.getElementById('btnToggleWebcam');
const btnSimulateDemo = document.getElementById('btnSimulateDemo');
const btnSimulateSelected = document.getElementById('btnSimulateSelected');
const btnTrainModel = document.getElementById('btnTrainModel');
const btnTTS = document.getElementById('btnTTS');
const btnClear = document.getElementById('btnClear');

const selectVocab = document.getElementById('selectVocab');
const valMotionEnergy = document.getElementById('valMotionEnergy');
const decisionStateBadge = document.getElementById('decisionStateBadge');

const tkHx = document.getElementById('tkHx');
const tkHy = document.getElementById('tkHy');
const tkMx = document.getElementById('tkMx');
const tkMy = document.getElementById('tkMy');
const tkRx = document.getElementById('tkRx');
const tkRy = document.getElementById('tkRy');

const currentLangLabel = document.getElementById('currentLangLabel');
const translatedText = document.getElementById('translatedText');
const wordBufferContainer = document.getElementById('wordBufferContainer');
const valDetectedWord = document.getElementById('valDetectedWord');
const valConfidencePct = document.getElementById('valConfidencePct');
const confidenceBarFill = document.getElementById('confidenceBarFill');
const valThreshold = document.getElementById('valThreshold');

// Init application
document.addEventListener('DOMContentLoaded', () => {
    loadVocabulary();
    initEventListeners();
    drawMotionGraph();
});

function initEventListeners() {
    btnToggleWebcam.addEventListener('click', toggleWebcam);
    btnSimulateDemo.addEventListener('click', () => simulateSign('water'));
    btnSimulateSelected.addEventListener('click', () => {
        const word = selectVocab.value;
        if (word) simulateSign(word);
    });
    btnTrainModel.addEventListener('click', trainModel);
    btnTTS.addEventListener('click', speakTranslation);
    btnClear.addEventListener('click', clearSentence);
}

// Load 50 ISL vocabulary into dropdown
async function loadVocabulary() {
    try {
        const res = await fetch('/api/vocabulary');
        const data = await res.json();
        selectVocab.innerHTML = '';
        data.vocabulary.forEach(item => {
            const opt = document.createElement('option');
            opt.value = item.english;
            opt.textContent = `${item.id + 1}. ${item.display_english} → ${item.tamil}`;
            selectVocab.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load vocabulary:', e);
    }
}

// Webcam controller
async function toggleWebcam() {
    if (isWebcamRunning) {
        stopWebcam();
    } else {
        startWebcam();
    }
}

async function startWebcam() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        videoEl.srcObject = mediaStream;
        isWebcamRunning = true;
        btnToggleWebcam.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Webcam';
        btnToggleWebcam.style.background = '#ef4444';
        requestAnimationFrame(processWebcamLoop);
    } catch (e) {
        alert('Webcam access error: ' + e.message);
    }
}

function stopWebcam() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    isWebcamRunning = false;
    btnToggleWebcam.innerHTML = '<i class="fa-solid fa-video"></i> Start Webcam';
    btnToggleWebcam.style.background = '#3b82f6';
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
}

// Main Frame Processing Loop is defined at bottom of file using live MediaPipe frame capture

// Update UI Widgets
function updateUI(data) {
    // Motion Energy & Threshold
    valMotionEnergy.textContent = data.motion_energy.toFixed(4);
    valThreshold.textContent = data.threshold.toFixed(4);

    // Motion History Graph
    motionHistory.shift();
    motionHistory.push(data.motion_energy);
    thresholdHistory.shift();
    thresholdHistory.push(data.threshold);
    drawMotionGraph();

    // 6D Token Values
    if (data.token && data.token.length >= 6) {
        tkHx.textContent = data.token[0].toFixed(2);
        tkHy.textContent = data.token[1].toFixed(2);
        tkMx.textContent = data.token[2].toFixed(2);
        tkMy.textContent = data.token[3].toFixed(2);
        tkRx.textContent = data.token[4].toFixed(2);
        tkRy.textContent = data.token[5].toFixed(2);
    }

    // Prediction Confidence Bar & Word
    if (data.prediction) {
        valDetectedWord.textContent = data.prediction.word.replace('_', ' ').toUpperCase();
        const confPct = Math.round(data.prediction.confidence * 100);
        valConfidencePct.textContent = `${confPct}%`;
        confidenceBarFill.style.width = `${confPct}%`;
    }

    // Early Decision State Badge
    if (data.early_decision) {
        const state = data.early_decision.state;
        decisionStateBadge.textContent = state;
        if (state === 'CONFIRMED' || data.early_decision.accepted) {
            decisionStateBadge.className = 'decision-badge confirmed';
        } else {
            decisionStateBadge.className = 'decision-badge';
        }
    }

    // Translation Output
    if (data.translation) {
        const text = data.translation.display_text || '<em>Waiting for sign gesture input...</em>';
        translatedText.innerHTML = text;
        lastTranslationText = data.translation.display_text || '';

        // Word Buffer Tags
        updateWordBuffer(data.translation.raw_words || []);
    }
}

function updateWordBuffer(words) {
    // Clear existing tags except buffer label
    const label = wordBufferContainer.querySelector('.buffer-label');
    wordBufferContainer.innerHTML = '';
    wordBufferContainer.appendChild(label);

    words.forEach(w => {
        const tag = document.createElement('span');
        tag.className = 'word-tag';
        tag.textContent = w.replace('_', ' ').toUpperCase();
        wordBufferContainer.appendChild(tag);
    });
}

// Draw Landmark Skeleton on Canvas
function drawSkeleton(hx, hy) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    const w = canvasEl.width;
    const h = canvasEl.height;

    // Body Skeleton Points
    const ls = { x: 0.4 * w, y: 0.35 * h };
    const rs = { x: 0.6 * w, y: 0.35 * h };
    const hand = { x: hx * w, y: hy * h };

    // Draw Bones
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 4;

    // Shoulders Line
    ctx.beginPath();
    ctx.moveTo(ls.x, ls.y);
    ctx.lineTo(rs.x, rs.y);
    ctx.stroke();

    // Right Arm to Hand
    ctx.beginPath();
    ctx.moveTo(rs.x, rs.y);
    ctx.lineTo(hand.x, hand.y);
    ctx.stroke();

    // Draw Joint Nodes
    ctx.fillStyle = '#06b6d4';
    [ls, rs].forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
        ctx.fill();
    });

    // Hand Center Node
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(hand.x, hand.y, 10, 0, 2 * Math.PI);
    ctx.fill();
}

// Draw Waveform Graph
function drawMotionGraph() {
    graphCtx.clearRect(0, 0, motionGraphCanvas.width, motionGraphCanvas.height);
    const w = motionGraphCanvas.width;
    const h = motionGraphCanvas.height;

    const maxVal = 0.1;

    // Draw Motion Energy Line
    graphCtx.strokeStyle = '#3b82f6';
    graphCtx.lineWidth = 2;
    graphCtx.beginPath();
    for (let i = 0; i < motionHistory.length; i++) {
        const x = (i / (motionHistory.length - 1)) * w;
        const y = h - (motionHistory[i] / maxVal) * h;
        if (i === 0) graphCtx.moveTo(x, y);
        else graphCtx.lineTo(x, y);
    }
    graphCtx.stroke();

    // Draw Threshold Line (Amber)
    graphCtx.strokeStyle = '#f59e0b';
    graphCtx.lineWidth = 1;
    graphCtx.setLineDash([4, 4]);
    graphCtx.beginPath();
    for (let i = 0; i < thresholdHistory.length; i++) {
        const x = (i / (thresholdHistory.length - 1)) * w;
        const y = h - (thresholdHistory[i] / maxVal) * h;
        if (i === 0) graphCtx.moveTo(x, y);
        else graphCtx.lineTo(x, y);
    }
    graphCtx.stroke();
    graphCtx.setLineDash([]);
}

// Language Toggle
async function setLanguage(lang) {
    currentLanguage = lang;
    document.getElementById('btnLangEng').classList.toggle('active', lang === 'english');
    document.getElementById('btnLangTam').classList.toggle('active', lang === 'tamil');
    currentLangLabel.textContent = lang === 'english' ? 'ENGLISH SENTENCE' : 'TAMIL TRANSLATION (தமிழ்)';

    try {
        const res = await fetch('/api/toggle_language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: lang })
        });
        const data = await res.json();
        if (data.translation) {
            translatedText.innerHTML = data.translation.display_text || '<em>Waiting for sign gesture input...</em>';
        }
    } catch (e) {
        console.error('Toggle language error:', e);
    }
}

// Clear Sentence Buffer
async function clearSentence() {
    try {
        const res = await fetch('/api/clear_sentence', { method: 'POST' });
        const data = await res.json();
        translatedText.innerHTML = '<em>Waiting for sign gesture input...</em>';
        updateWordBuffer([]);
    } catch (e) {
        console.error('Clear sentence error:', e);
    }
}

// Text-to-Speech (Web Speech API)
function speakTranslation() {
    if (!lastTranslationText) return;
    const utterance = new SpeechSynthesisUtterance(lastTranslationText);
    utterance.lang = currentLanguage === 'english' ? 'en-US' : 'ta-IN';
    window.speechSynthesis.speak(utterance);
}

// Simulate Gesture Sign
async function simulateSign(word) {
    try {
        const res = await fetch('/api/simulate_sign', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word })
        });
        const data = await res.json();
        if (data.last_frame_result) {
            updateUI(data.last_frame_result);
        }
    } catch (e) {
        console.error('Simulate sign error:', e);
    }
}

// Train Model API
async function trainModel() {
    btnTrainModel.disabled = true;
    btnTrainModel.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training...';
    try {
        const res = await fetch('/api/train', { method: 'POST' });
        const data = await res.json();
        alert(`Model trained successfully!\nBest Accuracy: ${(data.training_result.best_val_acc * 100).toFixed(1)}%`);
    } catch (e) {
        alert('Training failed: ' + e.message);
    } finally {
        btnTrainModel.disabled = false;
        btnTrainModel.innerHTML = '<i class="fa-solid fa-brain"></i> Train Model';
    }
}
// Hidden offscreen canvas for capturing webcam JPEG frames
const offscreenCanvas = document.createElement('canvas');
offscreenCanvas.width = 320;
offscreenCanvas.height = 240;
const offscreenCtx = offscreenCanvas.getContext('2d');

// Low-Latency Debug Variables & Queue Lock
let isProcessingFrame = false;
let processedFramesCount = 0;
let droppedFramesCount = 0;
let lastFrameTimestamp = performance.now();
let webcamFps = 0;
let mpFps = 0;
let roundTripLatencyMs = 0;
let latestDebugMetrics = {
    handsCount: 0,
    poseStatus: 'Inactive',
    landmarksCount: 0,
    queueStatus: 'LATEST / 0 BACKLOG'
};

// Main Frame Processing Loop with Queue Protection
async function processWebcamLoop() {
    if (!isWebcamRunning) return;

    const now = performance.now();
    const dtFrame = now - lastFrameTimestamp;
    lastFrameTimestamp = now;
    webcamFps = Math.round(1000 / Math.max(1, dtFrame));

    // CHANGE 1: FRAME QUEUE PROTECTION
    // If a request is already in progress, drop this frame immediately to avoid HTTP backlog
    if (isProcessingFrame) {
        droppedFramesCount++;
        if (isWebcamRunning) {
            setTimeout(() => requestAnimationFrame(processWebcamLoop), 33);
        }
        return;
    }

    frameId++;
    let imageBase64 = '';
    if (videoEl.readyState >= 2) {
        offscreenCtx.drawImage(
            videoEl, 0, 0, offscreenCanvas.width, offscreenCanvas.height
        );
        imageBase64 = offscreenCanvas.toDataURL('image/jpeg', 0.6);
    }

    if (imageBase64) {
        isProcessingFrame = true;
        const captureTime = performance.now();

        try {
            const res = await fetch('/api/process_frame_image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_base64: imageBase64,
                    frame_id: frameId,
                    timestamp_ms: captureTime
                }),
            });

            const data = await res.json();
            processedFramesCount++;
            const renderTime = performance.now();
            roundTripLatencyMs = Math.round(renderTime - captureTime);

            if (data.processing_time_ms) {
                mpFps = Math.round(1000 / Math.max(1, data.processing_time_ms));
            }

            updateUI(data);

            if (data.landmark_data) {
                // Update Debug Metrics
                const numHands = (data.landmark_data.hands || []).length;
                const posePresent = (data.landmark_data.pose && Object.keys(data.landmark_data.pose).length > 0);
                let lmCount = (numHands * 21);
                if (posePresent) lmCount += 6;

                latestDebugMetrics = {
                    handsCount: numHands,
                    poseStatus: posePresent ? 'Active' : 'Inactive',
                    landmarksCount: lmCount,
                    queueStatus: 'LATEST / 0 BACKLOG',
                    bufferStatus: data.buffer_status || 'N/A'
                };

                drawRealLandmarks(data.landmark_data);
            }
        } catch (e) {
            console.error('Frame process error:', e);
        } finally {
            isProcessingFrame = false;
        }
    }

    if (isWebcamRunning) {
        // Stream at 30 FPS without hardcoded artificial delays
        setTimeout(() => requestAnimationFrame(processWebcamLoop), 33);
    }
}

function drawRealLandmarks(landmarkData) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    const w = canvasEl.width;
    const h = canvasEl.height;

    // Draw Pose joints
    if (landmarkData.pose) {
        const p = landmarkData.pose;
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 3;

        if (p.LS && p.RS) {
            ctx.beginPath();
            ctx.moveTo(p.LS[0] * w, p.LS[1] * h);
            ctx.lineTo(p.RS[0] * w, p.RS[1] * h);
            ctx.stroke();
        }

        ctx.fillStyle = '#06b6d4';
        ['LS', 'RS', 'LE', 'RE', 'LW', 'RW'].forEach((k) => {
            if (p[k]) {
                ctx.beginPath();
                ctx.arc(p[k][0] * w, p[k][1] * h, 6, 0, 2 * Math.PI);
                ctx.fill();
            }
        });
    }

    // Draw Hand Keypoints (21 joints per hand)
    if (landmarkData.hands && landmarkData.hands.length > 0) {
        ctx.fillStyle = '#ec4899';
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 2;

        landmarkData.hands.forEach((hand) => {
            hand.forEach((pt) => {
                ctx.beginPath();
                ctx.arc(pt[0] * w, pt[1] * h, 4, 0, 2 * Math.PI);
                ctx.fill();
            });

            if (hand.length >= 21) {
                const wrist = hand[0];
                [4, 8, 12, 16, 20].forEach((tipIdx) => {
                    ctx.beginPath();
                    ctx.moveTo(wrist[0] * w, wrist[1] * h);
                    ctx.lineTo(hand[tipIdx][0] * w, hand[tipIdx][1] * h);
                    ctx.stroke();
                });
            }
        });
    }

    // Draw Debug HUD Overlay
    drawDebugHUD(w, h);
}

function drawDebugHUD(w, h) {
    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(10, 10, 260, 190);
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1;
    ctx.strokeRect(10, 10, 260, 190);

    ctx.font = '11px monospace';
    ctx.fillStyle = '#38bdf8';
    ctx.fillText('=== LANDMARK DEBUG HUD ===', 20, 26);

    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(`1. Webcam FPS:      ${webcamFps}`, 20, 42);
    ctx.fillText(`2. MediaPipe FPS:   ${mpFps}`, 20, 56);
    ctx.fillText(`3. Latency:         ${roundTripLatencyMs} ms`, 20, 70);
    ctx.fillText(`4. Hands Count:     ${latestDebugMetrics.handsCount}`, 20, 84);
    ctx.fillText(`5. Pose Status:     ${latestDebugMetrics.poseStatus}`, 20, 98);
    ctx.fillText(`6. Valid Landmarks: ${latestDebugMetrics.landmarksCount}`, 20, 112);
    ctx.fillText(`7. Processed:       ${processedFramesCount}`, 20, 126);
    ctx.fillText(`8. Dropped Frames:  ${droppedFramesCount}`, 20, 140);
    ctx.fillText(`9. Token Buffer:    ${latestDebugMetrics.bufferStatus || '25/25'}`, 20, 154);

    ctx.fillStyle = '#4ade80';
    ctx.fillText(`10. Queue Status:   ${latestDebugMetrics.queueStatus}`, 20, 168);
    ctx.restore();
}

