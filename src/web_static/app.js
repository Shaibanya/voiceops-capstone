const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");
const messageList = document.querySelector("#message-list");
const welcome = document.querySelector("#welcome");
const sendButton = document.querySelector("#send-button");
const audioInput = document.querySelector("#audio-input");
const recordButton = document.querySelector("#record-button");
const recordLabel = document.querySelector("#record-label");
const recordingTimer = document.querySelector("#recording-timer");
const toast = document.querySelector("#toast");
const statusLabel = document.querySelector("#status-label");
const serviceStatus = document.querySelector(".service-status");

let toastTimer;
let recording = null;
const MAX_RECORDING_SECONDS = 60;

function showError(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 4200);
}

function addMessage(role, text, options = {}) {
  welcome.hidden = true;
  const row = document.createElement("article");
  row.className = `message-row ${role}`;

  if (role === "assistant") {
    const avatar = document.createElement("span");
    avatar.className = "message-avatar";
    avatar.textContent = "Z";
    avatar.setAttribute("aria-hidden", "true");
    row.append(avatar);
  }

  const content = document.createElement("div");
  content.className = "message-content";
  const meta = document.createElement("p");
  meta.className = "message-meta";
  meta.textContent = role === "user" ? "You" : "ZentroMart AI";
  content.append(meta);

  if (options.transcript !== undefined) {
    const transcript = document.createElement("div");
    transcript.className = "transcript-label";
    transcript.textContent = "Voice message · transcript";
    content.append(transcript);
    addBubble(content, options.transcript);
    meta.textContent = "You · voice";
  } else {
    addBubble(content, text);
  }

  if (role === "assistant" && options.evidence?.length) {
    const citations = document.createElement("div");
    citations.className = "evidence-list";
    for (const source of options.evidence) {
      const chip = document.createElement("span");
      chip.className = "evidence-chip";
      chip.textContent = source;
      citations.append(chip);
    }
    content.append(citations);
  }

  if (options.audioUrl) {
    const player = document.createElement("audio");
    player.className = "audio-player";
    player.controls = true;
    player.src = options.audioUrl;
    player.setAttribute("aria-label", "AI voice reply");
    content.append(player);
  }

  row.append(content);
  messageList.append(row);
  row.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function addBubble(parent, text) {
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  parent.append(bubble);
}

function addTyping() {
  const row = document.createElement("article");
  row.className = "message-row assistant";
  row.id = "typing-indicator";
  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.textContent = "Z";
  const dots = document.createElement("div");
  dots.className = "message-bubble typing";
  dots.innerHTML = "<i></i><i></i><i></i>";
  row.append(avatar, dots);
  welcome.hidden = true;
  messageList.append(row);
  row.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function removeTyping() {
  document.querySelector("#typing-indicator")?.remove();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The request could not be completed.");
  return result;
}

async function sendMessage(message) {
  const cleanMessage = message.trim();
  if (!cleanMessage) return;
  addMessage("user", cleanMessage);
  messageInput.value = "";
  messageInput.style.height = "auto";
  sendButton.disabled = true;
  addTyping();
  try {
    const result = await postJson("/api/chat", { message: cleanMessage });
    addMessage("assistant", result.reply, { evidence: result.evidence });
  } catch (error) {
    showError(error.message);
  } finally {
    removeTyping();
    sendButton.disabled = false;
    messageInput.focus();
  }
}

function wavFile(samples, sampleRate) {
  const dataLength = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);
  const writeText = (offset, text) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, dataLength, true);
  let offset = 44;
  for (const sample of samples) {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
    offset += 2;
  }
  return new File([buffer], "microphone.wav", { type: "audio/wav" });
}

async function sendAudioFile(file) {
  const formData = new FormData();
  formData.append("audio", file);
  addTyping();
  recordButton.disabled = true;
  try {
    const response = await fetch("/api/voice", { method: "POST", body: formData });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The audio could not be processed.");
    addMessage("user", "", { transcript: result.transcript || "No speech detected." });
    const bytes = Uint8Array.from(atob(result.audio_base64), (char) => char.charCodeAt(0));
    const audioUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    addMessage("assistant", result.reply, {
      evidence: result.evidence,
      audioUrl,
    });
  } catch (error) {
    showError(error.message);
  } finally {
    removeTyping();
    recordButton.disabled = false;
  }
}

function setRecordingUi(isRecording) {
  recordButton.classList.toggle("recording", isRecording);
  recordButton.setAttribute("aria-pressed", String(isRecording));
  recordLabel.textContent = isRecording ? "Stop & send" : "Record voice";
  recordingTimer.textContent = isRecording ? "00:00" : "";
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) {
    showError("Microphone recording isn’t supported by this browser. You can upload an audio file instead.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const chunks = [];
    let active = true;
    processor.onaudioprocess = (event) => {
      if (!active) return;
      chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      event.outputBuffer.getChannelData(0).fill(0);
    };
    source.connect(processor);
    processor.connect(context.destination);

    const startedAt = Date.now();
    const timerId = window.setInterval(() => {
      const elapsed = Math.floor((Date.now() - startedAt) / 1000);
      recordingTimer.textContent = `00:${String(elapsed).padStart(2, "0")}`;
      if (elapsed >= MAX_RECORDING_SECONDS) stopRecording();
    }, 250);
    recording = { stream, context, source, processor, chunks, timerId, startedAt };
    setRecordingUi(true);
    addMessage("user", "Recording from microphone…");
  } catch (error) {
    const message = error.name === "NotAllowedError"
      ? "Microphone permission was denied. Allow microphone access or upload an audio file."
      : "Could not start the microphone. Check that it is connected, then try again.";
    showError(message);
  }
}

async function stopRecording() {
  if (!recording) return;
  const current = recording;
  recording = null;
  window.clearInterval(current.timerId);
  const duration = (Date.now() - current.startedAt) / 1000;
  current.processor.onaudioprocess = null;
  current.processor.disconnect();
  current.source.disconnect();
  current.stream.getTracks().forEach((track) => track.stop());
  await current.context.close();
  setRecordingUi(false);

  if (duration < 0.3 || current.chunks.length === 0) {
    showError("That recording was too short. Record a little longer and try again.");
    return;
  }

  const sampleCount = current.chunks.reduce((count, chunk) => count + chunk.length, 0);
  const samples = new Float32Array(sampleCount);
  let offset = 0;
  for (const chunk of current.chunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  await sendAudioFile(wavFile(samples, current.context.sampleRate));
}

recordButton.addEventListener("click", () => {
  if (recording) {
    stopRecording();
  } else {
    startRecording();
  }
});

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(messageInput.value);
});

messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 120)}px`;
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.message || ""));
});

document.querySelector("#new-chat").addEventListener("click", async () => {
  try {
    await postJson("/api/reset", {});
    messageList.replaceChildren();
    welcome.hidden = false;
    messageInput.focus();
  } catch (error) {
    showError(error.message);
  }
});

audioInput.addEventListener("change", async () => {
  const file = audioInput.files?.[0];
  if (!file) return;
  audioInput.value = "";
  await sendAudioFile(file);
});

fetch("/api/status")
  .then((response) => response.json())
  .then((status) => {
    const missing = [];
    if (!status.llm_configured) missing.push("LLM key");
    if (!status.voice_configured) missing.push("Gnani key");
    if (missing.length) {
      statusLabel.textContent = `${missing.join(" & ")} needed`;
      serviceStatus.classList.add("unavailable");
    } else {
      statusLabel.textContent = "Services ready";
    }
  })
  .catch(() => {
    statusLabel.textContent = "Status unavailable";
    serviceStatus.classList.add("unavailable");
  });
