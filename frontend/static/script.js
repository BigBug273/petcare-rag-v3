const form = document.querySelector("#ask-form");
const input = document.querySelector("#question");
const button = document.querySelector("#ask-button");
const messages = document.querySelector("#messages");
const exampleButtons = document.querySelectorAll(".examples button");
const MAX_MESSAGES = 20;

function renderIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function scrollMessages() {
  messages.scrollTop = messages.scrollHeight;
}

function icon(name) {
  const node = document.createElement("i");
  node.setAttribute("data-lucide", name);
  return node;
}

function trimMessages() {
  const bubbles = messages.querySelectorAll(".bubble:not(.loading)");
  const extra = bubbles.length - MAX_MESSAGES;
  for (let i = 0; i < extra; i += 1) {
    bubbles[i].remove();
  }
}

function createBubble(role, text, sources = []) {
  const bubble = document.createElement("article");
  bubble.className = `bubble ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.appendChild(icon(role === "user" ? "user-round" : "cat"));

  const body = document.createElement("div");
  body.className = "bubble-body";

  const name = document.createElement("strong");
  name.textContent = role === "user" ? "คุณ" : "น้องผู้ช่วย";

  const content = document.createElement("p");
  content.textContent = text;

  body.append(name, content);

  if (sources.length > 0) {
    body.appendChild(createSources(sources));
  }

  bubble.append(avatar, body);
  return bubble;
}

function createSources(sources) {
  const sourceWrap = document.createElement("div");
  sourceWrap.className = "sources";

  sources.forEach((source) => {
    const item = document.createElement("div");
    item.className = "source";

  const title = document.createElement("strong");
    const sourceIcon = icon(source.type === "cat" ? "cat" : "paw-print");
    const titleText = document.createElement("span");
    titleText.textContent = source.breed_name || "แหล่งข้อมูล";
    title.append(sourceIcon, titleText);

    const meta = document.createElement("div");
    const score = typeof source.score === "number" ? ` · ${(source.score * 100).toFixed(0)}%` : "";
    meta.textContent = `${(source.type || "pet").toUpperCase()}${score}`;

    const link = document.createElement("a");
    link.href = source.source_url || "#";
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = source.source_url || "ไม่มีลิงก์";

    item.append(title, meta, link);
    sourceWrap.appendChild(item);
  });

  return sourceWrap;
}

function addBubble(role, text, sources = []) {
  messages.appendChild(createBubble(role, text, sources));
  trimMessages();
  renderIcons();
  scrollMessages();
}

function addLoadingBubble() {
  const bubble = document.createElement("article");
  bubble.className = "bubble assistant loading";
  bubble.id = "loading-bubble";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.appendChild(icon("cat"));

  const body = document.createElement("div");
  body.className = "bubble-body";

  const name = document.createElement("strong");
  name.textContent = "น้องผู้ช่วยกำลังค้นข้อมูล";

  const dots = document.createElement("div");
  dots.className = "typing-dots";
  dots.innerHTML = "<span></span><span></span><span></span>";

  body.append(name, dots);
  bubble.append(avatar, body);
  messages.appendChild(bubble);
  renderIcons();
  scrollMessages();
}

function removeLoadingBubble() {
  document.querySelector("#loading-bubble")?.remove();
}

function setLoading(isLoading) {
  button.disabled = isLoading;
  input.disabled = isLoading;
  button.innerHTML = isLoading
    ? '<i data-lucide="loader-circle"></i> กำลังตอบ...'
    : '<i data-lucide="send"></i> Ask';
  renderIcons();
}

async function askQuestion(question) {
  addBubble("user", question);
  addLoadingBubble();
  setLoading(true);

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "ระบบมีปัญหาชั่วคราว");
    }

    removeLoadingBubble();
    addBubble("assistant", data.answer || "ยังไม่มีคำตอบ", data.sources || []);
  } catch (error) {
    removeLoadingBubble();
    addBubble("assistant", `ขอโทษนะคะ ${error.message}`);
  } finally {
    setLoading(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  askQuestion(question);
});

exampleButtons.forEach((example) => {
  example.addEventListener("click", () => {
    input.value = example.textContent;
    input.focus();
  });
});

renderIcons();
