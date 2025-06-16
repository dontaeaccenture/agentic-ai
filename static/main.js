  function switchTab(tabId, btn) {
    document.querySelectorAll('.tab-content').forEach(div => div.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (tabId === 'docsTab') loadDocs();
    if (tabId === 'metricsTab') refreshMetrics();

  }

  function setInput(text) {
    document.getElementById("input").value = text;
  }

  let chatInitialized = false;

  function send() {
    const input = document.getElementById("input");
    const chat = document.getElementById("chat");
    const message = input.value.trim();
    if (!message) return;

    if (!chatInitialized) {
      chat.innerHTML = "";
      chatInitialized = true;
    }

    chat.innerHTML += `<div class="chat-bubble user">🧑‍💼 <strong>You:</strong> ${message}</div>`;
    input.value = "";

    fetch("/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    })
    .then(res => res.json())
    .then(data => {
      const formatted = marked.parse(data.reply);
      chat.innerHTML += `<div class="chat-bubble ai">🤖 <strong>AI:</strong> ${formatted}</div>`;
      chat.scrollTop = chat.scrollHeight;
    });
  }

  async function loadDocs() {
    const docsDiv = document.getElementById("docs");
    docsDiv.textContent = "📄 Loading...";

    const res = await fetch("/docs");
    const data = await res.json();

    docsDiv.textContent = "";
    data.documents.forEach((doc, i) => {
      docsDiv.textContent += `📁 File: ${doc.file}\n📝 Preview:\n${doc.content}\n\n---\n\n`;
    });
  }

  function downloadPlan() {
    const chat = document.getElementById("chat").textContent;
    const blob = new Blob([chat], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "migration_plan.txt";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

document.getElementById("uploadForm").addEventListener("submit", async function(e) {
  e.preventDefault();
  const fileInput = document.getElementById("fileInput");
  const statusDiv = document.getElementById("uploadStatus");
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  statusDiv.style.display = "block";
  statusDiv.textContent = "⏳ Uploading to FAISS...\n";

  // Cool Matrix-style dots
  const interval = setInterval(() => {
    statusDiv.textContent += Math.random().toString(36).substring(2, 5) + "\n";
    statusDiv.scrollTop = statusDiv.scrollHeight;
  }, 100);

  try {
    const res = await fetch("/upload", {
      method: "POST",
      body: formData
    });
    const result = await res.json();
    clearInterval(interval);
    statusDiv.textContent += "\n✅ " + result.message;
  } catch (err) {
    clearInterval(interval);
    statusDiv.textContent += "\n❌ Upload failed. Please try again.";
    console.error(err);
  }
});
async function runAgentPlanner() {
  const input = document.getElementById("input").value.trim();
  const wrapper = document.getElementById("agentic-response");
  const content = document.getElementById("planner-content");

  if (!input) return alert("Please enter a message first.");

  wrapper.style.display = "block";
  content.innerHTML = "⏳ Generating agentic migration plan...";

  try {
    const res = await fetch("/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input })
    });

    const data = await res.json();
    content.innerHTML = marked.parse(data.reply);
  } catch (err) {
    content.innerHTML = "❌ Error fetching plan. Please try again.";
    console.error(err);
  }
}
async function loadHistory() {
  try {
    const res = await fetch("/history");
    const data = await res.json();
    const historyList = document.getElementById("chat-history");
    historyList.innerHTML = "";

    data.history.forEach(chat => {
      const li = document.createElement("li");
      li.style.marginBottom = "1rem";
      li.style.padding = "8px";
      li.style.border = "1px solid #ccc";
      li.style.borderRadius = "6px";
      li.style.cursor = "pointer";
      li.style.background = "#fff";

      const previewUser = chat.user_input.length > 60 ? chat.user_input.substring(0, 60) + "..." : chat.user_input;
      const previewAI = chat.ai_response.length > 60 ? chat.ai_response.substring(0, 60) + "..." : chat.ai_response;

      li.innerHTML = `<strong>User:</strong> ${previewUser}<br><strong>AI:</strong> ${previewAI}`;

      li.onclick = () => {
        document.getElementById("input").value = chat.user_input;

        // Render full conversation in chat window
        const chatWindow = document.getElementById("chat");
        chatWindow.innerHTML = `
          <div class="chat-bubble user">🧑‍💼 <strong>You:</strong> ${chat.user_input}</div>
          <div class="chat-bubble ai">🤖 <strong>AI:</strong> ${marked.parse(chat.ai_response)}</div>
        `;
      };

      historyList.appendChild(li);
    });
  } catch (err) {
    console.error("Failed to load history", err);
  }
}
  function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const toggleBtn = document.querySelector(".sidebar-toggle");
    if (sidebar.style.display === "none") {
      sidebar.style.display = "block";
      toggleBtn.innerHTML = "📂 Hide History";
    } else {
      sidebar.style.display = "none";
      toggleBtn.innerHTML = "📂 Show History";
    }
  }

  function startNewChat() {
    const chat = document.getElementById("chat");
    chat.innerHTML = `
      <div class="callout">
        <strong>🤖 Welcome to Agentic AI!</strong>
        <p>I’m your cloud migration assistant. I can:</p>
        <ul>
          <li>Generate 3-year AWS & Azure migration plans</li>
          <li>Prioritize workloads based on complexity</li>
          <li>Answer questions based on your uploaded documents</li>
        </ul>
        <p>Type a question or select a suggested prompt to begin.</p>
      </div>
    `;
    document.getElementById("input").value = "";
  }
function downloadPDF() {
  const chat = document.getElementById("chat");
  const originalHeight = chat.style.height;

  // Expand chat area to full height temporarily
  chat.style.height = 'auto';

  html2pdf()
    .set({
      margin: 10,
      filename: 'migration_plan.pdf',
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2, scrollY: 0 },
      jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
    })
    .from(chat)
    .save()
    .then(() => {
      chat.style.height = originalHeight; // Restore original height
    });
}
async function loadMetrics() {
  try {
    const res = await fetch("/metrics");
    const data = await res.json();
    document.getElementById("plansCount").textContent = data.plans_generated;
    document.getElementById("docsCount").textContent = data.documents_uploaded;
    document.getElementById("messagesCount").textContent = data.user_messages;
 
  } catch (err) {
    console.error("Failed to load metrics", err);
  }
}
async function refreshMetrics() {
  try {
    const res = await fetch("/metrics");
    const data = await res.json();
    document.getElementById("plansCount").textContent = data.plans_generated;
    document.getElementById("docsCount").textContent = data.documents_uploaded;
    document.getElementById("messagesCount").textContent = data.user_messages;
  
    document.getElementById("docsUploaded").textContent = data.documents_uploaded;
    document.getElementById("plansGenerated").textContent = data.plans_generated;
    document.getElementById("userMessages").textContent = data.user_messages;
  } catch (err) {
    console.error("Failed to fetch metrics:", err);
  }
}


  // 👇 Hide sidebar on load
  window.onload = () => {
    loadHistory();     
       refreshMetrics();  // 👈 ensure metrics load automatically

    toggleSidebar(); // hides it by default
      loadMetrics();  // 👈 Call this when page loads

  };

