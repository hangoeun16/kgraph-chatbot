/**
 * KGraph chat client
 *
 * Handles:
 * - SSE streaming for real-time response display
 * - Status message rendering during pipeline stages
 * - Auto-resizing textarea
 * - Graph visualization modal
 * - Memory clear
 */

const chat      = document.getElementById("chat");
const input     = document.getElementById("input");
const btnSend   = document.getElementById("btn-send");
const btnClear  = document.getElementById("btn-clear");
const btnGraph  = document.getElementById("btn-graph");
const statusBar = document.getElementById("status-bar");
const modal     = document.getElementById("graph-modal");
const btnClose  = document.getElementById("btn-close-modal");

let isStreaming = false;


// ---------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------

function addMessage(role, content) {
    // Remove welcome message on first interaction
    const welcome = chat.querySelector(".chat__welcome");
    if (welcome) welcome.remove();

    const div = document.createElement("div");
    div.classList.add("message", `message--${role}`);
    div.textContent = content;
    chat.appendChild(div);
    scrollToBottom();
    return div;
}

function createAssistantBubble() {
    const welcome = chat.querySelector(".chat__welcome");
    if (welcome) welcome.remove();

    const div = document.createElement("div");
    div.classList.add("message", "message--assistant");
    chat.appendChild(div);
    scrollToBottom();
    return div;
}

function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
}


// ---------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------

function showStatus(text) {
    if (!text) {
        statusBar.hidden = true;
        statusBar.textContent = "";
        return;
    }
    statusBar.textContent = text;
    statusBar.hidden = false;
}


// ---------------------------------------------------------------
// Send message via SSE
// ---------------------------------------------------------------

async function sendMessage() {
    const text = input.value.trim();
    if (!text || isStreaming) return;

    isStreaming = true;
    btnSend.disabled = true;
    input.value = "";
    resetTextareaHeight();

    addMessage("user", text);

    let bubble = null;

    try {
        const response = await fetch("/chat-stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;

                const data = JSON.parse(line.slice(6));

                switch (data.type) {
                    case "status":
                        showStatus(data.content);
                        break;

                    case "token":
                        if (!bubble) {
                            showStatus("");
                            bubble = createAssistantBubble();
                        }
                        bubble.textContent += data.content;
                        scrollToBottom();
                        break;

                    case "done":
                        showStatus("");
                        break;

                    case "rejected":
                        showStatus("");
                        addMessage("rejected", data.content);
                        break;

                    case "error":
                        showStatus("");
                        addMessage("error", data.content);
                        break;
                }
            }
        }
    } catch (err) {
        showStatus("");
        addMessage("error", "Connection failed. Please try again.");
        console.error("Stream error:", err);
    } finally {
        isStreaming = false;
        btnSend.disabled = false;
        input.focus();
    }
}


// ---------------------------------------------------------------
// Clear memory
// ---------------------------------------------------------------

async function clearMemory() {
    if (!confirm("Clear all memory? This cannot be undone.")) return;

    try {
        const res = await fetch("/clear", { method: "POST" });
        const data = await res.json();

        // Reset chat area
        chat.innerHTML = `
            <div class="chat__welcome">
                <p>Memory cleared. Start a new conversation.</p>
            </div>
        `;
    } catch (err) {
        addMessage("error", "Failed to clear memory.");
        console.error("Clear error:", err);
    }
}


// ---------------------------------------------------------------
// Graph modal
// ---------------------------------------------------------------

async function openGraph() {
    modal.showModal();

    try {
        const res = await fetch("/graph-data");
        const data = await res.json();

        const container = document.getElementById("graph-container");

        if (!data.nodes || data.nodes.length === 0) {
            container.innerHTML = `
                <p class="graph-container__empty">
                    No data yet. Start a conversation to build the graph.
                </p>
            `;
            return;
        }

        renderGraph(container, data);
    } catch (err) {
        console.error("Graph error:", err);
    }
}

function renderGraph(container, data) {
    const width = container.clientWidth;
    const height = container.clientHeight || 400;

    // Simple force-directed layout using SVG
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

    // Position nodes in a circle
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.35;
    const positions = {};

    data.nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / data.nodes.length - Math.PI / 2;
        positions[node.id] = {
            x: cx + radius * Math.cos(angle),
            y: cy + radius * Math.sin(angle),
        };
    });

    // Draw edges
    data.edges.forEach((edge) => {
        const from = positions[edge.from];
        const to = positions[edge.to];
        if (!from || !to) return;

        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", from.x);
        line.setAttribute("y1", from.y);
        line.setAttribute("x2", to.x);
        line.setAttribute("y2", to.y);
        line.classList.add("graph-edge");
        svg.appendChild(line);

        // Edge label
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", (from.x + to.x) / 2);
        label.setAttribute("y", (from.y + to.y) / 2 - 6);
        label.classList.add("graph-edge-label");
        label.setAttribute("text-anchor", "middle");
        label.textContent = edge.relation;
        svg.appendChild(label);
    });

    // Draw nodes
    data.nodes.forEach((node) => {
        const pos = positions[node.id];

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", pos.x);
        circle.setAttribute("cy", pos.y);
        circle.setAttribute("r", 20);
        circle.classList.add("graph-node");
        svg.appendChild(circle);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", pos.x);
        label.setAttribute("y", pos.y + 34);
        label.classList.add("graph-node-label");
        label.textContent = node.name;
        svg.appendChild(label);
    });

    container.innerHTML = "";
    container.appendChild(svg);
}


// ---------------------------------------------------------------
// Auto-resize textarea
// ---------------------------------------------------------------

function autoResize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
}

function resetTextareaHeight() {
    input.style.height = "auto";
}


// ---------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------

btnSend.addEventListener("click", sendMessage);

input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

input.addEventListener("input", autoResize);
btnClear.addEventListener("click", clearMemory);
btnGraph.addEventListener("click", openGraph);
btnClose.addEventListener("click", () => modal.close());

modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.close();
});
