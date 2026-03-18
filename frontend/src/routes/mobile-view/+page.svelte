<script lang="ts">
  import { v7 as uuid } from "uuid";
  import LoadingAnimation from "$lib/assets/loading.svg"
  import MobileQuestionForm from "$lib/components/MobileQuestionForm.svelte";
  import Footer from "$lib/components/Footer.svelte";
  import Bubble from "$lib/components/Bubble.svelte";
  import type { Message } from "$lib/types/Message"


  // --- Interfaces ---
  interface GenesysHandoff {
    action: string;
    deploymentId: string;
    region: string;
    token: string;
    reason: string;
    customAttributes?: Record<string, string>;
  }

  // --- State ---
  let { data }: { data: { messages?: Message[] } } = $props();
  
  let scrollContainer: HTMLElement | undefined = $state();
  let socket: WebSocket | null = $state(null);
  let sessionToken = $state("");

  let chatMessages = $state<Message[]>(data.messages ?? []);
  let ojaEnabled = $state(false);
  let isLoading = $state(false);
  let isLiveChat = $state(false);
  
  let currentInputText = $state("");
  let hasInputText = $state(false);

  let triageDisplay = $state({ active_service: "None", collected: {}, all_required: [] });

  // --- effect hooks ---
    // Show 'more options' or 'send' button
    $effect(() => {
      if (currentInputText !== "") {
          hasInputText = true;
        } else {
          hasInputText = false;
        }
    });

    // Auto-scroll effect
    $effect(() => {
        if (scrollContainer && chatMessages.length > 0) {
        // Small timeout ensures the DOM has rendered the new message before scrolling
        setTimeout(() => {
            scrollContainer!.scrollTo({
            top: scrollContainer!.scrollHeight,
            behavior: 'smooth'
            });
        }, 50);
        }
    });


  // --- Original Logic (Preserved) ---
  async function toggleOjaCapability() {
    const nextState = !ojaEnabled;
    try {
      const res = await fetch(`http://localhost:8000/agent/toggle?enabled=${nextState}`, {
        method: "POST",
        headers: { "Accept": "application/json" }
      });
      if (res.ok) {
        const data = await res.json();
        ojaEnabled = data.oja_enabled;
      }
    } catch (err) {
      console.error("Network error toggling capability", err);
    }
  }

  async function manualHandBack() {
    if (!socket) return;
    socket.close();
  }

  async function returnToAIAgent() {
    const transcript = chatMessages
      .filter((m) => (m.user === "Live Agent" || m.user === "You") && m.message)
      .map((m) => ({ role: m.user === "You" ? "user" : "assistant", text: m.message }));

    if (transcript.length === 0) {
      isLiveChat = false;
      return;
    }

    try {
      const res = await fetch("http://localhost:8000/handoff/back", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript })
      });
      const data = await res.json();
      chatMessages = [...chatMessages, {
        message: data.summary || "I'm back. How can I help you further?",
        user: "GOV.UK Chat",
        isSelf: false,
        id: uuid()
      }];
    } catch {
      chatMessages = [...chatMessages, {
        message: "I've reconnected, but I couldn't summarize the previous chat.",
        user: "System",
        isSelf: false,
        id: uuid()
      }];
    } finally {
      isLiveChat = false;
    }
  }

  function setupGenesysSocket(config: GenesysHandoff) {
    if (socket) socket.close();
    isLiveChat = true;
    sessionToken = config.token;
    const uri = `wss://webmessaging.${config.region}/v1?deploymentId=${config.deploymentId}`;
    const newSocket = new WebSocket(uri);
    socket = newSocket;

    newSocket.onopen = () => {
      newSocket.send(JSON.stringify({
        action: "configureSession",
        deploymentId: config.deploymentId,
        token: sessionToken,
        data: { customAttributes: config.customAttributes || {} }
      }));
    };

    newSocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.class === "SessionResponse" && data.code === 200) {
        const currentSessionHistory = chatMessages
          .map((m) => `${m.timestamp ? `[${m.timestamp}] ` : ""}${m.user}: ${m.message}`)
          .join('\n');
        const fullContext = `--- SYSTEM: HANDOFF ---\nREASON: ${config.reason}\n\nLOG:\n${currentSessionHistory}`;
        setTimeout(() => {
          newSocket.send(JSON.stringify({ action: "onMessage", token: sessionToken, message: { type: "Text", text: fullContext } }));
        }, 1000);
      }
      if (data.class === "StructuredMessage" && data.body?.text && data.body.direction === "Outbound") {
        chatMessages = [...chatMessages, {
          message: data.body.text,
          user: "Live Agent",
          isSelf: false,
          id: uuid(),
          timestamp: new Date().toLocaleTimeString()
        }];
      }
    };
    newSocket.onclose = () => { isLiveChat = false; socket = null; returnToAIAgent(); };
  }

  async function handleSendMessage(userText: string) {
    if (!userText.trim() || isLoading) return;
    chatMessages = [...chatMessages, { message: userText, user: "You", isSelf: true, id: uuid() }];
    currentInputText = ""; 

    if (isLiveChat && socket?.readyState === 1) {
      socket.send(JSON.stringify({ action: "onMessage", token: sessionToken, message: { type: "Text", text: userText } }));
      return;
    }

    isLoading = true;
    if (!ojaEnabled) {
      setTimeout(() => {
        chatMessages = [...chatMessages, {
          message: "I don't have the specific phone number for HMRC Pension Schemes Services in the guidance provided. You can find phone contact details for other HMRC services on the Contact HMRC page.",
          user: "GOV.UK Chat",
          isSelf: false,
          id: uuid()
        }];
        isLoading = false;
      }, 2000);
      return;
    }

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText })
      });
      const result = await res.json();
      if (result.response?.includes("initiate_live_handoff")) {
        const jsonStart = result.response.indexOf('{');
        const jsonEnd = result.response.lastIndexOf('}') + 1;
        setupGenesysSocket(JSON.parse(result.response.substring(jsonStart, jsonEnd)));
      } else {
        chatMessages = [...chatMessages, { message: result.response, user: "GOV.UK Chat", isSelf: false, id: uuid() }];
      }
    } catch {
      chatMessages = [...chatMessages, { message: "Connection error.", user: "System", isSelf: false, id: uuid() }];
    } finally {
      isLoading = false;
    }
  }

</script>

<div class="workspace">
  <div class="iphone-frame">
    <div class="status-bar">
      <span class="time">9:41</span>
      <div class="dynamic-island"></div> 
      <div class="status-icons">📶 5G 🔋</div>
    </div>

    <main class="ios-screen">

      <header class="ios-header-group">
        <div class="govuk-banner">
          <span class="phase-tag">BETA</span> 
          <p>GOV.UK Chat <span class="exp-text">Experimental</span></p>
        </div>

        {#if isLiveChat}
          <div class="handoff-header">
            <div class="handoff-info">
              <span class="online-indicator"></span>
              <p><strong>Advisor</strong></p>
            </div> 
            <button class="ios-close-btn" onclick={manualHandBack}>End</button>
          </div>
        {/if}
      </header>

      <div bind:this={scrollContainer} class="message-container">
        <div class="chat-feed">
            {#each chatMessages as m (m.id)}
                <Bubble message={m} />
            {/each} 
            {#if isLoading}
                <div class="ios-typing-container">
                <div class="ios-typing-bubble govuk-!-padding-left-3 govuk-!-padding-right-3">
                    <img src={LoadingAnimation} height="15%" width="15%" alt="loading animation" />
                    <p>GOVUK Chat is typing...</p>
                </div>
                </div>
            {/if}
        </div>
      </div>

      <footer class="ios-footer-group">
        <div class="input-pill">
            <MobileQuestionForm 
                bind:value={currentInputText} 
                onSend={handleSendMessage} 
                {isLoading}
            />
        </div>

        <div class="homebar govuk-!-padding-top-2">
            <Footer />
        </div>

      </footer>
    </main>
  </div>

  <aside class="control-panel">
    <div class="glass-card">
      <div class="pill-status {ojaEnabled ? 'active' : 'inactive'}">
        {ojaEnabled ? "OJ Enabled" : "OJ Disabled"}
      </div>
      <button class="govuk-button {ojaEnabled ? 'govuk-button--warning' : ''}" onclick={toggleOjaCapability}>
        {ojaEnabled ? 'Disable OJ' : 'Enable OJ'}
      </button>
    </div>
  </aside>
</div>

<style>
  /* GOV.UK Global Font Setup */
  :global(body), .workspace { 
    font-family: "GDS Transport", arial, sans-serif; 
    -webkit-font-smoothing: antialiased; 
    -moz-osx-font-smoothing: grayscale; 
    margin: 0;
  }

  .workspace { 
    display: flex; 
    justify-content: center; 
    align-items: center; 
    gap: 40px; 
    padding: 20px; 
    background: #f2f2f7; 
    height: 100vh;
    box-sizing: border-box;
  }
  
  /* iPhone Frame */
  .iphone-frame { 
    width: 375px; 
    height: 812px; /* Standard iPhone 13 Pro height */
    background: #000; 
    border: 12px solid #2c2c2e; 
    border-radius: 54px; 
    position: relative; 
    overflow: hidden; 
    box-shadow: 0 40px 100px -20px rgba(0,0,0,0.3);
  }

  .ios-screen { 
    height: 100%; 
    background: #e8edf4; 
    display: grid;
    grid-template-rows: auto 1fr auto; /* Header, Scroll Area, Footer */
  }
  
  /* Message Container - Fix for visibility */
  .message-container { 
    overflow-y: auto; 
    padding: 20px 12px;
    scroll-behavior: smooth;
    display: flex;
    flex-direction: column;
  }

  /* Chat feed starts at bottom but remains visible */
  .chat-feed { 
    display: flex; 
    flex-direction: column; 
    gap: 12px; 
    width: 100%; 
    justify-content: flex-end;
    min-height: min-content;
  }

/* UI Components */
.ios-header-group {
  z-index: 10;
  border-bottom: 1px solid #d1d1d6;
}
.status-bar {
  display: flex;
  justify-content: space-between;
  padding: 14px 28px 4px;
  background: #fff;
  font-size: 13px;
  font-weight: 600;
  position: relative;
  z-index: 20;
}
.dynamic-island {
  width: 110px;
  height: 30px;
  background: #000;
  border-radius: 20px;
  position: absolute;
  top: 8px;
  left: 50%;
  transform: translateX(-50%);
}

.govuk-banner {
  background: #f3f2f1;
  padding: 10px 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
}
.phase-tag {
  background: #1d70b8;
  color: white;
  padding: 2px 6px;
  font-weight: bold;
  font-size: 11px;
}

.handoff-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  background: #fff;
  border-top: 1px solid #eee;
}
.online-indicator {
  width: 8px;
  height: 8px;
  background: #4cd964;
  border-radius: 50%;
  display: inline-block;
  margin-right: 5px;
}
.ios-close-btn {
  color: #007aff;
  border: none;
  background: none;
  font-weight: 600;
  cursor: pointer;
  font-size: 14px;
}

/* Footer & Input */
.ios-footer-group {
  backdrop-filter: blur(10px);
  padding-bottom: 20px;
}
.input-pill {
  display: flex;
  flex: 1;
  border: 1px solid #d1d1d6;
  border-radius: 20px;
  padding: 4px 12px;
  min-height: 38px;
  align-items: center;
}

/* Typing Animation */
.ios-typing-bubble {
  background: #f2f2f7;
  border-radius: 15px;
  width: fit-content;
  display: flex;
  gap: 4px;
  align-items: center;
}

/* Control Panel */
.control-panel {
  width: 220px;
}
.glass-card {
  background: white;
  padding: 20px;
  border-radius: 16px;
  border: 1px solid #d1d1d6;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
}
.pill-status {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: bold;
  margin-bottom: 12px;
  display: inline-block;
}
.pill-status.active {
  background: #00703c;
  color: white;
}
.pill-status.inactive {
  background: #505a5f;
  color: white;
}

</style>