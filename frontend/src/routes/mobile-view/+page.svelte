<script lang="ts">
import { v7 as uuid } from "uuid";
import LoadingAnimation from "$lib/assets/loading.svg"
import MobileQuestionForm from "$lib/components/MobileQuestionForm.svelte";
import Footer from "$lib/components/Footer.svelte";
import Bubble from "$lib/components/Bubble.svelte";
import ConnectionBubble from "$lib/components/ConnectionBubble.svelte";
import rightSideIcon from "$lib/assets/rightside.svg"
import sendarrow from "$lib/assets/send-icon.svg"
import type { Message } from "$lib/types/Message"

let agentThoughts = $state<string[]>(["System initialized. Awaiting user input..."]);
let { data }: { data: { messages?: Message[] } } = $props();
let scrollContainer = $state<HTMLElement | undefined>(); // Fixed type syntax
let socket = $state<WebSocket | null>(null);
let sessionToken = $state("");
let chatMessages = $state<Message[]>(data.messages ?? []);
  
let ojaEnabled = $state(false); 
let isLoading = $state(false);
let isLiveChat = $state(false);
let currentInputText = $state("");
let hasInputText = $derived(currentInputText !== "");

$effect(() => {
  if (scrollContainer && chatMessages.length > 0) {
    // We use untrack or just a simple timeout to ensure the DOM has updated
    const timer = setTimeout(() => {
      scrollContainer?.scrollTo({
        top: scrollContainer.scrollHeight,
        behavior: 'smooth'
      });
    }, 50);
    return () => clearTimeout(timer); // Cleanup is good practice
  }
});

  // --- Logic ---
  async function toggleOjaCapability() {
    const nextState = !ojaEnabled;
    try {
      const res = await fetch(`http://localhost:8000/agent/toggle?enabled=${nextState}`, {
        method: "POST",
        headers: { "Accept": "application/json" }
      });
      if (res.ok) { ojaEnabled = nextState; }
    } catch (err) {
      console.error("Network error toggling capability", err);
    }
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

  // --- UPDATED SOCKET LOGIC WITH SUMMARY HANDOFF ---
  function setupGenesysSocket(config: any) {
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
      chatMessages = [...chatMessages, {
        message: "EVENT_HUMAN_CONNECTED",
        user: "System",
        isSelf: false,
        id: uuid()
      }];
    };

    newSocket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // DETECT SUCCESSFUL SESSION START TO SEND SUMMARY
      if (data.class === "SessionResponse" && data.code === 200) {
        // Build the transcript for the agent
        const currentSessionHistory = chatMessages
          .map((m) => `${m.user}: ${m.message}`)
          .join('\n');

        const fullContext = `--- SYSTEM: HANDOFF SUMMARY ---\n` +
                            `REASON: ${config.reason}\n\n` +
                            `PREVIOUS INTERACTION:\n${currentSessionHistory}`;

        // Delay slightly to ensure session is ready
        setTimeout(() => {
            newSocket.send(JSON.stringify({
                action: "onMessage",
                token: sessionToken,
                message: {
                    type: "Text",
                    text: fullContext,
                    channel: { metadata: { customAttributes: config.customAttributes } }
                }
            }));
        }, 1000);

        chatMessages = [...chatMessages, {
            message: "Conversation history and handoff reason have been shared with the advisor.",
            user: "System",
            isSelf: false,
            id: uuid()
        }];
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
          message: "I don't have the specific phone number for HMRC Pension Schemes Services in the guidance provided.The guidance shows that you \
    'can contact HMRC Pension Schemes Services by: using the online contact form writing to: Pension Schemes Services, HM Revenue and Customs, \
    'BX9 1GH, United Kingdom. You can find phone contact details for other HMRC services on the Contact HMRC page. GOV.UK Chat can make mistakes. \
    'Check GOV.UK pages for important information. GOV.UK pages used in this answer (links open in a new tab)'",
          user: "GOV.UK Chat",
          isSelf: false,
          id: uuid()
        }];
        isLoading = false;
      }, 800);
      return;
    }

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText })
      });
      const result = await res.json();
      if (result.logs) {
  agentThoughts = [...agentThoughts, ...result.logs]; 
}
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
      <div class="status-icons">
        <img src={rightSideIcon} alt="Signal" class="status-svg" /></div>
    </div>

    <main class="ios-screen">
      <header class="ios-header-group">
        <div class="caspar-header">
          <p><strong>GOV.UK Chat</strong></p>
        </div>
      </header>

      <div class="chat-container">
        <div bind:this={scrollContainer} class="message-container"> 
          <div class="chat-feed">
              {#each chatMessages as m (m.id)}
                  {#if m.message === "EVENT_HUMAN_CONNECTED"}
                      <ConnectionBubble agentType="human" agentName="Caspar" />
                  {:else}
                      <Bubble message={m} />
                  {/if}
              {/each} 
              {#if isLoading}
                  <div class="ios-typing-container">
                    <div class="ios-typing-bubble">
                        <img src={LoadingAnimation} height="15%" width="15%" alt="loading" />
                        <p>Generating your answer...</p>
                    </div>
                  </div>
              {/if}
          </div>
        </div>
<div class="input-area-blue">
  <div class="input-pill-container"> 
      <div class="input-pill">
          <MobileQuestionForm bind:value={currentInputText} onSend={handleSendMessage} {isLoading} />
      </div>
      
      <button 
  class="action-circle {hasInputText ? 'active' : ''}" 
  onclick={() => hasInputText && handleSendMessage(currentInputText)}
  disabled={isLoading && !hasInputText}
>
  {#if hasInputText}
    <img src={sendarrow} alt="Send" class="send-icon-white" />
  {:else}
    <div class="dots-container">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </div>
  {/if}
</button>
  </div>
</div>

      <footer class="ios-footer-group">
        <div class="homebar">
            <Footer /> 
        </div>
      </footer>
    </main>
  </div>

<div class="iphone-frame debug-frame">
  <div class="status-bar debug-status-top">
    <span class="time">{new Date().getHours()}:{new Date().getMinutes().toString().padStart(2, '0')}</span>
  </div>

  <main class="ios-screen debug-screen-container">
    <header class="ios-header-group debug-header-group">
      <div class="caspar-header">
        <p><strong style="color: black;">Agent Logic</strong></p>
      </div>
    </header>

    <div class="debug-scroll-area" bind:this={debugScrollContainer}>
      <div class="debug-log-container">
        {#each agentThoughts as thought}
          <div class="thought-entry">
            <span class="timestamp">{new Date().toLocaleTimeString()}</span>
            <p>{thought}</p>
          </div>
        {/each}
      </div>
    </div>
  </main>
</div>

  <aside class="control-panel">
    <div class="glass-card">
      <h3 class="govuk-heading-s">Onward Journey Toggle</h3>
      <p class="govuk-body-s">
      </p>
      <button 
        class="govuk-button {ojaEnabled ? 'govuk-button--warning' : ''}" 
        onclick={toggleOjaCapability}>
        {ojaEnabled ? 'Disable' : 'Enable'}
      </button>
    </div>
  </aside>
</div>

<style>
  :global(body), .workspace { font-family: "GDS Transport", arial, sans-serif; margin: 0; }
  .workspace { display: flex; justify-content: center; align-items: center; gap: 40px; padding: 20px; background: #f2f2f7; height: 100vh; box-sizing: border-box; }
  .iphone-frame { width: 375px; height: 812px; background: #000; border: 12px solid #2c2c2e; border-radius: 54px; position: relative; overflow: hidden; transform: scale(0.8); }
  
  .status-bar { display: flex; justify-content: space-between; padding: 14px 28px 4px; background: #E8F1F8; font-size: 13px; font-weight: 600; position: relative; z-index: 20; border:none; margin:0; }

  .ios-screen { height: 100%; background: #E8F1F8; display: flex; flex-direction: column; border: none; margin: 0; }  
  .ios-header-group { z-index: 3; border: none; background: #E8F1F8; margin: 0; padding: 0; }
  .caspar-header { padding: 12px 16px; color: black; text-align: center; }
  .caspar-header strong { font-weight: 700; font-size: 22px; display: block; }

  .chat-container { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .message-container { flex: 1; overflow-y: auto; padding: 20px 12px 10px 12px; display: flex; flex-direction: column; }
  .chat-feed { display: flex; flex-direction: column; gap: 12px; width: 100%; align-items: stretch; }

  .input-area-blue { padding: 10px 16px 20px 16px; background: transparent; }
  .input-pill { background: #ffffff; border-radius: 25px; padding: 4px 12px; min-height: 44px; display: flex; align-items: center; border: none; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }
  
  .ios-footer-group { background: #ffffff; border-top: 1px solid #d1d1d6; padding: 10px 0 30px 0; }
  .ios-typing-bubble { background: #E8F1F8; border-radius: 15px; padding: 8px 12px; color: #0F385C; display: flex; align-items: center; gap: 8px; }

  .control-panel { width: 220px; }
  .glass-card { background: white; padding: 20px; border-radius: 16px; border: 1px solid #d1d1d6; }


.input-pill-container {
    display: flex;
    align-items: flex-end; /* Keeps circle at bottom during multi-line input */
    gap: 10px;
    width: 100%;
    border: none;
  }

.action-circle {
    width: 50px;
    height: 50px;
    background: white; /* Always white */
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    border: none;
    flex-shrink: 0; 
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    cursor: pointer;
    padding: 0;
  }
.action-circle.active {
    width: 50px;
    height: 50px;
    background: #1D70B8; 
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    border: none;
    flex-shrink: 0; 
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    cursor: pointer;
    padding: 0;
  }

.dots-container {
    display: flex;
    color: white;
    gap: 3px;
  }
.input-pill {
    flex: 1;
    background: #ffffff;
    border-radius: 22px;
    padding: 4px 16px;
    min-height: 44px;
    display: flex;
    align-items: center;
    border: none;
  }

.dot {
    width: 5px;
    height: 5px;
    background-color: #1D70B8;
    border-radius: 50%;
  }

  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1.1); }
  }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1.1); }
  }
  /* Animations */
  @keyframes pulse {
    0% { transform: scale(1); opacity: 0.8; }
    70% { transform: scale(2.5); opacity: 0; }
    100% { transform: scale(2.5); opacity: 0; }
  }

  @keyframes slideDown {
    from { transform: translateY(-100%); }
    to { transform: translateY(0); }
  }


.debug-frame {
  border-color: #333; /* Darker frame */
  filter: sepia(0.2); /* Slight tint to distinguish */
}

.debug-scroll-area {
  flex: 1;
  background: #E8F1F8;
  color: black;
  font-family: "GDS Transport";
  padding: 15px; 
  font-size: 11px; 
  overflow-y: auto; /* Enables vertical scrolling  */
  overflow-x: hidden; /* Prevents horizontal scrolling */
  display: flex;
  flex-direction: column;
}

.debug-screen-container {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden; /* Keeps the header fixed while the area below scrolls */
}

.thought-entry p {
  margin: 0;
  padding: 0;
  white-space: pre-wrap;      /* Preserves line breaks but wraps text naturally */
  word-wrap: break-word;     /* Legacy support */
  overflow-wrap: break-word; /* Prevents long strings from breaking the layout */
  word-break: break-word;    /* Ensures text stays within the container boundaries */
  max-width: 100%;           /* Constrains width to the parent frame */
}

/* Update the scrollbar track and thumb as well */
.debug-scroll-area::-webkit-scrollbar {
  width: 6px;
}

.debug-scroll-area::-webkit-scrollbar-track {
  background: #E8F1F8;
}

.debug-scroll-area::-webkit-scrollbar-thumb {
  background: #1D70B8;
  border-radius: 10px;
}

.debug-screen {
  background: #E8F1F8; /* Dark terminal background */
  color: black; /* Classic "terminal green" text */
  font-family: "GDS Transport";
  padding: 15px;
  font-size: 11px;
}

.debug-screen {
  background: #E8F1F8;
  color: black;
  font-family: "GDS Transport";
  padding: 15px; 
  font-size: 11px; 
  overflow-y: auto; /* Enables the vertical scrollbar */
  scrollbar-width: thin; /* For Firefox */
  scrollbar-color: #1D70B8 #E8F1F8; /* For Firefox: green thumb, dark track */
}

/* Custom Scrollbar for Chrome, Safari, and Edge */
.debug-screen::-webkit-scrollbar {
  width: 6px;
}

.debug-screen::-webkit-scrollbar-track {
  background: #1c1c1e;
}

.debug-screen::-webkit-scrollbar-thumb {
  background: #32d74b;
  border-radius: 10px;
}

.debug-screen::-webkit-scrollbar-thumb:hover {
  background: #28a745; /* Slightly darker green on hover */
}

.thought-entry {
  border-left: 2px solid black;
  padding-left: 10px;
  margin-bottom: 12px;
  animation: fadeIn 0.3s ease-out;
}

.timestamp {
  color: #888;
  display: block;
  font-size: 9px;
}

</style>