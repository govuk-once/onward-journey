<script lang="ts">
  import { v7 as uuid } from "uuid";
  import QuestionForm from "$lib/components/QuestionForm.svelte";
  import SvelteMarkdown from "svelte-markdown";
  import LoadingCircle from "$lib/assets/loading.svg"
  // --- Interfaces ---
  interface GenesysHandoff {
    action: string;
    deploymentId: string;
    region: string;
    token: string;
    reason: string;
    customAttributes?: Record<string, string>;
  }

  interface Message {
    id: string;
    user: string;
    message: string;
    isSelf: boolean;
    timestamp?: string;
    agentId?: string;
  }

// --- New State for Capability Gating ---
let ojaEnabled = $state(false);
let currentInputText = $state("");
let triageDisplay = $state({
  active_service: "None",
  collected: {},
  all_required: []
});

// --- New Toggle Action ---
async function toggleOjaCapability() {
  const nextState = !ojaEnabled;
  try {
    const res = await fetch(`http://localhost:8000/agent/toggle?enabled=${nextState}`, {
      method: "POST"
    });
    if (res.ok) {
      ojaEnabled = nextState;
    }
  } catch (err) {
    console.error("Failed to toggle OJA:", err);
  }
}

  // --- State ---
  let scrollContainer: HTMLElement | undefined = $state();

let { data }: { data: { messages?: Message[] } } = $props();

// Use an effect to sync the prop to your state if the prop changes
let chatMessages = $state<Message[]>([]);

$effect(() => {
  if (data.messages) {
    chatMessages = data.messages;
  }
});

  let isLoading = $state(false);
  let isLiveChat = $state(false);
  let socket: WebSocket | null = $state(null);
  let sessionToken = $state("");

  // --- Actions ---
  function autoScroll(node: HTMLElement) {
    const observer = new ResizeObserver(() => {
      node.scrollTo({ top: node.scrollHeight, behavior: 'smooth' });
    });
    observer.observe(node);
    return {
      destroy() { observer.disconnect(); }
    };
  }

  async function manualHandBack() {
    if (!socket) return;
    socket.close();
  }

  async function returnToAIAgent() {
    const transcript = chatMessages
      .filter( (m: Message) => (m.user === "Live Agent" || m.user === "You") && m.message)
      .map( (m: Message) => ({
        role: m.user === "You" ? "user" : "assistant",
        text: m.message
      }));

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
      if (!res.ok) throw new Error("Server Error");

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
    if (socket) {
      socket.onopen = null;
      socket.onmessage = null;
      socket.onclose = null;
      socket.close();
    }

    isLiveChat = true;
    sessionToken = config.token;
    const uri = `wss://webmessaging.${config.region}/v1?deploymentId=${config.deploymentId}`;
    const newSocket = new WebSocket(uri);
    socket = newSocket;


    console.log("Session Token:", sessionToken);
    console.log("Attributes to send:", JSON.stringify(config.customAttributes));


    newSocket.onopen = () => {
      newSocket.send(JSON.stringify({
        action: "configureSession",
        deploymentId: config.deploymentId,
        token: sessionToken,
        data: {
          customAttributes: config.customAttributes || {}
        }
      }));
    };

  newSocket.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.class === "SessionResponse" && data.code === 200) {
        const currentSessionHistory = chatMessages
          .map((m: Message) => {
            const time = m.timestamp ? `[${m.timestamp}] ` : "";
            const agentId = m.agentId ? ` (Agent: ${m.agentId})` : "";
            return `${time}${m.user}${agentId}: ${m.message}`;
          })
          .join('\n');

        const fullContext = `--- SYSTEM: CONTINUED HANDOFF CONTEXT ---\n` +
                            `REASON: ${config.reason}\n\n` +
                            `PREVIOUS INTERACTION LOG:\n${currentSessionHistory}`;

        setTimeout(() => {
            newSocket.send(JSON.stringify({
                action: "onMessage",
                token: sessionToken,
                message: {
                    type: "Text",
                    text: fullContext,
                    channel: {
                        metadata: {
                            customAttributes: config.customAttributes
                        }
                    }
                }
            }));
        }, 1000);

        chatMessages = [...chatMessages, {
            message: "All previous history (including prior advisors) has been shared.",
            user: "System",
            isSelf: false,
            id: uuid(),
            timestamp: new Date().toLocaleTimeString()
        }];
    }

      if (data.class === "StructuredMessage" && data.body?.text && data.body.direction === "Outbound") {
        chatMessages = [...chatMessages, {
          message: data.body.text,
          user: "Live Agent",
          agentId: data.metadata?.externalContactId || "Advisor",
          isSelf: false,
          id: uuid(),
          timestamp: new Date().toLocaleTimeString()
        }];
      }
    };

    newSocket.onclose = () => {
      isLiveChat = false;
      socket = null;
      sessionToken="";
      returnToAIAgent();
    };
  }
async function handleSendMessage(userText: string) {
  if (!userText.trim() || isLoading) return;
  chatMessages = [...chatMessages, { message: userText, user: "You", isSelf: true, id: uuid() }];

  if (isLiveChat && socket?.readyState === 1) {
    // Standard Live Chat forwarding [cite: 49]
    socket.send(JSON.stringify({ action: "onMessage", token: sessionToken, message: { type: "Text", text: userText } }));
    return;
  }

  isLoading = true;
  
  // CUSTOM LOGIC: If Onward Journey Tool is OFF, simulate the GOV.UK Chat Beta response
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
    const data = await res.json();

    if (data.debug) {
      triageDisplay = data.debug;
    }
    
    // Handle handoff signals if OJA provides them
    if (data.response?.includes("initiate_live_handoff")) {
      const jsonStart = data.response.indexOf('{');
      const jsonEnd = data.response.lastIndexOf('}') + 1;
      const config = JSON.parse(data.response.substring(jsonStart, jsonEnd));
      setupGenesysSocket(config);
    } else {
      chatMessages = [...chatMessages, {
        message: data.response,
        user: "GOV.UK Chat",
        isSelf: false,
        id: uuid()
      }];
    }
  } catch {
    chatMessages = [...chatMessages, { message: "Connection error.", user: "System", isSelf: false, id: uuid() }];
  } finally {
    isLoading = false;
  }
}
</script>


<main class="app-conversation-layout__main">
  {#if isLiveChat}
    <div class="app-conversation-layout__header handoff-banner">
      <div class="banner-content">
        <p class="govuk-body"><strong>Live Chat:</strong> Connected to advisor.</p>
      </div>
      <div class="banner-actions">
        <button class="govuk-button govuk-button--small govuk-!-margin-right-2" onclick={manualHandBack}>
          Return to AI
        </button>
        <button class="govuk-button govuk-button--warning govuk-button--small" onclick={() => { socket?.close(); isLiveChat = false; }}>
          End Session
        </button>
      </div>
    </div>
  {#if triageDisplay.active_service !== "None"}
  <div class="govuk-inset-text app-conversation-layout__width-restrictor">
    <p class="govuk-body-s"><strong>Debug Triage:</strong> {triageDisplay.active_service}</p>
    <ul class="govuk-list govuk-list--bullet">
      {#each Object.entries(triageDisplay.collected) as [key, val] (key)}        <li>{key}: {val}</li>
      {/each}
    </ul>
  </div>
{/if}
  {/if}

<div bind:this={scrollContainer} use:autoScroll class="app-conversation-layout__wrapper app-conversation-layout__width-restrictor">
  <div class="message-feed">
    {#each chatMessages as m (m.id)}
      <div class="message-bubble {m.isSelf ? 'user' : 'agent'}">
        <p class="govuk-body-s"><strong>{m.user}:</strong></p>
        <div class="markdown-content">
          <SvelteMarkdown source={m.message || ""} />
        </div>
      </div>
    {/each}
  </div>
</div>
  {#if isLoading}
    <div class="loading-image app-conversation-layout__width-restrictor">
      <img src={LoadingCircle} alt="Loading Circle" />
      <span class="govuk-body govuk-hint govuk-!-margin-bottom-2">GOV.UK Chat is typing...</span>
    </div>
  {/if}
<div class="app-conversation-layout__fixed-footer app-conversation-layout__width-restrictor">
  <QuestionForm 
    onSend={handleSendMessage} 
    {isLoading} 
    value={currentInputText} 
  />
</div>
<div class="app-conversation-layout__form-region">
  <div class="app-conversation-layout__width-restrictor govuk-!-margin-top-0">
    <div class="capability-toggle-panel">
      <div class="toggle-info">
        <h3 class="govuk-heading-s govuk-!-margin-bottom-1">Agent Capability Control</h3>
        <p class="govuk-body-s govuk-hint">
          {ojaEnabled 
            ? "Onward Journey is active: Can access OJ KB and Live Chat." 
            : "Onward Journey is inactive: Limited to GOV.UK Chat Beta only."}
        </p>
      </div>
        <button 
          class="govuk-button {ojaEnabled ? 'govuk-button--warning' : ''}" 
          onclick={toggleOjaCapability}>
          {ojaEnabled ? 'Disable Onward Journey Tool' : 'Enable Onward Journey Tool'}
        </button>
      </div>
    </div>
</div>

<hr class="govuk-section-break govuk-section-break--m govuk-section-break--visible app-conversation-layout__width-restrictor">
</main>

<style>
.app-conversation-layout__header {
	padding: 10px 20px;
	border-bottom: 1px solid #b1b4b6;
	background: #ffffff;
	z-index: 10;
}
.handoff-banner {
	display: flex;
	justify-content: space-between;
	align-items: center;
	margin: 0;
	border-color: #1d70b8;
}
.app-conversation-layout__wrapper {
	flex: 1;
	overflow-y: auto;
}
:global(body) {
	font-family: "GDS Transport", arial, sans-serif;
	-webkit-font-smoothing: antialiased;
	-moz-osx-font-smoothing: grayscale;
}
.app-conversation-layout__main {
	display: flex;
	flex-direction: column;
	height: 100vh;
	/* background-color: #ffffff; */
  background-color: #e8f1f8;
	overflow: hidden;
}
.app-conversation-layout__wrapper {
	flex: 1;
	overflow-y: auto;
	padding: 20px;
	display: flex;
	flex-direction: column;
}
.govuk-body-s {
	font-size: 16px;
	margin-bottom: 8px;
	line-height: 1.3;
}
.app-conversation-layout__fixed-footer {
	border-top: 1px solid #b1b4b6;
	padding: 15px 0;
}
.handoff-banner {
	display: flex;
	justify-content: space-between;
	align-items: center;
	border-color: #1d70b8;
	margin-bottom: 20px;
}
.message-feed {
	display: flex;
	flex-direction: column;
	gap: 20px;
	width: 100%;
}
.message-bubble {
	padding: 15px;
	border-radius: 4px;
	max-width: 80%;
	line-height: 1.5;
}
.message-bubble.agent {
	background-color: #f3f2f1;
	border-left: 4px solid #1d70b8;
	align-self: flex-start;
}
.message-bubble.user {
	background-color: #005ea5;
	color: white;
	align-self: flex-end;
}
.message-bubble.user :global(strong) {
	color: white;
}
.capability-toggle-panel {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px;
  background-color: #f3f2f1;
  border: 1px solid #b1b4b6;
  border-radius: 4px;
}
.toggle-info h3 {
  margin-top: 0;
}
.govuk-section-break {
  margin: 20px auto;
  width: 100%;
}
.loading-image {
  display: flex;
  align-items: center;
}
</style>
