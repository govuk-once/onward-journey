<script lang="ts">
  import { v7 as uuid } from "uuid";
  import QuestionForm from "$lib/components/QuestionForm.svelte";
  import snarkdown from 'snarkdown';

  // --- Interfaces ---
  interface GenesysHandoff {
    action: string;
    deploymentId: string;
    region: string;
    token: string;
    reason: string; 
  }

  // --- State ---
  let scrollContainer: HTMLElement | undefined = $state(); 
  
  let { data } = $props();
  let chatMessages = $state(data.messages || []);
  
  let isLoading = $state(false); 
  let isLiveChat = $state(false); 
  let socket: WebSocket | null = $state(null); 
  let sessionToken = $state(""); 
  let handoffProcessed = $state(false); 
  
  let handoffPackage = $state({
      final_conversation_history: [
        { role: "user", content: [{ type: "text", text: "I'm trying to find the line for Pension Schemes." }] },
        { role: "assistant", content: [{ type: "text", text: "I don't have the specific phone number for HMRC Pension Schemes Services in the guidance provided.The guidance shows that you \
    'can contact HMRC Pension Schemes Services by: using the online contact form writing to: Pension Schemes Services, HM Revenue and Customs, \
    'BX9 1GH, United Kingdom. You can find phone contact details for other HMRC services on the Contact HMRC page. GOV.UK Chat can make mistakes. \
    'Check GOV.UK pages for important information. GOV.UK pages used in this answer (links open in a new tab)'" }] },
      ]
    }); 

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
      .filter(m => (m.user === "Live Agent" || m.user === "You") && m.message)
      .map(m => ({
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
        user: "GOV.UK Onward Journey Agent", 
        isSelf: false, 
        id: uuid() 
      }];
    } catch {
      // Removed unused 'err' variable
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

  async function triggerHandoffAnalysis() {
    isLoading = true;
    try {
      const res = await fetch("http://localhost:8000/handoff/process", { method: "POST" });
      if (!res.ok) throw new Error("Server error");
      const data = await res.json();
      chatMessages = [...chatMessages, {
        message: data.response,
        user: "GOV.UK Onward Journey Agent",
        isSelf: false,
        id: uuid()
      }];
      handoffProcessed = true;
    } catch {
      // Removed unused 'err' variable
      chatMessages = [...chatMessages, { message: "Error processing context.", user: "System", isSelf: false, id: uuid() }];
    } finally {
      isLoading = false;
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

    newSocket.onopen = () => {
      newSocket.send(JSON.stringify({
        action: "configureSession",
        deploymentId: config.deploymentId,
        token: sessionToken
      }));
    };

    newSocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.class === "SessionResponse" && data.code === 200) { 
          const currentSessionHistory = chatMessages
            .map(m => {
              const time = m.timestamp ? `[${m.timestamp}] ` : "";
              const agentId = m.agentId ? ` (Agent: ${m.agentId})` : "";
              return `${time}${m.user}${agentId}: ${m.message}`;
            })
            .join('\n');

          const fullContext = `--- SYSTEM: CONTINUED HANDOFF CONTEXT ---\n` +
                              `REASON: ${config.reason}\n\n` +
                              `PREVIOUS INTERACTION LOG:\n${currentSessionHistory}`;
          
          newSocket.send(JSON.stringify({
            action: "onMessage",
            token: sessionToken,
            message: { type: "Text", text: fullContext }
          }));

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
      socket.send(JSON.stringify({
        action: "onMessage",
        token: sessionToken,
        message: { type: "Text", text: userText }
      }));
      return;
    }

    isLoading = true;
    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText })
      });
      const data = await res.json();
      const responseText = data.response;

      if (responseText?.includes("initiate_live_handoff")) {
        const jsonStart = responseText.indexOf('{');
        const jsonEnd = responseText.lastIndexOf('}') + 1;
        const config: GenesysHandoff = JSON.parse(responseText.substring(jsonStart, jsonEnd));
        chatMessages = [...chatMessages, { message: "Transferring to a live agent...", user: "System", isSelf: false, id: uuid() }];
        setupGenesysSocket(config);
      } else {
        chatMessages = [...chatMessages, { 
          message: responseText, 
          user: "GOV.UK Onward Journey Agent", 
          isSelf: false, 
          id: uuid() 
        }];
      }
    } catch {
       // Removed unused 'err' variable [cite: 46]
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
  {/if}

  <div bind:this={scrollContainer} use:autoScroll class="app-conversation-layout__wrapper app-conversation-layout__width-restrictor">
    
    {#if handoffPackage.final_conversation_history.length > 0}
      <details class="govuk-details handoff-details">
        <summary class="govuk-details__summary">
          <span class="govuk-details__summary-text">View Previous Conversation (GOV.UK Chat)</span>
        </summary>
        <div class="govuk-details__text handoff-dropdown-container">
          <div class="history-list">
            {#each handoffPackage.final_conversation_history as history, i (i)}
              <p class="govuk-body-s">
                <strong>{history.role === 'user' ? 'User' : 'GOV.UK Chat'}:</strong> 
                {history.content[0].text}
              </p>
            {/each}
          </div>
          <div class="govuk-button-group govuk-!-margin-top-2">
            {#if !handoffProcessed}
              <button class="govuk-button govuk-button--small" onclick={triggerHandoffAnalysis} disabled={isLoading}>
                {isLoading ? 'Analyzing...' : 'Trigger Onward Journey'}
              </button>
            {:else}
              <strong class="govuk-tag govuk-tag--blue">Context Shared</strong>
            {/if}
          </div>
        </div>
      </details>
    {/if}

    <div class="message-feed">
      {#each chatMessages as m (m.id)}
        <div class="message-bubble {m.isSelf ? 'user' : 'agent'}">
          <p class="govuk-body-s"><strong>{m.user}:</strong></p>
          <div class="markdown-content">
            {@html snarkdown(m.message || "")}
          </div>
        </div>
      {/each}
    </div>
  </div>

  <div class="app-conversation-layout__fixed-footer app-conversation-layout__width-restrictor">
    {#if isLoading}
      <div class="govuk-body govuk-hint govuk-!-margin-bottom-2">GOV.UK Onward Journey Agent is typing...</div>
    {/if}
    <QuestionForm onSend={handleSendMessage} />
  </div>
</main>

<style>
/* ... Styles remain unchanged [cite: 55-71] ... */
.app-conversation-layout__header { padding: 10px 20px;
   border-bottom: 1px solid #b1b4b6; background: #ffffff; z-index: 10; }
.handoff-banner { display: flex; justify-content: space-between; align-items: center; margin: 0; border-color: #1d70b8; }
.app-conversation-layout__wrapper { flex: 1; overflow-y: auto; }
:global(body) { font-family: "GDS Transport", arial, sans-serif; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
.app-conversation-layout__main { display: flex; flex-direction: column; height: 100vh; background-color: #ffffff; overflow: hidden; }
.app-conversation-layout__wrapper { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; }
.history-list { max-height: 150px; overflow-y: auto; margin-bottom: 15px; padding-right: 10px; }
.govuk-body-s { font-size: 16px; margin-bottom: 8px; line-height: 1.3; }
.app-conversation-layout__fixed-footer { background: white; border-top: 1px solid #b1b4b6; padding: 15px 0; }
.handoff-banner { display: flex; justify-content: space-between; align-items: center; border-color: #1d70b8; margin-bottom: 20px; }
.message-feed { display: flex; flex-direction: column; gap: 20px; width: 100%; }
.message-bubble { padding: 15px; border-radius: 4px; max-width: 80%; line-height: 1.5; }
.message-bubble.agent { background-color: #f3f2f1; border-left: 4px solid #1d70b8; align-self: flex-start; }
.message-bubble.user { background-color: #005ea5; color: white; align-self: flex-end; }
.message-bubble.user :global(strong) { color: white; }
.markdown-content :global(strong) { font-weight: 700; }
.markdown-content :global(h3), .markdown-content :global(h4) { font-size: 19px; font-weight: bold; margin: 10px 0; display: block; color: #0b0c0c; }
.markdown-content :global(ul) { margin: 10px 0 10px 20px; list-style-type: disc; }
.markdown-content :global(li) { margin-bottom: 5px; }
.markdown-content :global(p) { margin-bottom: 10px; }
</style>