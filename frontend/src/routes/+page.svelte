<script lang="ts">
  import { v7 as uuid } from "uuid";
  import type { ListableConversationMessageProps } from "$lib/types/ConversationMessage";
  import ConversationMessageContainer from "$lib/components/ConversationMessageContainer.svelte";
  import QuestionForm from "$lib/components/QuestionForm.svelte";

  // --- Interfaces ---
  interface GenesysHandoff {
    action: string;
    deploymentId: string;
    region: string;
    token: string;
    reason: string;
  }

  import { tick } from 'svelte';

  let scrollContainer: HTMLElement | undefined = $state();

  // This function (action) will run whenever the element is created or updated
  function autoScroll(node: HTMLElement) {
    const observer = new ResizeObserver(() => {
      node.scrollTo({
        top: node.scrollHeight,
        behavior: 'smooth'
      });
    });

    observer.observe(node);

    return {
      destroy() {
        observer.disconnect();
      }
    };
  }
  // --- Reactive State ---
  let messages: ListableConversationMessageProps[] = $state([
    {
      message: "Hello! How can I help you with your Onward Journey today?",
      user: "GOV.UK Onward Journey Agent",
      isSelf: false,
      id: uuid()
    }
  ]);
  let isLoading = $state(false);
  let isLiveChat = $state(false);
  let socket: WebSocket | null = $state(null);
  let sessionToken = $state("");

  /**
   * Initializes the browser WebSocket to Genesys.
   * This replaces the terminal-based 'chat_relay' from the backend.
   */
  function setupGenesysSocket(config: GenesysHandoff) {
    isLiveChat = true;
    sessionToken = config.token;
    
    // Construct the Genesys WebSocket URI
    const uri = `wss://webmessaging.${config.region}/v1?deploymentId=${config.deploymentId}`;
    console.log("Connecting to Genesys at:", uri);
    
    socket = new WebSocket(uri);

    socket.onopen = function() {
      console.log("WebSocket Connection Opened");
      if (socket) {
        // Step 1: Configure the session
        socket.send(JSON.stringify({
          action: "configureSession",
          deploymentId: config.deploymentId,
          token: sessionToken
        }));

        // Step 2: Send initial handoff context to the live agent
        socket.send(JSON.stringify({
          action: "onMessage",
          token: sessionToken,
          message: { 
            type: "Text", 
            text: `System: Handoff initiated. Reason: ${config.reason}` 
          }
        }));
      }
    };

    socket.onmessage = function(event) {
      const data = JSON.parse(event.data);
      
      // 1. Check if it's a StructuredMessage
      // 2. Ensure it has a text body
      // 3. CRITICAL: Only show if direction is "Outbound" (from Agent to User)
      if (
        data.class === "StructuredMessage" && 
        data.body && 
        data.body.text && 
        data.body.direction === "Outbound" // Added this check
      ) {
        messages = [...messages, {
          message: data.body.text,
          user: "Live Agent",
          isSelf: false,
          id: uuid()
        }];
      }
    };

    socket.onclose = function() {
      console.log("WebSocket Closed");
      isLiveChat = false;
      messages = [...messages, {
        message: "Live chat session has ended.",
        user: "System",
        isSelf: false,
        id: uuid()
      }];
    };

    socket.onerror = function(err) {
      console.error("WebSocket Error:", err);
    };
  }

  /**
   * Main function to handle outgoing messages.
   */
  async function handleSendMessage(userText: string) {
    if (!userText.trim() || isLoading) return;

    // 1. Add user message to UI immediately
    messages = [...messages, {
      message: userText,
      user: "You",
      isSelf: true,
      id: uuid()
    }];

    // 2. Route to Live Chat if active (Bypasses AI Backend)
    if (isLiveChat && socket && socket.readyState === 1) {
      socket.send(JSON.stringify({
        action: "onMessage",
        token: sessionToken,
        message: { type: "Text", text: userText }
      }));
      return;
    }

    // 3. AI Backend Flow
    isLoading = true;
    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText })
      });

      if (!res.ok) throw new Error("Server error: " + res.status);

      const data = await res.json();
      const responseText = data.response;

      // 4. Check for Handoff Signal string returned by connect_to_live_chat tool
      if (responseText && responseText.includes("initiate_live_handoff")) {
        try {
          // Extract JSON from conversational AI text
          const jsonStart = responseText.indexOf('{');
          const jsonEnd = responseText.lastIndexOf('}') + 1;
          const jsonString = responseText.substring(jsonStart, jsonEnd);
          const config: GenesysHandoff = JSON.parse(jsonString);

          messages = [...messages, {
            message: "Transferring you to a live agent...",
            user: "System",
            isSelf: false,
            id: uuid()
          }];

          // Initialize the WebSocket connection in the browser
          setupGenesysSocket(config);
        } catch (parseError) {
          console.error("Handoff parse error:", parseError);
          messages = [...messages, { message: responseText, user: "Agent", isSelf: false, id: uuid() }];
        }
      } else {
        // Standard AI response
        messages = [...messages, {
          message: responseText,
          user: "GOV.UK Onward Journey Agent",
          isSelf: false,
          id: uuid()
        }];
      }
    } catch (err) {
      console.error("Connection failed:", err);
      messages = [...messages, {
        message: "Sorry, I'm having trouble connecting to the service.",
        user: "System",
        isSelf: false,
        id: uuid()
      }];
    } finally {
      isLoading = false;
    }
  }
</script>

<main class="app-conversation-layout__main">
  <div 
    bind:this={scrollContainer} 
    use:autoScroll
    class="app-conversation-layout__wrapper app-conversation-layout__width-restrictor"
  >
    {#if isLiveChat}
      <div class="govuk-inset-text handoff-banner">
        <p class="govuk-body"><strong>Live Chat:</strong> Connected to advisor.</p>
        <button class="govuk-button govuk-button--warning" onclick={() => socket?.close()}>End</button>
      </div>
    {/if}

    <ConversationMessageContainer {messages} />
  </div>

  <div class="app-conversation-layout__fixed-footer app-conversation-layout__width-restrictor">
    {#if isLoading}
      <div class="govuk-body govuk-hint govuk-!-margin-bottom-2">GOV.UK Onward Journey Agent is typing...</div>
    {/if}
    <QuestionForm onSend={handleSendMessage} />
  </div>
</main>

<style>
.app-conversation-layout__main {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 100px); /* Lock height to viewport [cite: 40] */
  overflow: hidden; /* Prevent page scroll [cite: 41] */
}

.app-conversation-layout__wrapper {
  flex: 1; /* Take all available middle space [cite: 42] */
  overflow-y: auto; /* Enable internal scrolling [cite: 43] */
  padding: 20px;
  display: flex;
  flex-direction: column;
}

.app-conversation-layout__fixed-footer {
  background: white;
  border-top: 1px solid #b1b4b6;
  padding: 15px 0;
}

.handoff-banner {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 0;
  border-color: #1d70b8;
}
</style>