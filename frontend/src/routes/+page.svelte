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
      // Listen for StructuredMessages from the Live Agent
      if (data.class === "StructuredMessage" && data.body && data.body.text) {
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

<main class="app-conversation-layout__main" id="main-content">
  <div class="app-conversation-layout__wrapper app-conversation-layout__width-restrictor">
    
    {#if isLiveChat}
      <div class="govuk-inset-text" style="margin-top: 0; border-color: #1d70b8;">
        <p class="govuk-body"><strong>Live Chat:</strong> You are now connected to a human advisor.</p>
      </div>
    {/if}

    <ConversationMessageContainer {messages} />

    <QuestionForm onSend={handleSendMessage} />

    {#if isLoading}
      <div class="govuk-body govuk-hint p-4">
        OJ Agent is typing...
      </div>
    {/if}
    
  </div>
</main>

<style>
  .app-conversation-layout__main {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  .app-conversation-layout__wrapper {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
  }
</style>