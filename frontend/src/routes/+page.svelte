<script lang="ts">
  import type { ListableConversationMessageProps } from "$lib/types/ConversationMessage";
  import ConversationMessageContainer from "$lib/components/ConversationMessageContainer.svelte";
  import QuestionForm from "$lib/components/QuestionForm.svelte";
  import type { SendMessageHandler } from "$lib/components/QuestionForm.svelte";
  import { v7 as uuid } from "uuid";
  import { GenesysClient } from "$lib/services/genesysClient.js";
  import { onMount } from "svelte";
  import { env } from "$env/dynamic/public";
  import { error } from "@sveltejs/kit";

  const sleep = (time: number) => new Promise(resolve => setTimeout(resolve, time))

  let sendMessageHandler: SendMessageHandler = $state((message) => {
    messages.push({
      message,
      isSelf: true,
      id: uuid()
    })
  })

  let showTypingIndicator: boolean = $state(false)

  onMount(() => {
    const websocketUrl = env.PUBLIC_SUPPORT_CHAT_URL;
    const deploymentKey = env.PUBLIC_DEPLOYMENT_KEY;

    if (!websocketUrl || !deploymentKey) {
      error(500, "Genesys configuration not set. Please set environment variables referenced in README.md");
    }

    const genesysClient = new GenesysClient({ websocketUrl, deploymentKey });

    // Log all messages in either direction for easy debugging
    genesysClient.on("rawMessage", (msg) => {
      console.log(msg);
    });

    genesysClient.connect().then(async () => {
      genesysClient.on("message", (msg) => {

        // When we receive a new message from the agent, show it on the page
        if (msg.direction == "Outbound" && msg.text) {
          messages.push({
            message: msg.text!,
            user: msg.channel?.from?.nickname ?? "Unknown " + msg.originatingEntity,
            image: msg.channel?.from?.image,
            isSelf: false,
            id: msg.id!
          })
        }

        // When we receive a typing indicator event, show the typing indicator for the duration specified
        if (msg.type == "Event" && msg.events?.find((e) => e.eventType == "Typing")) {
          const typingEvent = msg.events?.find((e) => e.eventType == "Typing")
          showTypingIndicator = true
          if (typingEvent && typingEvent.typing.duration) {
            setTimeout(() => {
              showTypingIndicator = false
            }, typingEvent.typing.duration)
          }
        }
      })

      const addToPage = sendMessageHandler;
      sendMessageHandler = (message) => {
        addToPage(message);
        genesysClient.sendMessage(message);
      }

      genesysClient.configureSession();
      await sleep(500);
      // We need to send an initial message to start the bot triage
      genesysClient.sendMessage("I am being transferred from Onward Journey");
    });

    // Cleanup on component destroy
    return () => {
      genesysClient.disconnect();
    };
  });

  // Static initial message to demonstrate what a handover might look like
  let messages: ListableConversationMessageProps[] = $state([
    {
      message: "I am connecting you to a support agent",
      user: "GOV.UK Onward Journey Agent",
      isSelf: false,
      id: uuid()
    }
  ]);
</script>

<main class="app-conversation-layout__main" id="main-content">
  <div class="app-conversation-layout__wrapper app-conversation-layout__width-restrictor">
    <ConversationMessageContainer messages={messages} showTypingIndicator={showTypingIndicator}/>

    <QuestionForm messageHandler={sendMessageHandler}/>
  </div>
</main>
