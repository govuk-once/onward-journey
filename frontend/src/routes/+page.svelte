<script lang="ts">
  import type { ListableConversationMessageProps } from "$lib/types/ConversationMessage";
  import ConversationMessageContainer from "$lib/components/ConversationMessageContainer.svelte";
  import QuestionForm from "$lib/components/QuestionForm.svelte";
  import type { SendMessageHandler } from "$lib/components/QuestionForm.svelte";
  import { v7 as uuid } from "uuid";
  import { GenesysClient } from "$lib/services/genesysClient.js";
  import { onMount } from "svelte";
  import { PUBLIC_SUPPORT_CHAT_URL as SUPPORT_CHAT_URL, PUBLIC_DEPLOYMENT_KEY as DEPLOYMENT_KEY } from "$env/static/public";

  const sleep = (time: number) => new Promise(resolve => setTimeout(resolve, time))

  let sendMessageHandler: SendMessageHandler = $state((message) => {
    messages.push({
      message,
      user: "You",
      isSelf: true,
      id: uuid()
    })
  })

  onMount(() => {
    const genesysClient = new GenesysClient({ websocketUrl: SUPPORT_CHAT_URL, deploymentKey: DEPLOYMENT_KEY });

    // Log all messages in either direction for easy debugging
    genesysClient.on("rawMessage", (msg) => {
      console.log(msg);
    });

    genesysClient.connect().then(async () => {
      console.log("Genesys client connected");

      genesysClient.on("message", (msg) => {
        if (msg.direction == "Outbound" && msg.text) {
          messages.push({
            message: msg.text,
            user: "Genesys support",
            isSelf: false,
            id: msg.id!
          })
        }
      })

      const addToPage = sendMessageHandler;
      sendMessageHandler = (message) => {
        addToPage(message);
        genesysClient.sendMessage(message);
      }

      genesysClient.configureSession();
      await sleep(500);
      genesysClient.sendMessage("Hi");
    });

    // Cleanup on component destroy
    return () => {
      genesysClient.disconnect();
    };
  });

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
    <ConversationMessageContainer messages={messages}/>

    <QuestionForm messageHandler={sendMessageHandler}/>
  </div>
</main>
