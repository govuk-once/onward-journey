<script lang="ts">
  import type { ListableConversationMessageProps } from "$lib/types/ConversationMessage";
  import ConversationMessage from "$lib/components/ConversationMessage.svelte";

  interface Props {
    messages: ListableConversationMessageProps[]
  }

  let { messages }: Props = $props();
</script>

<div class="app-conversation-layout__message-container">
  <ul id="app-conversation-layout__list">
    {#each messages as message (message.id)}
      <ConversationMessage {...message} />
    {/each}
  </ul>
</div>
