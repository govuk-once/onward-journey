<script lang="ts">
  import type { Props as ConversationMessageProps } from "$lib/components/ConversationMessage.svelte";
  import ConversationMessage from "$lib/components/ConversationMessage.svelte";

  interface Props {
    messages: ConversationMessageProps[]
  }

  let { messages }: Props = $props();
</script>

<div class="app-conversation-layout__message-container">
  <ul id="app-conversation-layout__list">
    {#each messages as message (message.message, message.user)}
      <ConversationMessage {...message} />
    {/each}
  </ul>
</div>
