<script lang="ts">
  import type { ConversationMessageProps } from "$lib/types/ConversationMessage"
  let { message, user, image, isSelf = true }: ConversationMessageProps = $props();
</script>

<li class="app-c-conversation-message">
  <div class="app-c-conversation-message__message {isSelf ? 'app-c-conversation-message__message--user-message' : 'app-c-conversation-message__message--govuk-message'}">
    <div class="app-c-conversation-message__body {isSelf ? 'app-c-conversation-message__body--user-message' : 'app-c-conversation-message__body--govuk-message'}">
      {#if image || user}
      <div class="app-c-conversation-message__header">
        {#if image}
        <img class="app-c-conversation-message__image" src={image} alt="" />
        {/if}
        {#if user}
        <div class="app-c-conversation-message__identifier">
          <strong>{user}</strong>
        </div>
        {/if}
      </div>
      {/if}
      <div class="govuk-body-m govuk-!-margin-bottom-0">{message}</div>
    </div>
  </div>
</li>
