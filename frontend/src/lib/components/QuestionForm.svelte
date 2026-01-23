<script lang="ts">
  import type { ListableConversationMessageProps } from "$lib/types/ConversationMessage";
  import { v7 as uuid } from "uuid";

  interface Props {
    messages: ListableConversationMessageProps[]
  }

  let { messages = $bindable() }: Props = $props()
  let message = $state("")
</script>

<div class="app-conversation-layout__form-region">
  <div class="app-c-question-form">
    <form class="app-c-question-form__form" onsubmit={() => {
      messages.push({
        message,
        user: "You",
        isSelf: true,
        id: uuid()
      })
      message = ""
    }}>

      <div class="app-c-question-form__form-group">
        <div class="app-c-question-form__textarea-wrapper">
          <textarea class="app-c-question-form__textarea" name="message" placeholder="Enter your question or message" rows=1 bind:value={message}></textarea>
        </div>
        <div class="app-c-question-form__button-wrapper">
          <button class="app-c-blue-button govuk-button app-c-blue-button--question-form">
            Start
          </button>
        </div>
      </div>
    </form>
  </div>
</div>
