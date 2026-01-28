<script lang="ts">
  export type SendMessageHandler = (message: string) => void;

  interface Props {
    messageHandler: SendMessageHandler
  }

  let { messageHandler }: Props = $props()
  let message = $state("")
</script>

<div class="app-conversation-layout__form-region">
  <div class="app-c-question-form">
    <form class="app-c-question-form__form" onsubmit={() => {
      messageHandler(message)
      message = ""
    }}>

      <div class="app-c-question-form__form-group">
        <div class="app-c-question-form__textarea-wrapper">
          <textarea
            class="app-c-question-form__textarea"
            name="message"
            placeholder="Enter your question or message"
            rows=1
            bind:value={message}
            onkeydown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                if (message.trim()) {
                  messageHandler(message)
                  message = ""
                }
              }
            }}
          ></textarea>
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