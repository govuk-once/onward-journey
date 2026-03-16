<script lang="ts">
  // Use $props() to receive the backend function from the parent
  let { onSend } = $props<{ onSend: (text: string) => void }>();

  let value = $state("");
  let hasValue = $state(false);

  function handleSubmit(e: SubmitEvent) {
    e.preventDefault();
    if (!value.trim()) return;

    // Trigger the parent's function
    onSend(value);

    // Clear the input
    value = "";
  }

  $effect(() => {
    if (value !== "") {
      hasValue = true;
    } else {
      hasValue = false;
    }
  })
</script>

<form onsubmit={handleSubmit} class="app-conversation-layout__form">
  <div class="govuk-form-group">
    <!-- <label class="govuk-label" for="onward-journey-input">
      Ask a question
    </label> -->
    <!-- <div class="flex gap-2">
      <input
        bind:value
        class="text-input app-c-question-form__textarea"
        id="onward-journey-input"
        type="text"
        placeholder="Ask a question..."
      />
      <button type="submit" class="app-c-blue-button app-c-blue-button--question-form govuk-button w-auto" data-module="govuk-button">
        Send
      </button>
    </div> -->
     <div class="text-input-wrapper flex gap-2">
      <input
        bind:value
        class="text-input-mobile app-c-question-form__textarea govuk-!-padding-right-2"
        id="onward-journey-input"
        type="text"
        placeholder="Ask a question..."
      />
      {#if hasValue} 
      <button type="submit" class="circular-button" data-module="govuk-button">
        ^
      </button>
      {:else}
      <button type="submit" class="circular-button" data-module="govuk-button">
        ...
      </button>
      {/if}
    </div>
  </div>
</form>

<style>
  .text-input {
    max-width: 90%;
    flex-grow: 1;
  }
  .text-input-mobile {
    max-width: 100%;
    flex-grow: 1;
    border: none;
  }
  .app-c-question-form__textarea {
    border-radius: 20px;
  }
  .text-input-wrapper {
    width: 100%;
    border: none;
    position: relative;
    display: inline-block;
  }
  .circular-button {
    width: 30px;
    height: 30px;
    position: absolute;
    top: 8px;
    right: 20px;
    border: none;
    background-color: transparent;
    cursor: pointer;
    border-radius: 50%;
    background-color:	#1d70b8;
    color: white;
  }
</style>