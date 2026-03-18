<script lang="ts">
  // Use $props() to receive the backend function from the parent
  let { value = $bindable(''), onSend, isLoading } = $props<{ value: string, onSend: (text: string) => void }>();

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
    if (value.trim() !== "") {
      hasValue = true;
    } else {
      hasValue = false;
    }
  })
</script>

<form onsubmit={handleSubmit} class="layout">
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
          ↑
        </button>
      {:else}
        <button 
            class="ios-send-btn" 
            disabled={!value.trim() || isLoading}
            aria-label="more options"
        > ...
        </button>
      {/if}
      
    </div>
  <!-- </div> -->
</form>

<style>
  .layout {
    flex-grow: 1;

  }
  .text-input-mobile {
    max-width: 100%;
    flex-grow: 1;
    border: none;
    border-radius: 20px;
  }
  .text-input-wrapper {
    width: 100%;
    border: none;
    position: relative;
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .app-c-question-form__textarea {
    border-radius: 20px;
  }
  .app-c-question-form__textarea:focus {
    outline: 1px solid black;
  }
  .ios-send-btn {
    width: 34px;
    height: 34px;
    background-color: #007aff;
    border-radius: 50%;
    border: none;
    color: white;
    flex-shrink: 1;
    margin-bottom: 3px;
}
  .circular-button {
    width: 32px;
    height: 32px;
    position: absolute;
    top: 7px;
    right: 20px;
    border: none;
    cursor: pointer;
    border-radius: 50%;
    background-color: #007aff;
    color: white;
  }
</style>