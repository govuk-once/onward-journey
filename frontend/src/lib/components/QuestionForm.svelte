<script lang="ts">
  // Use $props() to receive the backend function from the parent
  let { onSend } = $props<{ onSend: (text: string) => void }>();
  
  let value = $state("");

  function handleSubmit(e: SubmitEvent) {
    e.preventDefault();
    if (!value.trim()) return;

    // Trigger the parent's function
    onSend(value);
    
    // Clear the input
    value = "";
  }
</script>

<form onsubmit={handleSubmit} class="app-conversation-layout__form">
  <div class="govuk-form-group">
    <label class="govuk-label" for="onward-journey-input">
      Ask a question
    </label>
    <div class="flex gap-2">
      <input 
        bind:value 
        class="govuk-input" 
        id="onward-journey-input" 
        type="text"
      />
      <button type="submit" class="govuk-button" data-module="govuk-button">
        Send
      </button>
    </div>
  </div>
</form>