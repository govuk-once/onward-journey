<script lang="ts">
  let { value = $bindable(''), isLoading, onSend } = $props<{ 
                                                      value: string, 
                                                      isLoading: boolean
                                                      onSend: (text: string) => void 
                                                    }>();

  let hasValue: boolean = $state(false);
  let textArea = $state(null as HTMLTextAreaElement | null)

  function handleSubmit(e: Event) {
    e.preventDefault();
    if (!value.trim()) return;

    onSend(value);

    value = "";
  }

  const submitOnEnter = (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        handleSubmit(event)

        if (!textArea) {
          return;
        }
        textArea.style.height = "auto";
      }
    }


  const adjustTextAreaSize = () => {
    if (!textArea) {
      return;
    }
    
    textArea.style.height = "auto";
    textArea.style.height = `${textArea.scrollHeight}px`
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
     <div class="text-input-wrapper">
      <textarea
        bind:value
        bind:this={textArea}
        class="text-input-mobile app-c-question-form__textarea govuk-!-padding-right-2"
        id="onward-journey-input"
        placeholder="Ask a question..."
        rows=1
        oninput={adjustTextAreaSize}
        onkeydown={submitOnEnter}
      ></textarea>
      {#if hasValue} 
        <button type="submit" class="circular-button" data-module="govuk-button">
          ↑
        </button>
      {:else}
        <button 
            class="circular-button" 
            disabled={!value.trim() || isLoading}
            aria-label="more options"
            onclick={handleSubmit}
        > 
          ...
        </button>
      {/if}
      
    </div>
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
    height: auto;
    overflow: hidden;
    resize: none;
    line-height: 1.2;
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
  .circular-button {
    width: 34px;
    height: 34px;
    background-color: #007aff;
    border-radius: 50%;
    border: none;
    color: white;
    flex-shrink: 1;
    margin-bottom: 3px;
}
.circular-button:disabled {
  background-color: grey;
}
</style>