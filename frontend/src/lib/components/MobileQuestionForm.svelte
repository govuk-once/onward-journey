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
        onkeydown={submitOnEnter}
      ></textarea>
    </div>
</form>

<style>
  .layout {
    flex-grow: 1;

  }
.text-input-mobile {
    width: 100%;
    border: none;
    background: transparent;
    resize: none;
    font-size: 16px;
    line-height: 1.2;
    padding: 8px 0;
    outline: none !important; 
    box-shadow: none !important;
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
    outline:none;
  }
  .app-c-question-form__textarea:focus {
    border:none;
    outline: none;
  }
</style>