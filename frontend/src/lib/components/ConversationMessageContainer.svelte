<script>
  let { children } = $props();
</script>

<div class="app-conversation-layout__message-container">
  <ul id="app-conversation-layout__list">
    {@render children()}
  </ul>
</div>
