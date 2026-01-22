<script lang="ts">
	import ConversationMessage from "$lib/components/ConversationMessage.svelte";
  import ConversationMessageContainer from "$lib/components/ConversationMessageContainer.svelte";
  import QuestionForm from "$lib/components/QuestionForm.svelte";
</script>

<main class="app-conversation-layout__main" id="main-content">
	<div class="app-conversation-layout__wrapper app-conversation-layout__width-restrictor">
		<ConversationMessageContainer>
			<ConversationMessage message="Hello" user="GOV.UK Onward Journey Agent" isSelf={false} />
			<ConversationMessage message="I would like to talk to someone" user="You" isSelf={true} />
		</ConversationMessageContainer>


		<!-- TODO hook up an action to this from which just displays the user message on the page -->
		<QuestionForm />
	</div>
</main>
