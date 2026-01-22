<script lang="ts">
  import type { Props as ConversationMessageProps } from "$lib/components/ConversationMessage.svelte";
  import ConversationMessageContainer from "$lib/components/ConversationMessageContainer.svelte";
  import QuestionForm from "$lib/components/QuestionForm.svelte";

	let messages: ConversationMessageProps[] = $state([
		{
			message: "Hello",
			user: "GOV.UK Onward Journey Agent",
			isSelf: false
		},
		{
			message: "I would like to talk to someone",
			user: "You",
			isSelf: true
		}
	]);
</script>

<main class="app-conversation-layout__main" id="main-content">
	<div class="app-conversation-layout__wrapper app-conversation-layout__width-restrictor">
		<ConversationMessageContainer messages={messages}/>

		<!-- For now, this simply adds adds a message to the page when submitted
		 		 TODO: connect to genesys web messaging guest API and send the message there -->
		<QuestionForm bind:messages={messages}/>
	</div>
</main>
