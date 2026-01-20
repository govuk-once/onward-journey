<script lang="ts">
	import ConversationMessage from "$lib/components/ConversationMessage.svelte";
</script>

<div class="govuk-width-container">
	<main class="govuk-main-wrapper" id="main-content">
		<ul class="app-c-conversation-list">
			<ConversationMessage message="Hello" user={"GOV.UK Onward Journey Agent"} isSelf={false} />
			<ConversationMessage message="I would like to talk to someone" user={"You"} isSelf={true} />
		</ul>
	</main>
</div>

<style>

</style>
