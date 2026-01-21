<script lang="ts">
	import ConversationMessage from "$lib/components/ConversationMessage.svelte";
</script>

<main class="app-conversation-layout__main" id="main-content">
	<div class="app-conversation-layout__wrapper app-conversation-layout__width-restrictor">
		<div class="app-conversation-layout__message-container">
			<ul id="app-conversation-layout__list">
				<ConversationMessage message="Hello" user={"GOV.UK Onward Journey Agent"} isSelf={false} />
				<ConversationMessage message="I would like to talk to someone" user={"You"} isSelf={true} />
			</ul>
		</div>

		<div class="app-conversation-layout__form-region">
			<div class="app-c-question-form">
				<form class="app-c-question-form__form">
					<!-- TODO hook up an action to this from which just displays the user message on the page -->
					<div class="app-c-question-form__form-group">
						<div class="app-c-question-form__textarea-wrapper">
							<textarea class="app-c-question-form__textarea" name="message" placeholder="Enter your question or message" rows=1></textarea>
						</div>
						<div class="app-c-question-form__button-wrapper">
							<div class="app-c-blue-button govuk-button app-c-blue-button--question-form">
								Start
							</div>
						</div>
					</div>
				</form>
			</div>
		</div>
	</div>
</main>
