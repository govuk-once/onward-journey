<script lang="ts">
	interface Props {
		message: string;
		user?: string;
		isSelf: boolean;
	}

	let { message, user, isSelf = true }: Props = $props();
</script>

<li class="app-c-conversation-message">
	<div class="app-c-conversation-message__message {isSelf ? 'app-c-conversation-message__message--user-message' : 'app-c-conversation-message__message--govuk-message'}">
		<div class="app-c-conversation-message__body {isSelf ? 'app-c-conversation-message__body--user-message' : 'app-c-conversation-message__body--govuk-message'}">
			{#if user}
			<div class="app-c-conversation-message__identifier {isSelf ? 'govuk-visually-hidden' : ''}">
				<strong>{user}</strong>
			</div>
			{/if}
			<div class="govuk-body-m govuk-!-margin-bottom-0">{message}</div>
		</div>
	</div>
</li>
