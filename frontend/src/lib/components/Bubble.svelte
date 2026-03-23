<script lang="ts">
    import SvelteMarkdown from "svelte-markdown";
    import ConnectionBubble from "./ConnectionBubble.svelte";
    import type { Message } from "$lib/types/Message";

    let { message } = $props<{message: Message}>();
</script>

{#if message?.messageType === "agent-transfer" && message.agentType === "human"}
    <ConnectionBubble agentName={message.user} agentType={message.agentType}/>
{:else if message?.messageType === "agent-transfer" && message.agentType === "ai"}
    <ConnectionBubble agentName={message.user} agentType={message.agentType}/>\
{:else}
    <div class="ios-bubble {message.isSelf ? 'user' : 'agent'}">
        <div>
            <SvelteMarkdown source={message.message || ""} />
        </div>
    </div>
{/if}


<style>
    .ios-bubble { 
        padding: 12px 16px; 
        border-radius: 18px; 
        max-width: 85%; 
        font-size: 15px; 
        line-height: 1.5; 
        word-wrap: break-word; 
        box-sizing: border-box;
    }
    .ios-bubble.agent {
        background: white;
        color: black;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
    }
    .ios-bubble.user {
        background: #0F385C;
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 4px;
    }
</style>