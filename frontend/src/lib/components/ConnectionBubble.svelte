<script lang="ts">
    let { agentType, agentName } = $props<{ agentType: string, agentName: string }>();

    const imageOptions = {
        "human": {
            "alt": "Caspar Advisor",
            "message": (name: string) => `${name} has joined the chat`
        }
    };
    let profile = $derived(imageOptions[agentType as keyof typeof imageOptions]);
</script>

<div class="connection-row">
    <div class="bubble">
        <div class="wrapper">
            <div class="caspar-icon">C</div>
            <h4 class="govuk-!-margin-top-1 govuk-!-margin-bottom-2">
                {profile.message(agentName)}
            </h4>
        </div>
    </div>
</div>

<style>
    .connection-row {
        width: 100%;
        display: flex;
        justify-content: center;
    }
    .wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }
    .caspar-icon {
        background: #1D70B8; /* Caspar Blue */
        color: white;
        border-radius: 50%;
        height: 48px;
        width: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 8px;
    }
    .bubble { 
        padding: 16px; 
        border-radius: 12px;
        width: 100%; /* Full Width */
        background: white; 
        color: #000; 
        box-sizing: border-box;
    }
</style>