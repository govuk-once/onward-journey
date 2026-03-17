<script lang="ts">
    import humanAgentPhoto from "$lib/assets/humanAgent.png";
    import aiAgentPhoto from "$lib/assets/govuk-icon.png";

    let { agentType, agentName } = $props<{ 
                            agentType: string,
                            agentName: string, 
                        }>();

    const imageOptions: Record<string, { image: string; alt: string; message: (name: string) => string }> = {
        "ai": {
            "image": aiAgentPhoto,
            "alt": "GOVUK Chat Agent",
            "message": (name) => `You have been connected to the ${name}`
        },
        "human": {
            "image": humanAgentPhoto,
            "alt": "Photo of your human chat partner",
            "message": (name) => `${name} has joined the chat`
        }
    };
    
    let profile = $derived(imageOptions[agentType])
</script>


<div class="bubble">
    <div class="wrapper">
        <div class="heading">
            <img src={profile.image} alt={profile.alt} class="profile-image"/>
            <h4 class="govuk-!-margin-top-1 govuk-!-margin-bottom-2">{profile.message(agentName)}</h4>
        </div>
    </div>
</div>

<style>
    .wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .heading {
        border-bottom: 2px solid #e6e6e6;
    }
    .profile-image {
        background: #007aff;
        border-radius: 50%;
        height: 20%;
        width: 20%;
        
    }
    .bubble { 
        padding: 16px; 
        border-radius: 18px; 
        max-width: 85%; 
        font-size: 15px; 
        line-height: 1.5; 
        word-wrap: break-word; 
        box-sizing: border-box;
        background: #f2f2f7; 
        color: #000; 
        align-self: flex-start; 
        border-bottom-left-radius: 4px;
  }
</style>