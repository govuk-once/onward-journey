<script lang="ts">
    import homeIcon from "$lib/assets/FooterBar/home.svg";
    import chatIcon from "$lib/assets/FooterBar/Chat Icon.svg";
    import settingsIcon from "$lib/assets/FooterBar/settings.svg";

    console.log("home icon", homeIcon);
</script>

<div class="homebar-container">
    <div class="homebar-item">
        <img src={homeIcon} alt="Home icon" class="homebar-image" height="30px" width="30px" />
        <p>Home</p>
    </div>
    <div class="homebar-item">
        <img src={chatIcon} alt="Chat icon" class="homebar-image" height="30px" width="30px" />
        <p>Chat</p>
    </div>
    <div class="homebar-item">
        <img src={settingsIcon} alt="Settings icon" class="homebar-image" height="30px" width="30px" />
        <p>Settings</p>
    </div>
</div>

<style>
    .homebar-container {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .homebar-item {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        width: 33%;
        height: 85px;
        font-size: 14px;
    }
</style>