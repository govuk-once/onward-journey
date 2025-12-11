<script lang="ts">
	import favicon from "$lib/assets/favicon.svg";
	import { Header, PhaseBanner } from "@communitiesuk/svelte-component-library"

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<Header />
<PhaseBanner
	tagText="Experimental"
	bannerText="This service is an experimental prototype"
	linkText=""
/>

{@render children()}
