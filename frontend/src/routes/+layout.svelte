<script lang="ts">
  import { resolve } from "$app/paths";
  import GovUKLogo from "$lib/components/GovUKLogo.svelte";
  import "../app.scss";

  let { children } = $props();
</script>

<style>
  .govuk-header__product-name {
    margin-right: 10px;
  }
</style>

<svelte:head>
  <link rel="icon" href="/assets/rebrand/images/favicon.svg" />
</svelte:head>

<header class="govuk-header" data-module="govuk-header">
  <div class="govuk-header__container govuk-width-container">
    <div class="govuk-header__logo">
      <a href={resolve("/")} class="govuk-header__link govuk-header__link--homepage">
        <GovUKLogo />
        <span class="govuk-header__product-name">Agents Onward Journey</span>
        <span class="govuk-tag govuk-tag--yellow">
          Experimental
        </span>
      </a>
    </div>
  </div>
</header>

{@render children()}
