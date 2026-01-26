import { error } from "@sveltejs/kit";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async () => {
  const supportChatUrl = process.env["SUPPORT_CHAT_URL"];
  if (!supportChatUrl) {
    error(
      500,
      "Please provide a support chat URL via the SUPPORT_CHAT_URL environment variable",
    );
  }

  const deploymentKey = process.env["DEPLOYMENT_KEY"];
  if (!deploymentKey) {
    error(
      500,
      "Please provide a web messaging deployment key via the DEPLOYMENT_KEY environment variable",
    );
  }

  return {
    supportChatUrl,
    deploymentKey,
  };
};
