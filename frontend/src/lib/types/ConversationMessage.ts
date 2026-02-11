import type { Identified } from "./Identified";

export interface ConversationMessageProps {
  message: string;
  user?: string;
  image?: string;
  isSelf: boolean;
  timestamp?: string;
  agentId?: string;
}

export type ListableConversationMessageProps = ConversationMessageProps &
  Identified;
