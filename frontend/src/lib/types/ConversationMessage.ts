import type { Identified } from "./Identified";

export interface ConversationMessageProps {
  message: string;
  user?: string;
  image?: string;
  isSelf: boolean;
}

export type ListableConversationMessageProps = ConversationMessageProps &
  Identified;