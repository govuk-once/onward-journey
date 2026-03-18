export interface Message {
    id: string;
    user: string;
    message: string;
    isSelf: boolean;
    messageType?: string;
    agentType?: string;
    timestamp?: string;
    agentId?: string;
  }