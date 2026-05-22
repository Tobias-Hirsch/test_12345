import { defineStore } from 'pinia';
import { ref, Ref, toRaw } from 'vue';
import { get, post, del, postStream } from '@/services/apiService';

// Define interfaces for Conversation and Message based on expected backend structure
export interface Message {
  _id?: string;
  conversation_id?: string;
  sender: 'user' | 'bot';
  content: string;
  source_documents?: any[];
  time?: string;
  attachments?: any[];
  // Update search_results to be more flexible for frontend display
  search_results?: {
    source_documents?: Array<{
      rag_item_name: string;
      content: string;
      content_preview?: string;
      file_id?: number;
      filename?: string;
      source?: string;
    }>;
    online_results?: Array<{ title: string; href: string; body: string }>;
    [key: string]: any; // Allow for other properties
  };
  // Add a loading state for streaming messages
  loading?: boolean;
  thinkingProcess?: string; // Use ref for reactivity in Vue 3
  // Add search options to Message interface
  search_ai_active?: boolean;
  search_rosti_active?: boolean;
  search_online_active?: boolean;
  show_think_process?: boolean; // Add this line
  feedback?: 'like' | 'dislike' | null; // Add this for feedback
}
 
export interface Conversation {
  _id: string;
  title: string;
  created_at?: string;
  updated_at?: string;
  messages?: Message[]; // Changed from Ref<Message>[]
}

const SUMMARY_ARTIFACT_PATTERNS = [
  /multiple items of type/i,
  /requires a single string output/i,
  /single string output/i,
  /mehreren zeichenfolgen/i,
  /mehrere zeichenfolgen/i,
  /ein einzelner string/i,
];

const sanitizeMessageText = (content: unknown): string => {
  if (typeof content !== 'string') return '';

  const paragraphs = content.split(/\n\s*\n/);
  const kept: string[] = [];

  paragraphs.forEach((paragraph, index) => {
    const trimmed = paragraph.trim();
    const hasArtifact = SUMMARY_ARTIFACT_PATTERNS.some((pattern) => pattern.test(trimmed));

    if (hasArtifact) {
      if (kept.length > 0 && /^(hinweis|note)\s*:?\s*$/i.test(kept[kept.length - 1])) {
        kept.pop();
      }
      return;
    }

    if (/^(hinweis|note)\s*:?\s*$/i.test(trimmed)) {
      const nextParagraph = paragraphs[index + 1]?.trim() || '';
      if (SUMMARY_ARTIFACT_PATTERNS.some((pattern) => pattern.test(nextParagraph))) {
        return;
      }
    }

    if (trimmed) kept.push(trimmed);
  });

  return kept.join('\n\n').trim() || content;
};

const sanitizeSourceDocuments = (sourceDocuments: unknown): any[] => {
  if (!Array.isArray(sourceDocuments)) return [];

  return sourceDocuments.map((source) => {
    if (!source || typeof source !== 'object') return source;

    const sanitizedSource = { ...(source as Record<string, any>) };
    if (typeof sanitizedSource.summary === 'string') {
      const hasArtifact = SUMMARY_ARTIFACT_PATTERNS.some((pattern) => pattern.test(sanitizedSource.summary));
      if (hasArtifact) sanitizedSource.summary = 'No reliable summary available.';
    }
    return sanitizedSource;
  });
};
 
// Helper function to convert raw message data to Message interface (plain strings)
// Helper function to convert raw message data to Message interface (plain strings)
const convertMessageContent = (message: any): Message => {
  const sourceDocuments = sanitizeSourceDocuments(
    message?.search_results?.source_documents || message?.source_documents || []
  );

  return {
    ...message,
    sender: message.sender || 'bot', // Ensure sender is always present, default to 'bot'
    content: sanitizeMessageText(message.content || ''), // Ensure it's always a string
    thinkingProcess: message.thinkingProcess || '', // Ensure it's always a string
    source_documents: sourceDocuments,
    search_results: {
      ...(message.search_results || {}),
      source_documents: sourceDocuments,
    },
  };
};

export const useChatStore = defineStore('chat', () => {
  // State
  const conversations = ref<Conversation[]>([]);
  const currentConversation = ref<Conversation | null>(null);
  const messages = ref<Message[]>([]); // Changed from Ref<Message>[]
  const showThinkProcess = ref<boolean>(false); // New state to control visibility of thinking process
  const isSending = ref<boolean>(false); // Add sending state
 
  // Actions
  const fetchConversations = async () => {
    try {
      const response = await get('/chat/conversations/');
      // console.log("Response from /chat/conversations/:", response);
      if (response !== null && Array.isArray(response)) {
        conversations.value = response.map((conv: any) => ({
          ...conv,
          messages: conv.messages ? conv.messages.map((msg: any) => convertMessageContent(msg)) : undefined,
        })) as Conversation[];
      } else {
        // console.error("Received unexpected data for conversations:", response);
        conversations.value = [];
      }
    } catch (error) {
      // console.error("Error fetching conversations:", error);
      conversations.value = [];
    }
  };
 
  const selectConversation = async (conversation: Conversation) => {
    currentConversation.value = conversation;
    try {
      const response = await get(`/chat/conversations/${conversation._id}/messages/`);
      if (response !== null && Array.isArray(response)) {
        messages.value = response.map((msg: any) => convertMessageContent(msg)) as Message[];
      } else {
        // console.error("Received unexpected data for messages:", response);
        messages.value = [];
      }
    } catch (error) {
      // console.error(`Error fetching messages for conversation ${conversation._id}:`, error);
      messages.value = [];
    }
  };
 
  // Action to create a new conversation and add the first message
  const createConversationAndAddMessage = async (messageContent: string, attachments: any[], searchOptions: { search_ai_active: boolean; search_rosti_active: boolean; search_online_active: boolean; show_think_process: boolean }) => {
      try {
          // 1. Create a new conversation using the first message content as title
          const newConversationTitle = messageContent.substring(0, 50) + (messageContent.length > 50 ? '...' : ''); // Use first 50 chars as title
          const createResponse = await post('/chat/conversations/', { title: newConversationTitle });
          const newConversation = createResponse as Conversation;
 
          if (newConversation && newConversation._id) {
              // Set the newly created conversation as the current one
              currentConversation.value = newConversation;
 
              // 2. Add the first message to the new conversation
              const firstMessage: Message = {
                  sender: 'user',
                  content: messageContent,
                  attachments: attachments,
                  ...searchOptions
              };
              // Add the user message to the state immediately
              messages.value.push(firstMessage); // Removed ref()
  
              // Add a placeholder for the bot's streaming message
              const botMessagePlaceholder: Message = {
                  _id: `temp-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`, // Add a temporary ID
                  sender: 'bot',
                  content: '',
                  loading: true, // Indicate loading state
                  time: new Date().toISOString(), // Add timestamp
                  source_documents: [], // Initialize source documents
                  thinkingProcess: ''
              };
              messages.value.push(botMessagePlaceholder); // Removed ref()
  
              // Use postStream for the bot's response
              await postStream(`/chat/conversations/${newConversation._id}/messages/`, toRaw(firstMessage), (chunk) => {
                const botMessage = messages.value.find(msg => msg._id === botMessagePlaceholder._id);
                if (botMessage) {
                    const lines = chunk.split('\n').filter((line: string) => line.length > 0);
                    let eventName = 'message';
                    let eventData = '';
            
                    lines.forEach((line: string) => {
                        if (line.startsWith('event:')) {
                            eventName = line.substring(6).trim();
                        } else if (line.startsWith('data:')) {
                            eventData = line.substring(5).trim();
                        }
                    });
            
                    if (eventData) {
                        try {
                            const parsedData = JSON.parse(eventData);
                            switch (eventName) {
                                case 'thought':
                                    if (!botMessage.thinkingProcess) botMessage.thinkingProcess = '';
                                    botMessage.thinkingProcess += parsedData; // parsedData is a string
                                    break;
                                case 'text':
                                    botMessage.content += parsedData; // parsedData is a string
                                    break;
                                case 'metadata':
                                    if (!botMessage.search_results) botMessage.search_results = {};
                                    botMessage.search_results.source_documents = sanitizeSourceDocuments(parsedData.source_documents);
                                    break;
                                case 'error':
                                     console.error("SSE Error:", parsedData.error);
                                     botMessage.content += `\n\nError: ${parsedData.error}`;
                                     break;
                            }
                        } catch (e) {
                            console.error("Failed to parse JSON from SSE event:", eventName, eventData, e);
                        }
                    }
                }
              });
  
              // After the stream is complete, update the bot message state
              const botMessage = messages.value.find(msg => msg._id === botMessagePlaceholder._id) as Message;
              if (botMessage) {
                  botMessage.loading = false; // Remove loading state
                  console.log(">>> [STATE] Final bot message object after stream (new conversation):", JSON.stringify(toRaw(botMessage), null, 2));
              }
  
              // Add the new conversation to the conversations list and select it
              conversations.value.unshift(newConversation);
              // Refresh messages to get the actual IDs from backend
              if (currentConversation.value) {
                await selectConversation(currentConversation.value);
              }
  
          } else {
              // console.error("Failed to create new conversation: Invalid response data.", createResponse);
              // Optionally, display an error message to the user
              throw new Error("Failed to create new conversation.");
          }
      } catch (error) {
          // console.error("Error creating new conversation and adding message:", error);
          // Handle error
          throw error; // Re-throw to allow calling component to handle
      }
  };
  
  
  const addMessageToConversation = async (conversationId: string, message: Message) => {
    // Add the user message to the state immediately
    messages.value.push(message);
  
    // Add a placeholder for the bot's streaming message
    const botMessagePlaceholder: Message = {
        _id: `temp-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`, // Add a temporary ID
        sender: 'bot',
        content: '',
        loading: true, // Indicate loading state
        time: new Date().toISOString(), // Add timestamp
        source_documents: [], // Initialize source documents
        thinkingProcess: ''
    };
    messages.value.push(botMessagePlaceholder);
  
    try {
      // Use postStream for the bot's response
      await postStream(`/chat/conversations/${conversationId}/messages/`, toRaw(message), (chunk) => {
        const botMessage = messages.value.find(msg => msg._id === botMessagePlaceholder._id);
        if (botMessage) {
            const lines = chunk.split('\n').filter((line: string) => line.length > 0);
            let eventName = 'message';
            let eventData = '';
    
            lines.forEach((line: string) => {
                if (line.startsWith('event:')) {
                    eventName = line.substring(6).trim();
                } else if (line.startsWith('data:')) {
                    eventData = line.substring(5).trim();
                }
            });
    
            if (eventData) {
                try {
                    const parsedData = JSON.parse(eventData);
                    switch (eventName) {
                        case 'thought':
                            if (!botMessage.thinkingProcess) botMessage.thinkingProcess = '';
                            botMessage.thinkingProcess += parsedData; // parsedData is a string
                            break;
                        case 'text':
                            botMessage.content += parsedData; // parsedData is a string
                            break;
                        case 'metadata':
                            if (!botMessage.search_results) botMessage.search_results = {};
                            botMessage.search_results.source_documents = sanitizeSourceDocuments(parsedData.source_documents);
                            break;
                        case 'error':
                             console.error("SSE Error:", parsedData.error);
                             botMessage.content += `\n\nError: ${parsedData.error}`;
                             break;
                    }
                } catch (e) {
                    console.error("Failed to parse JSON from SSE event:", eventName, eventData, e);
                }
            }
        }
      });
   
      // After the stream is complete, update the bot message state
      const botMessage = messages.value.find(msg => msg._id === botMessagePlaceholder._id) as Message;
      if (botMessage) {
          botMessage.loading = false; // Remove loading state
          console.log(">>> [STATE] Final bot message object after stream (existing conversation):", JSON.stringify(toRaw(botMessage), null, 2));
      }
      // Refresh conversations to update the list on the side, but don't re-select.
      await fetchConversations();
   
    } catch (error: any) {
      // console.error(`Error adding message to conversation ${conversationId}:`, error);
      // Find the bot message placeholder and update its content with an error message
      const botMessage = messages.value.find(msg => msg._id === botMessagePlaceholder._id) as Message;
      if (botMessage) {
          botMessage.content = `Error: ${error.message}`;
          botMessage.loading = false; // Remove loading state
      }
    }
  };
 
  // Action to update conversation title
  const updateConversationTitle = async (conversationId: string, newTitle: string) => {
      try {
          const response = await post(`/chat/conversations/${conversationId}/title`, { title: newTitle });
          // Assuming backend returns the updated conversation
          const updatedConversation = response as Conversation;
          // Find and update the conversation in the list
          const index = conversations.value.findIndex(conv => conv._id === conversationId);
          if (index !== -1) {
              conversations.value[index].title = updatedConversation.title;
          }
          // If the updated conversation is the current one, update its title too
          if (currentConversation.value?._id === conversationId) {
              currentConversation.value.title = updatedConversation.title;
          }
      } catch (error) {
          // console.error(`Error updating conversation title ${conversationId}:`, error);
          // Handle error
          throw error;
      }
  };
 
  // Action to delete a conversation
  const deleteConversation = async (conversationId: string) => {
      try {
          await del(`/chat/conversations/${conversationId}/`);
          // Remove the conversation from the list
          conversations.value = conversations.value.filter(conv => conv._id !== conversationId);
          // If the deleted conversation was the current one, clear currentConversation and messages
          if (currentConversation.value?._id === conversationId) {
              currentConversation.value = null;
              messages.value = [];
          }
      } catch (error) {
          // console.error(`Error deleting conversation ${conversationId}:`, error);
          // Handle error
          throw error;
      }
  };
 
 
  const clearCurrentConversation = () => {
      currentConversation.value = null;
      messages.value = [];
  };
 
  return {
    conversations,
    currentConversation,
    messages,
    showThinkProcess,
    isSending,
    fetchConversations,
    selectConversation,
    addMessageToConversation,
    createConversationAndAddMessage,
    updateConversationTitle,
    deleteConversation,
    clearCurrentConversation,
  };
});
