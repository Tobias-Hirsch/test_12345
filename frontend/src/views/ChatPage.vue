<template>
  <div class="chat-container">
    <div class="chat-area">
      <!-- Chat-Nachrichtenbereich -->
      <div class="messages-container" ref="messagesContainerRef">
        <div v-if="messages && !messages.some((message: Message) => message.sender === 'user')" class="welcome-top">
          <div class="welcome-message">Welcome to the Multi-Agent Chat!</div>
        </div>
        <template v-for="message in formattedMessages" :key="message.key">
          <div :class="['message-bubble', message.placement]">
            <div class="avatar">
              <img :src="message.avatar" alt="avatar" />
            </div>
            <div class="content-wrapper">
              <div v-if="message.thinkingProcess && showThinkProcess" class="thinking-process-header">
                <strong>Thinking Process:</strong>
                <div class="thinking-process-content">{{ message.thinkingProcess }}</div>
              </div>
              <div class="content">{{ message.content }}</div>
              <div v-if="message.loading" class="loading-indicator">...</div>
            </div>
          </div>
        </template>
      </div>
      <!-- Eingabebereich -->
      <div class="input-area">
        <div class="input-container input-flex">
          <textarea
            v-model="userInput"
            placeholder="Nachricht eingeben ..."
            class="textarea-input"
            rows="1"
            @keyup.enter.prevent="sendMessage"
            @input="adjustTextareaHeight"
            ref="textareaInputRef"
          ></textarea>
        </div>
        <div class="search-switches">
          <div class="switch-group-left">
            <span
              class="search-switch"
              :class="{ active: searchAIActive }"
              @click="searchAIActive = !searchAIActive"
            >{{ $t('rostiChat.searchAI') }}</span>
            <span
              class="search-switch"
              :class="{ active: searchRagActive }"
              @click="searchRagActive = !searchRagActive"
            >{{ $t('rostiChat.searchRostiData') }}</span>
            <span
              class="search-switch"
              :class="{ active: searchOnlineActive }"
              @click="searchOnlineActive = !searchOnlineActive"
            >{{ $t('rostiChat.searchOnline') }}</span>
            <span
              class="search-switch"
              :class="{ active: showThinkProcess }"
              @click="showThinkProcess = !showThinkProcess"
            >{{ $t('rostiChat.showThinkProcess') }}</span>
          </div>
          <div class="switch-group-right">
            <!-- File upload icon with tooltip -->
            <span
              class="attachment-icon sendbar-icon"
              @click="triggerFileInput"
              :title="$t('chatPage.attachmentTooltip')"
            >📎</span>
            <input type="file" multiple @change="handleFileChange" ref="fileInputRef" style="display: none;">
            <button type="button" class="send-btn" @click="sendMessage" :disabled="isLoading || !userInput.trim()">Senden</button>
          </div>
        </div>
        <!-- Display selected files -->
        <div v-if="selectedFiles.length" class="selected-files-preview">
          <span>Selected Files: </span>
          <span
            v-for="(file, index) in selectedFiles"
            :key="index"
            class="file-tag"
          >
            {{ file.name }} <span @click="removeSelectedFile(index)" style="cursor: pointer;">x</span>
          </span>
        </div>
      </div>
    </div>

    <!-- Source Documents Sidebar (simplified for ChatPage.vue) -->
    <div class="source-documents-sidebar" v-if="latestBotMessageSourceDocuments.length > 0">
      <div class="sidebar-header">
        <h3>Source Documents</h3>
      </div>
      <div class="sidebar-content">
        <div v-if="latestBotMessageSourceDocuments.length > 0">
          <h4>Sources:</h4>
          <ul>
            <li v-for="(source, index) in latestBotMessageSourceDocuments" :key="index">
              <strong>{{ source.rag_item_name || 'Unknown Source' }}</strong>: {{ source.content }}
            </li>
          </ul>
        </div>
        <div v-else>
          <p>No source documents found for the latest bot response.</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, nextTick, computed } from 'vue';
import { agentChat } from '@/services/apiService'; // Import the new agentChat service
import { useAuthStore } from '@/stores/auth'; // Import useAuthStore
import { storeToRefs } from 'pinia'; // Import storeToRefs

// Define Message interface to match backend/app/schemas/schemas.py and RostiChatInterface.vue
interface Message {
  _id?: string; // Optional, as it might not be present for user messages initially
  conversation_id?: string;
  sender: 'user' | 'bot';
  content: string;
  timestamp: string;
  attachments?: any[]; // Placeholder for attachments
  source_documents?: any[]; // Placeholder for source documents
  thinkingProcess?: string; // For displaying agent thoughts
  loading?: boolean; // To indicate if the bot is still typing/processing
}

const messages = ref<Message[]>([]);
const userInput = ref('');
const isLoading = ref(false);
const messagesContainerRef = ref<HTMLElement | null>(null);
const textareaInputRef = ref<HTMLTextAreaElement | null>(null); // Ref for the textarea

// Auth store for user information
const authStore = useAuthStore();
const { user } = storeToRefs(authStore);

// Feature toggles (similar to RostiChatInterface.vue)
const searchAIActive = ref(true); // Corresponds to reasoning/planning
const searchRagActive = ref(false); // Corresponds to RAG
const searchOnlineActive = ref(false); // Corresponds to Search Agent
const showThinkProcess = ref(false); // To display agent thoughts

// Default avatars (simplified for ChatPage.vue)
const selectedUserAvatar = ref('https://avatars.githubusercontent.com/u/76239030?v=4');
const selectedBotAvatar = ref('https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png');

onMounted(() => {
  // Optional: Load messages from local storage or a backend API if needed
  // For now, start with an empty chat
});

watch(messages, async () => {
  await nextTick();
  if (messagesContainerRef.value) {
    const container = messagesContainerRef.value;
    container.scrollTop = container.scrollHeight;
  }
}, { deep: true });

const addMessage = (message: Message) => {
  messages.value.push(message);
};

const updateLastBotMessage = (chunk: any) => {
  const lastMessage = messages.value[messages.value.length - 1];
  if (lastMessage && lastMessage.sender === 'bot' && lastMessage.loading) {
    // Append content
    // The backend now sends 'output' and 'thoughts' within the chunk,
    // and the final formatted response has 'final_user_response' and 'agent_thoughts'.
    // We need to handle both intermediate and final formats.
    if (chunk.output) { // Intermediate output from orchestrator
      if (typeof chunk.output === 'string') {
        lastMessage.content += chunk.output;
      } else if (chunk.output.final_user_response) { // Final output from formatting agent
        lastMessage.content += chunk.output.final_user_response;
      }
    }

    // Update thinking process
    if (chunk.thoughts && Array.isArray(chunk.thoughts)) { // Intermediate thoughts from orchestrator
      lastMessage.thinkingProcess = chunk.thoughts.join('\n');
    } else if (chunk.output && chunk.output.agent_thoughts && Array.isArray(chunk.output.agent_thoughts)) { // Final thoughts from formatting agent
      lastMessage.thinkingProcess = chunk.output.agent_thoughts.join('\n');
    }

    // Update status
    if (chunk.status === 'completed') {
      lastMessage.loading = false;
    }
    // Update source documents if available (assuming they come in intermediate chunks)
    if (chunk.source_documents) {
      lastMessage.source_documents = chunk.source_documents;
    }
  } else {
    // If the last message is not a loading bot message, create a new one
    // This handles the very first chunk or if a previous message was completed
    let content = '';
    let thinkingProcess = undefined;
    let sourceDocuments = [];

    if (chunk.output) {
      if (typeof chunk.output === 'string') {
        content = chunk.output;
      } else if (chunk.output.final_user_response) {
        content = chunk.output.final_user_response;
      }
    }

    if (chunk.thoughts && Array.isArray(chunk.thoughts)) {
      thinkingProcess = chunk.thoughts.join('\n');
    } else if (chunk.output && chunk.output.agent_thoughts && Array.isArray(chunk.output.agent_thoughts)) {
      thinkingProcess = chunk.output.agent_thoughts.join('\n');
    }

    if (chunk.source_documents) {
      sourceDocuments = chunk.source_documents;
    }

    addMessage({
      sender: 'bot',
      content: content,
      timestamp: new Date().toISOString(),
      thinkingProcess: thinkingProcess,
      loading: chunk.status !== 'completed',
      source_documents: sourceDocuments,
    });
  }
};

const sendMessage = async () => {
  if (!userInput.value.trim()) return;

  const userMessageContent = userInput.value;
  addMessage({
    sender: 'user',
    content: userMessageContent,
    timestamp: new Date().toISOString(),
  });
  userInput.value = '';
  isLoading.value = true;

  // Reset textarea height
  if (textareaInputRef.value) {
    textareaInputRef.value.style.height = 'auto';
  }

  // Add a placeholder bot message to show loading state
  addMessage({
    sender: 'bot',
    content: '',
    timestamp: new Date().toISOString(),
    loading: true,
    thinkingProcess: 'Thinking...',
  });

  try {
    await agentChat(
      {
        message: userMessageContent,
        display_thoughts: showThinkProcess.value,
        search_ai_active: searchAIActive.value,
        search_rag_active: searchRagActive.value,
        search_online_active: searchOnlineActive.value,
        files: selectedFiles.value, // Pass selected files
      },
      (chunkString) => {
        // SSE chunks are prefixed with "data: ". We need to remove this prefix before parsing.
        const jsonString = chunkString.startsWith('data: ') ? chunkString.substring(6) : chunkString;
        try {
          const chunk = JSON.parse(jsonString);
          updateLastBotMessage(chunk);
        } catch (e) {
          console.error("Failed to parse chunk as JSON:", e, chunkString);
          // If it's not valid JSON, it might be a partial chunk or plain text.
          // For now, we'll just append it as raw content to the last message.
          // A more robust solution might buffer chunks until a complete JSON object is formed.
          const lastMessage = messages.value[messages.value.length - 1];
          if (lastMessage && lastMessage.sender === 'bot' && lastMessage.loading) {
            lastMessage.content += chunkString;
          }
        }
      },
      (error) => {
        // Handle errors during streaming
        console.error('Streaming error:', error);
        const lastMessage = messages.value[messages.value.length - 1];
        if (lastMessage && lastMessage.sender === 'bot' && lastMessage.loading) {
          lastMessage.content = `Error: ${error.message || error}`;
          lastMessage.loading = false;
        } else {
          addMessage({
            sender: 'bot',
            content: `Error: ${error.message || error}`,
            timestamp: new Date().toISOString(),
          });
        }
        isLoading.value = false;
      }
    );
  } catch (e: any) {
    console.error('Agent chat initiation error:', e);
    const lastMessage = messages.value[messages.value.length - 1];
    if (lastMessage && lastMessage.sender === 'bot' && lastMessage.loading) {
      lastMessage.content = `Failed to send message: ${e.message || e}`;
      lastMessage.loading = false;
    } else {
      addMessage({
        sender: 'bot',
        content: `Failed to send message: ${e.message || e}`,
        timestamp: new Date().toISOString(),
      });
    }
  } finally {
    isLoading.value = false;
    // Ensure the last bot message's loading state is false after completion or error
    const lastMessage = messages.value[messages.value.length - 1];
    if (lastMessage && lastMessage.sender === 'bot') {
      lastMessage.loading = false;
    }
  }
};

// Computed property to format messages for display, similar to RostiChatInterface.vue
const formattedMessages = computed(() => {
  return messages.value.map((message: Message, index: number) => {
    return {
      key: message._id || index,
      sender: message.sender,
      placement: (message.sender === 'user' ? 'end' : 'start') as "end" | "start",
      content: message.content,
      loading: message.loading || false,
      thinkingProcess: message.thinkingProcess,
      avatar: message.sender === 'user' ? selectedUserAvatar.value : selectedBotAvatar.value,
      avatarSize: '32px',
      avatarGap: '12px',
      source_documents: message.source_documents || [],
    };
  });
});

// Computed property to get source documents from the latest bot message
const latestBotMessageSourceDocuments = computed(() => {
  if (!messages.value || messages.value.length === 0) {
    return [];
  }
  const latestBotMessage = messages.value.slice().reverse().find((message: Message) => message.sender === 'bot');
  return latestBotMessage?.source_documents || [];
});

// Placeholder for file input ref and related functions (not fully implemented in ChatPage.vue for simplicity)
const fileInputRef = ref<HTMLInputElement | null>(null);
const selectedFiles = ref<File[]>([]);

// Function to adjust textarea height
const adjustTextareaHeight = () => {
  if (textareaInputRef.value) {
    textareaInputRef.value.style.height = 'auto'; // Reset height to recalculate
    textareaInputRef.value.style.height = `${textareaInputRef.value.scrollHeight}px`;
  }
};

const triggerFileInput = () => {
  // This functionality is not fully implemented in ChatPage.vue
  // ElMessage.info('File upload is not yet fully integrated into this chat page.');
  fileInputRef.value?.click();
};
const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (target.files) {
    selectedFiles.value = Array.from(target.files);
    // In a real scenario, you'd upload these files and pass their IDs/references to the backend
    console.log('Selected files:', selectedFiles.value);
  }
};
const removeSelectedFile = (index: number) => {
  selectedFiles.value.splice(index, 1);
};
</script>

<style scoped>
@import './ChatPage.vue.css';
</style>
