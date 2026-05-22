<!--
Developer: Jinglu Han
mailbox: admin@de-manufacturing.cn
-->

<template>
  <div class="chat-container" :class="{ 'dark-mode': isDarkMode }">
    <!-- Chatbereich -->
    <div class="chat-area">
      <!-- Statusanzeige -->
      <div v-if="isSending || retryState.isRetrying" class="status-indicator">
        <div class="loading-spinner" />
        <span class="status-text">
          <template v-if="retryState.isRetrying">
            {{ retryState.progress || `Wiederholungsversuch läuft ... (${retryState.retryCount}/3)` }}
          </template>
          <template v-else-if="isSending">
            {{ retryState.progress || 'Nachricht wird gesendet ...' }}
          </template>
        </span>
      </div>

      <!-- FehlerKommentaräge -->
      <div v-if="retryState.error" class="error-recovery-panel">
        <div class="error-message">
          <el-icon class="error-icon"><WarningFilled /></el-icon>
          <span>Nachricht konnte nicht gesendet werden: {{ retryState.error }}</span>
        </div>
        
        <!-- Kommentaräge -->
        <div v-if="retryState.partialResponse" class="recovery-options">
          <div class="partial-response-preview">
            <h4>Teilantwort empfangen:</h4>
            <div class="response-preview">
              {{ retryState.partialResponse.substring(0, 200) }}
              <span v-if="retryState.partialResponse.length > 200">...</span>
            </div>
          </div>
          
          <div class="recovery-actions">
            <el-button 
              @click="usePartialResponse" 
              type="info"
              size="small"
              :disabled="isSending"
            >
              Teilantwort verwenden
            </el-button>
            <el-button 
              @click="retryLastMessage" 
              type="primary"
              size="small"
              :disabled="!canRetry || isSending"
            >
              Erneut senden
            </el-button>
          </div>
        </div>

        <!-- Kommentaräge -->
        <div v-else class="retry-actions">
          <el-button 
            @click="retryLastMessage" 
            type="primary"
            size="small"
            :disabled="!canRetry || isSending"
          >
            Nachricht erneut senden
          </el-button>
          <el-button 
            @click="dismissError" 
            type="info"
            size="small"
          >
            Abbrechen
          </el-button>
        </div>
      </div>
      
      <!-- Chat-Nachrichtenbereich -->
      <div class="messages-container" ref="messagesContainerRef">
        <div v-if="messages && !messages.some((message: Message) => message.sender === 'user')" class="welcome-top">
          <div class="welcome-message">{{ t('rostiChat.welcome') }}</div>
        </div>
        <template v-for="message in formattedMessages" :key="message.key">
          <div :class="['message-bubble', message.placement]">
            <div class="avatar" :style="{ width: message.avatarSize, height: message.avatarSize, marginRight: message.avatarGap }"
                 @click="console.log('Clicked avatar sender:', message.sender); openAvatarDialog(message.sender)">
              <img :src="message.avatar" alt="avatar" />
            </div>
            <div class="content-wrapper">
              <!-- <div v-if="message.thinkingProcess && showThinkProcess" class="thinking-process-header">
                <strong>{{ t('rostiChat.thinkingProcess') }}:</strong>
                <div class="thinking-process-content">{{ message.thinkingProcess }}</div>
              </div> -->
              <div
                v-if="message.sender === 'bot'"
                class="content bot-content"
                v-html="markdownToHtml(message.content)"
              ></div>
              <div v-else class="content">{{ message.content }}</div>
              <div v-if="message.loading" class="loading-indicator">...</div>

              <!-- Attachments Display -->
              <div v-if="message.attachments && message.attachments.length > 0" class="message-attachments">
                <h4>{{ t('rostiChat.attachments') }}:</h4>
                <ul>
                  <li v-for="attachment in message.attachments" :key="attachment._id">
                    <el-tag size="small" type="info" class="attachment-tag">
                      <el-icon><Paperclip /></el-icon>
                      <span>{{ attachment.filename }} ({{ formatBytes(attachment.size) }})</span>
                      <el-tooltip :content="t('rostiChat.downloadAttachment')" placement="top">
                        <el-icon class="action-icon" @click="handleAttachmentAction('download', message.messageId, attachment._id, attachment.filename, attachment.download_url)"><Download /></el-icon>
                      </el-tooltip>
                      <el-tooltip :content="t('rostiChat.previewAttachment')" placement="top">
                        <el-icon class="action-icon" @click="handleAttachmentAction('preview', message.messageId, attachment._id, attachment.filename, attachment.download_url)"><Link /></el-icon>
                      </el-tooltip>
                      <el-tooltip :content="t('rostiChat.deleteAttachment')" placement="top">
                        <el-icon class="action-icon" @click="handleAttachmentAction('delete', message.messageId, attachment._id)"><Delete /></el-icon>
                      </el-tooltip>
                    </el-tag>
                  </li>
                </ul>
              </div>

              <!-- Search Results Display -->
              <div v-if="message.search_results && (message.search_results.source_documents?.length > 0 || message.search_results.online_results?.length > 0)" class="message-search-results">
                <div v-if="message.search_results.source_documents && message.search_results.source_documents.length > 0" class="search-results-section">
                  <h4>{{ t('rostiChat.sourcesLabel') }}:</h4>
                  <ul>
                    <li v-for="(source, index) in message.search_results.source_documents" :key="index">
                      <strong>{{ source.rag_item_name }}</strong>: {{ source.source }}
                      <a v-if="source.file_id" @click.prevent="previewFile(source.file_id, source.filename || source.source)" href="#" class="source-preview-link">
                        [{{ t('rostiChat.preview') }}]
                      </a>
                    </li>
                  </ul>
                </div>
                <div v-if="message.search_results.online_results && message.search_results.online_results.length > 0" class="search-results-section">
                  <h4>{{ t('rostiChat.onlineResultsLabel') }}:</h4>
                  <ul>
                    <li v-for="(result, index) in message.search_results.online_results" :key="index">
                      <strong><a :href="result.href" target="_blank">{{ result.title }}</a></strong>: {{ result.body }}
                    </li>
                  </ul>
                </div>
              </div>

              <!-- Aktionen für Bot-Nachrichten -->
              <div v-if="message.sender === 'bot' && !message.loading" class="message-actions">
                <el-tooltip :content="t('rostiChat.copyMessage')" placement="top">
                  <el-icon class="action-icon" @click="copyMessage(message.content)"><CopyDocument /></el-icon>
                </el-tooltip>
                <el-tooltip :content="t('rostiChat.regenerateMessage')" placement="top">
                  <el-icon class="action-icon" @click="() => { regenerateMessage(); nextTick(() => sendMessage()); }"><Refresh /></el-icon>
                </el-tooltip>
                <el-tooltip :content="t('rostiChat.exportToWord')" placement="top">
                  <el-icon class="action-icon" @click="exportToWord(message.content)"><Document /></el-icon>
                </el-tooltip>
                <!-- Feedback Buttons -->
                <el-tooltip :content="t('rostiChat.like')" placement="top">
                  <el-icon class="action-icon" @click="handleFeedback(message.messageId, 'like')"><Top /></el-icon>
                </el-tooltip>
                <el-tooltip :content="t('rostiChat.dislike')" placement="top">
                  <el-icon class="action-icon" @click="handleFeedback(message.messageId, 'dislike')"><Bottom /></el-icon>
                </el-tooltip>
              </div>
              <!-- Aktionen für Nutzer-Nachrichten -->
              <div v-if="message.sender === 'user'" class="message-actions">
                <el-tooltip :content="t('rostiChat.editMessage')" placement="top">
                  <el-icon class="action-icon" @click="editMessage(message.content)"><Edit /></el-icon>
                </el-tooltip>
                <el-tooltip :content="t('rostiChat.copyMessage')" placement="top">
                  <el-icon class="action-icon" @click="copyMessage(message.content)"><CopyDocument /></el-icon>
                </el-tooltip>
              </div>
            </div>
          </div>
        </template>
      </div>
     <!-- Eingabebereich -->
      <div class="input-area">
        <div class="input-container input-flex">
          <el-input
            v-model="userInput"
            :placeholder="t('rostiChat.inputPlaceholder')"
            class="textarea-input"
            @keyup.enter="sendMessage"
            type="textarea"
            :autosize="{ minRows: 3, maxRows: 5 }"
          />
        </div>
        <div class="search-switches">
          <div class="switch-group-left">
            <span
              class="search-switch"
              :class="{ active: searchAIActive }"
              @click="searchAIActive = !searchAIActive"
            >{{ t('rostiChat.searchAI') }}</span>
            <span
              class="search-switch"
              :class="{ active: searchRostiActive }"
              @click="searchRostiActive = !searchRostiActive"
            >{{ t('rostiChat.searchRostiData') }}</span>
            <span
              class="search-switch"
              :class="{ active: searchOnlineActive }"
              @click="searchOnlineActive = !searchOnlineActive"
            >{{ t('rostiChat.searchOnline') }}</span>
            <!-- <span
              class="search-switch"
              :class="{ active: showThinkProcess }"
              @click="showThinkProcess = !showThinkProcess"
            >{{ t('rostiChat.showThinkProcess') }}</span> -->
          </div>
          <div class="switch-group-right">
            <!-- <el-tooltip :content="t('knowledgeBase.title')" placement="top">
              <FolderOpened class="attachment-icon sendbar-icon" @click="showKbManager = true" />
            </el-tooltip> -->
            <el-tooltip :content="t('rostiChat.uploadTip')" placement="top">
              <Paperclip class="attachment-icon sendbar-icon" @click="triggerFileInput" />
            </el-tooltip>
            <!-- Hidden native file input -->
            <input type="file" multiple @change="handleFileChange" ref="fileInputRef" style="display: none;">
            <el-button 
              type="primary" 
              class="send-btn" 
              @click="() => { console.log('Send button clicked'); sendMessage(); }"
              :disabled="isSending"
              :loading="isSending"
            >
              <template v-if="isSending">
                {{ retryState.isRetrying ? 'Wird erneut versucht ...' : 'Wird gesendet ...' }}
              </template>
              <template v-else>
                {{ t('rostiChat.send') }}
              </template>
            </el-button>
          </div>
        </div>
        <!-- Display selected files -->
        <div v-if="selectedFiles.length" class="selected-files-preview">
          <span>{{ t('rostiChat.selectedFiles') }}: </span>
          <el-tag
            v-for="(file, index) in selectedFiles"
            :key="index"
            closable
            @close="removeSelectedFile(index)"
            type="info"
            size="small"
            class="file-tag"
          >
            {{ file.name }}
          </el-tag>
        </div>
      </div>
    </div>
 
   </div>
 

   <!-- Avatar Selection Dialog -->
   <el-dialog v-model="showAvatarDialog" title="Select Avatar" width="30%">
     <div class="avatar-options">
      <el-avatar
        v-for="(avatarUrl, index) in avatarOptions"
        :key="index"
        :size="50"
        :src="avatarUrl"
        class="avatar-option"
        @click="selectAvatar(avatarUrl)"
      />
    </div>
    <div class="upload-avatar-section">
      <el-button type="primary" @click="avatarFileInputRef?.click()">
        {{ t('rostiChat.uploadCustomAvatar') }}
      </el-button>
      <input type="file" ref="avatarFileInputRef" @change="handleAvatarFileChange" style="display: none;" accept="image/jpeg,image/png,image/gif">
      <p class="upload-tip">{{ t('rostiChat.avatarUploadTip') }}</p>
    </div>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, inject, computed, watch, nextTick, onMounted } from 'vue';
import { storeToRefs } from 'pinia';
import { Paperclip, Link, Download, Delete, Edit, CopyDocument, Refresh, Document, Top, Bottom, WarningFilled } from '@element-plus/icons-vue';
import { useI18n } from 'vue-i18n';
import { ElTooltip, ElTag, ElMessage, ElMessageBox, ElButton, ElDialog, ElAvatar, ElIcon } from 'element-plus';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { useChatStore, Message } from '@/stores/chat';
import { useAuthStore } from '@/stores/auth';
import { usePreviewStore } from '@/stores/preview'; // Import the new preview store
import { previewFile as previewFileApi } from '@/services/apiService'; // Import the new API service function
import './RostiChatInterface.vue.css';

// Import composables
import { useAvatarManagement } from '@/utils/useAvatarManagement';
import { useAttachmentHandling } from '@/utils/useAttachmentHandling';
import { useResilientChatSending } from '@/utils/useResilientChatSending';
import { useMessageActions } from '@/utils/useMessageActions';

const markdownToHtml = (markdown: string): string => {
  if (!markdown) return '';

  let processedText = markdown;

  // Process tables
  processedText = processedText.replace(
    /^\|(.+)\|\s*\n\|( *[-:]+ *\|)+\s*\n((?:\|.*\|\s*\n?)*)/gm,
    (match: string, header: string, separator: string, body: string) => {
      const headerCells = header.split('|').map((cell: string) => cell.trim()).filter(Boolean);
      const tableHtml = `
        <table class="custom-table">
          <thead>
            <tr>
              ${headerCells.map((cell: string) => `<th>${cell}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${body.split('\n').filter((row: string) => row.trim()).map((row: string) => {
              const rowCells = row.split('|').map((cell: string) => cell.trim()).filter(Boolean);
              return `<tr>${rowCells.map((cell: string) => `<td>${cell}</td>`).join('')}</tr>`;
            }).join('')}
          </tbody>
        </table>
      `;
      return tableHtml;
    }
  );

  // Process block-level formulas $$...$$
  processedText = processedText.replace(/\$\$([\s\S]+?)\$\$/g, (match, formula) => {
    try {
      return katex.renderToString(formula, {
        throwOnError: false,
        displayMode: true
      });
    } catch (e) {
      console.error(e);
      return match; // Return original string on error
    }
  });

  // Process code blocks first to avoid conflicts
  processedText = processedText.replace(/```(\w*)\n([\s\S]+?)\n```/g, (match, lang, code) => {
    const escapedCode = code.replace(/</g, '<').replace(/>/g, '>');
    return `<pre><code class="language-${lang}">${escapedCode.trim()}</code></pre>`;
  });

  // Process headings (h1 to h6) - MUST be before list processing
  processedText = processedText.replace(/^###### (.*$)/gim, '<h6>$1</h6>');
  processedText = processedText.replace(/^##### (.*$)/gim, '<h5>$1</h5>');
  processedText = processedText.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
  processedText = processedText.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  processedText = processedText.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  processedText = processedText.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // Process unordered lists
  processedText = processedText.replace(/^- (.*)/gm, (match) => {
    return `<li>${match.substring(2)}</li>`;
  });
  processedText = processedText.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>').replace(/<\/ul>\n?<ul>/g, '');

  // Process horizontal rules
  processedText = processedText.replace(/^(---|___|\*\*\*)\s*$/gm, '<hr>');

  // Process bold and italic
  processedText = processedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  processedText = processedText.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // Process inline code
  processedText = processedText.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Process newlines to <br>, but not inside <pre>, <ul>, or <table>
  const parts = processedText.split(/(<pre>[\s\S]*?<\/pre>|<ul>[\s\S]*?<\/ul>|<table[\s\S]*?<\/table>)/g);
  const result = parts.map(part => {
    if (part.startsWith('<pre>') || part.startsWith('<ul>') || part.startsWith('<table')) {
      // also remove <br> inside these blocks
      return part.replace(/<br\s*\/?>/g, '');
    }
    return part.trim().replace(/\n/g, '<br>');
  }).join('');

  return result;
};


// Destructure from composables
const { showAvatarDialog, selectedUserAvatar, selectedBotAvatar, avatarFileInputRef, avatarFile, avatarOptions, openAvatarDialog, selectAvatar, handleAvatarFileChange } = useAvatarManagement();
const { fileInputRef, selectedFiles, formatBytes, handleAttachmentAction, triggerFileInput, handleFileChange, removeSelectedFile } = useAttachmentHandling();
const { userInput, editMessage, copyMessage, regenerateMessage, exportToWord, handleFeedback } = useMessageActions();
const { 
  searchAIActive, 
  searchRostiActive, 
  searchOnlineActive, 
  sendMessage,
  retryState,
  canRetry,
  isSending,
  retryLastMessage,
  usePartialResponse,
  dismissError
} = useResilientChatSending(userInput, selectedFiles);

// State for Knowledge Base Manager

const previewStore = usePreviewStore();

const previewFile = async (fileId: number, filename:string) => {
  if (!fileId) {
    ElMessage.error('No file ID provided.');
    return;
  }

  try {
    // Use the new, centralized API function
    const blob = await previewFileApi(fileId);
    const url = window.URL.createObjectURL(blob);
    
    // Use the new global preview store
    const fileType = filename.split('.').pop() || 'unknown';
    previewStore.show(url, fileType, filename);

  } catch (error: any) {
    console.error('Error previewing file:', error);
    ElMessage.error(`Could not preview file: ${error.message}`);
    previewStore.showError(`Could not preview file: ${error.message}`);
  }
};

const { t } = useI18n();
const isDarkMode = inject('isDarkMode', false);

// Use the chat store
const chatStore = useChatStore();

// Use storeToRefs to maintain reactivity for state properties
const { conversations, currentConversation, messages, showThinkProcess } = storeToRefs(chatStore);

// Use the auth store
const authStore = useAuthStore();
const { user } = storeToRefs(authStore); // Get user from auth store

const messagesContainerRef = ref<HTMLElement | null>(null);

watch(messages, async () => {
  await nextTick();
  if (messagesContainerRef.value) {
    const container = messagesContainerRef.value;
    container.scrollTop = container.scrollHeight; // Always scroll to bottom
  }
}, { deep: true, immediate: true });
onMounted(async () => {
  await chatStore.fetchConversations();
  // Optionally, select the most recent conversation or a default one
  if (chatStore.conversations.length > 0) {
    await chatStore.selectConversation(chatStore.conversations[0]);
  }
});

// Computed property to check if there is any user message
const hasUserMessage = computed(() => {
  return messages.value && messages.value.some((message: Message) => message.sender === 'user');
});

// Computed property to format messages for BubbleList
const formattedMessages = computed(() => {
  if (!messages.value) {
    return [];
  }
  return messages.value.map((message: Message, index: number) => {
    return {
      key: message._id || index,
      messageId: message._id, // Pass message ID for attachment actions
      sender: message.sender, // Explicitly include sender
      role: message.sender === 'user' ? 'user' : 'ai',
      placement: (message.sender === 'user' ? 'end' : 'start') as "end" | "start",
      content: message.content,
      loading: message.loading || false,
      thinkingProcess: message.thinkingProcess,
      shape: 'corner',
      variant: message.sender === 'user' ? 'outlined' : 'filled',
      isMarkdown: false,
      typing: false,
      isFog: false,
      avatar: message.sender === 'user' ? selectedUserAvatar.value : selectedBotAvatar.value, // Use selectedBotAvatar for bot messages
      avatarSize: '32px',
      avatarGap: '12px',
      attachments: Array.isArray(message.attachments) ? message.attachments : [], // Ensure attachments is always an array
      search_results: message.search_results as any, // Pass search results with correct type
    };
  });
});

</script>


