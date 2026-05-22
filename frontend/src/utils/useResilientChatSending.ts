import { ref, Ref, computed } from 'vue';
import { ElMessage } from 'element-plus';
import { useI18n } from 'vue-i18n';
import { useChatStore, Message } from '@/stores/chat';
import { uploadFiles } from '@/services/apiService';
import { storeToRefs } from 'pinia';

// Kommentar
interface RetryState {
  isRetrying: boolean
  retryCount: number
  error: string | null
  progress: string
  partialResponse: string
  lastHeartbeat: number
}

// Kommentar
interface Attachment {
  _id?: string | null;
  filename: string;
  bucket_name: string;
  object_name: string;
  size: number;
  content_type: string;
  upload_timestamp: string;
  download_url?: string;
}

export function useResilientChatSending(userInput: Ref<string>, selectedFiles: Ref<File[]>) {
  const { t } = useI18n();
  const chatStore = useChatStore();
  const { currentConversation, messages, showThinkProcess } = storeToRefs(chatStore);

  const searchAIActive = ref(true);
  const searchRostiActive = ref(false);
  const searchOnlineActive = ref(false);
  const MAX_WORD_COUNT = 200000;

  // Kommentar
  const retryState = ref<RetryState>({
    isRetrying: false,
    retryCount: 0,
    error: null,
    progress: '',
    partialResponse: '',
    lastHeartbeat: Date.now()
  });

  // SpeichernKommentar
  const lastMessageData = ref<any>(null);
  const lastBotMessageId = ref<string | null>(null);

  // Kommentar
  const retryConfig = {
    maxRetries: 3,
    retryDelay: 2000,
    timeoutMs: 900000, // 15Hinweis
    backoffMultiplier: 1.5,
    heartbeatTimeout: 30000 // 30Hinweis
  };

  // Kommentar
  const canRetry = computed(() => 
    !chatStore.isSending && 
    retryState.value.error && 
    retryState.value.retryCount < retryConfig.maxRetries
  );

  const isSending = computed(() => chatStore.isSending || retryState.value.isRetrying);

  // Kommentar
  const resetRetryState = () => {
    retryState.value = {
      isRetrying: false,
      retryCount: 0,
      error: null,
      progress: '',
      partialResponse: '',
      lastHeartbeat: Date.now()
    };
  };

  // Kommentar
  const isNetworkError = (error: any): boolean => {
    if (!error) return false;
    
    const errorString = error.toString().toLowerCase();
    const networkErrors = [
      'network error',
      'err_incomplete_chunked_encoding',
      'err_connection_reset',
      'err_connection_aborted',
      'fetch_error',
      'abort',
      'timeout',
      'streaming error'
    ];
    
    return networkErrors.some(errorType => errorString.includes(errorType));
  };

  // Kommentar
  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));



  // Kommentar
  const sendMessage = async (isRetry: boolean = false) => {
    console.log('resilient sendMessage called. userInput:', userInput.value, 'selectedFiles:', selectedFiles.value, 'isRetry:', isRetry);
    
    if (!isRetry && !userInput.value.trim() && selectedFiles.value.length === 0) {
      console.log('No user input or selected files. Aborting sendMessage.');
      return;
    }

    // Kommentar
    chatStore.isSending = true;

    // Kommentar
    if (!isRetry) {
      resetRetryState();
    }

    // Kommentar
    let currentWordCount = 0;
    if (chatStore.messages && Array.isArray(chatStore.messages)) {
      currentWordCount = chatStore.messages.reduce((count: number, message: Message) => {
        const words = message.content ? message.content.trim().split(/\s+/).filter((word: string) => word.length > 0) : [];
        return count + words.length;
      }, 0);
    }

    if (chatStore.currentConversation && currentWordCount >= MAX_WORD_COUNT) {
      ElMessage.warning(t('rostiChat.conversationLimitReached'));
      chatStore.clearCurrentConversation();
    }

    let uploadedFilesInfo: Attachment[] = [];
    let messageContent = '';

    try {
      // Kommentar
      if (!isRetry) {
        // Kommentar
        if (selectedFiles.value.length > 0) {
          const formData = new FormData();
          selectedFiles.value.forEach((file: File) => {
            formData.append('files', file);
          });

          // Kommentar
          const conversationId = chatStore.currentConversation?._id;
          const response = await uploadFiles(formData, conversationId);
          console.log("Upload response:", response);
          uploadedFilesInfo = response.uploaded_files as Attachment[];
          selectedFiles.value = [];
        }

        messageContent = userInput.value.trim();
        
        // SpeichernKommentar
        lastMessageData.value = {
          sender: 'user',
          content: messageContent,
          attachments: uploadedFilesInfo,
          search_ai_active: searchAIActive.value,
          search_rosti_active: searchRostiActive.value,
          search_online_active: searchOnlineActive.value,
          show_think_process: showThinkProcess.value,
        };
      } else {
        // Kommentar
        if (!lastMessageData.value) {
          ElMessage.error('Fehler bei der Verarbeitung');
          return;
        }
        messageContent = lastMessageData.value.content;
        uploadedFilesInfo = lastMessageData.value.attachments || [];
      }

      // SendenKommentar
      if (messageContent || uploadedFilesInfo.length > 0) {
        const searchOptions = {
          search_ai_active: searchAIActive.value,
          search_rosti_active: searchRostiActive.value,
          search_online_active: searchOnlineActive.value,
          show_think_process: showThinkProcess.value,
        };

        // Kommentar
        // Kommentar
        if (chatStore.currentConversation) {
          // Kommentar
          console.log('Adding message to existing conversation:', chatStore.currentConversation._id);
          
          const newMessage: Message = {
            sender: 'user',
            content: messageContent,
            attachments: uploadedFilesInfo,
            ...searchOptions
          };
          
          await chatStore.addMessageToConversation(chatStore.currentConversation._id, newMessage);
        } else {
          // Kommentar
          console.log('Creating new conversation and adding message');
          await chatStore.createConversationAndAddMessage(messageContent, uploadedFilesInfo, searchOptions);
        }
        
        if (!isRetry) {
          userInput.value = '';
        }
        
        // Kommentar
        resetRetryState();
      }

    } catch (error: any) {
      console.error("Error in message sending:", error);
      
      // Kommentar
      if (isNetworkError(error)) {
        retryState.value.error = error.message;
        retryState.value.retryCount = 0; // Hinweis
        ElMessage.warning('Nachricht konnte nicht gesendet werden, Warnhinweis');
      } else {
        // Kommentar
        console.log('Non-network error, letting chat store handle it');
      }
    } finally {
      // Kommentar
      chatStore.isSending = false;
    }
  };

  // Kommentar
  const retryLastMessage = async () => {
    if (!canRetry.value) {
      ElMessage.warning('Warnhinweis');
      return;
    }
    
    ElMessage.info('Hinweis');
    await sendMessage(true);
  };

  // Teilantwort verwenden
  const usePartialResponse = () => {
    if (!retryState.value.partialResponse) {
      ElMessage.warning('Warnhinweis');
      return;
    }

    const botMessage = messages.value.find((msg: any) => msg._id === lastBotMessageId.value) as Message;
    if (botMessage) {
      botMessage.content = retryState.value.partialResponse;
      botMessage.loading = false;
    }
    
    resetRetryState();
    ElMessage.success('Hinweis');
  };

  // AbbrechenFehlerStatus
  const dismissError = () => {
    resetRetryState();
  };

  return {
    searchAIActive,
    searchRostiActive,
    searchOnlineActive,
    MAX_WORD_COUNT,
    sendMessage,
    
    // Kommentar
    retryState: computed(() => retryState.value),
    canRetry,
    isSending,
    retryLastMessage,
    usePartialResponse,
    dismissError,
    resetRetryState
  };
}