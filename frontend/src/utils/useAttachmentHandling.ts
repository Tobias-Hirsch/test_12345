import { ref } from 'vue';
import { ElMessage, ElMessageBox } from 'element-plus';
import { useI18n } from 'vue-i18n';
import { useChatStore, Message } from '@/stores/chat';
import { storeToRefs } from 'pinia';
import { del, uploadFiles } from '@/services/apiService';

// Define interface for Attachment based on backend schema
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

export function useAttachmentHandling() {
  const { t } = useI18n();
  const chatStore = useChatStore();
  const { currentConversation, messages } = storeToRefs(chatStore);

  const fileInputRef = ref<HTMLInputElement | null>(null);
  const selectedFiles = ref<File[]>([]);
  const allowedFileTypes = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
  ];

  const MAX_ATTACHMENT_COUNT = 5;
  const MAX_TOTAL_ATTACHMENT_SIZE = 100 * 1024 * 1024; // 100MB in bytes

  const formatBytes = (bytes: number, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  const handleAttachmentAction = async (action: 'preview' | 'download' | 'delete', messageId: string | undefined, attachmentId: string | undefined, filename?: string, downloadUrl?: string) => {
    if (!messageId || !attachmentId || attachmentId === 'None' || !currentConversation.value?._id) {
      console.error("Missing information for attachment action.");
      ElMessage.error("Die Anhangsinformationen sind unvollständig. Die Aktion kann nicht ausgeführt werden. Bitte kontaktieren Sie den Administrator."); // More specific error message
      return;
    }

    const conversationId = currentConversation.value._id;

    if (action === 'preview' || action === 'download') {
      if (!filename) {
        console.error("Missing filename for preview/download.");
        ElMessage.error("Could not preview/download file due to missing filename.");
        return;
      }
      const urlToUse = downloadUrl || `/api/chat/conversations/${conversationId}/messages/${messageId}/attachments/${attachmentId}/download/`;

      if (action === 'preview') {
        window.open(urlToUse, '_blank');
      } else if (action === 'download') {
        const link = document.createElement('a');
        link.href = urlToUse;
        link.setAttribute('download', filename);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } else if (action === 'delete') {
      ElMessageBox.confirm(
        t('rostiChat.confirmDeleteAttachment'),
        t('rostiChat.warning'),
        {
          confirmButtonText: t('rostiChat.delete'),
          cancelButtonText: t('rostiChat.cancel'),
          type: 'warning',
        }
      ).then(async () => {
        try {
          await del(`/chat/conversations/${conversationId}/messages/${messageId}/attachments/${attachmentId}/`);
          const messageIndex = messages.value.findIndex((msg: Message) => msg._id === messageId);
          if (messageIndex !== -1 && messages.value[messageIndex].attachments) {
            messages.value[messageIndex].attachments = messages.value[messageIndex].attachments!.filter((att: Attachment) => att._id !== attachmentId);
          }
          ElMessage({
            type: 'success',
            message: t('rostiChat.deleteAttachmentSuccess'),
          });
        } catch (error: any) {
          console.error("Error deleting attachment:", error);
          ElMessage.error(`${t('rostiChat.deleteAttachmentFailed')}: ${error.message}`);
        }
      }).catch(() => {
        // User cancelled deletion
      });
    }
  };

  const triggerFileInput = () => {
    fileInputRef.value?.click();
  };

  const handleFileChange = (event: Event) => {
    const target = event.target as HTMLInputElement;
    if (target.files) {
      let files = Array.from(target.files);
      const validFiles: File[] = [];
      let currentAttachmentCount = 0;
      let currentTotalAttachmentSize = 0;

      if (files.length > MAX_ATTACHMENT_COUNT) {
        ElMessage.warning(`You can select a maximum of ${MAX_ATTACHMENT_COUNT} files at once. Processing the first ${MAX_ATTACHMENT_COUNT}.`);
        files = files.slice(0, MAX_ATTACHMENT_COUNT);
      }

      if (currentConversation.value && currentConversation.value.messages) {
        currentConversation.value.messages.forEach((message: Message) => {
          if (message.attachments) {
            currentAttachmentCount += message.attachments.length;
            message.attachments.forEach((attachment: Attachment) => {
              currentTotalAttachmentSize += attachment.size || 0;
            });
          }
        });
      }

      for (const file of files) {
        if (!allowedFileTypes.includes(file.type)) {
          console.warn(`Skipping file "${file.name}": Anhänge sind nur in den Formaten PDF, DOCX und XLSX erlaubt.`);
          ElMessage.warning(`Anhänge sind nur in den Formaten PDF, DOCX und XLSX erlaubt.`);
          continue;
        }

        if (currentAttachmentCount + selectedFiles.value.length + validFiles.length >= MAX_ATTACHMENT_COUNT) {
          console.warn(`Skipping file "${file.name}": Exceeds maximum total attachment count (${MAX_ATTACHMENT_COUNT}).`);
          ElMessage.warning(`Skipping file "${file.name}": Exceeds maximum total attachment count (${MAX_ATTACHMENT_COUNT}).`);
          continue;
        }

        const sizeOfCurrentlySelectedAndValid = selectedFiles.value.reduce((sum: number, f: File) => sum + f.size, 0) + validFiles.reduce((sum: number, f: File) => sum + f.size, 0);
        if (currentTotalAttachmentSize + sizeOfCurrentlySelectedAndValid + file.size > MAX_TOTAL_ATTACHMENT_SIZE) {
          console.warn(`Skipping file "${file.name}": Exceeds maximum total attachment size (${MAX_TOTAL_ATTACHMENT_SIZE / (1024 * 1024)}MB).`);
          ElMessage.warning(`Skipping file "${file.name}": Exceeds maximum total attachment size (${MAX_TOTAL_ATTACHMENT_SIZE / (1024 * 1024)}MB).`);
          continue;
        }
        validFiles.push(file);
      }

      selectedFiles.value = [...selectedFiles.value, ...validFiles];

    } else {
      selectedFiles.value = [];
    }
    if (fileInputRef.value) {
      fileInputRef.value.value = '';
    }
  };

  const removeSelectedFile = (index: number) => {
    selectedFiles.value.splice(index, 1);
  };

  return {
    fileInputRef,
    selectedFiles,
    formatBytes,
    handleAttachmentAction,
    triggerFileInput,
    handleFileChange,
    removeSelectedFile,
  };
}