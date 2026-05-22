<template>
  <!-- Chat-Nachrichtenbereich -->
  <div class="messages-container" ref="messagesContainerRef">
    <div v-if="messages && !messages.some((message: Message) => message.sender === 'user')" class="welcome-top">
      <div class="welcome-message">{{ t('rostiChat.welcome') }}</div>
    </div>
    <template v-for="message in formattedMessages" :key="message.key">
      <div v-if="message.loading" :class="['message-bubble', message.placement]">
        <div class="avatar" :style="{ width: message.avatarSize, height: message.avatarSize, marginRight: message.avatarGap }">
          <img :src="message.avatar" alt="avatar" />
        </div>
        <div class="content-wrapper">
          <div v-if="message.thinkingProcess && showThinkProcess" class="thinking-process-header">
            <strong>{{ t('rostiChat.thinkingProcess') }}:</strong>
            <div class="thinking-process-content">{{ message.thinkingProcess }}</div>
          </div>
          <div class="content">{{ message.content }}</div>
          <div class="loading-indicator">...</div>
        </div>
      </div>
      <Bubble
        v-else
        :key="message.key"
        :content="message.content"
        :placement="message.placement"
        :avatar="message.avatar"
        :avatar-size="message.avatarSize"
        :avatar-gap="message.avatarGap"
      >
        <template v-if="message.thinkingProcess && showThinkProcess" #header>
          <div class="thinking-process-header">
            <strong>{{ t('rostiChat.thinkingProcess') }}:</strong>
          </div>
        </template>
        <template v-if="message.thinkingProcess && showThinkProcess" #default>
          <div class="thinking-process-content">
            {{ message.thinkingProcess }}
          </div>
        </template>
      </Bubble>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue';
import { useI18n } from 'vue-i18n';
import Bubble from '@/components/Bubble.vue'; // Assuming Bubble component path

interface Message {
  key: string;
  content: string;
  sender: 'user' | 'ai';
  loading?: boolean;
  thinkingProcess?: string;
  placement: 'left' | 'right';
  avatar: string;
  avatarSize: string;
  avatarGap: string;
}

const props = defineProps<{
  messages: Message[];
  showThinkProcess: boolean;
}>();

const { t } = useI18n();
const messagesContainerRef = ref<HTMLElement | null>(null);

const formattedMessages = computed(() => {
  return props.messages.map(msg => ({
    ...msg,
    placement: msg.sender === 'user' ? 'right' : 'left',
    avatar: msg.sender === 'user' ? '/avatar/user.png' : '/avatar/ai.png', // Adjust paths as needed
    avatarSize: '32px',
    avatarGap: '8px',
  }));
});

watch(
  () => props.messages,
  () => {
    nextTick(() => {
      if (messagesContainerRef.value) {
        messagesContainerRef.value.scrollTop = messagesContainerRef.value.scrollHeight;
      }
    });
  },
  { deep: true }
);
</script>

<style scoped>
.messages-container {
  flex-grow: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.welcome-top {
  text-align: center;
  margin-bottom: 20px;
}

.welcome-message {
  font-size: 1.2em;
  color: #666;
}

.message-bubble {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  max-width: 80%;
}

.message-bubble.right {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.message-bubble.left {
  align-self: flex-start;
}

.avatar {
  flex-shrink: 0;
}

.avatar img {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  object-fit: cover;
}

.content-wrapper {
  background-color: #f0f0f0;
  padding: 10px 15px;
  border-radius: 10px;
  word-break: break-word;
  white-space: pre-wrap;
}

.message-bubble.right .content-wrapper {
  background-color: #e0f7fa;
}

.loading-indicator {
  font-style: italic;
  color: #999;
}

.thinking-process-header {
  font-size: 0.9em;
  color: #555;
  margin-bottom: 5px;
}

.thinking-process-content {
  font-size: 0.85em;
  color: #777;
  background-color: #f9f9f9;
  border-left: 3px solid #ccc;
  padding-left: 10px;
  margin-top: 5px;
}
</style>
