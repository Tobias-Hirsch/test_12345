<template>
  <div class="queryrag-root">
    <div class="queryrag-main">
      <div class="queryrag-history">
        <div
          v-for="(item, idx) in history"
          :key="item.id"
          class="queryrag-history-item"
        >
          <div class="history-meta">
            <span class="history-title">Conversation {{ idx + 1 }}</span>
            <el-button link icon="el-icon-delete" @click="removeHistory(item.id)" circle size="small" />
          </div>
          <div class="history-content">
            <div class="history-question">{{ item.question }}</div>
            <div v-if="item.attachment" class="history-attachment">
              <el-link :href="item.attachment.url" target="_blank">{{ item.attachment.name }}</el-link>
            </div>
            <div v-if="item.answer" class="history-answer">{{ item.answer }}</div>
          </div>
        </div>
      </div>
      <div class="queryrag-dialogue">
        <div class="queryrag-dialogue-main">
          <div class="queryrag-welcome" v-if="!history.length">
            <span>How can I help you today?</span>
          </div>
          <div v-else class="queryrag-last-answer">
            <div v-if="history.length > 0">
              <div class="last-question">{{ history[history.length-1]?.question }}</div>
              <div v-if="history[history.length-1]?.attachment" class="last-attachment">
                <el-link :href="history[history.length-1]?.attachment?.url" target="_blank">{{ history[history.length-1]?.attachment?.name }}</el-link>
              </div>
              <div class="last-answer">{{ history[history.length-1]?.answer }}</div>
            </div>
          </div>
        </div>
        <div class="queryrag-inputbar">
          <el-input
            v-model="inputText"
            :placeholder="'Please input your question here'"
            class="queryrag-input"
            @keyup.enter="handleSend"
            clearable
            type="textarea"
            :rows="4"
          />
          <el-upload
            class="queryrag-upload"
            :show-file-list="false"
            :before-upload="beforeUpload"
          >
            <el-button class="upload-btn" link><i class="el-icon-paperclip"></i></el-button>
          </el-upload>
          <el-switch v-model="switchA" active-text="Search AI" class="queryrag-switch" />
          <el-switch v-model="switchB" active-text="Search Rosti Data" class="queryrag-switch" />
          <el-switch v-model="switchC" active-text="Search Online" class="queryrag-switch" />
          <el-button type="primary" @click="handleSend" :disabled="!inputText.trim()" class="queryrag-send">Send</el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref } from 'vue';
import { ElButton, ElInput, ElSwitch, ElUpload, ElLink, ElMessage } from 'element-plus';

interface HistoryItem {
  id: number;
  question: string;
  answer: string;
  attachment?: { name: string; url: string };
}

export default defineComponent({
  name: 'QueryRAG',
  components: {
    ElButton,
    ElInput,
    ElSwitch,
    ElUpload,
    ElLink,
  },
  setup() {
    const inputText = ref('');
    const switchA = ref(true);
    const switchB = ref(false);
    const switchC = ref(false);
    const history = ref<HistoryItem[]>([]);
    let nextId = 1;
    const attachment = ref<{ name: string; url: string } | null>(null);

    const beforeUpload = (file: File) => {
      const url = URL.createObjectURL(file);
      attachment.value = { name: file.name, url };
      ElMessage.success('Attachment uploaded');
      return false; // prevent auto upload
    };

    const handleSend = () => {
      if (!inputText.value.trim()) return;
      // Kommentar
      const answer = 'This is a mock answer.';
      history.value.push({
        id: nextId++,
        question: inputText.value,
        answer,
        attachment: attachment.value ? { ...attachment.value } : undefined,
      });
      inputText.value = '';
      attachment.value = null;
    };

    const removeHistory = (id: number) => {
      history.value = history.value.filter(item => item.id !== id);
    };

    return {
      inputText,
      switchA,
      switchB,
      switchC,
      history,
      beforeUpload,
      handleSend,
      removeHistory,
    };
  },
});
</script>

<style scoped>
.queryrag-root {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: stretch;
  background: #fff;
  box-sizing: border-box;
}
.queryrag-main {
  width: 100%;
  max-width: 100vw;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-end;
}
.queryrag-history {
  max-height: 40%;
  overflow-y: auto;
  margin-bottom: 20px;
}
.queryrag-history-item {
  background: #f8f9fa;
  border-radius: 8px;
  margin-bottom: 10px;
  padding: 12px 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  position: relative;
}
.history-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.history-title {
  font-weight: bold;
  color: #333;
}
.history-content {
  font-size: 15px;
  color: #444;
}
.history-question {
  margin-bottom: 4px;
}
.history-attachment {
  margin-bottom: 4px;
}
.history-answer {
  color: #1a73e8;
}
.queryrag-dialogue {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.08);
  padding: 32px 24px 16px 24px;
  display: flex;
  flex-direction: column;
  min-height: 180px;
  height: 50%;
  max-height: 50%;
  justify-content: flex-end;
  margin: 0 auto;
  width: 700px;
  max-width: 98vw;
}
.queryrag-dialogue-main {
  flex: 1;
  margin-bottom: 16px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}
.queryrag-welcome {
  text-align: center;
  color: #222;
  font-size: 1.2em;
  margin-top: 40px;
}
.queryrag-last-answer {
  text-align: left;
}
.last-question {
  font-weight: bold;
  margin-bottom: 6px;
}
.last-attachment {
  margin-bottom: 6px;
}
.last-answer {
  color: #1a73e8;
}
.queryrag-inputbar {
  display: flex;
  align-items: center;
  background: #666;
  border-radius: 16px;
  padding: 12px 16px;
  gap: 8px;
  margin-top: 0;
}
.queryrag-input {
  flex: 1;
  background: transparent;
  color: #fff;
  min-height: 80px;
  max-height: 160px;
  resize: vertical;
}
.queryrag-upload .upload-btn {
  color: #ffd600;
  font-size: 22px;
  padding: 0 8px;
}
.queryrag-switch {
  margin: 0 4px;
}
.queryrag-send {
  margin-left: 8px;
}
</style>