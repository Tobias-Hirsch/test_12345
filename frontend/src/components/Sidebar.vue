<!--
Developer: Jinglu Han
mailbox: admin@de-manufacturing.cn
-->

<template>
    <el-aside :width="isAsideCollapsed ? '0px' : asideWidth" v-if="isAuthenticated" class="aside-menu">
      <div class="aside-content" v-if="!isAsideCollapsed">
        <div class="aside-logo">
          <img src="/image/rosti_ai_logo.png" alt="Logo" class="logo-img" />
        </div>
        <el-menu
          v-if="showAdminMenu"
          default-active="1"
          class="el-menu-vertical-demo"
          :key="isAsideCollapsed ? 'collapsed' : 'expanded'"
        >
          <!-- <el-menu-item index="3">
              <router-link to="/rag-intro" style="text-decoration: none; color: inherit;">
                <el-icon><Menu /></el-icon>
                <span>{{ $t('sidebar.ragIntro') }}</span>
              </router-link>
          </el-menu-item> -->
          <el-menu-item index="4">
              <router-link to="/query" style="text-decoration: none; color: inherit;">
                <el-icon><Menu /></el-icon>
                <span>{{ $t('sidebar.queryRAG') }}</span>
              </router-link>
          </el-menu-item>
            <el-menu-item index="5">
              <router-link to="/rag" style="text-decoration: none; color: inherit;">
                <el-icon><Menu /></el-icon>
                <span>{{ $t('sidebar.rag') }}</span>
              </router-link>
          </el-menu-item>
            <!-- User and Role Management Menu Group -->
            <el-sub-menu index="user-management">
              <template #title>
                <el-icon><Menu /></el-icon>
                <span>{{ $t('sidebar.userManagementGroup') }}</span>
              </template>
              <el-menu-item index="user-management-users">
                <router-link to="/user-management" style="text-decoration: none; color: inherit;">
                  <el-icon><Menu /></el-icon>
                  <span>{{ $t('sidebar.userManagement') }}</span>
                </router-link>
              </el-menu-item>
              <el-menu-item index="user-management-roles">
                <router-link to="/role-management" style="text-decoration: none; color: inherit;">
                  <el-icon><Menu /></el-icon>
                  <span>{{ $t('sidebar.roleManagement') }}</span>
                </router-link>
              </el-menu-item>
              <el-menu-item index="user-management-policy-management">
                <router-link to="/policy-management" style="text-decoration: none; color: inherit;">
                  <el-icon><Menu /></el-icon>
                  <span>{{ $t('sidebar.policyManagement') }}</span>
                </router-link>
              </el-menu-item>
              <el-menu-item index="user-management-system-settings">
                <router-link to="/system-settings" style="text-decoration: none; color: inherit;">
                  <el-icon><Menu /></el-icon>
                  <span>{{ $t('sidebar.systemSettings') }}</span>
                </router-link>
              </el-menu-item>
            </el-sub-menu>
        </el-menu>
        <el-divider v-if="showAdminMenu" class="sidebar-divider" />
        <div class="sidebar-extra">
          <template v-if="showRostiChatSidebar">
            <div class="logo-area">
              <div> 
                <el-button class="new-chat-btn new-chat" type="primary" @click="chatStore.clearCurrentConversation()">
                  {{ $t('rostiChat.newChat') }}
                </el-button>
              </div>
            </div>
            <div class="history-section">
              <el-menu
                :default-active="currentConversation ? currentConversation._id : ''"
                class="history-menu"
              >
                <el-menu-item
                  v-for="conv in conversations"
                  :key="conv._id"
                  :index="conv._id"
                  @click="chatStore.selectConversation(conv)"
                  class="history-menu-item"
                >
                  <div class="history-item-content">
                    <span class="history-item-title">{{ conv.title }}</span>
                    <div class="conversation-actions">
                      <el-icon class="action-icon rename-icon" @click.stop="handleRenameConversation(conv)"><EditPen /></el-icon>
                      <el-icon class="action-icon delete-icon" @click.stop="handleDeleteConversation(conv)"><Delete /></el-icon>
                    </div>
                  </div>
                </el-menu-item>
              </el-menu>
            </div>
          </template>
          <slot name="sidebar-extra" />
        </div>
        <div class="sidebar-bottom-buttons">
          <el-button class="help-btn" @click="showHelp">{{ $t('rostiChat.help') }}</el-button>
          <el-button class="lang-toggle-btn" @click="toggleLanguage" type="default">DE/EN</el-button>
        </div>
      </div>
      <div class="resizer" @mousedown="startResize" v-if="!isAsideCollapsed"></div>
    </el-aside>
  </template>
  
  <script setup lang="ts">
  import { ref, computed, inject, onBeforeUnmount } from 'vue';
  import { useRoute, useRouter } from 'vue-router';
  import { useI18n } from 'vue-i18n';
  import { useAuthStore } from '@/stores/auth';
  import { useChatStore } from '@/stores/chat';
  import { storeToRefs } from 'pinia';
  import { ElMessageBox, ElMessage } from 'element-plus';
  import { EditPen, Delete, Menu } from '@element-plus/icons-vue';
  import { useAppStore } from '@/stores/app';
  
  const authStore = useAuthStore();
  const { user } = storeToRefs(authStore); // Use storeToRefs for reactive state
  const chatStore = useChatStore();
  const { conversations, currentConversation } = storeToRefs(chatStore);
  const isAuthenticated = computed(() => authStore.isAuthenticated);
  
  const showAdminMenu = computed(() => {
    const roles = user.value?.roles;
    if (!Array.isArray(roles) || roles.length === 0) {
      return false;
    }
    return roles.some(role =>
      role.name.toLowerCase() === 'admin' ||
      role.name.toLowerCase() === 'content_manager'
    );
  });
  
  const isAsideCollapsed = inject('isAsideCollapsed', ref(false));
  const route = useRoute();
  const router = useRouter();
  const { t, locale } = useI18n();
  const appStore = useAppStore();
  
  const helpDocMap: Record<string, string> = {
    '/rag-intro': '/frontend/chat-rag/rag-intro',
    '/query': '/frontend/chat-rag/rosti-chat-interface',
    '/rag': '/frontend/chat-rag/rag-list',
    '/user-management': '/frontend/system-settings/user-management',
    '/role-management': '/frontend/system-settings/role-management',
    '/permission-management': '/frontend/system-settings/permission-management',
    '/system-settings': '/frontend/system-settings/system-settings',
    '/user-profile': '/frontend/system-settings/user-profile',
    '/activate-account': '/frontend/authentication/activate-account',
    '/forgot-password': '/frontend/authentication/forgot-password',
    '/login': '/frontend/authentication/login',
    '/register': '/frontend/authentication/register',
    '/reset-password': '/frontend/authentication/reset-password',
  };
  
  const showRostiChatSidebar = computed(() => route.name === 'QueryRAG');
  
  function showHelp() {
    // const currentPath = route.path;
    // let docPath = helpDocMap[currentPath] || '/';
    // const langPrefix = locale.value === 'en' ? '/en' : '';
    // // The base URL for VitePress, which should correspond to the Nginx location.
    // const vitepressBaseUrl = import.meta.env.VITE_VITEPRESS_BASE_URL || '/docs';
    
    // // Construct the URL to point to the VitePress site, letting it handle the routing.
    // // We append the language and the specific document path. VitePress will resolve this.
    // const fullDocUrl = `${window.location.origin}${vitepressBaseUrl}${langPrefix}`;
    // window.open(fullDocUrl, '_blank');

    // For now, we will just navigate to the RAG intro page
    // This will be replaced with the actual help documentation URL later.
    // Jinglu Han on 17.07.2025, because the VitePress site is not yet ready.
    // Navigate to the RAG intro page within the app
    // Use router.resolve to generate the URL for the target route.
    // This ensures that it works correctly with the SPA's routing mechanism.
    const routeData = router.resolve({ path: '/rag-intro' });
    window.open(routeData.href, '_blank');
  }
  
  const toggleLanguage = () => {
    const newLang = locale.value === 'en' ? 'de' : 'en';
    locale.value = newLang;
    localStorage.setItem('language', newLang);
    appStore.setLanguage(newLang);
  };
  
  const handleRenameConversation = async (conversation: any) => {
      ElMessageBox.prompt(t('rostiChat.promptNewTitle'), t('rostiChat.renameConversation'), {
          confirmButtonText: t('rostiChat.confirm'),
          cancelButtonText: t('rostiChat.cancel'),
          inputValue: conversation.title,
          inputPlaceholder: t('rostiChat.newTitlePlaceholder'),
      }).then(async ({ value }) => {
          if (value && value.trim() !== conversation.title) {
              try {
                  await chatStore.updateConversationTitle(conversation._id, value.trim());
                  ElMessage.success(t('rostiChat.renameSuccess'));
              } catch (error: any) {
                  ElMessage.error(`${t('rostiChat.renameFailed')}: ${error.message}`);
              }
          }
      }).catch(() => {});
  };
  
  const handleDeleteConversation = async (conversation: any) => {
      ElMessageBox.confirm(
          t('rostiChat.confirmDeleteConversation', { title: conversation.title }),
          t('rostiChat.warning'),
          {
              confirmButtonText: t('rostiChat.delete'),
              cancelButtonText: t('rostiChat.cancel'),
          }
      ).then(async () => {
          try {
              await chatStore.deleteConversation(conversation._id);
              ElMessage.success(t('rostiChat.deleteSuccess'));
          } catch (error: any) {
              ElMessage.error(`${t('rostiChat.deleteFailed')}: ${error.message}`);
          }
      }).catch(() => {});
  };
  
  // --- Resizable Sidebar Logic ---
  const asideWidth = ref('250px');
  const isResizing = ref(false);
  
  const startResize = (event: MouseEvent) => {
    isResizing.value = true;
    document.body.style.cursor = 'col-resize';
    document.addEventListener('mousemove', doResize);
    document.addEventListener('mouseup', stopResize);
  };
  
  const doResize = (event: MouseEvent) => {
    if (isResizing.value) {
      const newWidth = Math.max(200, Math.min(event.clientX, 500));
      asideWidth.value = `${newWidth}px`;
    }
  };
  
  const stopResize = () => {
    isResizing.value = false;
    document.body.style.cursor = '';
    document.removeEventListener('mousemove', doResize);
    document.removeEventListener('mouseup', stopResize);
  };
  
  onBeforeUnmount(() => {
    document.removeEventListener('mousemove', doResize);
    document.removeEventListener('mouseup', stopResize);
  });
  </script>
  
  <style scoped>
  .aside-menu {
    background-color: var(--el-menu-bg-color);
    color: var(--el-menu-text-color);
    height: 100%;
    display: flex;
    position: relative;
    transition: width 0.1s ease;
    overflow: visible;
  }
  
  .aside-content {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
    overflow-x: hidden;
  }
  
  .resizer {
    width: 5px;
    height: 100%;
    background: transparent;
    position: absolute;
    right: 0;
    top: 0;
    cursor: col-resize;
    z-index: 10;
  }
  
  .aside-logo {
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px;
    box-sizing: border-box;
  }
  
  .logo-img {
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
  }
  
  .el-menu-vertical-demo {
    border-right: none;
    overflow-y: auto;
  }
  
  .sidebar-extra {
    padding: 10px;
    border-top: 1px solid var(--el-border-color-light);
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }
  
  .history-section {
    flex-grow: 1;
    overflow-y: auto;
  }
  
  .sidebar-bottom-buttons {
    margin-top: auto;
    padding: 10px;
    border-top: 1px solid var(--el-border-color-light);
    display: flex;
    gap: 10px;
  }
  
  .help-btn, .lang-toggle-btn {
    flex: 1;
    margin: 0 !important;
  }
  
  .history-menu {
    border-right: none; /* Remove the border from the nested menu */
    background-color: transparent; /* Inherit background color */
  }

  .history-menu-item {
    padding: 0 10px !important; /* Adjust padding to align with other items */
  }
  
  .history-item-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    overflow: hidden;
  }
  
  .history-item-title {
    flex-grow: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-right: 10px;
  }
  
  .conversation-actions {
    display: flex;
    gap: 5px;
  }
  
  .action-icon {
    cursor: pointer;
    color: var(--el-text-color-secondary);
  }
  
  .action-icon:hover {
    color: var(--el-color-primary);
  }
  
  .new-chat-btn {
    width: 100%;
  }
  </style>

