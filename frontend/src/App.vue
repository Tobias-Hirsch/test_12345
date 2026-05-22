<template>
  <el-config-provider :locale="currentLocale">
    <div id="app" :class="{ 'dark-mode': isDarkMode }">
      <el-container style="height: 100vh;">
        <Sidebar />
        <el-container direction="vertical">
          <AppHeader />
          <el-main>
            <router-view :key="$i18n.locale" />
          </el-main>
        </el-container>
      </el-container>
      <!-- Place the FilePreviewModal here, at the root level, to ensure it can overlay everything -->
      <FilePreviewModal />
    </div>
  </el-config-provider>
</template>

<script lang="ts">
import { defineComponent, ref, computed, provide } from 'vue';
import { useRouter } from 'vue-router';
import { ElConfigProvider, ElSwitch, ElButton, ElSelect, ElOption } from 'element-plus';
import { Sunny, Moon, Menu } from '@element-plus/icons-vue';
import { useI18n } from 'vue-i18n';
import enLocale from 'element-plus/es/locale/lang/en';
import deLocale from 'element-plus/es/locale/lang/de';
import { useAuthStore } from './stores/auth';
import Sidebar from './components/Sidebar.vue';
import AppHeader from './components/AppHeader.vue';
import FilePreviewModal from './components/FilePreviewModal.vue';

export default defineComponent({
  name: 'App',
  components: {
    ElConfigProvider,
    ElSwitch,
    ElButton,
    ElSelect,
    ElOption,
    Sunny,
    Moon,
    Menu,
    Sidebar,
    AppHeader,
    FilePreviewModal,
  },
  setup() {
    const router = useRouter();
    const { locale } = useI18n();
    const authStore = useAuthStore();

    const selectedLanguage = ref(locale.value);

    const changeLanguage = (lang: string) => {
      locale.value = lang;
      localStorage.setItem('language', lang);
    };

    const isDarkMode = ref(false);

    const currentLocale = computed(() => {
      return locale.value === 'en' ? enLocale : deLocale;
    });

    const toggleTheme = (value: any) => {
      const multiThemeEnv = import.meta.env.VITE_MULTI_THEME;
      const multiThemeEnabled = multiThemeEnv === 'true' || multiThemeEnv === true;

      if (multiThemeEnabled) {
        isDarkMode.value = value as boolean;
        document.body.classList.toggle('dark-mode', value as boolean);
      } else {
        // If multi-theme is disabled, force light mode and do nothing on toggle.
        isDarkMode.value = false;
        document.body.classList.remove('dark-mode');
      }
    };

    // Initialize theme based on the setting
    if (import.meta.env.VITE_MULTI_THEME !== 'true' && import.meta.env.VITE_MULTI_THEME !== true) {
      isDarkMode.value = false;
      document.body.classList.remove('dark-mode');
    }

    const isAuthenticated = computed(() => authStore.isAuthenticated);

    const handleLogout = () => {
      authStore.logout();
    };

    const goToProfile = () => {
      router.push('/profile');
    };

    // Menü ein- und ausklappen
    const isAsideCollapsed = ref(false);
    const toggleAside = () => {
      isAsideCollapsed.value = !isAsideCollapsed.value;
    };

    provide('changeLanguage', changeLanguage);
    provide('isDarkMode', isDarkMode);
    provide('toggleTheme', toggleTheme);
    provide('isAsideCollapsed', isAsideCollapsed);
    provide('toggleAside', toggleAside);

    return {
      isDarkMode,
      toggleTheme,
      currentLocale,
      Sunny,
      Moon,
      isAuthenticated,
      handleLogout,
      selectedLanguage,
      changeLanguage,
      goToProfile,
      isAsideCollapsed,
      toggleAside,
      Menu,
    };
  },
});
</script>

<style>
body {
  margin: 0;
  padding: 0;
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: #ffffff;
  color: #2c3e50;
  transition: background-color 0.3s, color 0.3s;
}
body.dark-mode {
  background-color: #1a1a1a;
  color: #cccccc;
}
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  height: 80px;
  min-height: 80px;
  background: #fff;
}
.dark-mode .app-header {
  background: #1a1a1a;
}
.header-content {
  display: flex;
  align-items: center;
  height: 100%;
}
.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 100%;
}
.language-switcher-app {
  margin-right: 15px;
}
.language-icon-app {
  width: 20px;
  height: 20px;
  cursor: pointer;
}
.dark-mode .aside-menu {
  background: #1a1a1a !important;
}
.dark-mode .aside-logo {
  background: #1a1a1a !important;
  border-bottom: 1px solid #222 !important;
}
.dark-mode .el-menu-vertical-demo {
  background-color: #1a1a1a !important;
  color: var(--el-text-color-primary);
}
.dark-mode .el-menu-vertical-demo .el-menu-item {
    color: var(--el-text-color-primary);
}
.dark-mode .el-menu-vertical-demo .el-menu-item.is-active {
    color: var(--el-color-primary);
}
.aside-logo {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 80px;
  background: #fff;
  border-bottom: 1px solid #eee;
}
.logo-img {
  max-width: 120px;
  max-height: 60px;
  object-fit: contain;
}
.aside-menu {
  transition: width 0.3s;
  overflow: hidden;
}
.aside-toggle-btn {
  font-size: 20px;
}
</style>