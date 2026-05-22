<template>
  <div class="login-container">
    <!-- Language Switcher -->
    <el-dropdown @command="handleLocaleChange" class="language-switcher">
      <img src="/image/planet.png" alt="Language Switcher Icon" class="language-icon">
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item command="en">English</el-dropdown-item>
          <el-dropdown-item command="de">Deutsch</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
    <div class="login-content">
      <img src="/image/rosti_logo.jpg" alt=" Logo" class="logo">
      <!-- <div class="logo-text">From concept to reality</div> -->
      <el-card class="login-card">
        <el-form :model="loginForm" :rules="loginRules" ref="loginFormRef" label-position="top">
          <el-form-item :label="$t('login.username')" prop="username">
            <el-input v-model="loginForm.username" :placeholder="$t('login.usernamePlaceholder')"></el-input>
          </el-form-item>
          <el-form-item :label="$t('login.password')" prop="password">
            <el-input type="password" v-model="loginForm.password" :placeholder="$t('login.passwordPlaceholder')" show-password></el-input>
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="handleLogin" style="width: 100%;">{{ $t('login.loginButton') }}</el-button>
          </el-form-item>
        </el-form>
        <el-divider>{{ $t('login.or') }}</el-divider>

        <div class="social-login-buttons">

          <el-button v-if="enabledAuthProviders.google" @click="handleThirdPartyLogin('google')" class="social-button google-button">
            <img src="/image/google_logo.png" alt="Google" class="social-icon"> Google
          </el-button>

          <el-button v-if="enabledAuthProviders.microsoft" @click="handleThirdPartyLogin('microsoft')" class="social-button microsoft-button">
            <img src="/image/microsoft_logo.png" alt="Microsoft" class="social-icon"> Microsoft
          </el-button>

          <el-button v-if="enabledAuthProviders.oauth_generic" @click="handleThirdPartyLogin('oauth_generic')" class="social-button oauth-button">
            <img src="/image/oauth_logo.png" alt="OAuth" class="social-icon"> OAuth
          </el-button>
        </div>
      </el-card>
      <div class="link-buttons"> <!-- Move links outside the card -->
        <el-button link @click="goToRegister">{{ $t('login.registerLink') }}</el-button>
        <el-button link @click="goToForgotPassword">{{ $t('login.forgotPasswordLink') }}</el-button>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted } from 'vue'; // Removed nextTick, kept onMounted
import { useRouter } from 'vue-router';
import { ElForm, ElFormItem, ElInput, ElButton, ElCard, ElMessage, ElDivider } from 'element-plus';
import type { FormInstance, FormRules } from 'element-plus';
import { useI18n } from 'vue-i18n';
import { useAuthStore } from '../stores/auth';
import { login } from '../services/authService';
import axios from 'axios'; // Keep only one axios import
import { API_BASE_URL } from '../services/apiService'; // Import API_BASE_URL

export default defineComponent({
  name: 'Login',
  components: {
    ElForm,
    ElFormItem,
    ElInput,
    ElButton,
    ElCard,
  },
  setup() {
    const router = useRouter();
    const { t } = useI18n();
    const authStore = useAuthStore(); // Use the auth store

    const { locale } = useI18n(); // Destructure locale from useI18n
    const currentLocale = ref(locale.value); // Create a ref for the current locale

    const handleLocaleChange = (newLocale: string) => {
      locale.value = newLocale; // Update the locale
      // Optionally save the locale preference, e.g., in localStorage
      localStorage.setItem('language', newLocale);
    };

    const loginFormRef = ref<FormInstance>();
    const loginForm = ref({
      username: '',
      password: '',
    });

    const enabledAuthProviders = ref({
      google: false,
      microsoft: false,
      oauth_generic: false,
    });

    const fetchEnabledAuthProviders = async () => {
      // console.log('Attempting to fetch enabled auth providers...');
      try {
        const response = await axios.get(`${API_BASE_URL}/auth/enabled-providers`);
        enabledAuthProviders.value = response.data as { google: boolean; microsoft: boolean; oauth_generic: boolean; };
        // console.log('Successfully fetched enabled auth providers:', enabledAuthProviders.value);
      } catch (error) {
        console.error('Failed to fetch enabled auth providers:', error);
      }
    };

    const handleThirdPartyLogin = (provider: string) => {
      // Redirect to backend OAuth initiation endpoint
      window.location.href = `${API_BASE_URL}/auth/${provider}/login`;
    };

    onMounted(() => {
      fetchEnabledAuthProviders();
    });

    const loginRules: FormRules = {
      username: [
        { required: true, message: t('login.usernameRequired'), trigger: 'blur' },
      ],
      password: [
        { required: true, message: t('login.passwordRequired'), trigger: 'blur' },
      ],
    };

    const handleLogin = async () => {
      const form = loginFormRef.value;
      if (!form) return;

      form.validate(async (valid) => {
        if (valid) {
          // console.log('Login form submitted:', loginForm.value);
          try {
            // Call the login action from the store
            await authStore.login(loginForm.value);
            ElMessage.success(t('login.loginSuccess'));
            // console.log('Login successful, attempting redirection...');
            // Redirect to a protected page after successful login
            router.push('/'); // Use replace instead of push
          } catch (error: any) {
            // console.error('Login failed:', error);
            // Check for specific inactive user error detail
            if (error.message && error.message.includes('inactive_user')) {
              ElMessage.error('Der aktuelle Benutzer existiert nicht oder ist nicht aktiviert'); // Specific message for inactive users
            } else {
              ElMessage.error(error.message || t('login.loginFailed')); // Generic login failed message
            }
          }
        } else {
          // console.log('Login form validation failed');
        }
      });
    };

    const goToRegister = () => {
      router.push('/register');
    };

    const goToForgotPassword = () => {
      router.push('/forgot-password');
    };

    return {
      loginFormRef,
      loginForm,
      loginRules,
      handleLogin,
      goToRegister,
      goToForgotPassword,
      t, // Expose t for template usage
      currentLocale, // Expose currentLocale
      handleLocaleChange, // Expose handleLocaleChange
      enabledAuthProviders, // Expose enabledAuthProviders
      handleThirdPartyLogin, // Expose handleThirdPartyLogin
    };
  },
});
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background-color: var(--el-bg-color-page);
}

.social-login-buttons {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 20px;
}

.social-button {
  width: 100%;
  display: flex;
  align-items: center;
  /* Removed: justify-content: center; */
  padding: 10px 20px;
  font-size: 16px;
  border-radius: 4px;
  cursor: pointer;
  padding-left: 40px; /* Adjust this value as needed for visual alignment */
}

.social-icon {
  width: 20px;
  height: 20px;
  margin-right: 8px;
}

.google-button {
  background-color: #DB4437; /* Google Red */
  color: white;
  border: 1px solid #DB4437;
}

.google-button:hover {
  background-color: #c23321;
  border-color: #c23321;
}

.microsoft-button {
  background-color: #0078D4; /* Microsoft Blue */
  color: white;
  border: 1px solid #0078D4;
}

.microsoft-button:hover {
  background-color: #005a9e;
  border-color: #005a9e;
}

.oauth-button {
  background-color: #6A5ACD; /* SlateBlue */
  color: white;
  border: 1px solid #6A5ACD;
}

.oauth-button:hover {
  background-color: #5346a3;
  border-color: #5346a3;
}

.login-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative; /* Add position relative for absolute positioning of switcher */
}

.language-switcher {
  position: absolute;
  top: 20px; /* Adjust position as needed */
  right: 20px; /* Adjust position as needed */
  width: 120px; /* Adjust width as needed */
}

.language-icon {
  width: 24px; /* Adjust size as needed */
  height: 24px; /* Adjust size as needed */
  cursor: pointer;
}

.logo {
  width: 200px; /* Adjust size as needed */
  margin-bottom: 10px;
}

.logo-text {
  font-size: 1em;
  color: #007bff; /* Adjust color as needed */
  margin-bottom: 20px;
}

.login-card {
  width: 400px;
  max-width: 90%;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  background-color: var(--el-bg-color-overlay);
  color: var(--el-text-color-primary);
  padding: 20px; /* Add padding inside the card */
}

/* Adjust form item margin if needed */
.el-form-item {
  margin-bottom: 20px;
}

/* Hide labels for username and password */
.el-form-item__label {
  display: none;
}




/* Revert the login button styling to default Element Plus primary button */
.login-card .el-button--primary {
  /* Remove custom styles */
  padding: 12px 20px; /* Default Element Plus button padding */
  height: auto;
}

.link-buttons {
  text-align: center;
  margin-top: 20px; /* Increase space above the links */
}

.link-buttons .el-button {
  margin: 0 10px; /* Increase space between the buttons */
}
</style>