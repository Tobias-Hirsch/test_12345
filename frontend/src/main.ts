import { createApp } from 'vue';
import 'vue-demi/lib/index'
import App from './App.vue';
import ElementPlus from 'element-plus';
import * as ElementPlusIconsVue from '@element-plus/icons-vue';
import 'element-plus/dist/index.css';
import './styles/variables.css'; // Import theme variables
import router from './router'; // Import the router
import i18n from './locales'; // Import the i18n configuration
import { createPinia } from 'pinia'; // Import createPinia
import { useAppStore } from './stores/app'; // Import the app store

const app = createApp(App);
const pinia = createPinia(); // Create a Pinia instance

// Initialize Pinia and the app store
app.use(pinia);
const appStore = useAppStore();
appStore.setLanguage(localStorage.getItem('language') || 'de'); // Set initial language from localStorage

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
  }
  
app.use(ElementPlus);
app.use(router); // Use the router
app.use(i18n); // Use the i18n configuration

app.mount('#app');