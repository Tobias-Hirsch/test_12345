import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useAppStore = defineStore('app', () => {
  const language = ref(localStorage.getItem('language') || 'de');

  function setLanguage(lang: string) {
    language.value = lang;
    localStorage.setItem('language', lang);
  }

  return { language, setLanguage };
});