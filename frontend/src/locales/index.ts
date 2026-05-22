import { createI18n } from 'vue-i18n';

// Sprachdateien importieren
import en from './en.json';
import de from './de.json';

const messages = {
  en,
  de,
};

const determineLocale = () => {
  // Zuerst localStorage prüfen
  const savedLanguage = localStorage.getItem('language');
  if (savedLanguage) {
    return savedLanguage;
  }

  // Falls keine Sprache gespeichert ist, Browsersprache verwenden
  const browserLanguage = navigator.language.split('-')[0]; // Get primary language tag
  const supportedLocales = Object.keys(messages);
  if (supportedLocales.includes(browserLanguage)) {
    return browserLanguage;
  }

  // Fallback auf Standardsprache
  return 'de';
};

const i18n = createI18n({
  legacy: false, // Use Composition API
  locale: determineLocale(), // Set locale based on localStorage or browser language
  fallbackLocale: 'de', // Fallback locale
  messages,
});

export default i18n;