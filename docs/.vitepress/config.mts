import { defineConfig } from 'vitepress'

export default defineConfig({
  // Kommentar
  // Kommentar
  base: '/docs/', // Set the base path for deployment
  head: [
    ['meta', { property: 'og:locale', content: 'en_US' }],
    ['meta', { property: 'og:locale:alternate', content: 'zh_CN' }],
  ],

  themeConfig: {}, // Hinweis
  locales: {
    '/': { // Hinweis'/'
      label: 'Beschriftung',
      lang: 'zh-CN',
      title: "Rosti Titel",
      description: "Rosti Hinweis",
      link: '/', // Hinweis
      themeConfig: {
        i18nRouting: true, // Hinweis
        nav: [
          { text: 'Hinweis', link: '/' }, // Hinweis
          { text: 'Hinweis', link: '/frontend/' },
          { text: 'Hinweis', link: '/backend/' },
          { text: 'Hinweis', link: '/settings/' },
          { text: 'Hinweis', link: '/tech-specs/' }
        ],
        sidebar: {
          '/frontend/': [
            {
              text: 'Hinweis',
              items: [
                { text: 'Hinweis', link: '/frontend/' },
                { text: 'BenutzerHinweis', link: '/frontend/user-interface' },
                { text: 'Hinweis', link: '/frontend/data-display' }
              ]
            },
            {
              text: 'Hinweis',
              items: [
                { text: 'Hinweis', link: '/frontend/authentication/activate-account' },
                { text: 'Passwort vergessen', link: '/frontend/authentication/forgot-password' },
                { text: 'BenutzerAnmelden', link: '/frontend/authentication/login' },
                { text: 'BenutzerRegistrieren', link: '/frontend/authentication/register' },
                { text: 'Passwort zurücksetzen', link: '/frontend/authentication/reset-password' }
              ]
            },
            {
              text: 'Hinweis',
              items: [
                { text: 'Hinweis', link: '/frontend/chat-rag/chat-page' },
                { text: 'Chat-Nachrichtenbereich', link: '/frontend/chat-rag/chat-message-area' },
                { text: 'RAG AbfragenHinweis', link: '/frontend/chat-rag/query-rag' },
                { text: 'RAG BearbeitenHinweis', link: '/frontend/chat-rag/rag-edit' },
                { text: 'RAG Hinweis', link: '/frontend/chat-rag/rag-intro' },
                { text: 'RAG Hinweis', link: '/frontend/chat-rag/rag-list' },
                { text: 'Rosti Hinweis', link: '/frontend/chat-rag/rosti-chat-interface' },
                { text: 'Hinweis', link: '/frontend/chat-rag/upload-file' },
                { text: 'Hinweis', link: '/frontend/chat-rag/components/file-preview-drawer' },
                { text: 'RAG Hinweis', link: '/frontend/chat-rag/components/rag-embedding' },
                { text: 'RAG Hinweis', link: '/frontend/chat-rag/components/rag-file-list' },
                { text: 'RAG Hinweis', link: '/frontend/chat-rag/components/rag-form' }
              ]
            },
            {
              text: 'Systemeinstellungen',
              items: [
                { text: 'Richtlinienverwaltung', link: '/frontend/system-settings/policy-management' },
                { text: 'Systemeinstellungen', link: '/frontend/system-settings/system-settings' },
                { text: 'BenutzerHinweis', link: '/frontend/system-settings/user-profile' },
                { text: 'Rechteverwaltung', link: '/frontend/system-settings/permission-management' },
                { text: 'Rollenverwaltung', link: '/frontend/system-settings/role-management' },
                { text: 'Benutzerverwaltung', link: '/frontend/system-settings/user-management' }
              ]
            },
            {
              text: 'Fehlerhinweis',
              items: [
                { text: 'Fehler bei der Verarbeitung', link: '/frontend/error-pages/general-error' },
                { text: '404 Fehler bei der Verarbeitung', link: '/frontend/error-pages/not-found' },
                { text: 'BerechtigungFehler bei der Verarbeitung', link: '/frontend/error-pages/permission-denied' }
              ]
            }
          ],
          '/backend/': [
            {
              text: 'Hinweis',
              items: [
                { text: 'Hinweis', link: '/backend/' },
                { text: 'main.py', link: '/backend/main-py' },
                { text: 'Dockerfile', link: '/backend/dockerfile' },
                { text: 'requirements.txt', link: '/backend/requirements-txt' }
              ]
            },
            {
              text: 'Hinweis',
              items: [
                { text: 'Hinweis', link: '/backend/core/config' },
                { text: 'Hinweis', link: '/backend/core/security' }
              ]
            },
            {
              text: 'Hinweis',
              items: [
                { text: 'Hinweis', link: '/backend/models/database' }
              ]
            },
            {
              text: 'Hinweis',
              items: [
                { text: 'MinIO Hinweis', link: '/backend/modules/minio-module' },
                { text: 'Milvus Hinweis', link: '/backend/modules/milvus-module' },
                { text: 'MongoDB Hinweis', link: '/backend/modules/mongodb-module' },
                { text: 'MySQL Hinweis', link: '/backend/modules/mysql-module' },
                { text: 'Ollama Hinweis', link: '/backend/modules/ollama-module' },
                { text: 'MagicPDF MinIO', link: '/backend/modules/magicpdf-minio' }
              ]
            },
            {
              text: 'RAG Hinweis',
              items: [
                { text: 'Hinweis', link: '/backend/rag-knowledge/embedding-service' },
                { text: 'Hinweis', link: '/backend/rag-knowledge/generic-knowledge' }
              ]
            },
            {
              text: 'API Hinweis',
              items: [
                { text: 'Hinweis', link: '/backend/routers/authentication' },
                { text: 'Captcha', link: '/backend/routers/captcha' },
                { text: 'Hinweis', link: '/backend/routers/chat' },
                { text: 'Hinweis', link: '/backend/routers/files' },
                { text: 'RAG', link: '/backend/routers/rag' },
                { text: 'Hinweis', link: '/backend/routers/embeddings' },
                { text: 'Rolle', link: '/backend/routers/roles' },
                { text: 'Berechtigung', link: '/backend/routers/permissions' },
                { text: 'BenutzerRolle', link: '/backend/routers/user-roles' },
                { text: 'Benutzer', link: '/backend/routers/users' },
                { text: 'Hinweis', link: '/backend/routers/settings' },
                { text: 'SMTP', link: '/backend/routers/smtp' },
                { text: 'Hinweis', link: '/backend/routers/policies' },
                { text: 'Hinweis', link: '/backend/routers/agent-chat' },
                { text: 'Hinweis', link: '/backend/routers/chatpage' }
              ]
            },
            {
              text: 'Hinweis',
              items: [
                { text: 'Hinweis', link: '/backend/schemas/schemas' },
                { text: 'Hinweis', link: '/backend/schemas/chat-schemas' }
              ]
            },
            {
              text: 'Hinweis',
              items: [
                { text: 'Hinweis', link: '/backend/services/auth' },
                { text: 'Hinweis', link: '/backend/services/auth-thirdparty' },
                { text: 'Hinweis', link: '/backend/services/email' },
                { text: 'RAG Hinweis', link: '/backend/services/rag-file-service' },
                { text: 'Hinweis', link: '/backend/services/conversation-cleaner' },
                { text: 'Hinweis', link: '/backend/services/inactive-user-cleaner' },
                { text: 'Hinweis', link: '/backend/services/logging' },
                { text: 'MSAD LDAP', link: '/backend/services/msad-ldap' },
                { text: 'Ollama DeepSeek', link: '/backend/services/ollama-deepseek' },
                { text: 'Ollama Hinweis', link: '/backend/services/ollama-service' },
                { text: 'RAG BerechtigungHinweis', link: '/backend/services/rag-permission-service' },
                { text: 'Hinweis', link: '/backend/services/chat-data-service' },
                { text: 'ABAC Hinweis', link: '/backend/services/abac-attribute-extractor' },
                { text: 'ABAC Hinweis', link: '/backend/services/abac-functions' },
                { text: 'ABAC Hinweis', link: '/backend/services/abac-policy-evaluator' }
              ]
            },
            {
              text: 'Hinweis',
              items: [
                { text: 'PDF Hinweis', link: '/backend/tools/pdf' },
                { text: 'DokumenteHinweis', link: '/backend/tools/deal-document' },
                { text: 'Excel Hinweis', link: '/backend/tools/exlsx' },
                { text: 'PyMuPDF Hinweis', link: '/backend/tools/inpymupdf' },
                { text: 'Hinweis', link: '/backend/tools/retry-tools' },
                { text: 'Hinweis', link: '/backend/tools/search-online-tools' },
                { text: 'Hinweis', link: '/backend/tools/split-tools' },
                { text: 'Word Hinweis', link: '/backend/tools/word' }
              ]
            },
            {
              text: 'LLM Hinweis',
              items: [
                { text: 'Hinweis', link: '/backend/llm/chain' },
                { text: 'LLM Hinweis', link: '/backend/llm/llm' }
              ]
            }
          ],
          '/settings/': [
            {
              text: 'Hinweis',
              items: [
                { text: 'Hinweis', link: '/settings/' },
                { text: 'BenutzerHinweis', link: '/settings/user-settings' }
              ]
            }
          ],
          '/tech-specs/': [
            {
              text: 'Hinweis',
              items: [
                { text: 'Hinweis', link: '/tech-specs/' },
                { text: 'Hinweis', link: '/tech-specs/architecture' }
              ]
            }
          ]
        },
        // socialLinks: [
        //   { icon: 'github', link: 'https://github.com/your-org/your-repo' }
        // ],
        footer: {
          message: 'Hinweis',
          copyright: 'Hinweis© 2017-Hinweis'
        }
      }
    },
    '/en/': {
      label: 'English',
      lang: 'en-US',
      title: "Rosti Product Documentation",
      description: "Rosti Product User Manual and Technical Specifications.",
      link: '/en/',
      themeConfig: {
        i18nRouting: true, // Hinweis
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Frontend Features', link: '/en/frontend/' },
          { text: 'Backend Features', link: '/en/backend/' },
          { text: 'Settings', link: '/en/settings/' },
          { text: 'Tech Specs', link: '/en/tech-specs/' }
        ],
        sidebar: {
          '/en/frontend/': [
            {
              text: 'Frontend Features',
              items: [
                { text: 'Overview', link: '/en/frontend/' },
                { text: 'User Interface', link: '/en/frontend/user-interface' },
                { text: 'Data Display', link: '/en/frontend/data-display' }
              ]
            },
            {
              text: 'Authentication & User Management',
              items: [
                { text: 'Account Activation', link: '/en/frontend/authentication/activate-account' },
                { text: 'Forgot Password', link: '/en/frontend/authentication/forgot-password' },
                { text: 'User Login', link: '/en/frontend/authentication/login' },
                { text: 'User Registration', link: '/en/frontend/authentication/register' },
                { text: 'Reset Password', link: '/en/frontend/authentication/reset-password' }
              ]
            },
            {
              text: 'Chat & RAG',
              items: [
                { text: 'Chat Page', link: '/en/frontend/chat-rag/chat-page' },
                { text: 'Chat Message Area', link: '/en/frontend/chat-rag/chat-message-area' },
                { text: 'RAG Query Page', link: '/en/frontend/chat-rag/query-rag' },
                { text: 'RAG Edit Page', link: '/en/frontend/chat-rag/rag-edit' },
                { text: 'RAG Intro Page', link: '/en/frontend/chat-rag/rag-intro' },
                { text: 'RAG File List', link: '/en/frontend/chat-rag/rag-list' },
                { text: 'Rosti Chat Interface Component', link: '/en/frontend/chat-rag/rosti-chat-interface' },
                { text: 'File Upload Page', link: '/en/frontend/chat-rag/upload-file' },
                { text: 'File Preview Component', link: '/en/frontend/chat-rag/components/file-preview-drawer' },
                { text: 'RAG Embedding Component', link: '/en/frontend/chat-rag/components/rag-embedding' },
                { text: 'RAG File List Component', link: '/en/frontend/chat-rag/components/rag-file-list' },
                { text: 'RAG Form Component', link: '/en/frontend/chat-rag/components/rag-form' }
              ]
            },
            {
              text: 'System Settings',
              items: [
                { text: 'Policy Management', link: '/en/frontend/system-settings/policy-management' },
                { text: 'System Settings', link: '/en/frontend/system-settings/system-settings' },
                { text: 'User Profile', link: '/en/settings/user-settings' },
                { text: 'Permission Management', link: '/en/frontend/system-settings/permission-management' },
                { text: 'Role Management', link: '/en/frontend/system-settings/role-management' },
                { text: 'User Management', link: '/en/frontend/system-settings/user-management' }
              ]
            },
            {
              text: 'Error Pages',
              items: [
                { text: 'General Error Page', link: '/en/frontend/error-pages/general-error' },
                { text: '404 Page Not Found', link: '/en/frontend/error-pages/not-found' },
                { text: 'Permission Denied Page', link: '/en/frontend/error-pages/permission-denied' }
              ]
            }
          ],
          '/en/backend/': [
            {
              text: 'Backend Features',
              items: [
                { text: 'Overview', link: '/en/backend/' },
                { text: 'main.py', link: '/en/backend/main-py' },
                { text: 'Dockerfile', link: '/en/backend/dockerfile' },
                { text: 'requirements.txt', link: '/en/backend/requirements-txt' }
              ]
            },
            {
              text: 'Core Modules',
              items: [
                { text: 'Configuration', link: '/en/backend/core/config' },
                { text: 'Security', link: '/en/backend/core/security' }
              ]
            },
            {
              text: 'Database Models',
              items: [
                { text: 'Database', link: '/en/backend/models/database' }
              ]
            },
            {
              text: 'Modules',
              items: [
                { text: 'MinIO Module', link: '/en/backend/modules/minio-module' },
                { text: 'Milvus Module', link: '/en/backend/modules/milvus-module' },
                { text: 'MongoDB Module', link: '/en/backend/modules/mongodb-module' },
                { text: 'MySQL Module', link: '/en/backend/modules/mysql-module' },
                { text: 'Ollama Module', link: '/en/backend/modules/ollama-module' },
                { text: 'MagicPDF MinIO', link: '/en/backend/modules/magicpdf-minio' }
              ]
            },
            {
              text: 'RAG Knowledge',
              items: [
                { text: 'Embedding Service', link: '/en/backend/rag-knowledge/embedding-service' },
                { text: 'Generic Knowledge', link: '/en/backend/rag-knowledge/generic-knowledge' }
              ]
            },
            {
              text: 'API Routes',
              items: [
                { text: 'Authentication', link: '/en/backend/routers/authentication' },
                { text: 'Captcha', link: '/en/backend/routers/captcha' },
                { text: 'Chat', link: '/en/backend/routers/chat' },
                { text: 'Files', link: '/en/backend/routers/files' },
                { text: 'RAG', link: '/en/backend/routers/rag' },
                { text: 'Embeddings', link: '/en/backend/routers/embeddings' },
                { text: 'Roles', link: '/en/backend/routers/roles' },
                { text: 'Permissions', link: '/en/backend/routers/permissions' },
                { text: 'User Roles', link: '/en/backend/routers/user-roles' },
                { text: 'Users', link: '/en/backend/routers/users' },
                { text: 'Settings', link: '/en/backend/routers/settings' },
                { text: 'SMTP', link: '/en/backend/routers/smtp' },
                { text: 'Policies', link: '/en/backend/routers/policies' },
                { text: 'Agent Chat', link: '/en/backend/routers/agent-chat' },
                { text: 'Chat Page', link: '/en/backend/routers/chatpage' }
              ]
            },
            {
              text: 'Data Schemas',
              items: [
                { text: 'General Schemas', link: '/en/backend/schemas/schemas' },
                { text: 'Chat Schemas', link: '/en/backend/schemas/chat-schemas' }
              ]
            },
            {
              text: 'Business Services',
              items: [
                { text: 'Authentication Service', link: '/en/backend/services/auth' },
                { text: 'Third-Party Authentication', link: '/en/backend/services/auth-thirdparty' },
                { text: 'Email Service', link: '/en/backend/services/email' },
                { text: 'RAG File Service', link: '/en/backend/services/rag-file-service' },
                { text: 'Conversation Cleaner', link: '/en/backend/services/conversation-cleaner' },
                { text: 'Inactive User Cleaner', link: '/en/backend/services/inactive-user-cleaner' },
                { text: 'Logging Service', link: '/en/backend/services/logging' },
                { text: 'MSAD LDAP', link: '/en/backend/services/msad-ldap' },
                { text: 'Ollama DeepSeek', link: '/en/backend/services/ollama-deepseek' },
                { text: 'Ollama Service', link: '/en/backend/services/ollama-service' },
                { text: 'RAG Permission Service', link: '/en/backend/services/rag-permission-service' },
                { text: 'Chat Data Service', link: '/en/backend/services/chat-data-service' },
                { text: 'ABAC Attribute Extractor', link: '/en/backend/services/abac-attribute-extractor' },
                { text: 'ABAC Functions', link: '/en/backend/services/abac-functions' },
                { text: 'ABAC Policy Evaluator', link: '/en/backend/services/abac-policy-evaluator' }
              ]
            },
            {
              text: 'Tools',
              items: [
                { text: 'PDF Tool', link: '/en/backend/tools/pdf' },
                { text: 'Document Processing Tool', link: '/en/backend/tools/deal-document' },
                { text: 'Excel Tool', link: '/en/backend/tools/exlsx' },
                { text: 'PyMuPDF Tool', link: '/en/backend/tools/inpymupdf' },
                { text: 'Retry Tool', link: '/en/backend/tools/retry-tools' },
                { text: 'Online Search Tool', link: '/en/backend/tools/search-online-tools' },
                { text: 'Split Tool', link: '/en/backend/tools/split-tools' },
                { text: 'Word Tool', link: '/en/backend/tools/word' }
              ]
            },
            {
              text: 'LLM Related',
              items: [
                { text: 'Chain', link: '/en/backend/llm/chain' },
                { text: 'LLM Client', link: '/en/backend/llm/llm' }
              ]
            }
          ],
          '/en/settings/': [
            {
              text: 'Settings',
              items: [
                { text: 'Overview', link: '/en/settings/' },
                { text: 'User Settings', link: '/en/settings/user-settings' }
              ]
            }
          ],
          '/en/tech-specs/': [
            {
              text: 'Technical Specifications',
              items: [
                { text: 'Overview', link: '/en/tech-specs/' },
                { text: 'System Architecture', link: '/en/tech-specs/architecture' }
              ]
            }
          ]
        },
        // socialLinks: [
        //   { icon: 'github', link: 'https://github.com/your-org/your-repo' }
        // ],
        footer: {
          message: 'Released under the MIT License.',
          copyright: 'Copyright © 2017-present Shanghai De Manufacturing IT Co., Ltd.'
        }
      }
    }
  }
})
