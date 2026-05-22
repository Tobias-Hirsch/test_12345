import { useAuthStore } from '../stores/auth';
import { getAuthHeaders } from './authService'; // Re-use the helper to get auth headers
import { toRaw } from 'vue';

export const API_BASE_URL = '/api'; // Export API_BASE_URL

export interface RAGResultItem {
  source: string;
  score: number;
  text: string;
}

export interface QueryResult {
  query: string;
  results: RAGResultItem[];
}

export interface AgentChatRequest {
  message: string;
  display_thoughts?: boolean;
  search_ai_active?: boolean;
  search_rag_active?: boolean;
  search_online_active?: boolean;
  files?: File[];
}

// Interfaces for RAG Data
export interface RagDataBase {
  name: string;
  description?: string;
}

export interface RagDataCreate extends RagDataBase {}

export interface RagData extends RagDataBase {
  id: number;
  is_active: number;
  created_at: string; // Assuming string representation of datetime from backend
  updated_at?: string; // Assuming string representation of datetime from backend
  access_level?: string; // Add access_level for granular permissions
}

export interface RagDataUpdate {
  name?: string;
  description?: string;
  is_active?: number;
}


export interface RagDataListResponse {
  rag_data: RagData[];
  can_create: boolean;
}

export interface FileGist {
  id: number;
  filename: string;
  gist: string;
  created_at: string; // Assuming string representation of datetime from backend
  rag_id: number;
  download_url?: string; // Add download_url to store the MinIO pre-signed URL
}


const handleJsonResponse = async (response: Response): Promise<any> => {
  if (!response.ok) {
    // If response is not ok, try to parse JSON for error details
    try {
      const error = await response.json();
      // For 403 errors, throw a specific message that can be caught by the component
      if (response.status === 403) {
        throw new Error(error.detail || 'Permission Denied');
      }
      throw new Error(error.detail || `API request failed with status ${response.status}`);
    } catch (e) {
      // If parsing JSON fails for an error response, throw a generic error
      throw new Error(`API request failed with status ${response.status}`);
    }
  }

  // If response is ok, check content type before parsing JSON
  const contentType = response.headers.get('Content-Type');
  if (contentType && contentType.includes('application/json')) {
    try {
      return await response.json();
    } catch (e) {
      console.error("Failed to parse JSON from successful response:", e);
      // Return null if JSON parsing fails for a successful response
      return null;
    }
  } else {
    // If response is ok but not JSON, return null
    console.warn(`Received non-JSON response with status ${response.status}. Content-Type: ${contentType}`);
    return null;
  }
};

export const uploadFile = async (file: File): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);

  const headers = getAuthHeaders();

  const response = await fetch(`${API_BASE_URL}/rag/uploadfile`, {
    method: 'POST',
    headers: {
      ...headers,
      // 'Content-Type': 'multipart/form-data', // Fetch API sets this automatically with FormData
    },
    body: formData,
  });
  return handleJsonResponse(response);
};

export const queryRAG = async (queryText: string): Promise<QueryResult> => {
  const headers = getAuthHeaders();

  const response = await fetch(`${API_BASE_URL}/embeddings/query?query_text=${encodeURIComponent(queryText)}`, {
    method: 'GET',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

// RAG Data API Calls
export const createRagData = async (ragData: RagDataCreate): Promise<RagData> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/rag`, {
    method: 'POST',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(ragData),
  });
  return handleJsonResponse(response);
};

export const getRagDataList = async (skip: number = 0, limit: number = 100): Promise<RagDataListResponse> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/rag?skip=${skip}&limit=${limit}`, {
    method: 'GET',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

export const getRagDataById = async (ragId: number): Promise<RagData> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/rag/${ragId}`, {
    method: 'GET',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

export const updateRagData = async (ragId: number, ragDataUpdate: RagDataUpdate): Promise<RagData> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/rag/${ragId}`, {
    method: 'PUT',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(ragDataUpdate),
  });
  return handleJsonResponse(response);
};

export const deleteRagData = async (ragId: number): Promise<any> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/rag/${ragId}`, {
    method: 'DELETE',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

export const uploadRagFile = async (ragId: number, file: File): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);

  const headers = getAuthHeaders();

  const response = await fetch(`${API_BASE_URL}/rag/${ragId}/uploadfile`, {
    method: 'POST',
    headers: {
      ...headers,
      // 'Content-Type': 'multipart/form-data', // Fetch API sets this automatically with FormData
    },
    body: formData,
  });
  return handleJsonResponse(response);
};

export const listRagFiles = async (ragId: number): Promise<FileGist[]> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/rag/${ragId}/files`, {
    method: 'GET',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

export const deleteRagFile = async (fileId: number): Promise<any> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/rag/files/${fileId}`, {
    method: 'DELETE',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

export const previewFile = async (fileId: number): Promise<Blob> => {
  const headers = getAuthHeaders();

  const response = await fetch(`${API_BASE_URL}/files/${fileId}/content`, {
    method: 'GET',
    headers: {
      ...headers,
    },
  });

  if (!response.ok) {
    // Try to get more detailed error from response body
    const errorData = await response.json().catch(() => ({ detail: `Failed to fetch file: ${response.statusText}` }));
    throw new Error(errorData.detail || `Failed to fetch file: ${response.statusText}`);
  }

  return response.blob();
};
export const previewRagFile = async (fileId: number): Promise<any> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/rag/files/${fileId}/preview`, {
    method: 'GET',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

export const embedRagFiles = async (ragId: number, fileIds: number[]): Promise<any> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/rag/${ragId}/embed_files`, {
    method: 'POST',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ file_ids: fileIds }),
  });
  return handleJsonResponse(response);
};


// --- Knowledge Base Management API Calls ---

export const listKnowledgeBaseDocuments = async (ragId: number): Promise<FileGist[]> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/rag/${ragId}/documents`, {
    method: 'GET',
    headers: { ...headers, 'Content-Type': 'application/json' },
  });
  return handleJsonResponse(response);
};

export const uploadFilesToKnowledgeBase = async (ragId: number, files: File[]): Promise<FileGist[]> => {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/files/${ragId}/upload`, {
    method: 'POST',
    headers: { ...headers }, // Content-Type is set automatically for FormData
    body: formData,
  });
  return handleJsonResponse(response);
};

export const triggerKnowledgeBaseEmbedding = async (ragId: number, fileIds: number[]): Promise<any> => {
  // This function reuses the existing embedRagFiles logic
  return embedRagFiles(ragId, fileIds);
};

export const deleteKnowledgeBaseDocument = async (ragId: number, documentId: number): Promise<any> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/rag/${ragId}/documents/${documentId}`, {
    method: 'DELETE',
    headers: { ...headers, 'Content-Type': 'application/json' },
  });
  return handleJsonResponse(response);
};


// Interfaces for User Management
export interface User {
  id: number;
  username: string;
  email?: string;
  phone?: string;
  department: string;
  first_name?: string;
  surname?: string;
  avatar?: string; // Add avatar field
  is_active: number;
  activation_token?: string;
  activation_expires_at?: string; // Assuming string representation of datetime
  roles: Role[]; // Add roles to User interface
}

export interface Role {
  id: number;
  name: string;
  description?: string;
  created_at: string; // Assuming string representation of datetime
  updated_at?: string; // Assuming string representation of datetime
  // The 'permissions' field is removed as permissions are now decoupled from roles
  // and handled by ABAC policies.
}

export interface UserRoleAssign {
  user_id: number;
  role_id: number;
}


// User Management API Calls
export const getUsers = async (skip: number = 0, limit: number = 100): Promise<User[]> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/users?skip=${skip}&limit=${limit}`, { // TODO: Update endpoint path if needed
    method: 'GET',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

export const deleteUser = async (userId: number): Promise<any> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/users/${userId}`, {
    method: 'DELETE',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

export const updateUser = async (userId: number, userDataUpdate: Partial<User>): Promise<User> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/users/${userId}`, {
    method: 'PUT',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(userDataUpdate),
  });
  return handleJsonResponse(response);
};


// Role Management API Calls
export const getRoles = async (skip: number = 0, limit: number = 100): Promise<Role[]> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/roles?skip=${skip}&limit=${limit}`, {
    method: 'GET',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

export const createRole = async (role: { name: string; description?: string }): Promise<Role> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/roles`, {
    method: 'POST',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(role),
  });
  return handleJsonResponse(response);
};



// User-Role Association API Calls
export const assignRoleToUser = async (userRole: UserRoleAssign): Promise<any> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/user-roles`, {
    method: 'POST',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(userRole),
  });
  return handleJsonResponse(response);
};

export const removeRoleFromUser = async (userRole: UserRoleAssign): Promise<any> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/user-roles`, {
    method: 'DELETE',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(userRole),
  });
  return handleJsonResponse(response);
};

export const getUserRoles = async (userId: number): Promise<Role[]> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/user-roles/user/${userId}/roles`, {
    method: 'GET',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

// ABAC Permission Check
export interface CheckPermissionRequest {
  action: string;
  resource_type: string;
  resource_id?: string;
}

export interface CheckPermissionResponse {
  allowed: boolean;
}

export const checkAbacPermission = async (request: CheckPermissionRequest): Promise<CheckPermissionResponse> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/abac/check-permission`, {
    method: 'POST',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });
  return handleJsonResponse(response);
};

export const agentChat = async (
  request: AgentChatRequest,
  onChunk: (chunk: string) => void,
  onError: (error: any) => void
): Promise<void> => {
  const headers = getAuthHeaders();
  const formData = new FormData();

  formData.append('message', request.message);
  // Explicitly convert booleans to "true" or "false" strings
  if (request.display_thoughts !== undefined) {
    formData.append('display_thoughts', request.display_thoughts ? 'true' : 'false');
  }
  if (request.search_ai_active !== undefined) {
    formData.append('search_ai_active', request.search_ai_active ? 'true' : 'false');
  }
  if (request.search_rag_active !== undefined) {
    formData.append('search_rag_active', request.search_rag_active ? 'true' : 'false');
  }
  if (request.search_online_active !== undefined) {
    formData.append('search_online_active', request.search_online_active ? 'true' : 'false');
  }

  // Append files if they exist
  if (request.files && request.files.length > 0) {
    request.files.forEach((file) => {
      formData.append('files', file);
    });
  } else {
    // If no files are provided, append an empty blob to ensure the backend receives an empty list
    // This is a workaround for FastAPI's Pydantic validation of List[UploadFile]
    formData.append('files', new Blob([]), '');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/agent_chat`, {
      method: 'POST',
      headers: {
        ...headers,
        // 'Content-Type': 'multipart/form-data', // Fetch API sets this automatically with FormData
      },
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      // Stringify errorData.detail to ensure it's a readable string, especially for 422 validation errors
      const errorMessage = typeof errorData.detail === 'object'
                           ? JSON.stringify(errorData.detail)
                           : errorData.detail;
      throw new Error(errorMessage || `HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder('utf-8');

    if (!reader) {
      throw new Error('Failed to get reader from response body.');
    }

    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        // Process any remaining data in the buffer
        if (buffer.trim()) {
          const lines = buffer.split('\n\n').filter(line => line.trim() !== '');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              onChunk(line); // Pass the full 'data: {json}' string to onChunk
            }
          }
        }
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE messages from the buffer
      let lastNewlineIndex;
      while ((lastNewlineIndex = buffer.indexOf('\n\n')) !== -1) {
        const message = buffer.substring(0, lastNewlineIndex);
        buffer = buffer.substring(lastNewlineIndex + 2); // +2 for '\n\n'

        if (message.startsWith('data: ')) {
          onChunk(message); // Pass the full 'data: {json}' string to onChunk
        }
      }
    }
  } catch (error) {
    console.error("Streaming error:", error);
    onError(error);
  }
};

export const get = async (url: string, params?: Record<string, any>): Promise<any> => {
  const headers = getAuthHeaders();
  // Automatically remove trailing slashes from the URL, but ensure the base URL isn't just "/"
  const sanitizedUrl = url.length > 1 && url.endsWith('/') ? url.slice(0, -1) : url;
  let fullUrl = `${API_BASE_URL}${sanitizedUrl}`;

  if (params) {
    const queryParams = new URLSearchParams(params).toString();
    if (queryParams) {
      fullUrl += `?${queryParams}`;
    }
  }

  const response = await fetch(fullUrl, {
    method: 'GET',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

export const post = async (url: string, data: any): Promise<any> => {
  const headers = getAuthHeaders();
  const sanitizedUrl = url.length > 1 && url.endsWith('/') ? url.slice(0, -1) : url;
  const response = await fetch(`${API_BASE_URL}${sanitizedUrl}`, {
    method: 'POST',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });
  return handleJsonResponse(response);
};

export const put = async (url: string, data: any): Promise<any> => {
  const headers = getAuthHeaders();
  const sanitizedUrl = url.length > 1 && url.endsWith('/') ? url.slice(0, -1) : url;
  const response = await fetch(`${API_BASE_URL}${sanitizedUrl}`, {
    method: 'PUT',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });
  return handleJsonResponse(response);
};
export const del = async (url: string): Promise<any> => {
  const headers = getAuthHeaders();
  const sanitizedUrl = url.length > 1 && url.endsWith('/') ? url.slice(0, -1) : url;
  const response = await fetch(`${API_BASE_URL}${sanitizedUrl}`, {
    method: 'DELETE',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};


// Kommentar
const isNetworkError = (error: any): boolean => {
  if (!error) return false;
  
  const errorString = error.toString().toLowerCase();
  const networkErrors = [
    'network error',
    'err_incomplete_chunked_encoding',
    'err_connection_reset',
    'err_connection_aborted',
    'fetch_error',
    'abort',
    'timeout',
    'streaming error'
  ];
  
  return networkErrors.some(errorType => errorString.includes(errorType));
};

export const postStream = async (url: string, data: any, onChunk: (chunk: string) => void): Promise<void> => {
  const headers = getAuthHeaders();
  const maxRetries = 3;
  const retryDelay = 2000;
  const backoffMultiplier = 1.5;
  
  let attempt = 0;
  let accumulatedContent = ''; // Hinweis
  
  while (attempt <= maxRetries) {
    try {
      console.log(`Stream attempt ${attempt + 1}/${maxRetries + 1}`);
      
      const response = await fetch(`${API_BASE_URL}${url}`, {
        method: 'POST',
        headers: {
          ...headers,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');

      if (!reader) {
        throw new Error('Failed to get reader from response body.');
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        const chunk = decoder.decode(value, { stream: true });
        
        // Kommentar
        if (chunk.includes('event: text')) {
          const dataMatch = chunk.match(/data: "([^"]+)"/);
          if (dataMatch) {
            accumulatedContent += dataMatch[1];
          }
        }
        
        onChunk(chunk);
      }
      
      console.log('Stream completed successfully');
      return; // Hinweis
      
    } catch (error: any) {
      console.error(`Stream attempt ${attempt + 1} failed:`, error);
      
      // Kommentar
      if (!isNetworkError(error) || attempt >= maxRetries) {
        console.error('Max retries reached or non-network error, giving up');
        throw error;
      }
      
      attempt++;
      console.log(`Will retry in ${retryDelay * Math.pow(backoffMultiplier, attempt - 1)}ms...`);
      
      // Kommentar
      const delay = retryDelay * Math.pow(backoffMultiplier, attempt - 1);
      await new Promise(resolve => setTimeout(resolve, delay));
      
      // Kommentar
      if (accumulatedContent) {
        console.log(`Resuming from accumulated content: ${accumulatedContent.length} characters`);
        // Kommentar
      }
    }
  }
  
  throw new Error(`Fehler bei der Verarbeitung${maxRetries}Fehler bei der Verarbeitung`);
};

export const uploadFiles = async (formData: FormData, conversationId?: string): Promise<any> => {
  const headers = getAuthHeaders();
  let url = `${API_BASE_URL}/chat/upload_files`;
  if (conversationId) {
    url += `?conversation_id=${conversationId}`;
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      ...headers,
      // 'Content-Type' is automatically set by the browser for FormData
    },
    body: formData,
  });
  return handleJsonResponse(response);
};

export const getSettings = async (): Promise<Record<string, string>> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/settings`, {
    method: 'GET',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
  });
  return handleJsonResponse(response);
};

export const updateBotAvatar = async (avatarUrl: string): Promise<any> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/settings/bot-avatar`, {
    method: 'PUT',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ avatar_url: avatarUrl }),
  });
  return handleJsonResponse(response);
};

export const uploadAvatar = async (file: File): Promise<{ avatar_url: string }> => {
  const formData = new FormData();
  formData.append('file', file);

  const headers = getAuthHeaders();

  const response = await fetch(`${API_BASE_URL}/files/upload_avatar`, {
    method: 'POST',
    headers: {
      ...headers,
    },
    body: formData,
  });
  return handleJsonResponse(response);
};

export const updateSettings = async (settings: Record<string, string>): Promise<any> => {
  const headers = getAuthHeaders();
  const response = await fetch(`${API_BASE_URL}/settings`, {
    method: 'PUT',
    headers: {
      ...headers,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(settings),
  });
  return handleJsonResponse(response);
};
