// Kommentar
import { ref, computed } from 'vue'

export interface RetryConfig {
  maxRetries: number
  retryDelay: number
  timeoutMs: number
  backoffMultiplier: number
}

export interface StreamingState {
  isStreaming: boolean
  isRetrying: boolean
  retryCount: number
  error: string | null
  lastResponse: string
}

export function useStreamingWithRetry(defaultConfig: Partial<RetryConfig> = {}) {
  const config: RetryConfig = {
    maxRetries: 3,
    retryDelay: 2000,
    timeoutMs: 600000, // 10Hinweis
    backoffMultiplier: 1.5,
    ...defaultConfig
  }

  const state = ref<StreamingState>({
    isStreaming: false,
    isRetrying: false,
    retryCount: 0,
    error: null,
    lastResponse: ''
  })

  const canRetry = computed(() => 
    state.value.retryCount < config.maxRetries && 
    !state.value.isStreaming
  )

  const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

  const isNetworkError = (error: any): boolean => {
    if (!error) return false
    
    const errorString = error.toString().toLowerCase()
    const networkErrors = [
      'network error',
      'err_incomplete_chunked_encoding',
      'err_connection_reset',
      'err_connection_aborted',
      'fetch_error',
      'abort',
      'timeout'
    ]
    
    return networkErrors.some(errorType => errorString.includes(errorType))
  }

  const executeWithRetry = async <T>(
    operation: () => Promise<T>,
    onProgress?: (chunk: string) => void,
    onRetry?: (retryCount: number, error: any) => void
  ): Promise<T> => {
    state.value.error = null
    state.value.retryCount = 0

    while (state.value.retryCount <= config.maxRetries) {
      try {
        state.value.isStreaming = true
        state.value.error = null

        // Kommentar
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error('Request timeout')), config.timeoutMs)
        })

        const result = await Promise.race([
          operation(),
          timeoutPromise
        ])

        // Kommentar
        state.value.isStreaming = false
        state.value.isRetrying = false
        return result

      } catch (error) {
        console.error(`Streaming attempt ${state.value.retryCount + 1} failed:`, error)
        
        state.value.error = error instanceof Error ? error.message : String(error)
        
        // Kommentar
        if (!isNetworkError(error) || state.value.retryCount >= config.maxRetries) {
          state.value.isStreaming = false
          state.value.isRetrying = false
          throw error
        }

        state.value.retryCount++
        state.value.isRetrying = true
        
        // Kommentar
        onRetry?.(state.value.retryCount, error)

        // Kommentar
        const delay = config.retryDelay * Math.pow(config.backoffMultiplier, state.value.retryCount - 1)
        await sleep(delay)
      }
    }

    state.value.isStreaming = false
    state.value.isRetrying = false
    throw new Error(`Max retries (${config.maxRetries}) exceeded`)
  }

  const reset = () => {
    state.value.isStreaming = false
    state.value.isRetrying = false
    state.value.retryCount = 0
    state.value.error = null
    state.value.lastResponse = ''
  }

  return {
    state: computed(() => state.value),
    canRetry,
    executeWithRetry,
    reset,
    config
  }
}