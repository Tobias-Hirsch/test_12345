// Kommentar
export class ResilientStreamingApi {
  private baseUrl: string
  private defaultHeaders: HeadersInit

  constructor(baseUrl: string = '/api', defaultHeaders: HeadersInit = {}) {
    this.baseUrl = baseUrl
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      ...defaultHeaders
    }
  }

  private async makeRequest(
    url: string, 
    options: RequestInit = {}
  ): Promise<Response> {
    const response = await fetch(`${this.baseUrl}${url}`, {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers
      }
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    return response
  }

  // Kommentar
  async uploadWithRetry(
    url: string,
    formData: FormData,
    onProgress?: (progress: number) => void
  ): Promise<any> {
    const maxRetries = 2
    let attempt = 0
    
    while (attempt <= maxRetries) {
      try {
        const response = await fetch(`${this.baseUrl}${url}`, {
          method: 'POST',
          body: formData,
          headers: {
            // Kommentar
            ...Object.fromEntries(
              Object.entries(this.defaultHeaders).filter(([key]) => 
                key.toLowerCase() !== 'content-type'
              )
            )
          }
        })

        if (!response.ok) {
          throw new Error(`Upload failed: HTTP ${response.status}`)
        }

        return await response.json()
      } catch (error) {
        attempt++
        if (attempt > maxRetries) {
          throw error
        }
        
        // Kommentar
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt))
      }
    }
  }

  // Kommentar
  async requestWithRetry<T>(
    url: string,
    options: RequestInit = {},
    retries: number = 3
  ): Promise<T> {
    let attempt = 0
    
    while (attempt <= retries) {
      try {
        const response = await this.makeRequest(url, options)
        return await response.json()
      } catch (error) {
        attempt++
        if (attempt > retries) {
          throw error
        }
        
        // Kommentar
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt))
      }
    }
    
    throw new Error('Max retries exceeded')
  }
}