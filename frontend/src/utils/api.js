/**
 * utils/api.js — Axios-based API client
 * All calls go to /api/* which Vite dev-server proxies to Flask on port 5000.
 *
 * IMPORTANT: inference can take 10-60 seconds on CPU.
 * timeout is set to 5 minutes to accommodate this.
 */
import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  timeout: 300000,   // 5 minutes — inference on CPU can take 30-60 s
})

// Response interceptor — extract data on success, extract error message on failure
client.interceptors.response.use(
  (res) => res.data,
  (err) => {
    let msg = 'Network error — could not reach the API server.'

    if (err.code === 'ECONNABORTED') {
      msg = 'Request timed out. Inference may still be running — try again in a moment.'
    } else if (!err.response) {
      msg = 'Cannot connect to the API server. Make sure "python api.py" is running on port 5000.'
    } else {
      const data = err.response.data
      if (typeof data === 'object' && data?.error) {
        msg = data.error
      } else if (typeof data === 'string' && data.length < 500) {
        msg = data
      } else {
        msg = `Server error ${err.response.status}: ${err.response.statusText}`
      }
    }

    return Promise.reject(new Error(msg))
  }
)

export const api = {
  health:       ()         => client.get('/health'),
  stats:        ()         => client.get('/stats'),
  modelInfo:    ()         => client.get('/model-info'),
  edaStats:     ()         => client.get('/eda-stats'),
  history:      ()         => client.get('/history'),
  clearHistory: ()         => client.post('/history/clear'),

  predict(formData) {
    return client.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      // 5-minute timeout specifically for predict
      timeout: 300000,
    })
  },
}

export default api
