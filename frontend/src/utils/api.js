/**
 * utils/api.js — Axios-based API client
 * All calls go to /api/* which Vite proxies to Flask on port 5000.
 */
import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  timeout: 120000, // 2 min for inference
})

// Response interceptor — unwrap data, surface errors
client.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const msg =
      err.response?.data?.error ||
      err.response?.statusText ||
      err.message ||
      'Network error'
    return Promise.reject(new Error(msg))
  }
)

export const api = {
  health:      ()     => client.get('/health'),
  stats:       ()     => client.get('/stats'),
  modelInfo:   ()     => client.get('/model-info'),
  edaStats:    ()     => client.get('/eda-stats'),
  history:     ()     => client.get('/history'),
  clearHistory:()     => client.post('/history/clear'),

  predict(formData) {
    return client.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
}

export default api
