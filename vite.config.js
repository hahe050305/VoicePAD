import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],

  server: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,

    // <-- put the watcher options here
    watch: {
      // Ignore everything that lives inside Android / iOS build output folders
      ignored: ['**/android/**', '**/ios/**'],
    },
  },

  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },

  worker: {
    format: 'es',
  },
})