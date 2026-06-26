import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  build: {
    // three.js is inherently large but isolated in its own lazy chunk.
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        // Split heavy vendors into their own chunks for better caching and to
        // keep the main bundle small (three.js is also lazy-loaded via the
        // Viewer3D dynamic import).
        manualChunks: {
          three: ['three', '@react-three/fiber', '@react-three/drei'],
          charts: ['recharts'],
          react: ['react', 'react-dom'],
        },
      },
    },
  },
})
