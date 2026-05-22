import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { codeInspectorPlugin } from 'code-inspector-plugin';
import { visualizer } from 'rollup-plugin-visualizer';

// https://vitejs.dev/config/
export default defineConfig({
  // debug setting jl on 26.April.2025 // https://vitejs.dev/guide/build.html#debugging-source-maps
  build: {
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('docx')) {
            return 'docx';
          }
          if (id.includes('element-plus')) {
            return 'element-plus';
          }
        },
      },
    },
  },
  // enables de debugging of the source map
  plugins: [
      vue(),
      codeInspectorPlugin({
      bundler: 'vite',
    }),
    visualizer({
      open: false, // Automatically open the report in your browser
      gzipSize: true, // Show gzip size
      brotliSize: true, // Show brotli size
    }),
  ],
  resolve: {
    alias: {
      '@': '/src',
    },
  },
  server: {
    port: 8080, // You can change the port if needed
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000', // Assuming backend runs on port 8000
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});