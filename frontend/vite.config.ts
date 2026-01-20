import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	css: {
		preprocessorOptions: {
			scss: {
				loadPaths: ['node_modules'],
				quietDeps: true,
				silenceDeprecations: ['import']
			}
		}
	}
});
