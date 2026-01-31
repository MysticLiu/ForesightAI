<script setup lang="ts">
import SubscriptionForm from '../components/SubscriptionForm.vue';

const { $md } = useNuxtApp();
const config = useRuntimeConfig();

const text = ref(
  `
## what is this?

your personal intelligence agency. ForesightAI delivers a daily brief of everything important happening in the world, with *actual analysis* beyond headlines.

get the "what", "why", and "so what" behind global events in 15min.

### how it works
- scrapes hundreds of news sources
- clusters related articles using AI
- separates facts from disputed claims
- identifies what actually matters
- delivers one clean & engaging brief

### why this exists

presidents get daily briefings - everything they need to know, perfectly prepared by a team of analysts. now with AI, everyone can have that too.

what would've required an *entire* intelligence agency now runs on AI and open-source tools.

just what's happening and why it matters. no middleman deciding what reaches you.
`.trim()
);

useSEO({
  title: 'home | ForesightAI',
  description:
    'a daily brief of everything important happening, with actual analysis beyond headlines',
  ogImage: `${config.public.WORKER_API}/openGraph/default`,
  ogUrl: `https://foresight-ai-frontend.vercel.app/`,
});
</script>

<template>
  <div class="flex flex-col gap-6">
    <div class="prose" v-html="$md.render(text)" />

    <!-- Subscription area -->
    <div class="mt-4 pt-8 pb-6 border-t border-gray-300">
      <SubscriptionForm />
    </div>
  </div>
</template>
