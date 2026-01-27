<script setup lang="ts">
import SubscriptionForm from '../components/SubscriptionForm.vue';

const { $md } = useNuxtApp();
const config = useRuntimeConfig();

const text = ref(
  `
## what is this?

my personal intelligence agency. i built a daily brief of everything important happening that i care about, with *actual analysis* beyond headlines.

gives me the "what", "why", and "so what" behind global events in 15min.

### how it works
- scrapes hundreds of news sources
- clusters related articles
- separates facts from disputed claims
- spots bias patterns
- runs targeted web searches to find context and fill knowledge gaps
- identifies what actually matters
- delivers one clean & engaging brief

### why i built this

always thought it was cool how presidents get those daily briefings - everything they need to know, perfectly prepared by a team of analysts. figured *why can't i have that too*?

now with ai, i can. what would've required an *entire* intelligence agency now costs about *a dollar a day* in compute.

this brief is tuned to what i care about (geopolitics, french news, tech, actual good news).

just what's happening and why it matters. no middleman deciding what reaches me.
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
