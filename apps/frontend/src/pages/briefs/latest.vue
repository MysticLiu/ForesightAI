<script setup lang="ts">
useSEO({
  title: 'latest report | ForesightAI',
  description:
    'a daily brief of everything important happening, with actual analysis beyond headlines',
  ogImage: `${useRuntimeConfig().public.WORKER_API}/og/default`,
  ogUrl: `https://foresight-ai-frontend.vercel.app/latest`,
});

// redirect to the latest report
const { data: latestSlug, error } = await useFetch('/api/briefs/latest');
if (error.value !== null) {
  throw createError({ statusCode: 500 });
}
if (latestSlug.value !== null) {
  await navigateTo(`/briefs/${latestSlug.value}`);
}
</script>

<template>
  <div>
    <p>Redirecting to the latest report...</p>
  </div>
</template>
