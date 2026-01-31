import app from './app';

export type Env = {
  // Bindings
  SCRAPE_RSS_FEED: Workflow;
  PROCESS_ARTICLES: Workflow;

  // Secrets (optional - only needed for browser rendering, which is a paid feature)
  CLOUDFLARE_BROWSER_RENDERING_API_TOKEN?: string;
  CLOUDFLARE_ACCOUNT_ID?: string;

  DATABASE_URL: string;

  GOOGLE_API_KEY: string;
  GOOGLE_BASE_URL: string;

  FORESIGHTAI_SECRET_KEY: string;

  // Feature flags
  ENABLE_BROWSER_RENDERING?: string; // Set to "true" to enable paid browser rendering
};

export default {
  fetch: app.fetch,
  async scheduled({ cron }: ScheduledController, env: Env, ctx: ExecutionContext) {
    // - Every hour (at minute 4): trigger scrapping of RSS feeds
    if (cron === '4 * * * *') {
      await env.SCRAPE_RSS_FEED.create({ id: crypto.randomUUID() });
      console.log('Starting RSS feed scraping...');
      return;
    }
  },
} satisfies ExportedHandler<Env>;

export { ScrapeRssFeed } from './workflows/rssFeed.workflow';
export { ProcessArticles } from './workflows/processArticles.workflow';
