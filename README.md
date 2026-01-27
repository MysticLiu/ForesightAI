# ForesightAI: Your Personal Intelligence Agency

[![Build Status](https://img.shields.io/github/actions/workflow/status/MysticLiu/ForesightAI/deploy-services.yaml?branch=main)](https://github.com/MysticLiu/ForesightAI/actions/workflows/deploy-services.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Presidential-level intelligence briefings, built with AI, tailored for you.**

ForesightAI cuts through news noise by scraping hundreds of sources, analyzing stories with AI, and delivering concise, personalized daily briefs.

## Why It Exists

Presidents get tailored daily intelligence briefs. Now with AI, you can too. ForesightAI delivers:

- Key global events filtered by relevance
- Context and underlying drivers
- Analysis of implications
- Open-source transparency

Built for the curious who want depth beyond headlines without the time sink.

## Key Features

- **Source Coverage**: Hundreds of diverse news sources
- **AI Analysis**: Multi-stage LLM processing (Gemini) for article and cluster analysis
- **Smart Clustering**: Embeddings + UMAP + HDBSCAN to group related articles
- **Personalized Briefing**: Daily brief with analytical voice and continuity tracking
- **Web Interface**: Clean Nuxt 3 frontend

## How It Works

1. **Scraping**: Cloudflare Workers fetch RSS feeds, store metadata
2. **Processing**: Extract text, analyze with Gemini for relevance and structure
3. **Brief Generation**: Cluster articles, generate analysis, synthesize final brief
4. **Frontend**: Display briefs via Nuxt/Vercel

## Tech Stack

- **Infrastructure**: Turborepo, Cloudflare (Workers, Workflows), Vercel
- **Backend**: Hono, TypeScript, PostgreSQL (Neon), Drizzle
- **AI/ML**: Gemini models, multilingual-e5-small embeddings, UMAP, HDBSCAN
- **Frontend**: Nuxt 3, Vue 3, Tailwind

## Setup

See [SETUP.md](./SETUP.md) for detailed setup instructions.

**Prerequisites**: Node.js v22+, pnpm v9.15+, Python 3.10+, PostgreSQL, Cloudflare account, Google AI API key

```bash
git clone https://github.com/MysticLiu/ForesightAI.git
cd ForesightAI
pnpm install
# Configure .env files (see SETUP.md)
# Add GitHub secrets and push to main to deploy
```

## License

MIT License - See [LICENSE](./LICENSE) file for details.

---

*Forked from [iliane5/meridian](https://github.com/iliane5/meridian) - Original concept by Iliane Amadou*
