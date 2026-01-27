# Meridian - Free Tier Setup Guide

This guide walks you through setting up Meridian with **$0/month** costs using free tiers of all services.

## Prerequisites

- **Node.js 22+** - [Download](https://nodejs.org/)
- **pnpm 9.15.4** - Install with: `npm install -g pnpm@9.15.4`
- **Python 3.10+** - [Download](https://www.python.org/) (for brief generation)

## Step 1: Get Free Services

### 1.1 PostgreSQL Database (Neon - Free)

1. Go to [https://neon.tech](https://neon.tech)
2. Sign up for a free account
3. Create a new project (name: "meridian" or any name)
4. Copy your connection string from the dashboard:
   ```
   postgresql://username:password@ep-xxx-xxx-123456.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

**Free tier includes:** 0.5 GB storage, 191 compute hours/month

### 1.2 Google Gemini API (Free)

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key (looks like `AIzaSy...`)

**Free tier includes:**
- `gemini-2.0-flash`: 15 requests/minute, 1M tokens/day
- `gemini-2.5-pro`: 5 requests/minute, 25K tokens/day

### 1.3 Cloudflare Account (Free)

1. Go to [https://dash.cloudflare.com](https://dash.cloudflare.com)
2. Sign up for a free account
3. You'll use this later to deploy workers

**Free tier includes:** 100K requests/day, 1000 workflow executions/day

### 1.4 Generate a Secret Key

Run this command to generate a random API secret:
```bash
openssl rand -hex 32
```

Save this value - you'll use it for API authentication.

---

## Step 2: Clone and Install

```bash
cd /home/user/ForesightAI
pnpm install
```

---

## Step 3: Configure Environment Files

### 3.1 Database Package

```bash
cp packages/database/.env.example packages/database/.env
```

Edit `packages/database/.env`:
```env
DATABASE_URL="your-neon-connection-string"
```

### 3.2 Frontend App

```bash
cp apps/frontend/.env.example apps/frontend/.env
```

Edit `apps/frontend/.env`:
```env
NUXT_DATABASE_URL="your-neon-connection-string"
NUXT_PUBLIC_WORKER_API="http://localhost:8787"
```

### 3.3 Scrapers Worker (Local Development)

Create `apps/scrapers/.dev.vars`:
```env
DATABASE_URL="your-neon-connection-string"
GOOGLE_API_KEY="your-gemini-api-key"
GOOGLE_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
MERIDIAN_SECRET_KEY="your-generated-secret"
```

### 3.4 Python Briefs

```bash
cp apps/briefs/.env.example apps/briefs/.env
```

Edit `apps/briefs/.env`:
```env
GOOGLE_API_KEY="your-gemini-api-key"
MERIDIAN_SECRET_KEY="your-generated-secret"
MERIDIAN_API_URL="http://localhost:8787"
```

---

## Step 4: Set Up Database

### 4.1 Run Migrations

```bash
cd packages/database
pnpm migrate
```

### 4.2 Seed RSS Sources

Connect to your Neon database (using their SQL editor or psql) and run:
```bash
psql "your-neon-connection-string" -f packages/database/seed.sql
```

Or use Neon's web console to paste and run the contents of `packages/database/seed.sql`.

---

## Step 5: Run Locally

You need 2 terminals:

### Terminal 1: Start Scrapers Worker

```bash
cd apps/scrapers
npx wrangler login  # First time only - opens browser to authenticate
npx wrangler dev
```

This starts the worker at `http://localhost:8787`

### Terminal 2: Start Frontend

```bash
pnpm dev --filter=@meridian/frontend
```

This starts the frontend at `http://localhost:3000`

---

## Step 6: Trigger First Scrape

### 6.1 Manually Trigger RSS Scraping

Open your browser or use curl:
```bash
curl "http://localhost:8787/trigger-rss?token=YOUR_SECRET_KEY"
```

Replace `YOUR_SECRET_KEY` with your generated secret.

### 6.2 Wait for Processing

The workflow will:
1. Fetch all RSS feeds (takes 1-5 minutes)
2. Store articles in the database
3. Automatically trigger article processing
4. Process articles with Gemini AI (takes 5-15 minutes depending on volume)

You can check progress in the wrangler terminal output.

---

## Step 7: Generate Your First Brief

### 7.1 Set Up Python Environment

```bash
cd apps/briefs
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 Run the Notebook

```bash
jupyter notebook reportV5.ipynb
```

Or use VS Code with the Jupyter extension.

**Run all cells** - this will:
1. Fetch processed articles from the API
2. Generate embeddings using a local model
3. Cluster articles by topic
4. Generate analysis with Gemini
5. Create the final markdown brief
6. POST it to your worker API

---

## Step 8: View Your Brief

1. Go to `http://localhost:3000`
2. Click "View Briefs" or navigate to `/briefs`
3. Your first brief should appear!

---

## Deployment (Optional)

### Deploy Scrapers to Cloudflare

```bash
cd apps/scrapers

# Set production secrets
wrangler secret put DATABASE_URL
wrangler secret put GOOGLE_API_KEY
wrangler secret put GOOGLE_BASE_URL
wrangler secret put MERIDIAN_SECRET_KEY

# Deploy
wrangler deploy --env production
```

### Deploy Frontend

The frontend can be deployed to:
- **Cloudflare Pages** (free)
- **Vercel** (free tier)
- **Netlify** (free tier)

---

## Cost Summary

| Service | Free Tier | Monthly Cost |
|---------|-----------|--------------|
| Neon PostgreSQL | 0.5 GB, 191 compute hrs | $0 |
| Google Gemini API | 1M tokens/day | $0 |
| Cloudflare Workers | 100K requests/day | $0 |
| Cloudflare Workflows | 1000 executions/day | $0 |
| **Total** | | **$0** |

---

## Troubleshooting

### "wrangler: command not found"
```bash
npm install -g wrangler
# or use npx wrangler
```

### Database connection errors
- Ensure your Neon database is in the same region as your Cloudflare worker
- Check that `?sslmode=require` is in your connection string

### Gemini rate limits
- The free tier has limits. If you hit them, wait a minute and try again
- Consider processing fewer articles per run (edit `processArticles.workflow.ts`)

### Articles failing to fetch
- Some sites block scrapers. The free tier doesn't include browser rendering
- Most major news sites work fine with the light fetch method
- Check `fail_reason` column in the articles table for details

---

## What's Not Included in Free Tier

1. **Browser Rendering** - Paywalled sites (NYT, Reuters, etc.) may fail
   - To enable: Set `ENABLE_BROWSER_RENDERING=true` and add Cloudflare tokens
   - Cost: ~$0.02 per 1000 requests

2. **MailerLite Newsletter** - Optional email subscriptions
   - Free tier: 1,000 subscribers, 12,000 emails/month
   - App works fully without it
