# ForesightAI Frontend

This is the Nuxt 3 frontend application for the [ForesightAI project](https://github.com/MysticLiu/ForesightAI) (your personal AI intelligence agency). It provides the web interface for viewing generated intelligence briefs.

Built with:

- [Nuxt 3](https://nuxt.com/) (Vue 3)
- [Tailwind CSS](https://tailwindcss.com/) (with Radix UI colors)
- [TypeScript](https://www.typescriptlang.org/)

## Key Features

- Displays daily intelligence briefs with rich formatting (`/briefs/[slug]`).
- Interactive Table of Contents for easy navigation within briefs.
- Subscription form for updates (`/`).
- Consumes the ForesightAI API (via Nitro server routes in `/server/api`).

## Setup

See the [main project SETUP.md](../../SETUP.md) for full setup details.

## Development Server

Start the Nuxt development server (usually on `http://localhost:3000`):

```bash
pnpm dev
```

## Production Build

Build the application for production:

```bash
pnpm build
```

## Deployment

This application is deployed on [Vercel](https://vercel.com/).
