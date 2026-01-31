-- Seed file for ForesightAI RSS sources
-- These are free, open RSS feeds that work well with the light fetch method
-- scrape_frequency: 1=hourly, 2=4hrs, 3=6hrs, 4=daily

-- Clear existing sources (optional - comment out if you want to preserve existing)
-- TRUNCATE sources CASCADE;

INSERT INTO sources (url, name, scrape_frequency, paywall, category) VALUES
-- Major News (International)
('https://feeds.bbci.co.uk/news/world/rss.xml', 'BBC World News', 1, false, 'world'),
('https://feeds.bbci.co.uk/news/technology/rss.xml', 'BBC Technology', 2, false, 'tech'),
('https://www.aljazeera.com/xml/rss/all.xml', 'Al Jazeera', 1, false, 'world'),
('https://rss.dw.com/xml/rss-en-all', 'Deutsche Welle', 2, false, 'world'),

-- US News
('https://feeds.npr.org/1001/rss.xml', 'NPR News', 1, false, 'us'),
('https://feeds.npr.org/1014/rss.xml', 'NPR Politics', 2, false, 'politics'),
('https://www.vox.com/rss/index.xml', 'Vox', 2, false, 'us'),

-- Technology
('https://www.theverge.com/rss/index.xml', 'The Verge', 1, false, 'tech'),
('https://techcrunch.com/feed/', 'TechCrunch', 1, false, 'tech'),
('https://www.wired.com/feed/rss', 'Wired', 2, false, 'tech'),
('https://arstechnica.com/feed/', 'Ars Technica', 2, false, 'tech'),
('https://feeds.feedburner.com/TheHackersNews', 'The Hacker News', 2, false, 'tech'),

-- Business & Finance
('https://feeds.bloomberg.com/markets/news.rss', 'Bloomberg Markets', 1, false, 'finance'),
('https://www.cnbc.com/id/100003114/device/rss/rss.html', 'CNBC', 1, false, 'finance'),

-- Science
('https://www.sciencedaily.com/rss/all.xml', 'Science Daily', 3, false, 'science'),
('https://phys.org/rss-feed/', 'Phys.org', 3, false, 'science'),
('https://www.nature.com/nature.rss', 'Nature', 4, false, 'science'),

-- Europe
('https://www.euronews.com/rss?level=theme&name=news', 'Euronews', 2, false, 'europe'),
('https://www.theguardian.com/world/rss', 'The Guardian World', 1, false, 'world'),
('https://www.theguardian.com/technology/rss', 'The Guardian Tech', 2, false, 'tech'),

-- Asia
('https://www.scmp.com/rss/91/feed', 'South China Morning Post', 2, false, 'asia'),
('https://japantoday.com/feed', 'Japan Today', 3, false, 'asia'),

-- Middle East
('https://www.timesofisrael.com/feed/', 'Times of Israel', 2, false, 'middle-east'),

-- Africa
('https://www.news24.com/news24/rss', 'News24 South Africa', 3, false, 'africa')

ON CONFLICT (url) DO NOTHING;
