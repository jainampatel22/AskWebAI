const express = require('express');
const cheerio = require('cheerio');
const axios = require('axios');
const { Pinecone } = require('@pinecone-database/pinecone');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const dotenv = require('dotenv');
const cors = require('cors');

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

// Initialize clients
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Constants
const MAX_METADATA_SIZE = 40900;
const MAX_TOKENS = 3000;
const MAX_DEPTH = 3;
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

// Utility functions
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function estimateTokens(text) {
    return Math.ceil(text.length / 4);
}

function normalizeUrl(baseUrl, link) {
    try {
        if (!link) return null;
        link = link.split('#')[0].split('?')[0];
        
        const skipExtensions = ['.pdf', '.jpg', '.png', '.gif', '.css', '.js'];
        const skipProtocols = ['mailto:', 'tel:', 'ftp:', 'file:'];
        
        if (skipExtensions.some(ext => link.toLowerCase().endsWith(ext)) ||
            skipProtocols.some(protocol => link.toLowerCase().startsWith(protocol))) {
            return null;
        }

        const parsed = new URL(link, baseUrl);
        const baseUrlParsed = new URL(baseUrl);
        
        if (parsed.hostname !== baseUrlParsed.hostname) {
            return null;
        }

        return parsed.toString();
    } catch (error) {
        return null;
    }
}

function cleanText(text) {
    return text
        .replace(/\s+/g, ' ')
        .replace(/[\n\r\t]/g, ' ')
        .trim();
}

// Content extraction
const extractContent = ($) => {
    let contentObj = {
        title: '',
        metaDescription: '',
        mainContent: '',
        textContent: new Set(),
        structuredData: {}
    };

    contentObj.title = $('title').text().trim();
    contentObj.metaDescription = $('meta[name="description"]').attr('content') || '';
    
    $('script[type="application/ld+json"]').each((_, element) => {
        try {
            const jsonData = JSON.parse($(element).html());
            contentObj.structuredData = { ...contentObj.structuredData, ...jsonData };
        } catch (e) {
            console.log('Error parsing JSON-LD:', e.message);
        }
    });

    const processTextContent = (element) => {
        const text = $(element).text().trim();
        if (text.length > 20) {
            contentObj.textContent.add(cleanText(text));
        }
    };

    $('script, style, noscript, iframe, img, svg, header, footer, nav').remove();

    const prioritySelectors = [
        'article', 'main', 'section', '.content', '#content',
        '.post', '.article', '[role="main"]', '[role="article"]'
    ];

    prioritySelectors.forEach(selector => {
        $(selector).each((_, element) => processTextContent(element));
    });

    $('h1, h2, h3, h4, h5, h6, p').each((_, element) => processTextContent(element));

    $('li').each((_, element) => {
        const text = $(element).text().trim();
        if (text.length > 50) {
            processTextContent(element);
        }
    });

    $('body').contents().each((_, element) => {
        if (element.type === 'text') {
            const text = $(element).text().trim();
            if (text.length > 20) {
                contentObj.textContent.add(cleanText(text));
            }
        }
    });

    contentObj.mainContent = Array.from(contentObj.textContent).join('\n\n');
    return contentObj;
};

// Web scraping function
async function scrapeWebsite(url, visitedUrls = new Set(), depth = 0, retryCount = 0) {
    if (visitedUrls.has(url) || depth > MAX_DEPTH) return null;
    visitedUrls.add(url);

    try {
        const response = await axios.get(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            },
            timeout: 10000
        });

        const $ = cheerio.load(response.data);
        const contentObj = extractContent($);

        const internalLinks = new Set();
        $('a').each((_, element) => {
            const href = $(element).attr('href');
            const normalizedUrl = normalizeUrl(url, href);
            if (normalizedUrl) {
                internalLinks.add(normalizedUrl);
            }
        });

        console.log(`✅ Scraped: ${url}`);
        return {
            url,
            ...contentObj,
            internalLinks: Array.from(internalLinks)
        };
    } catch (error) {
        console.error(`❌ Error scraping ${url}:`, error.message);
        if (retryCount < MAX_RETRIES) {
            console.log(`Retrying (${retryCount + 1}/${MAX_RETRIES})...`);
            await sleep(RETRY_DELAY);
            return scrapeWebsite(url, visitedUrls, depth, retryCount + 1);
        }
        return null;
    }
}

// Vector generation
async function generateVector({ text }) {
    const model = genAI.getGenerativeModel({ model: "text-embedding-004" });
    const result = await model.embedContent(text);
    return result.embedding.values;
}

// Content chunking
function chunkTextBySize(text, maxSize = MAX_METADATA_SIZE) {
    let chunks = [];
    let currentChunk = '';
    const sentences = text.split(/[.!?]+/);

    for (let sentence of sentences) {
        sentence = sentence.trim();
        if (!sentence) continue;

        const potentialChunk = currentChunk ? currentChunk + '. ' + sentence : sentence;
        const chunkSize = Buffer.byteLength(potentialChunk, 'utf8');

        if (chunkSize > maxSize) {
            if (currentChunk) chunks.push(currentChunk);
            currentChunk = sentence;
        } else {
            currentChunk = potentialChunk;
        }
    }

    if (currentChunk) chunks.push(currentChunk);
    return chunks;
}

// Recursive ingestion
async function ingestRecursive(url, visitedUrls = new Set(), depth = 0) {
    if (depth > MAX_DEPTH) return;

    const result = await scrapeWebsite(url, visitedUrls, depth);
    if (!result) return;

    const index = pc.Index("dbtest");

    const contentChunks = chunkTextBySize(result.mainContent);
    for (let i = 0; i < contentChunks.length; i++) {
        const chunk = contentChunks[i];
        if (chunk.length < 100) continue;

        const embeddingContent = await generateVector({ text: chunk });

        const metadata = {
            source: 'content',
            content: chunk,
            url: result.url,
            title: result.title,
            metaDescription: result.metaDescription,
            timestamp: new Date().toISOString()
        };

        await index.upsert([{
            id: `content_${Date.now()}_${i}`,
            values: embeddingContent,
            metadata
        }]);

        console.log(`📤 Inserted chunk ${i + 1}/${contentChunks.length}`);
        await sleep(100);
    }

    for (let link of result.internalLinks) {
        await sleep(500);
        await ingestRecursive(link, visitedUrls, depth + 1);
    }
}

// Question handling
async function chat(question) {
    const index = pc.Index("dbtest");
    const questionEmbedding = await generateVector({ text: question });
    
    const result = await index.query({
        vector: questionEmbedding,
        topK: 5,
        includeMetadata: true
    });

    const context = result.matches
        .map(match => match.metadata.content)
        .filter(content => content && content.length > 100)
        .join('\n\n');

    return context;
}

async function generateAnswer(question, context) {
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });

    let finalContext = context;
    while (estimateTokens(question + finalContext) > MAX_TOKENS) {
        finalContext = finalContext.slice(0, -1000);
    }

    const prompt = `You are an assistant with full context of the question. Answer accurately based on the provided context. If the context doesn't contain enough information to answer the question, say so directly.

    Question: ${question}

    Context: ${finalContext}

    Answer:`;

    const result = await model.generateContent(prompt);
    return result.response.text();
}

// Single API endpoint
app.post('/api/query', async (req, res) => {
    try {
        const { url, question } = req.body;
        
        if (!url || !question) {
            return res.status(400).json({ 
                success: false,
                error: 'Missing parameters',
                message: 'Both URL and question are required'
            });
        }

        console.log(`🔍 Processing URL: ${url}`);
        console.log(`❓ Question: ${question}`);

        // Step 1: Scrape and ingest content
        const visitedUrls = new Set();
        await ingestRecursive(url, visitedUrls);
        console.log(`📚 Processed ${visitedUrls.size} pages`);

        // Step 2: Get context and generate answer
        console.log('🤔 Generating answer...');
        const context = await chat(question);
        
        if (!context || context.trim().length === 0) {
            return res.json({
                success: true,
                answer: "I couldn't find any relevant information in the scraped data to answer your question.",
                metadata: {
                    pagesProcessed: visitedUrls.size,
                    url: url
                }
            });
        }

        const answer = await generateAnswer(question, context);
        
        res.json({
            success: true,
            answer: answer,
            metadata: {
                pagesProcessed: visitedUrls.size,
                url: url,
                processedAt: new Date().toISOString()
            }
        });

    } catch (error) {
        console.error('❌ Error:', error);
        res.status(500).json({ 
            success: false,
            error: 'Failed to process request',
            message: error.message 
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
});