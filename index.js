const express = require('express');
const cheerio = require('cheerio');
const axios = require('axios');
const { Pinecone } = require('@pinecone-database/pinecone');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const dotenv = require('dotenv');
const cors = require('cors');
const Redis = require('ioredis');
const crypto = require('crypto');
const CACHE_EXPIRY_SECONDS = 3600;
dotenv.config();

const app = express();

const corsOptions = {
    origin: ["https://webseer.vercel.app"],
    methods: ["GET", "POST", "PUT", "DELETE"],
    allowedHeaders: ["Content-Type", "Authorization"],
    exposedHeaders: ["Custom-Header"],
    credentials: true,
    optionsSuccessStatus: 200,
};
app.use(cors(corsOptions));

app.use(express.json());
const redis = new Redis({
    host: 'tender-mosquito-42163.upstash.io',
    port: 6379,
    password: 'AaSzAAIjcDE4YjQ3YzI2ZWMzMTc0NzY5YmY0ODRkY2U4OGUxMWNiZHAxMA',
    tls: {}
});

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

// Create a consistent namespace from a URL
function createNamespaceFromUrl(url) {
    // Extract domain and path
    const urlObj = new URL(url);
    const domain = urlObj.hostname.replace(/\./g, '_');
    
    // Hash the full URL to ensure uniqueness and valid namespace format
    const hash = crypto.createHash('md5').update(url).digest('hex').substring(0, 10);
    
    // Create namespace with domain and hash
    return `${domain}_${hash}`;
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

function validateAndFormatUrl(url) {
    try {
        // Remove any remaining URL encoding
        let cleanUrl = url.replace(/%..,/g, '');
        
        // If the URL doesn't start with http:// or https://, add https://
        if (!cleanUrl.startsWith('http://') && !cleanUrl.startsWith('https://')) {
            cleanUrl = 'https://' + cleanUrl;
        }
        
        // Create and validate URL object
        const urlObject = new URL(cleanUrl);
        return urlObject.toString();
    } catch (error) {
        throw new Error(`Invalid URL format: ${url}`);
    }
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
            timeout: 30000,
            maxContentLength: 10000000,
            maxBodyLength: 10000000
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
            await sleep(RETRY_DELAY * (retryCount + 1));
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
function chunkText(text, maxBytes = 10000) {
    const encoder = new TextEncoder();
    const words = text.split(" ");
    let chunks = [];
    let currentChunk = [];

    let currentSize = 0;

    for (const word of words) {
        const wordSize = encoder.encode(word + " ").length; // Get byte size

        if (currentSize + wordSize > maxBytes) {
            chunks.push(currentChunk.join(" "));
            currentChunk = [];
            currentSize = 0;
        }

        currentChunk.push(word);
        currentSize += wordSize;
    }

    if (currentChunk.length > 0) {
        chunks.push(currentChunk.join(" "));
    }

    return chunks;
}


// Recursive ingestion
async function ingestRecursive(url, visitedUrls = new Set(), depth = 0, namespace, maxInternalLinks = 5) {
    if (depth > MAX_DEPTH) return;

    const result = await scrapeWebsite(url, visitedUrls, depth);
    if (!result) return;

    const index = pc.Index("final-trial");

    const contentChunks = chunkText(result.mainContent);
    for (let i = 0; i < contentChunks.length; i++) {
        const chunk = contentChunks[i];
        if (chunk.length < 100) continue;

        try {
            const embeddingContent = await generateVector({ text: chunk });

            const metadata = {
                content: chunk,
                url: result.url,
                title: result.title,
                timestamp: new Date().toISOString()
            };

            await index.namespace(namespace).upsert([{
                id: `${namespace}_${Date.now()}_${i}`,
                values: embeddingContent,
                metadata
            }]);
            
            console.log(`📤 Inserted chunk ${i + 1}/${contentChunks.length} into namespace: ${namespace}`);
            await sleep(100);
        } catch (error) {
            console.error(`Error inserting chunk ${i}:`, error.message);
            continue;
        }
    }
    
    // Limit the number of internal links scraped
    const linksToScrape = result.internalLinks.slice(0, maxInternalLinks);
    
    for (let link of linksToScrape) {
        await sleep(1000);
        await ingestRecursive(link, visitedUrls, depth + 1, namespace, maxInternalLinks);
    }

    return {
        pageCount: visitedUrls.size,
        namespace: namespace
    };
}

// Question handling
async function chat(question, namespace) {
    const index = pc.Index("final-trial");
    
    const questionEmbedding = await generateVector({ text: question });
    try {
        const result = await index.namespace(namespace).query({
            vector: questionEmbedding,
            topK: 5,
            includeMetadata: true,
            // Query only within this namespace
        });

        const context = result.matches
            .map(match => match.metadata.content)
            .filter(content => content && content.length > 100)
            .join('\n\n');

        return context;
    } catch (error) {
        console.error(`Error querying Pinecone namespace ${namespace}:`, error);
        return '';
    }
}

// Answer generation
async function generateAnswer(question, context) {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" }, {
        apiVersion: 'v1beta',
    });

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

// API endpoint
app.post('/api/ask', async (req, res) => {
    try {
        let { url, question } = req.body;
        
        if (!url || !question) {
            return res.status(400).json({ 
                success: false,
                error: 'Missing parameters',
                message: 'Both URL and question are required'
            });
        }

        // Clean and validate URL
        try {
            url = validateAndFormatUrl(url);
        } catch (error) {
            return res.status(400).json({
                success: false,
                error: 'Invalid URL',
                message: error.message
            });
        }

        console.log(`🔍 Processing URL: ${url}`);
        console.log(`❓ Question: ${question}`);

        // Generate a consistent namespace for this URL
        const namespace = createNamespaceFromUrl(url);
        console.log(`📁 Using namespace: ${namespace}`);

        // Check Redis cache first
        const redisKey = `${namespace}:${question}`;
        const cachedAnswer = await redis.get(redisKey);
        
        if (cachedAnswer) {
            console.log("✅ Retrieved from Redis cache");
            return res.json(JSON.parse(cachedAnswer));
        }

        // Check if namespace exists in Pinecone
        const index = pc.Index("final-trial");
        const stats = await index.describeIndexStats();
        const namespaces = stats.namespaces || {};
        const urlNeedsIngestion = !namespaces[namespace] || namespaces[namespace].vectorCount === 0;

        let ingestData;
        if (urlNeedsIngestion) {
            console.log(`🔄 No data found for this URL, ingesting content...`);
            
            // Scrape and ingest content
            const visitedUrls = new Set();
            ingestData = await ingestRecursive(url, visitedUrls, 0, namespace);
            
            console.log(`📚 Processed ${visitedUrls.size} pages into namespace: ${namespace}`);
            
            if (visitedUrls.size === 0) {
                return res.status(400).json({
                    success: false,
                    error: 'Scraping failed',
                    message: 'Unable to scrape content from the provided URL'
                });
            }
        } else {
            console.log(`📚 Found existing namespace with ${namespaces[namespace].vectorCount} vectors`);
        }

        // Get context and generate answer
        console.log(`🤔 Generating answer for namespace: ${namespace}...`);
        const context = await chat(question, namespace);
        
        if (!context || context.trim().length === 0) {
            const response = {
                success: true,
                answer: "Sorry, I couldn't find any relevant information to answer your question. You can try asking diffrent Question!",
                metadata: {
                    namespace: namespace,
                    url: url,
                    processedAt: new Date().toISOString()
                }
            };
            
            // Cache the response
            await redis.set(redisKey, JSON.stringify(response), 'EX', CACHE_EXPIRY_SECONDS);
            
            return res.json(response);
        }

        const answer = await generateAnswer(question, context);
        
        const response = {
            success: true,
            answer: answer,
            metadata: {
                namespace: namespace,
                url: url,
                processedAt: new Date().toISOString()
            }
        };
        
        // Cache the response
        await redis.set(redisKey, JSON.stringify(response), 'EX', CACHE_EXPIRY_SECONDS);
        
        res.json(response);

    } catch (error) {
        console.error('❌ Error:', error);
        res.status(500).json({ 
            success: false,
            error: 'Failed to process request',
            message: error.message 
        });
    }
});


const ping = ()=>{
const url = 'https://askweb-backend.onrender.com'
setInterval(async()=>{
    try {
        const response = await fetch(url)
        console("keeping backend alive ping",response.status)
    } catch (error) {
        console.log("ping failed",error)
    }
},36000000)
}
ping()

const PORT = process.env.PORT || 4000;

app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
});