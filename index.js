const express = require('express');
const cheerio = require('cheerio');
const axios = require('axios');
const { Pinecone } = require('@pinecone-database/pinecone');
const dotenv = require('dotenv');
const cors = require('cors');

const crypto = require('crypto');
const CACHE_EXPIRY_SECONDS = 3600;
dotenv.config();

const app = express();

const corsOptions = {
    origin: ['https://webseer.vercel.app'],
    allowedHeaders: ["Content-Type", "Authorization"],
};
app.use(cors(corsOptions));
app.use(express.json());

// Initialize clients
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

// Constants
const MAX_METADATA_SIZE = 40900;
const MAX_TOKENS = 8000; // Mistral context window
const MAX_DEPTH = 3;
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

// Mistral API configuration
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;
const MISTRAL_BASE_URL = 'https://api.mistral.ai/v1';

// Rate limiting for API calls
const API_RATE_LIMIT = {
    requestsPerMinute: 30, // Mistral has higher limits
    minDelayBetweenCalls: 2000, // 2 seconds between calls
};

let apiCallTracker = {
    calls: [],
    lastCall: 0
};

// Enhanced rate limiting function
async function rateLimitedApiCall(apiFunction, ...args) {
    const now = Date.now();
    
    // Clean old calls (older than 1 minute)
    apiCallTracker.calls = apiCallTracker.calls.filter(time => now - time < 60000);
    
    // Check if we've exceeded rate limits
    if (apiCallTracker.calls.length >= API_RATE_LIMIT.requestsPerMinute) {
        const waitTime = 60000 - (now - apiCallTracker.calls[0]);
        console.log(`⏳ Rate limit reached. Waiting ${Math.ceil(waitTime/1000)} seconds...`);
        await sleep(waitTime);
        return rateLimitedApiCall(apiFunction, ...args);
    }
    
    // Ensure minimum delay between calls
    const timeSinceLastCall = now - apiCallTracker.lastCall;
    if (timeSinceLastCall < API_RATE_LIMIT.minDelayBetweenCalls) {
        const waitTime = API_RATE_LIMIT.minDelayBetweenCalls - timeSinceLastCall;
        console.log(`⏳ Enforcing minimum delay. Waiting ${Math.ceil(waitTime/1000)} seconds...`);
        await sleep(waitTime);
    }
    
    try {
        const result = await apiFunction(...args);
        apiCallTracker.calls.push(Date.now());
        apiCallTracker.lastCall = Date.now();
        return result;
    } catch (error) {
        if (error.response?.status === 429 || error.message.includes('429')) {
            const retryDelay = 60000; // Wait 60s on rate limit
            console.log(`🚫 429 Error - Waiting ${Math.ceil(retryDelay/1000)} seconds before retry...`);
            await sleep(retryDelay);
            return rateLimitedApiCall(apiFunction, ...args);
        }
        throw error;
    }
}

// Utility functions
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Create a consistent namespace from a URL
function createNamespaceFromUrl(url) {
    const urlObj = new URL(url);
    const domain = urlObj.hostname.replace(/\./g, '_');
    const hash = crypto.createHash('md5').update(url).digest('hex').substring(0, 10);
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
        let cleanUrl = url.replace(/%..,/g, '');
        
        if (!cleanUrl.startsWith('http://') && !cleanUrl.startsWith('https://')) {
            cleanUrl = 'https://' + cleanUrl;
        }
        
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

// Mistral AI embedding function
async function generateVector({ text }) {
    const vectorFunction = async () => {
        try {
            const response = await axios.post(
                `${MISTRAL_BASE_URL}/embeddings`,
                {
                    model: 'mistral-embed',
                    input: [text],
                    encoding_format: 'float'
                },
                {
                    headers: {
                        'Authorization': `Bearer ${MISTRAL_API_KEY}`,
                        'Content-Type': 'application/json'
                    }
                }
            );
            
            return response.data.data[0].embedding;
        } catch (error) {
            console.error('Mistral embedding error:', error.response?.data || error.message);
            throw error;
        }
    };
    
    return await rateLimitedApiCall(vectorFunction);
}

// Batch embedding function for efficiency
async function generateVectorsBatch(texts) {
    const batchFunction = async () => {
        try {
            const response = await axios.post(
                `${MISTRAL_BASE_URL}/embeddings`,
                {
                    model: 'mistral-embed',
                    input: texts,
                    encoding_format: 'float'
                },
                {
                    headers: {
                        'Authorization': `Bearer ${MISTRAL_API_KEY}`,
                        'Content-Type': 'application/json'
                    }
                }
            );
            
            return response.data.data.map(item => item.embedding);
        } catch (error) {
            console.error('Mistral batch embedding error:', error.response?.data || error.message);
            throw error;
        }
    };
    
    return await rateLimitedApiCall(batchFunction);
}

// Content chunking optimized for Mistral's 8K context
function chunkText(text, maxTokens = 6000) { // Leave room for other tokens
    const words = text.split(" ");
    let chunks = [];
    let currentChunk = [];
    let currentTokens = 0;

    for (const word of words) {
        const wordTokens = Math.ceil(word.length / 4); // Rough token estimation

        if (currentTokens + wordTokens > maxTokens) {
            if (currentChunk.length > 0) {
                chunks.push(currentChunk.join(" "));
                currentChunk = [];
                currentTokens = 0;
            }
        }

        currentChunk.push(word);
        currentTokens += wordTokens;
    }

    if (currentChunk.length > 0) {
        chunks.push(currentChunk.join(" "));
    }

    return chunks;
}

// Optimized recursive ingestion with batch processing
async function ingestRecursive(url, visitedUrls = new Set(), depth = 0, namespace, maxInternalLinks = 3) {
    if (depth > MAX_DEPTH) return;

    const result = await scrapeWebsite(url, visitedUrls, depth);
    if (!result) return;

    const index = pc.Index("askweb-2");
    const contentChunks = chunkText(result.mainContent);
    
    // Limit chunks per page
    const limitedChunks = contentChunks.slice(0, 8).filter(chunk => chunk.length > 100);
    
    if (limitedChunks.length === 0) return;

    try {
        console.log(`🔄 Processing ${limitedChunks.length} chunks for ${url}`);
        
        // Use batch processing for efficiency
        const batchSize = 5; // Process in smaller batches to avoid timeouts
        for (let i = 0; i < limitedChunks.length; i += batchSize) {
            const batch = limitedChunks.slice(i, i + batchSize);
            const embeddings = await generateVectorsBatch(batch);
            
            const upsertData = batch.map((chunk, idx) => ({
                id: `${namespace}_${Date.now()}_${i + idx}`,
                values: embeddings[idx],
                metadata: {
                    content: chunk,
                    url: result.url,
                    title: result.title,
                    timestamp: new Date().toISOString()
                }
            }));

            await index.namespace(namespace).upsert(upsertData);
            console.log(`📤 Inserted batch ${Math.floor(i/batchSize) + 1} into namespace: ${namespace}`);
        }
    } catch (error) {
        console.error(`Error processing chunks for ${url}:`, error.message);
    }
    
    // Process internal links
    const linksToScrape = result.internalLinks.slice(0, maxInternalLinks);
    
    for (let link of linksToScrape) {
        await sleep(2000);
        await ingestRecursive(link, visitedUrls, depth + 1, namespace, maxInternalLinks);
    }

    return {
        pageCount: visitedUrls.size,
        namespace: namespace
    };
}

// Question handling with Mistral embeddings
async function chat(question, namespace) {
    const index = pc.Index("askweb-2");
    
    const questionEmbedding = await generateVector({ text: question });
    try {
        const result = await index.namespace(namespace).query({
            vector: questionEmbedding,
            topK: 5,
            includeMetadata: true,
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

// Answer generation using Mistral AI chat completions
async function generateAnswer(question, context) {
    const answerFunction = async () => {
        try {
            // Trim context if too long
            let finalContext = context;
            while (estimateTokens(question + finalContext) > MAX_TOKENS - 1000) { // Leave room for response
                finalContext = finalContext.slice(0, -1000);
            }

            const response = await axios.post(
                `${MISTRAL_BASE_URL}/chat/completions`,
                {
                    model: 'mistral-small-latest', // Using Mistral Small for text generation
                    messages: [
                        {
                            role: 'system',
                            content: "You are an assistant with full context of the question. Answer accurately and concisely."
                        },
                        {
                            role: 'user',
                            content: `Question: ${question}\n\nContext: ${finalContext}\n\nAnswer:`
                        }
                    ],
                    max_tokens: 1000,
                    temperature: 0.3
                },
                {
                    headers: {
                        'Authorization': `Bearer ${MISTRAL_API_KEY}`,
                        'Content-Type': 'application/json'
                    }
                }
            );

            return response.data.choices[0].message.content;
        } catch (error) {
            console.error('Mistral chat completion error:', error.response?.data || error.message);
            throw error;
        }
    };

    try {
        return await rateLimitedApiCall(answerFunction);
    } catch (error) {
        console.error('❌ API Error:', error);

        if (error.response?.status === 429 || error.message.includes('429')) {
            return "Sorry, I'm experiencing high load. Please try again later.";
        } else if (error.response?.status === 401 || error.response?.status === 403) {
            return "Authentication error. Please check the API configuration.";
        } else if (error.response?.status >= 500) {
            return "The AI service is currently unavailable. Please try again later.";
        } else if (error.code === 'ENOTFOUND' || error.code === 'ETIMEDOUT') {
            return "Network error occurred. Please check your internet connection.";
        } else {
            return "An unexpected error occurred while generating the answer. Please try again.";
        }
    }
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

        const namespace = createNamespaceFromUrl(url);
        console.log(`📁 Using namespace: ${namespace}`);

        const index = pc.Index("askweb-2");
        const stats = await index.describeIndexStats();
        const namespaces = stats.namespaces || {};
        const urlNeedsIngestion = !namespaces[namespace] || namespaces[namespace].vectorCount === 0;

        let ingestData;
        if (urlNeedsIngestion) {
            console.log(`🔄 No data found for this URL, ingesting content...`);
            
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

        console.log(`🤔 Generating answer for namespace: ${namespace}...`);
        const context = await chat(question, namespace);
        
        if (!context || context.trim().length === 0) {
            const response = {
                success: true,
                answer: "Sorry, I couldn't find any relevant information to answer your question. You can try asking a different question!",
                metadata: {
                    namespace: namespace,
                    url: url,
                    processedAt: new Date().toISOString()
                }
            };
            
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

const ping = () => {
    const url = 'https://askweb-backend.onrender.com'
    setInterval(async() => {
        try {
            const response = await fetch(url)
            console.log("keeping backend alive ping", response.status)
        } catch (error) {
            console.log("ping failed", error)
        }
    }, 36000000)
}
ping()

const PORT = process.env.PORT || 4000;

app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
});