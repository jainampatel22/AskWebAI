const cheerio = require('cheerio');
const axios = require('axios');
const { Pinecone } = require('@pinecone-database/pinecone');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const dotenv = require('dotenv');
const urlParser = require('url');
const path = require('path');

dotenv.config();

// Initialize clients
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Constants
const MAX_METADATA_SIZE = 40900;
const MAX_TOKENS = 3000;
const MAX_DEPTH = 3;
const VISITED_URLS = new Set();
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
        
        // Remove hash fragments and query parameters
        link = link.split('#')[0].split('?')[0];
        
        // Skip certain file types and protocols
        const skipExtensions = ['.pdf', '.jpg', '.png', '.gif', '.css', '.js'];
        const skipProtocols = ['mailto:', 'tel:', 'ftp:', 'file:'];
        
        if (skipExtensions.some(ext => link.toLowerCase().endsWith(ext)) ||
            skipProtocols.some(protocol => link.toLowerCase().startsWith(protocol))) {
            return null;
        }

        const parsed = new URL(link, baseUrl);
        
        // Ensure same domain
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

    // Extract META information
    contentObj.title = $('title').text().trim();
    contentObj.metaDescription = $('meta[name="description"]').attr('content') || '';
    
    // Extract JSON-LD structured data
    $('script[type="application/ld+json"]').each((_, element) => {
        try {
            const jsonData = JSON.parse($(element).html());
            contentObj.structuredData = { ...contentObj.structuredData, ...jsonData };
        } catch (e) {
            console.log('Error parsing JSON-LD:', e.message);
        }
    });

    // Function to process text content
    const processTextContent = (element) => {
        const text = $(element).text().trim();
        if (text.length > 20) { // Ignore very short texts
            contentObj.textContent.add(cleanText(text));
        }
    };

    // Remove unwanted elements
    $('script, style, noscript, iframe, img, svg, header, footer, nav').remove();

    // Priority elements
    const prioritySelectors = [
        'article',
        'main',
        'section',
        '.content',
        '#content',
        '.post',
        '.article',
        '[role="main"]',
        '[role="article"]'
    ];

    // Process priority content first
    prioritySelectors.forEach(selector => {
        $(selector).each((_, element) => {
            processTextContent(element);
        });
    });

    // Process headings and paragraphs
    $('h1, h2, h3, h4, h5, h6, p').each((_, element) => {
        processTextContent(element);
    });

    // Process list items with substantial content
    $('li').each((_, element) => {
        const text = $(element).text().trim();
        if (text.length > 50) { // Only substantial list items
            processTextContent(element);
        }
    });

    // Process any remaining text nodes in the body
    $('body').contents().each((_, element) => {
        if (element.type === 'text') {
            const text = $(element).text().trim();
            if (text.length > 20) {
                contentObj.textContent.add(cleanText(text));
            }
        }
    });

    // Combine all unique text content
    contentObj.mainContent = Array.from(contentObj.textContent).join('\n\n');

    return contentObj;
};

// Web scraping function
async function scrapeWebsite(url, depth = 0, retryCount = 0) {
    if (VISITED_URLS.has(url) || depth > MAX_DEPTH) return null;
    VISITED_URLS.add(url);

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

        // Extract links
        const internalLinks = new Set();
        $('a').each((_, element) => {
            const href = $(element).attr('href');
            const normalizedUrl = normalizeUrl(url, href);
            if (normalizedUrl) {
                internalLinks.add(normalizedUrl);
            }
        });

        console.log(`✅ Scraped: ${url}`);
        console.log(`- Title: ${contentObj.title}`);
        console.log(`- Content Length: ${contentObj.mainContent.length} chars`);
        console.log(`- Internal Links: ${internalLinks.size}`);

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
            return scrapeWebsite(url, depth, retryCount + 1);
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
async function ingestRecursive(url, depth = 0) {
    if (depth > MAX_DEPTH) return;

    const result = await scrapeWebsite(url, depth);
    if (!result) return;

    const index = pc.Index("dbtest");

    const contentChunks = chunkTextBySize(result.mainContent);
    for (let i = 0; i < contentChunks.length; i++) {
        const chunk = contentChunks[i];
        if (chunk.length < 100) continue; // Skip very small chunks

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
        await sleep(100); // Rate limiting
    }

    // Process internal links with rate limiting
    for (let link of result.internalLinks) {
        await sleep(500); // Respect robots.txt timing
        await ingestRecursive(link, depth + 1);
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

async function handleUserQuestion(question) {
    const context = await chat(question);

    if (!context || context.trim().length === 0) {
        return "I couldn't find any relevant information in the scraped data to answer your question.";
    }

    const answer = await generateAnswer(question, context);
    return answer;
}

// Main execution
async function main() {
    const startUrl = 'https://ankit.10xdevlab.in/'; 
    
    console.log('🕷️ Starting web scraping...');
    await ingestRecursive(startUrl,startUrl);
    
    const questions = [
        "who is dinesh",
       
    ];

    console.log('\n🤖 Answering questions...');
    for (let question of questions) {
        const answer = await handleUserQuestion(question,startUrl);
        console.log(`\n❓ Q: ${question}\n💬 A: ${answer}\n---`);
    }
}

main().catch(console.error);