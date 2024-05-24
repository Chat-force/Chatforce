
const express = require('express');
const bodyParser = require('body-parser');
const dotenv = require('dotenv');
const { Configuration, OpenAIApi } = require('openai');
const { Pinecone } = require('@pinecone-database/pinecone'); // Correct import for Pinecone

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Initialize OpenAI
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_REGION, // Use the new PINECONE_REGION variable
});

const index = pinecone.index('quickstart'); // Use your actual index name here

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Function to process and upsert document
async function processDocument(content, name) {
  const embedding = await getEmbedding(content);
  await index.upsert([
    {
      id: name,
      values: embedding,
      metadata: { category: 'document' } // You can customize metadata as needed
    }
  ]);
}

app.post('/upload', async (req, res) => {
  const { fileContent, fileName } = req.body;

  if (!fileContent || !fileName) {
    return res.status(400).send('File content and name are required.');
  }

  try {
    // Process and store the document
    await processDocument(fileContent, fileName);
    res.send('File successfully uploaded and processed');
  } catch (error) {
    res.status(500).send(`Error processing file: ${error.message}`);
  }
});

app.post('/chat', async (req, res) => {
  const { message } = req.body;

  if (!message) {
    return res.status(400).send('Message is required.');
  }

  try {
    // Retrieve the most relevant document using Pinecone
    const queryResponse = await index.query({
      topK: 1,
      includeValues: true,
      vector: await getEmbedding(message),
    });

    if (!queryResponse.matches.length) {
      return res.status(200).send({ response: 'No relevant documents found' });
    }

    const documentContent = queryResponse.matches[0].values;

    // Use OpenAI to generate a response based on the document content
    const response = await openai.createCompletion({
      engine: 'text-davinci-003',
      prompt: `The user asked: ${message}\nBased on the following document, provide a detailed answer:\n${documentContent}`,
      maxTokens: 150,
    });

    const answer = response.choices[0].text.trim();
    res.send({ response: answer });
  } catch (error) {
    res.status(500).send(`Error processing chat request: ${error.message}`);
  }
});

async function getEmbedding(text) {
  const response = await openai.createEmbedding({
    input: text,
    model: 'text-embedding-ada-002',
  });
  return response.data[0].embedding;
}

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});



