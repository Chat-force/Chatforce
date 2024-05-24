const express = require('express');
const bodyParser = require('body-parser');
const dotenv = require('dotenv');
const { Configuration, OpenAIApi } = require('openai');
const pinecone = require('pinecone-client');
const path = require('path');


dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));


// Initialize OpenAI
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

// Initialize Pinecone
pinecone.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_REGION, // Use the new PINECONE_REGION variable
});

const index = pinecone.Index('documents');

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});


app.post('/upload', (req, res) => {
  const { fileContent, fileName } = req.body;

  if (!fileContent || !fileName) {
    return res.status(400).send('File content and name are required.');
  }

  // Process and store the document
  processDocument(fileContent, fileName)
    .then(() => res.send('File successfully uploaded and processed'))
    .catch(error => res.status(500).send(`Error processing file: ${error.message}`));
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

async function processDocument(content, name) {
  const embedding = await getEmbedding(content);
  await index.upsert([{ id: name, values: embedding }]);
}

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
