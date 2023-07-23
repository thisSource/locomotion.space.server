// Load all necessary modules from Node.js and the langchain library
require("dotenv").config();
const http = require("http");
const WebSocket = require("ws");
const { ChatOpenAI } = require("langchain/chat_models/openai");
const { LLMChain } = require("langchain/chains");
const {
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  ChatPromptTemplate,
} = require("langchain/prompts");
const { SupabaseVectorStore } = require("langchain/vectorstores/supabase");
const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { createClient } = require("@supabase/supabase-js");
const { ConversationSummaryMemory } = require("langchain/memory");
const { v4: uuidv4 } = require("uuid");
const { OpenAI } = require("langchain/llms/openai");

// Sanitization function to clean up content
const sanitizeContent = (content) => {
  return content.replace(/[{}]/g, "");
};

// Define class to handle memory storage using Supabase
class SupabaseMemoryStore {
  constructor(client, tableName) {
    this.client = client; // Supabase client
    this.tableName = tableName; // Table in the database to use
    this.conversationSummaryMemories = {}; // Store for conversation summaries
  }

  // Get or create a memory for a client
  getMemoryForClient(client_id) {
    if (!(client_id in this.conversationSummaryMemories)) {
      this.conversationSummaryMemories[client_id] =
        new ConversationSummaryMemory({
          memoryKey: "chat_history",
          llm: new OpenAI({
            // modelName: "gpt-3.5-turbo",
            modelName: "gpt-4-0613",
            temperature: 0,
          }),
        });
    }
    return this.conversationSummaryMemories[client_id];
  }

  async saveContext(input, output, client_id) {
    // Get or create a memory for the client
    const memory = this.getMemoryForClient(client_id);

    // Update conversation summary
    await memory.saveContext({ input: input }, { output: output });

    // Load conversation summary
    const conversationSummary = await memory.loadMemoryVariables();

    // Store summary in the database
    const row = {
      input: "", // store empty string
      output: "", // store empty string
      summary: conversationSummary.chat_history,
      client_id: client_id,
    };

    const { data, error } = await this.client.from(this.tableName).upsert(row, {
      conflictFields: ["client_id"],
      updateColumns: ["input", "output", "summary"],
    });
  }

  // Load memory variables from the database
  async loadMemoryVariables(client_id) {
    const { data, error } = await this.client
      .from(this.tableName)
      .select("summary")
      .eq("client_id", client_id)
      .limit(1);

    if (data && data.length > 0) {
      return { chat_history: data[0].summary };
    } else {
      return { chat_history: "" };
    }
  }
}

// Create HTTP server and WebSocket server
const server = http.createServer();
const wss = new WebSocket.Server({ server });

const privateKey = process.env.SUPABASE_PRIVATE_KEY;
const url = process.env.SUPABASE_URL;
const client = createClient(url, privateKey);

const memory = new SupabaseMemoryStore(client, "memory");
const vectorStore = new SupabaseVectorStore(new OpenAIEmbeddings(), {
  client,
  tableName: "book_and_claim",
});

// Listen for new WebSocket connections
wss.on("connection", (ws) => {
  const client_id = uuidv4(); // Generate a unique identifier for the client

  // Listen for messages from the client
  ws.on("message", async (message) => {
    try {
      const statusMsg1 = { status: "Processing your message" };
      console.log(statusMsg1); // Log to console
      ws.send(JSON.stringify(statusMsg1)); // Notify frontend that message is being processed

      const question = JSON.parse(message).question; // Parse the incoming message
      const chatHistory = await memory.loadMemoryVariables(client_id); // Load chat history

      // Initialize a new ChatOpenAI instance
      const chat = new ChatOpenAI({
        modelName: "gpt-4-0613",
        streaming: true,
        callbacks: [
          {
            handleLLMNewToken(token) {
              const statusMsg2 = { status: "Received new token", token: token };
              ws.send(JSON.stringify(statusMsg2)); // Notify frontend about new token
            },
          },
        ],
      });

      // Perform similarity search based on the question and format the results
      const searchResults = await vectorStore.similaritySearch(question, 5);
      const statusMsg3 = { status: "Performing similarity search" };
      console.log(statusMsg3); // Log to console
      ws.send(JSON.stringify(statusMsg3)); // Notify frontend that similarity search is done

      const documentPrompts = searchResults
        .map(
          (doc, i) =>
            `Document ${i + 1}: Titled '${doc.metadata.source}' on page ${
              doc.metadata.page
            }, it says: "${sanitizeContent(doc.pageContent)}".`
        )
        .join("\n");

      const expertPrompt = ChatPromptTemplate.fromPromptMessages([
        SystemMessagePromptTemplate.fromTemplate(
          `Previous conversation:\n${chatHistory.chat_history}\n` +
            "You are an AI assistant that embodies the persona of a friendly Sustainable Fashion Expert. Your expertise lies in sustainable supply chain management, textiles, future fashion, and you're adept at making complex subjects understandable. This includes, but is not limited to, topics like sustainable materials, carbon reduction, and textile technologies. The answers you provide are clear, informative, and tailored to be easily grasped by anyone, regardless of their prior knowledge. You may refer to the following documents to aid in answering questions:\n" +
            documentPrompts
        ),
        HumanMessagePromptTemplate.fromTemplate(
          `Based on the documents provided, please give an answer to the following question: ${question}`
        ),
      ]);

      const chain = new LLMChain({ prompt: expertPrompt, llm: chat });
      const statusMsg4 = { status: "Recieving the AI response" };
      console.log(statusMsg4); // Log to console
      ws.send(JSON.stringify(statusMsg4)); // Send status message to frontend
      const response = await chain.call({ text: question });

      const statusMsg5 = { status: "Formatting the response and adding to conversation memory" };
      console.log(statusMsg5); // Log to console
      ws.send(JSON.stringify(statusMsg5)); // Send status message to frontend

      await memory.saveContext(question, response.text, client_id);
      
      const responseMsg = {
        data: response,
        metadata: searchResults.map((doc) => doc.metadata),
      };
      const statusMsg6 = { status: "Finished" };
      ws.send(JSON.stringify(statusMsg6)); // Send status message to frontend

      console.log(statusMsg6); // Log to console
      // console.log(responseMsg);  // Log to console
      ws.send(JSON.stringify(responseMsg));
    } catch (error) {
      // Error handling
      console.error(error);
      if (error.error && error.error.code === "context_length_exceeded") {
        const errorMsg1 = {
          data: {
            message:
              "The context length has exceeded the model limit. Please reduce the length of your query.",
            type: "error",
          },
          metadata: [{ error: "context_length_exceeded" }],
        };
        console.log(errorMsg1); // Log to console
        ws.send(JSON.stringify(errorMsg1));
      } else {
        const errorMsg2 = {
          data: {
            message: "An unexpected error occurred. Please try again.",
            type: "error",
          },
          metadata: [{ error: "unexpected_error" }],
        };
        console.log(errorMsg2); // Log to console
        ws.send(JSON.stringify(errorMsg2));
      }
    }
  });
});

// Listen on the specified port
const PORT = process.env.PORT || 8080;
server.listen(PORT, () => {
  console.log(`WebSocket server is running on port ${PORT}`);
});
