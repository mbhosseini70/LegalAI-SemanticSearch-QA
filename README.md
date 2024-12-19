# Semantic Search and Question Answering for Legal Documents

## Overview
This project aims to create a fully automated pipeline that transforms legal documents into a searchable knowledge base. The system enables users to ask precise legal questions and retrieve accurate answers with contextual information while ensuring adherence to legal standards and terminologies.

---

## Key Features
- **Semantic Processing of Legal Documents:** Efficiently preprocesses, splits, and embeds legal documents for semantic retrieval.
- **Title and Summary Generation:** Automatically generates concise and contextually relevant titles or summaries for document chunks using large language models (LLMs).
- **Vector Database Construction:** Builds and persists a vector database using embeddings for high-performance document retrieval.
- **Custom Legal QA System:** Provides precise answers to legal queries with relevant context.
- **Cosine Similarity for Contextual Queries:** Ranks document chunks based on relevance using cosine similarity.

---

## Techniques Used
- **Natural Language Processing (NLP):** Leverages Sentence Transformers and the LegalBERT model for semantic embeddings.
- **Document Splitting:** Uses `RecursiveCharacterTextSplitter` to create manageable document chunks.
- **Vector-Based Retrieval:** Employs the Chroma vector database for semantic search.
- **Cosine Similarity:** Ranks document chunks based on user query relevance.
- **Few-Shot Learning:** Implements few-shot prompting techniques with an LLM (Cohere) for title and summary generation.

---

## Tools Used
- **LangChain:** For document splitting, retrieval chain, and prompt handling.
- **Hugging Face Transformers:** For the LegalBERT model.
- **Sentence Transformers:** For efficient embedding generation.
- **Chroma:** As a vector database for persistent storage and retrieval.
- **Cohere LLM:** For natural language understanding and title/summary generation.
- **Scikit-learn:** For implementing cosine similarity metrics.

---

## Architecture
The code architecture is designed to provide an end-to-end solution for processing legal documents and answering queries. Below is an overview of the modular components:

### 1. **Embedding Model Layer**
- **`LegalBERTEmbeddings` Class:**
  - Utilizes the SentenceTransformer library to load the LegalBERT model.
  - Generates semantic embeddings optimized for legal contexts.
  - Represents the semantic meaning of the text in a high-dimensional vector space.

### 2. **Document Preprocessing Layer**
- **`DocumentProcessor` Class:**
  - **Input Handling:** Processes `.pdf` or `.txt` files using appropriate loaders (`PyPDFLoader` for PDFs, `TextLoader` for text files).
  - **Document Splitting:** Splits documents into smaller chunks using `RecursiveCharacterTextSplitter` for efficient embedding and search.
  - **Title Generation:** Utilizes Cohere LLM to generate concise and relevant titles for each chunk.
  - **Vector Store Initialization:** Stores embeddings persistently using Chroma for semantic retrieval.

### 3. **Retrieval Layer**
- **Cosine Similarity-Based Retrieval:**
  - Computes cosine similarity between query embeddings and document chunk embeddings.
  - Ranks and retrieves top-matching chunks for answering user queries.

### 4. **Question Answering Layer**
- **`QAChain` Class:**
  - Initializes a retrieval-based chain to fetch relevant document chunks.
  - Uses Cohere LLM with structured prompts to generate precise answers based on retrieved context.
  - Ensures strict adherence to responding directly to user queries without unnecessary information.

### 5. **Query Execution Layer**
- **Independent Query Execution:**
  - Accepts user queries.
  - Retrieves relevant chunks and passes them as context to the LLM.
  - Outputs precise answers for each query.

---

## How to Use
1. **Install Dependencies:** Ensure you have Python installed along with the required libraries (`LangChain`, `Hugging Face Transformers`, `Sentence Transformers`, `Chroma`, `Cohere`, and `Scikit-learn`).
2. **Run the Code:**
   - Upload your legal documents (.pdf or .txt).
   - Execute the preprocessing pipeline to split and embed documents.
   - Query the system with your legal question and retrieve accurate answers.
3. **Customize:** Modify components like the embedding model or retrieval strategy to suit your specific needs.

---

## Future Work
- Extend support for multilingual legal documents.
- Implement advanced ranking techniques for better query matching.
- Incorporate additional legal-specific LLMs for enhanced performance.

---

Feel free to contribute to this project or reach out for collaboration!
