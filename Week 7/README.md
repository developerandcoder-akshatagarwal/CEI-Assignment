# Week 7 Assignment – RAG-Based Document Question Answering System

## Introduction

In this assignment, I developed a simple Retrieval-Augmented Generation (RAG) system that can answer questions from custom documents. Instead of generating answers only from the language model's knowledge, the system first retrieves relevant information from the uploaded documents and then generates an answer based on that context.

## Tools and Technologies

* Python
* LangChain
* Google Gemini
* PyPDF
* Pinecone

## Working

The system loads documents, splits them into smaller chunks, and converts them into embeddings. These embeddings are stored in a vector database. When a user asks a question, the system retrieves the most relevant chunks and uses Gemini to generate an accurate answer based on the retrieved content.

## Conclusion

This project helped me understand how modern AI assistants use retrieval and generation together to provide more accurate answers from custom documents. It also gave me practical experience with embeddings, vector search, and large language models.
