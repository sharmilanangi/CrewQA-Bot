# CrewQA-Bot
Personalized Crew Agents to QA, Factcheck and summarize 
## Overview
CrewQA-Bot is a personalized agent system for question answering, fact checking and summarization tasks.

## Features
- Customizable response length (short/medium/long)
- Adjustable expertise levels (naive/intermediate/expert)
- API integrations with:
  - Groq
  - Google Serper
  - Gemini

## Environment Setup
Required environment variables: 

Create a `.env` file in the root directory with the following variables:

GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
GEMINI_API_KEY=your_gemini_api_key

## Usage

Tested with Postman API Platform for API endpoints:
```
python api.py
```