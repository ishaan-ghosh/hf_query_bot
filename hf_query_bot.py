#!/usr/bin/env python3
"""
Hugging Face Query Agent for Agentverse

This agent receives natural language queries about Hugging Face models from other agents,
processes them using the HF query logic, and returns matching models to the sender.

It integrates with the agentverse platform using the uAgents library, allowing other agents
to communicate with it through the agentverse messaging protocol.
"""
import sys
import subprocess

# Import uAgents library
from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low
# Import AI Engine for compatibility
from ai_engine import UAgentResponse, UAgentResponseType

# Import huggingface_hub, install if not available
try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("huggingface_hub not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import HfApi, login

# Import rich, install if not available
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
    from rich.markdown import Markdown
except ImportError:
    print("rich not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
    from rich.markdown import Markdown

import json
import re
import requests
import os
import difflib
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from uuid import uuid4

# Import chat protocol
from chat_protocol import chat_proto, create_text_chat, ChatMessage, ChatAcknowledgement

# Initialize console for rich output
console = Console()

# Create the HF Query Agent
hf_query_agent = Agent(
    name="HuggingFaceQueryAgent",
    seed="hf-query-agent-seed",
)

# Fund the agent if needed
fund_agent_if_low(hf_query_agent.wallet.address())

# Import shared message models
from message_models import QueryRequest, ModelResult, QueryResponse

# Hardcoded configuration
CONFIG = {
    "api_tokens": {
        "asi1_token": "sk_3b8a4da4b7154410936dc08ae4a3f30d89dd7d09a49940f3a5a6e4c463ca8ec1",
        "hf_token": ""
    },
    "defaults": {
        "limit": 10,
        "sort": "downloads",
        "direction": -1,
        "gated": False
    },
    "display": {
        "show_explanation": True,
        "show_parameters": True,
        "color_output": True
    }
}

# Constants for API keys
DEFAULT_HF_TOKEN = CONFIG["api_tokens"]["hf_token"]
DEFAULT_ASI1_TOKEN = CONFIG["api_tokens"]["asi1_token"]

def load_config():
    """
    Hardcoded config is used. This function returns the hardcoded config.
    
    Returns:
        dict: Configuration dictionary
    """
    console.print("[bold green]Using hardcoded configuration.[/bold green]")
    return CONFIG

# Load configuration (hardcoded)
CONFIG = load_config()

# Task mapping
TASK_MAPPING = {
    "sentiment analysis": "text-classification",
    "text classification": "text-classification",
    "named entity recognition": "token-classification",
    "ner": "token-classification",
    "question answering": "question-answering",
    "qa": "question-answering",
    "summarization": "summarization",
    "translation": "translation",
    "text generation": "text-generation",
    "language generation": "text-generation",
    "fill mask": "fill-mask",
    "masked language modeling": "fill-mask",
    "sentence similarity": "sentence-similarity",
    "image classification": "image-classification",
    "object detection": "object-detection",
    "speech recognition": "automatic-speech-recognition",
    "asr": "automatic-speech-recognition",
    "text to speech": "text-to-speech",
    "tts": "text-to-speech",
}

# Library mapping
LIBRARY_MAPPING = {
    "pytorch": "pytorch",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "jax": "jax",
    "flax": "jax",
    "safetensors": "safetensors",
    "onnx": "onnx",
    "transformers": "transformers",
    "diffusers": "diffusers",
    "sentence transformers": "sentence-transformers",
}

# Language mapping
LANGUAGE_MAPPING = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "chinese": "zh",
    "japanese": "ja",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi",
    "multilingual": "multilingual",
    "multi-language": "multilingual",
    "multiple languages": "multilingual",
}

# Concept mapping
CONCEPT_MAPPING = {
    "lightweight": {"sort": "downloads", "inference": "warm"},
    "small": {"sort": "downloads", "inference": "warm"},
    "fast": {"inference": "warm"},
    "quick": {"inference": "warm"},
    "production-ready": {"inference": "warm", "gated": False},
    "production ready": {"inference": "warm", "gated": False},
    "recent": {"sort": "last_modified", "direction": -1},
    "new": {"sort": "last_modified", "direction": -1},
    "popular": {"sort": "downloads", "direction": -1},
    "most used": {"sort": "downloads", "direction": -1},
    "trending": {"sort": "trending_score", "direction": -1},
    "best": {"sort": "downloads", "direction": -1},
    "state-of-the-art": {"sort": "downloads", "direction": -1},
    "state of the art": {"sort": "downloads", "direction": -1},
    "sota": {"sort": "downloads", "direction": -1},
}

def preprocess_query(query: str) -> str:
    """
    Preprocess the query to enhance LLM understanding.
    
    Args:
        query (str): The natural language query from the user
        
    Returns:
        str: Preprocessed query
    """
    # Convert to lowercase for consistency
    query = query.lower()
    
    # Expand common abbreviations
    abbreviations = {
        "nlp": "natural language processing",
        "cv": "computer vision",
        "qa": "question answering",
        "ner": "named entity recognition",
        "mt": "machine translation",
        "tts": "text to speech",
        "asr": "automatic speech recognition",
        "ocr": "optical character recognition"
    }
    
    for abbr, expansion in abbreviations.items():
        # Replace only standalone abbreviations (with word boundaries)
        query = re.sub(r'\b' + abbr + r'\b', expansion, query)
    
    # Add context for domain-specific terms
    domain_terms = {
        "bert": "BERT language model",
        "gpt": "GPT language model",
        "t5": "T5 language model",
        "llama": "LLaMA language model",
        "roberta": "RoBERTa language model",
        "distilbert": "DistilBERT language model"
    }
    
    for term, context in domain_terms.items():
        if re.search(r'\b' + term + r'\b', query, re.IGNORECASE):
            query += f" (Note: {term} refers to {context})"
    
    return query

def get_system_prompt() -> str:
    """
    Get the system prompt for the LLM.
    
    Returns:
        str: The system prompt
    """
    return """You are an expert AI assistant specialized in parsing natural language queries for searching Hugging Face models. Your task is to extract structured search parameters from user queries to find the most relevant models on the Hugging Face Hub.

## YOUR ROLE:
- Analyze the user's natural language query about Hugging Face models
- Extract explicit and implicit search parameters
- Convert these parameters into a structured JSON format that can be used for API calls
- Infer reasonable defaults for important parameters when they're not explicitly mentioned
- Apply contextual reasoning to understand the user's intent
- Be careful not to over-constrain searches with too many filters
- Prioritize the most important parameters based on the query context

## AVAILABLE SEARCH PARAMETERS:
1. task: The primary task the model is designed for
   - text-classification: Models for classifying text into categories
   - token-classification: Models for classifying individual tokens (NER, POS tagging)
   - question-answering: Models for answering questions based on context
   - summarization: Models for summarizing longer texts
   - translation: Models for translating between languages
   - text2text-generation: Models for generating text based on input text
   - text-generation: Models for generating text (like GPT, LLaMA)
   - fill-mask: Models for filling in masked tokens (like BERT)
   - sentence-similarity: Models for comparing sentence similarities
   - image-classification: Models for classifying images
   - object-detection: Models for detecting objects in images
   - audio-classification: Models for classifying audio
   - automatic-speech-recognition: Models for transcribing speech to text
   - text-to-speech: Models for converting text to speech
   - image-to-text: Models for generating text descriptions of images
   - text-to-image: Models for generating images from text descriptions
   - conversational: Models for dialogue and conversation
   - feature-extraction: Models for extracting features from inputs
   - image-segmentation: Models for segmenting images
   - depth-estimation: Models for estimating depth from images
   - reinforcement-learning: Models for reinforcement learning
   - robotics: Models for robotics applications
   - tabular-classification: Models for classifying tabular data
   - tabular-regression: Models for regression on tabular data
   - time-series-forecasting: Models for forecasting time series
   - visual-question-answering: Models for answering questions about images
   - table-question-answering: Models for answering questions about tables

2. library: The framework or library the model uses
   - pytorch: PyTorch models
   - tensorflow: TensorFlow models
   - jax: JAX/Flax models
   - safetensors: Models using safetensors format
   - onnx: ONNX format models
   - transformers: Models compatible with Hugging Face Transformers
   - diffusers: Models compatible with Diffusers library
   - sentence-transformers: Models for sentence embeddings
   - keras: Keras models
   - timm: Timm (PyTorch Image Models) library
   - allennlp: AllenNLP models

3. language: The language(s) the model supports
   - en: English
   - fr: French
   - de: German
   - es: Spanish
   - zh: Chinese
   - ja: Japanese
   - ru: Russian
   - ar: Arabic
   - hi: Hindi
   - it: Italian
   - ko: Korean
   - pt: Portuguese
   - multilingual: Supporting multiple languages

4. inference: The inference availability status
   - warm: Ready for immediate inference (optimized for production)
   - cold: Needs to be loaded first
   - frozen: Not available for inference

5. gated: Whether the model is gated (requires acceptance of terms)
   - true: Gated models
   - false: Non-gated models

6. sort: How to sort the results
   - downloads: Sort by number of downloads
   - trending_score: Sort by trending popularity
   - last_modified: Sort by last update date
   - created_at: Sort by creation date
   - likes: Sort by number of likes

7. direction: Sort direction
   - -1: Descending order (highest first)
   - 1: Ascending order (lowest first)
   Note: For "likes" sorting, only descending order (-1) is supported by the API

8. limit: Maximum number of results to return (default: 10)

9. search: Free text search term to find in model name or description

10. author: Filter by model author or organization

11. tags: Filter by specific tags

## SPECIAL CONCEPTS TO RECOGNIZE:
- "lightweight" or "small": Models with smaller size, often faster
- "state-of-the-art" or "best": High-performing models, sort by downloads
- "fast" or "quick": Models optimized for speed, prefer inference="warm"
- "production-ready": Models suitable for production, prefer inference="warm" and gated=false
- "recent" or "new": Recently updated models, sort by last_modified
- "popular": Widely used models, sort by downloads
- "trending": Currently popular models, sort by trending_score

## IMPORTANT GUIDELINES:
- Do not add too many constraints that might result in zero matches
- For specialized domains (medical, financial, legal, etc.), focus on the task and search term, avoid adding too many other constraints
- When handling complex queries, prioritize 2-3 most important parameters rather than including all possible filters
- For queries about specific model types or architectures, prioritize the search term over other constraints

## DOMAIN-SPECIFIC GUIDELINES:
- For financial domain queries: Prioritize search terms over tags, avoid adding inference constraints
- For medical domain queries: Focus on search terms and task, avoid language constraints
- For legal domain queries: Prioritize search terms, avoid adding specialized tags
- For scientific domain queries: Focus on task type, avoid overly specific search terms
- For mobile/edge computing queries: Focus on model size and inference speed, not specific architectures

## QUERY INTERPRETATION STRATEGY:
- Start with minimal constraints (1-2 key parameters)
- Only add additional constraints if they are explicitly mentioned in the query
- For ambiguous queries, prefer broader interpretations that will return more results
- When in doubt between two possible interpretations, choose the less restrictive one

## OUTPUT FORMAT:
You must respond with a valid JSON object containing the extracted parameters. For example:

{
  "task": "text-classification",
  "library": "pytorch",
  "language": "en",
  "inference": "warm",
  "gated": false,
  "sort": "downloads",
  "direction": -1,
  "limit": 10,
  "search": "bert",
  "author": null,
  "tags": ["sentiment-analysis"]
}

## EXAMPLES:

Query: "Find me BERT models for sentiment analysis"
Response:
{
  "task": "text-classification",
  "library": null,
  "language": null,
  "inference": null,
  "gated": false,
  "sort": "downloads",
  "direction": -1,
  "limit": 10,
  "search": "bert",
  "author": null,
  "tags": ["sentiment-analysis"]
}

Query: "I need a lightweight model for French text classification with good inference speed"
Response:
{
  "task": "text-classification",
  "library": null,
  "language": "fr",
  "inference": "warm",
  "gated": false,
  "sort": "downloads",
  "direction": -1,
  "limit": 10,
  "search": null,
  "author": null,
  "tags": null
}

Query: "Show me the most popular PyTorch models for named entity recognition that support multiple languages"
Response:
{
  "task": "token-classification",
  "library": "pytorch",
  "language": "multilingual",
  "inference": null,
  "gated": false,
  "sort": "downloads",
  "direction": -1,
  "limit": 10,
  "search": null,
  "author": null,
  "tags": ["named-entity-recognition"]
}

Query: "I'm looking for a recently updated question answering model that works well with medical text"
Response:
{
  "task": "question-answering",
  "library": null,
  "language": "en",
  "inference": null,
  "gated": false,
  "sort": "last_modified",
  "direction": -1,
  "limit": 10,
  "search": "medical",
  "author": null,
  "tags": null
}

Query: "Find production-ready text generation models by Google that aren't gated"
Response:
{
  "task": "text-generation",
  "library": null,
  "language": null,
  "inference": "warm",
  "gated": false,
  "sort": "downloads",
  "direction": -1,
  "limit": 10,
  "search": null,
  "author": "google",
  "tags": null
}

Now parse the following query and extract the relevant parameters:
"""
def parse_query_with_llm(query: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a natural language query using the LLM to extract search parameters.
    
    Args:
        query (str): The natural language query from the user
        api_key (str, optional): ASI1 API key
        
    Returns:
        dict: Dictionary of extracted search parameters
    """
    # Preprocess the query
    processed_query = preprocess_query(query)
    
    # Get the system prompt
    system_prompt = get_system_prompt()
    
    # Prepare the API request
    url = "https://api.asi1.ai/v1/chat/completions"
    
    payload = json.dumps({
        "model": "asi1-mini",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": processed_query
            }
        ],
        "temperature": 0,  # Use 0 for deterministic results
        "max_tokens": 1000
    })
    
    # Use the provided API key or the one from config
    token = api_key or DEFAULT_ASI1_TOKEN
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    # Send the request to the LLM API
    try:
        console.print("[bold blue]Sending query to LLM for parsing...[/bold blue]")
        response = requests.post(url, headers=headers, data=payload)
        
        if response.status_code != 200:
            console.print(f"[bold red]Error calling LLM API: {response.text}[/bold red]")
            return get_default_parameters()
        
        # Parse the response
        response_data = response.json()
        llm_response = response_data['choices'][0]['message']['content']
        
        try:
            # Extract the JSON object from the response
            params = json.loads(llm_response)
            
            # Validate and clean up parameters
            return validate_parameters(params)
        except json.JSONDecodeError:
            # If the response is not valid JSON, try to extract JSON using regex
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                try:
                    params = json.loads(json_match.group(0))
                    return validate_parameters(params)
                except:
                    console.print("[bold red]Failed to parse JSON from LLM response[/bold red]")
            else:
                console.print("[bold red]No JSON found in LLM response[/bold red]")
                console.print("[bold yellow]Using default parameters and inferring from query...[/bold yellow]")
            
            # If all else fails, try to infer parameters from the query using enhanced inference
            inferred_params = enhanced_parameter_inference(query)
            if inferred_params:
                console.print("[bold green]Successfully inferred parameters from query using enhanced inference[/bold green]")
                return inferred_params
            else:
                console.print("[bold yellow]Using default parameters[/bold yellow]")
                return get_default_parameters()
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return get_default_parameters()
    
def validate_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean up parameters extracted from the LLM response.
    
    Args:
        params (dict): Dictionary of parameters extracted from the LLM
        
    Returns:
        dict: Cleaned and validated parameters
    """
    valid_params = {}
    
    # Validate task
    if "task" in params and params["task"]:
        valid_tasks = [
            "text-classification", "token-classification", "question-answering",
            "summarization", "translation", "text2text-generation", "text-generation",
            "fill-mask", "sentence-similarity", "image-classification", "object-detection",
            "audio-classification", "automatic-speech-recognition", "text-to-speech",
            "image-to-text", "text-to-image"
        ]
        if params["task"] in valid_tasks:
            valid_params["task"] = params["task"]
    
    # Validate library
    if "library" in params and params["library"]:
        valid_libraries = [
            "pytorch", "tensorflow", "jax", "safetensors", "onnx",
            "transformers", "diffusers", "sentence-transformers"
        ]
        if params["library"] in valid_libraries:
            valid_params["library"] = params["library"]
    
    # Validate language
    if "language" in params and params["language"]:
        valid_languages = [
            "en", "fr", "de", "es", "zh", "ja", "ru", "ar", "hi", "multilingual"
        ]
        if params["language"] in valid_languages:
            valid_params["language"] = params["language"]
    
    # Validate inference
    if "inference" in params and params["inference"]:
        valid_inference = ["warm", "cold", "frozen"]
        if params["inference"] in valid_inference:
            valid_params["inference"] = params["inference"]
    
    # Validate gated
    if "gated" in params and params["gated"] is not None:
        valid_params["gated"] = bool(params["gated"])
    
    # Validate sort
    if "sort" in params and params["sort"]:
        valid_sort = ["downloads", "trending_score", "last_modified", "created_at", "likes"]
        if params["sort"] in valid_sort:
            valid_params["sort"] = params["sort"]
    
    # Validate direction
    if "direction" in params and params["direction"] is not None:
        if params["direction"] in [-1, 1]:
            # Handle special case for likes sorting
            if "sort" in params and params["sort"] == "likes" and params["direction"] == 1:
                # For likes, only descending sort is supported
                valid_params["direction"] = -1
            else:
                valid_params["direction"] = params["direction"]
    
    # Validate limit
    if "limit" in params and params["limit"] is not None:
        try:
            limit = int(params["limit"])
            if 1 <= limit <= 100:
                valid_params["limit"] = limit
            else:
                valid_params["limit"] = 10
        except:
            valid_params["limit"] = 10
    
    # Copy other parameters
    for param in ["search", "author"]:
        if param in params and params[param]:
            valid_params[param] = params[param]
    
    # Handle tags
    if "tags" in params and params["tags"]:
        if isinstance(params["tags"], list):
            valid_params["tags"] = params["tags"]
        elif isinstance(params["tags"], str):
            valid_params["tags"] = [params["tags"]]
    
    return valid_params

def enhanced_parameter_inference(query: str) -> Dict[str, Any]:
    """
    Enhanced parameter inference for when LLM parsing fails.
    Uses more sophisticated NLP techniques to extract parameters.
    
    Args:
        query (str): The natural language query
        
    Returns:
        dict: Dictionary of inferred parameters
    """
    # Use regex patterns for common query structures
    task_pattern = re.compile(r"(?:for|find|show)\s+(?:models\s+for)?\s*(\w+(?:\s+\w+)*)")
    domain_pattern = re.compile(r"(?:in|for|on)\s+(financial|medical|legal|scientific|mobile)")
    
    # Extract potential tasks
    task_match = task_pattern.search(query.lower())
    task_text = task_match.group(1) if task_match else None
    
    # Map extracted text to actual tasks using similarity matching
    task_mapping = {
        "text classification": "text-classification",
        "sentiment analysis": "text-classification",
        "named entity recognition": "token-classification",
        "ner": "token-classification",
        "question answering": "question-answering",
        "qa": "question-answering",
        "summarization": "summarization",
        "summarize": "summarization",
        "translation": "translation",
        "translate": "translation",
        "text generation": "text-generation",
        "generate text": "text-generation",
        "language model": "text-generation",
        "llm": "text-generation",
        "fill mask": "fill-mask",
        "masked": "fill-mask",
        "sentence similarity": "sentence-similarity",
        "similar": "sentence-similarity",
        "image classification": "image-classification",
        "classify image": "image-classification",
        "object detection": "object-detection",
        "detect object": "object-detection",
        "speech recognition": "automatic-speech-recognition",
        "transcribe": "automatic-speech-recognition",
        "text to speech": "text-to-speech",
        "tts": "text-to-speech",
        "image generation": "text-to-image",
        "generate image": "text-to-image",
        "conversation": "conversational",
        "chat": "conversational",
        "dialogue": "conversational",
        "feature extraction": "feature-extraction",
        "embeddings": "feature-extraction",
        "image segmentation": "image-segmentation",
        "segment image": "image-segmentation",
        "depth estimation": "depth-estimation",
        "3d depth": "depth-estimation",
        "reinforcement learning": "reinforcement-learning",
        "rl": "reinforcement-learning",
        "robotics": "robotics",
        "robot control": "robotics",
        "tabular classification": "tabular-classification",
        "classify tabular": "tabular-classification",
        "tabular regression": "tabular-regression",
        "regression": "tabular-regression",
        "time series": "time-series-forecasting",
        "forecasting": "time-series-forecasting",
        "predict future": "time-series-forecasting",
        "visual qa": "visual-question-answering",
        "visual question answering": "visual-question-answering",
        "table qa": "table-question-answering",
        "table question answering": "table-question-answering",
        "image to text": "image-to-text",
        "image captioning": "image-to-text"
    }
    
    # Initialize parameters with defaults
    params = get_default_parameters()
    
    # Find the closest task match if we have a task text
    if task_text:
        best_match = None
        best_score = 0
        for key, value in task_mapping.items():
            similarity = difflib.SequenceMatcher(None, task_text, key).ratio()
            if similarity > best_score:
                best_score = similarity
                best_match = value
        
        if best_score > 0.7:  # Threshold for a good match
            params["task"] = best_match
    
    # Extract domain information
    domain_match = domain_pattern.search(query.lower())
    domain = domain_match.group(1) if domain_match else None
    
    # Apply domain-specific parameter adjustments
    if domain:
        if domain == "financial":
            # For financial domain: prioritize search terms, avoid inference constraints
            if "financial" in query.lower() or "finance" in query.lower():
                params["search"] = "financial"
        elif domain == "medical":
            # For medical domain: focus on search terms and task
            if "medical" in query.lower() or "health" in query.lower():
                params["search"] = "medical"
        elif domain == "legal":
            # For legal domain: prioritize search terms
            if "legal" in query.lower() or "law" in query.lower():
                params["search"] = "legal"
        elif domain == "scientific":
            # For scientific domain: focus on task type
            if "scientific" in query.lower() or "science" in query.lower():
                params["search"] = "scientific"
        elif domain == "mobile":
            # For mobile domain: focus on model size and inference speed
            params["inference"] = "warm"
    
    # Preprocess the query
    query = query.lower()
    
    # Try to infer task using direct keyword matching as a fallback
    if "task" not in params:
        for keyword, task in task_mapping.items():
            if keyword in query:
                params["task"] = task
                break
    
    # Try to infer library
    for keyword, library in {
        "pytorch": "pytorch",
        "torch": "pytorch",
        "tensorflow": "tensorflow",
        "tf": "tensorflow",
        "jax": "jax",
        "flax": "jax",
        "safetensors": "safetensors",
        "onnx": "onnx",
        "transformers": "transformers",
        "diffusers": "diffusers",
        "sentence transformers": "sentence-transformers"
    }.items():
        if keyword in query:
            params["library"] = library
            break
    
    # Try to infer language
    for keyword, language in {
        "english": "en",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "chinese": "zh",
        "japanese": "ja",
        "russian": "ru",
        "arabic": "ar",
        "hindi": "hi",
        "multilingual": "multilingual",
        "multi-language": "multilingual",
        "multiple languages": "multilingual"
    }.items():
        if keyword in query:
            params["language"] = language
            break
    
    # Try to infer sort and direction with more comprehensive keywords
    sort_keywords = {
        "downloads": ["best", "top", "popular", "most used", "highest", "most downloaded", "most common", "widely used"],
        "last_modified": ["newest", "recent", "latest", "updated", "just released", "new", "fresh"],
        "created_at": ["oldest", "earliest", "first", "original", "initial", "classic"],
        "trending_score": ["trending", "hot", "current", "viral", "rising", "popular now"],
        "likes": ["liked", "favorite", "favourite", "most liked", "highly rated", "well-received"]
    }
    
    for sort_type, keywords in sort_keywords.items():
        if any(keyword in query for keyword in keywords):
            params["sort"] = sort_type
            # Default to descending for most sorts
            params["direction"] = -1
            # But use ascending for "oldest" queries
            if sort_type == "created_at" and any(k in query for k in ["oldest", "earliest", "first"]):
                params["direction"] = 1
            break
    
    # Try to infer search term with more comprehensive model names
    model_names = {
        "bert": ["bert", "roberta", "distilbert", "albert", "electra", "deberta"],
        "gpt": ["gpt", "gpt2", "gpt3", "gpt4", "chatgpt", "gpt-j", "gpt-neo"],
        "t5": ["t5", "flan-t5", "t5-base", "t5-large", "t5-small", "t5-3b", "t5-11b"],
        "llama": ["llama", "llama2", "llama3", "meta-llama", "llama-7b", "llama-13b", "llama-70b"],
        "falcon": ["falcon", "tiiuae", "falcon-7b", "falcon-40b", "falcon-180b"],
        "mistral": ["mistral", "mixtral", "mistral-7b", "mixtral-8x7b"],
        "stable diffusion": ["stable diffusion", "sdxl", "sd", "stable-diffusion-xl", "sdxl-turbo"],
        "clip": ["clip", "openai clip", "open-clip", "clip-vit"],
        "whisper": ["whisper", "openai whisper", "whisper-large", "whisper-small"],
        "bloom": ["bloom", "bloomz", "bloom-560m", "bloom-7b1"],
        "gemma": ["gemma", "google-gemma", "gemma-7b", "gemma-2b"],
        "phi": ["phi", "phi-1", "phi-2", "microsoft-phi"],
        "mpt": ["mpt", "mpt-7b", "mpt-30b"],
        "yi": ["yi", "yi-6b", "yi-34b"],
        "qwen": ["qwen", "qwen-7b", "qwen-14b"],
        "vicuna": ["vicuna", "vicuna-7b", "vicuna-13b"]
    }
    
    for model_family, variants in model_names.items():
        if any(variant in query.lower() for variant in variants):
            params["search"] = model_family
            break
    
    # Try to infer author with more comprehensive patterns
    author_patterns = [
        r"by\s+([a-zA-Z0-9_-]+)",
        r"from\s+([a-zA-Z0-9_-]+)",
        r"created by\s+([a-zA-Z0-9_-]+)",
        r"developed by\s+([a-zA-Z0-9_-]+)",
        r"([a-zA-Z0-9_-]+)'s model"
    ]
    
    for pattern in author_patterns:
        author_match = re.search(pattern, query)
        if author_match:
            author = author_match.group(1).lower()
            # Map common organization names to their HF usernames
            org_mapping = {
                "google": "google",
                "meta": "meta",
                "facebook": "facebook",
                "microsoft": "microsoft",
                "openai": "openai",
                "huggingface": "huggingface",
                "stability": "stabilityai",
                "anthropic": "anthropic",
                "mistral": "mistralai"
            }
            params["author"] = org_mapping.get(author, author)
            break
    
    # Try to infer inference with more comprehensive keywords
    inference_keywords = {
        "warm": ["fast", "quick", "production", "ready", "immediate", "optimized", "efficient", "low latency", "real-time"],
        "cold": ["standard", "normal", "regular", "default"],
        "frozen": ["archived", "deprecated", "legacy"]
    }
    
    for inference_type, keywords in inference_keywords.items():
        if any(keyword in query for keyword in keywords):
            params["inference"] = inference_type
            break
    
    # Try to infer gated status with more comprehensive keywords
    gated_keywords = {
        False: ["free", "open", "not gated", "ungated", "public", "accessible", "available", "unrestricted", "open source", "permissive"],
        True: ["gated", "restricted", "limited", "access control", "approval required", "permission needed", "login required"]
    }
    
    for gated_status, keywords in gated_keywords.items():
        if any(keyword in query for keyword in keywords):
            params["gated"] = gated_status
            break
    
    # Try to infer limit with more comprehensive patterns
    limit_patterns = [
        r"(\d+)\s+(?:results|models)",
        r"show\s+(\d+)",
        r"limit\s+(?:to\s+)?(\d+)",
        r"top\s+(\d+)",
        r"first\s+(\d+)"
    ]
    
    for pattern in limit_patterns:
        limit_match = re.search(pattern, query)
        if limit_match:
            try:
                params["limit"] = min(int(limit_match.group(1)), 100)
                break
            except ValueError:
                pass
    
    return params

def get_default_parameters() -> Dict[str, Any]:
    """
    Get default parameters when extraction fails.
    
    Returns:
        dict: Dictionary of default parameters
    """
    return {
        "sort": "downloads",
        "direction": -1,
        "limit": 10,
        "gated": False
    }

def build_query(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a Hugging Face API query from extracted parameters.
    
    Args:
        params (dict): Dictionary of parameters extracted from the query
        
    Returns:
        dict: Dictionary of parameters for the Hugging Face API
    """
    # Make a copy of the parameters to avoid modifying the original
    params_copy = params.copy()
    
    # Count the number of restrictive filters
    restrictive_filters = 0
    
    # Check for potentially restrictive combinations
    has_search = "search" in params_copy and params_copy["search"]
    has_tags = "tags" in params_copy and params_copy["tags"]
    has_task = "task" in params_copy
    has_library = "library" in params_copy
    has_language = "language" in params_copy
    has_inference = "inference" in params_copy
    
    # Count restrictive filters
    if has_search:
        restrictive_filters += 1
    if has_tags:
        restrictive_filters += 1
    if has_task:
        restrictive_filters += 1
    if has_library:
        restrictive_filters += 1
    if has_language:
        restrictive_filters += 1
    if has_inference:
        restrictive_filters += 1
    
    # If we have too many restrictive filters, prioritize the most important ones
    if restrictive_filters > 3:
        console.print("[bold yellow]Warning: Query has many restrictive filters. Prioritizing the most important ones.[/bold yellow]")
        
        # Prioritize filters based on importance
        # For specialized searches, search term and task are usually most important
        priority_filters = []
        
        if has_search:
            priority_filters.append("search")
        if has_task:
            priority_filters.append("task")
        if has_tags:
            priority_filters.append("tags")
        if has_library:
            priority_filters.append("library")
        if has_language:
            priority_filters.append("language")
        if has_inference:
            priority_filters.append("inference")
        
        # Keep only the top 3 priority filters
        priority_filters = priority_filters[:3]
        
        console.print(f"[bold yellow]Using priority filters: {', '.join(priority_filters)}[/bold yellow]")
        
        # Remove non-priority filters
        for filter_name in ["search", "tags", "task", "library", "language", "inference"]:
            if filter_name not in priority_filters and filter_name in params_copy:
                console.print(f"[bold yellow]Removing filter: {filter_name}[/bold yellow]")
                params_copy.pop(filter_name)
    
    # Build the API parameters
    api_params = {}
    
    # Map task
    if "task" in params_copy:
        api_params["task"] = params_copy["task"]
    
    # Map library
    if "library" in params_copy:
        api_params["library"] = params_copy["library"]
    
    # Map language
    if "language" in params_copy:
        api_params["language"] = params_copy["language"]
    
    # Map search term
    if "search" in params_copy:
        api_params["search"] = params_copy["search"]
    
    # Map inference
    if "inference" in params_copy:
        api_params["inference"] = params_copy["inference"]
    
    # Map gated
    if "gated" in params_copy:
        api_params["gated"] = params_copy["gated"]
    
    # Map author
    if "author" in params_copy:
        api_params["author"] = params_copy["author"]
    
    # Map tags
    if "tags" in params_copy:
        api_params["filter"] = params_copy["tags"]
    
    # Map sorting
    if "sort" in params_copy:
        api_params["sort"] = params_copy["sort"]
        
        # Handle special case for likes sorting
        # The Hugging Face API only supports descending sort for likes
        if params_copy["sort"] == "likes" and params_copy.get("direction", -1) == 1:
            console.print("[bold yellow]Warning: Ascending sort for likes is not supported by the Hugging Face API.[/bold yellow]")
            console.print("[bold yellow]Using descending sort instead.[/bold yellow]")
            api_params["direction"] = -1
        else:
            api_params["direction"] = params_copy.get("direction", -1)
    
    # Map limit
    api_params["limit"] = params_copy.get("limit", 10)
    
    # Add expansion properties
    api_params["expand"] = [
        "downloads",
        "safetensors",
        "tags",
        "pipeline_tag",
        "library_name",
        "inference",
        "gated",
        "likes"
    ]
    
    return api_params

def explain_parameter_extraction(query: str, params: Dict[str, Any]) -> str:
    """
    Generate an explanation of how the query was interpreted.
    
    Args:
        query (str): The original query
        params (dict): The extracted parameters
        
    Returns:
        str: Human-readable explanation
    """
    explanation = [f"Query: \"{query}\"", ""]
    explanation.append("Interpreted as:")
    
    if "task" in params:
        explanation.append(f"- Looking for models for {params['task'].replace('-', ' ')} tasks")
    
    if "library" in params:
        explanation.append(f"- Using the {params['library']} library")
    
    if "language" in params:
        lang_name = {
            "en": "English",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "zh": "Chinese",
            "ja": "Japanese",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
            "multilingual": "multiple languages"
        }.get(params["language"], params["language"])
        explanation.append(f"- Supporting {lang_name}")
    
    if "inference" in params:
        inference_desc = {
            "warm": "ready for immediate inference (production-ready)",
            "cold": "requiring loading before inference",
            "frozen": "not available for inference"
        }.get(params["inference"], params["inference"])
        explanation.append(f"- Models that are {inference_desc}")
    
    if "gated" in params:
        gated_desc = "requiring access approval" if params["gated"] else "freely accessible without approval"
        explanation.append(f"- Models that are {gated_desc}")
    
    if "sort" in params:
        sort_desc = {
            "downloads": "most downloaded",
            "trending_score": "currently trending",
            "last_modified": "recently updated",
            "created_at": "newly created",
            "likes": "most liked"
        }.get(params["sort"], params["sort"])
        
        # Handle special case for likes sorting
        if params["sort"] == "likes":
            # For likes, only descending sort is supported
            direction = "highest first"
        else:
            direction = "highest first" if params.get("direction", -1) == -1 else "lowest first"
            
        explanation.append(f"- Sorted by {sort_desc} ({direction})")
    
    if "search" in params and params["search"]:
        explanation.append(f"- Containing '{params['search']}' in the name or description")
    
    if "author" in params and params["author"]:
        explanation.append(f"- Created by {params['author']}")
    
    if "tags" in params and params["tags"]:
        tags_str = ", ".join([f"'{tag}'" for tag in params["tags"]])
        explanation.append(f"- Tagged with {tags_str}")
    
    if "limit" in params:
        explanation.append(f"- Showing up to {params['limit']} results")
    
    return "\n".join(explanation)

def search_datasets(query: str, params: Dict[str, Any], api: HfApi) -> List[Any]:
    """
    Search for datasets using a two-phase approach.
    
    Args:
        query (str): The original query
        params (dict): The extracted parameters
        api (HfApi): The Hugging Face API instance
        
    Returns:
        list: List of DatasetInfo objects
    """
    console.print("[bold blue]Searching for datasets...[/bold blue]")
    
    # Extract dataset-specific parameters
    dataset_params = {
        "sort": params.get("sort", "downloads"),
        "direction": params.get("direction", -1),
        "limit": params.get("limit", 50)  # Get more results initially
    }
    
    # Map task to task_categories if present
    if "task" in params:
        dataset_params["filter"] = f"task_categories:{params['task']}"
    
    # Add language filter if present
    if "language" in params and params["language"]:
        dataset_params["language"] = params["language"]
    
    # Add search term if present
    if "search" in params and params["search"]:
        dataset_params["search"] = params["search"]
    
    # Add author filter if present
    if "author" in params and params["author"]:
        dataset_params["author"] = params["author"]
    
    console.print("[bold blue]Phase 1: Searching with minimal constraints...[/bold blue]")
    
    # Execute search
    try:
        datasets = list(api.list_datasets(**dataset_params))
        console.print(f"[bold blue]Found {len(datasets)} datasets in phase 1.[/bold blue]")
        
        # Phase 2: Apply additional filters in memory if needed
        if len(datasets) > 20:
            console.print("[bold blue]Phase 2: Applying additional filters in memory...[/bold blue]")
            filtered_datasets = []
            
            for dataset in datasets:
                include_dataset = True
                
                # Apply additional filters here if needed
                # For example, filter by size_categories, multilinguality, etc.
                
                if include_dataset:
                    filtered_datasets.append(dataset)
            
            console.print(f"[bold blue]Found {len(filtered_datasets)} datasets after applying filters.[/bold blue]")
            return filtered_datasets[:params.get("limit", 10)]
        
        return datasets[:params.get("limit", 10)]
    except Exception as e:
        console.print(f"[bold red]Error searching for datasets: {str(e)}[/bold red]")
        return []
    
def search_spaces(query: str, params: Dict[str, Any], api: HfApi) -> List[Any]:
    """
    Search for spaces using a two-phase approach.
    
    Args:
        query (str): The original query
        params (dict): The extracted parameters
        api (HfApi): The Hugging Face API instance
        
    Returns:
        list: List of SpaceInfo objects
    """
    console.print("[bold blue]Searching for spaces...[/bold blue]")
    
    # Extract space-specific parameters
    space_params = {
        "sort": params.get("sort", "downloads"),
        "direction": params.get("direction", -1),
        "limit": params.get("limit", 50)  # Get more results initially
    }
    
    # Map task to SDK if present (e.g., "gradio", "streamlit")
    if "sdk" in params:
        space_params["filter"] = params["sdk"]
    
    # Add search term if present
    if "search" in params and params["search"]:
        space_params["search"] = params["search"]
    
    # Add author filter if present
    if "author" in params and params["author"]:
        space_params["author"] = params["author"]
    
    console.print("[bold blue]Phase 1: Searching with minimal constraints...[/bold blue]")
    
    # Execute search
    try:
        spaces = list(api.list_spaces(**space_params))
        console.print(f"[bold blue]Found {len(spaces)} spaces in phase 1.[/bold blue]")
        
        # Phase 2: Apply additional filters in memory if needed
        if len(spaces) > 20:
            console.print("[bold blue]Phase 2: Applying additional filters in memory...[/bold blue]")
            filtered_spaces = []
            
            for space in spaces:
                include_space = True
                
                # Apply additional filters here if needed
                
                if include_space:
                    filtered_spaces.append(space)
            
            console.print(f"[bold blue]Found {len(filtered_spaces)} spaces after applying filters.[/bold blue]")
            return filtered_spaces[:params.get("limit", 10)]
        
        return spaces[:params.get("limit", 10)]
    except Exception as e:
        console.print(f"[bold red]Error searching for spaces: {str(e)}[/bold red]")
        return []

def two_phase_search(query: str, params: Dict[str, Any], api: HfApi) -> List[Any]:
    """
    Implement a two-phase search strategy that starts with minimal constraints
    and then refines results if too many are returned.
    
    Args:
        query (str): The original query
        params (dict): The extracted parameters
        api (HfApi): The Hugging Face API instance
        
    Returns:
        list: List of ModelInfo objects
    """
    console.print("[bold blue]Using two-phase search strategy...[/bold blue]")
    
    # Phase 1: Search with minimal constraints
    minimal_params = {
        "sort": params.get("sort", "downloads"),
        "direction": params.get("direction", -1),
        "limit": 50  # Get more results initially
    }
    
    # Add the most important parameter based on query type
    if "task" in params:
        minimal_params["task"] = params["task"]
    elif "search" in params:
        minimal_params["search"] = params["search"]
    
    console.print("[bold blue]Phase 1: Searching with minimal constraints...[/bold blue]")
    
    # Execute minimal search
    minimal_api_params = build_query(minimal_params)
    models = list(api.list_models(**minimal_api_params))
    
    console.print(f"[bold blue]Found {len(models)} models in phase 1.[/bold blue]")
    
    # Phase 2: If too many results, apply additional filters in memory
    if len(models) > 20:
        console.print("[bold blue]Phase 2: Applying additional filters in memory...[/bold blue]")
        filtered_models = []
        
        for model in models:
            # Apply additional filters in memory
            include_model = True
            
            # Language filter
            if "language" in params and params["language"] and hasattr(model, "card_data") and hasattr(model.card_data, "language"):
                model_languages = model.card_data.language
                if isinstance(model_languages, list):
                    if params["language"] not in model_languages and params["language"] != "multilingual":
                        include_model = False
                elif model_languages and model_languages != params["language"] and params["language"] != "multilingual":
                    include_model = False
            
            # Library filter
            if "library" in params and params["library"] and model.library_name:
                if model.library_name != params["library"]:
                    include_model = False
            
            # Inference filter
            if "inference" in params and params["inference"] and model.inference:
                if model.inference != params["inference"]:
                    include_model = False
            
            # Gated filter
            if "gated" in params and params["gated"] is not None:
                if model.gated != params["gated"]:
                    include_model = False
            
            # Author filter
            if "author" in params and params["author"]:
                if not model.id.startswith(params["author"] + "/"):
                    include_model = False
            
            # Tags filter
            if "tags" in params and params["tags"]:
                if not hasattr(model, "tags") or not model.tags:
                    include_model = False
                else:
                    model_tags = [tag.lower() for tag in model.tags]
                    for tag in params["tags"]:
                        if tag.lower() not in model_tags:
                            include_model = False
                            break
            
            if include_model:
                filtered_models.append(model)
        
        console.print(f"[bold blue]Found {len(filtered_models)} models after applying filters.[/bold blue]")
        return filtered_models[:params.get("limit", 10)]
    
    return models[:params.get("limit", 10)]

def log_user_feedback(query: str, params: Dict[str, Any], results: List[Any], feedback: str) -> None:
    """
    Log user feedback to improve future searches.
    
    Args:
        query (str): The original query
        params (dict): The extracted parameters
        results (list): The search results
        feedback (str): User feedback
    """
    feedback_data = {
        "query": query,
        "parameters": params,
        "num_results": len(results),
        "feedback": feedback,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Create feedback directory if it doesn't exist
    feedback_dir = Path(__file__).parent.parent / "feedback"
    feedback_dir.mkdir(exist_ok=True)
    
    # Append to feedback log
    feedback_file = feedback_dir / "user_feedback.jsonl"
    with open(feedback_file, "a") as f:
        f.write(json.dumps(feedback_data) + "\n")
    
    console.print(f"[bold green]Feedback logged to {feedback_file}[/bold green]")

def display_results(models: List[Any]) -> None:
    """
    Display search results in a formatted table.
    This function is kept for debugging purposes when running the agent locally.
    
    Args:
        models (list): List of ModelInfo objects
    """
    if not models:
        console.print("[bold red]No models found matching your criteria.[/bold red]")
        return
    
    # Create a table
    table = Table(
        title="Hugging Face Models",
        box=box.ROUNDED,
        header_style="bold magenta",
        show_lines=True
    )
    
    # Add columns
    table.add_column("Model ID", style="cyan")
    table.add_column("Downloads", justify="right")
    table.add_column("Library", style="green")
    table.add_column("Task", style="yellow")
    table.add_column("Inference", style="blue")
    table.add_column("Safetensors", justify="center")
    table.add_column("Gated", justify="center")
    
    # Add rows
    for model in models:
        table.add_row(
            model.id,
            f"{model.downloads:,}" if model.downloads else "N/A",
            model.library_name or "N/A",
            model.pipeline_tag or "N/A",
            model.inference or "N/A",
            "✓" if hasattr(model, 'safetensors') and model.safetensors else "✗",
            "✓" if hasattr(model, 'gated') and model.gated else "✗"
        )
    
    # Print the table
    console.print(table)

# Define a protocol for the Hugging Face query
hf_protocol = Protocol("HuggingFaceQueryProtocol", version="1.0")

@hf_query_agent.on_message(model=QueryRequest, replies={UAgentResponse})
async def handle_query(ctx: Context, sender: str, msg: QueryRequest):
    """
    Handle incoming query requests from other agents.
    Process the query and send back the results.
    """
    ctx.logger.info(f"Received query from {sender}: {msg.query}")
    
    # Use tokens from config
    hf_token = DEFAULT_HF_TOKEN
    asi1_token = DEFAULT_ASI1_TOKEN
    
    # Check if ASI1 token is available
    if not asi1_token:
        error_msg = "Error: ASI1 API token is not configured."
        ctx.logger.error(error_msg)
        await ctx.send(sender, QueryResponse(
            explanation="Failed to process query due to missing API token.",
            results=[],
            error=error_msg
        ))
        return
    
    try:
        # Parse query using LLM
        ctx.logger.info("Parsing query with LLM...")
        params = parse_query_with_llm(msg.query, asi1_token)
        
        # Override limit if specified in the request
        if msg.limit:
            params["limit"] = msg.limit
        
        # Generate explanation of parameter extraction
        explanation = explain_parameter_extraction(msg.query, params)
        ctx.logger.info(f"Query interpretation: {explanation}")
        
        # Build API query
        api_params = build_query(params)
        
        # Determine the resource type to search for
        resource_type = msg.resource_type
        
        # Initialize the API
        api = HfApi(token=hf_token)
        
        # Login if token is provided
        if hf_token:
            login(token=hf_token)
        
        # Search for the specified resource type
        if resource_type == "models":
            ctx.logger.info("Searching for models...")
            results = two_phase_search(msg.query, params, api)
            ctx.logger.info(f"Found {len(results)} models matching the criteria.")
            
            # Convert results to ModelResult objects
            model_results = []
            for model in results:
                # Convert safetensors and gated to strings
                safetensors_value = "false"
                if hasattr(model, 'safetensors'):
                    if model.safetensors is not None:
                        safetensors_value = str(model.safetensors).lower()
                
                gated_value = "false"
                if hasattr(model, 'gated'):
                    if model.gated is not None:
                        gated_value = str(model.gated).lower()
                
                model_results.append(ModelResult(
                    id=model.id,
                    downloads=model.downloads,
                    library_name=model.library_name,
                    pipeline_tag=model.pipeline_tag,
                    inference=model.inference,
                    safetensors=safetensors_value,
                    gated=gated_value
                ))
            
            # Send both the standard QueryResponse and an AI Engine compatible response
            await ctx.send(sender, QueryResponse(
                explanation=explanation,
                results=model_results,
                error=None
            ))
            
            # Also send an AI Engine compatible response
            response_text = f"{explanation}\n\nFound {len(model_results)} models matching your query."
            await ctx.send(sender, UAgentResponse(
                message=response_text,
                type=UAgentResponseType.FINAL
            ))
        elif resource_type == "datasets":
            ctx.logger.info("Searching for datasets...")
            results = search_datasets(msg.query, params, api)
            ctx.logger.info(f"Found {len(results)} datasets matching the criteria.")
            
            # For datasets, we'll just send back basic info as ModelResult objects
            dataset_results = []
            for dataset in results:
                # Handle potential None values
                downloads_value = None
                if hasattr(dataset, 'downloads'):
                    downloads_value = dataset.downloads
                
                dataset_results.append(ModelResult(
                    id=dataset.id,
                    downloads=downloads_value,
                    library_name=None,
                    pipeline_tag=None,
                    inference=None,
                    safetensors="false",  # Use string value
                    gated="false"  # Use string value
                ))
            
            # Send both the standard QueryResponse and an AI Engine compatible response
            await ctx.send(sender, QueryResponse(
                explanation=explanation,
                results=dataset_results,
                error=None
            ))
            
            # Also send an AI Engine compatible response
            response_text = f"{explanation}\n\nFound {len(dataset_results)} datasets matching your query."
            await ctx.send(sender, UAgentResponse(
                message=response_text,
                type=UAgentResponseType.FINAL
            ))
        elif resource_type == "spaces":
            ctx.logger.info("Searching for spaces...")
            results = search_spaces(msg.query, params, api)
            ctx.logger.info(f"Found {len(results)} spaces matching the criteria.")
            
            # For spaces, we'll just send back basic info as ModelResult objects
            space_results = []
            for space in results:
                # Handle potential None values
                downloads_value = None
                if hasattr(space, 'likes'):
                    downloads_value = space.likes
                
                library_name_value = None
                if hasattr(space, 'sdk'):
                    library_name_value = space.sdk
                
                space_results.append(ModelResult(
                    id=space.id,
                    downloads=downloads_value,
                    library_name=library_name_value,
                    pipeline_tag=None,
                    inference=None,
                    safetensors="false",  # Use string value
                    gated="false"  # Use string value
                ))
            
            # Send both the standard QueryResponse and an AI Engine compatible response
            await ctx.send(sender, QueryResponse(
                explanation=explanation,
                results=space_results,
                error=None
            ))
            
            # Also send an AI Engine compatible response
            response_text = f"{explanation}\n\nFound {len(space_results)} spaces matching your query."
            await ctx.send(sender, UAgentResponse(
                message=response_text,
                type=UAgentResponseType.FINAL
            ))
        else:
            error_msg = f"Error: Unknown resource type '{resource_type}'"
            ctx.logger.error(error_msg)
            
            # Send both the standard QueryResponse and an AI Engine compatible error response
            await ctx.send(sender, QueryResponse(
                explanation="Failed to process query due to unknown resource type.",
                results=[],
                error=error_msg
            ))
            
            # Also send an AI Engine compatible error response
            await ctx.send(sender, UAgentResponse(
                message=f"Error: Unknown resource type '{resource_type}'",
                type=UAgentResponseType.ERROR
            ))
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        ctx.logger.error(error_msg)
        
        # Send both the standard QueryResponse and an AI Engine compatible error response
        await ctx.send(sender, QueryResponse(
            explanation="An error occurred while processing your query.",
            results=[],
            error=error_msg
        ))
        
        # Also send an AI Engine compatible error response
        await ctx.send(sender, UAgentResponse(
            message=f"Error: {error_msg}",
            type=UAgentResponseType.ERROR
        ))

# Chat protocol handler for text messages (v0.2.0)
@chat_proto.on_message(ChatMessage, replies={UAgentResponse})
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages and convert them to query requests"""
    ctx.logger.info(f"Received chat message from {sender}")
    
    # Extract the text content from the message
    # In v0.2.0, content items have a 'type' field
    text_content = next((item for item in msg.content if hasattr(item, 'type') and item.type == 'text'), None)
    
    if text_content:
        query_text = text_content.text
        ctx.logger.info(f"Query text: {query_text}")
        
        # Send acknowledgement (v0.2.0 format)
        await ctx.send(
            sender,
            ChatAcknowledgement(
                timestamp=datetime.now(),
                acknowledged_msg_id=msg.msg_id
            )
        )
        
        # Process as a query request
        try:
            # Use the advanced query parsing with LLM
            params = parse_query_with_llm(query_text, DEFAULT_ASI1_TOKEN)
            
            # Generate explanation of parameter extraction
            explanation = explain_parameter_extraction(query_text, params)
            ctx.logger.info(f"Query interpretation: {explanation}")
            
            # Build API query
            api_params = build_query(params)
            
            # Initialize the API
            api = HfApi(token=DEFAULT_HF_TOKEN)
            
            # Login if token is provided
            if DEFAULT_HF_TOKEN:
                login(token=DEFAULT_HF_TOKEN)
            
            # Use two-phase search for better results
            results = two_phase_search(query_text, params, api)
            
            # Format the results as a text response
            response_text = f"{explanation}\n\nFound {len(results)} models matching your query:\n\n"
            
            for i, model in enumerate(results, 1):
                response_text += f"{i}. {model.id}\n"
                response_text += f"   Downloads: {model.downloads if model.downloads else 'N/A'}\n"
                response_text += f"   Library: {model.library_name or 'N/A'}\n"
                response_text += f"   Task: {model.pipeline_tag or 'N/A'}\n"
                if hasattr(model, 'inference') and model.inference:
                    response_text += f"   Inference: {model.inference}\n"
                if hasattr(model, 'safetensors'):
                    response_text += f"   Safetensors: {'Yes' if model.safetensors else 'No'}\n"
                if hasattr(model, 'gated'):
                    response_text += f"   Gated: {'Yes' if model.gated else 'No'}\n"
                response_text += "\n"
            
            # Send the response back as a UAgentResponse (AI Engine compatible)
            await ctx.send(sender, UAgentResponse(
                message=response_text,
                type=UAgentResponseType.FINAL
            ))
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            ctx.logger.error(error_msg)
            await ctx.send(sender, UAgentResponse(
                message=f"Error: {error_msg}",
                type=UAgentResponseType.ERROR
            ))

# Include the protocols in the agent
hf_query_agent.include(hf_protocol)
hf_query_agent.include(chat_proto)

# Add startup message
@hf_query_agent.on_event("startup")
async def startup(ctx: Context):
    """Log when the agent starts up"""
    ctx.logger.info("Hugging Face Query Agent is now running!")
    ctx.logger.info(f"Agent address: {hf_query_agent.address}")

if __name__ == "__main__":
    hf_query_agent.run()
