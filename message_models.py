#!/usr/bin/env python3
"""
Shared message models for Hugging Face Query Agent and Test Agent

This file contains the message models used for communication between agents.
"""

from uagents import Model
from typing import List, Optional

# Protocol-based models for AI Engine compatibility
class QueryRequest(Model):
    """Model for sending queries to the HF Query Agent"""
    query: str
    resource_type: str = "models"  # Options: "models", "datasets", "spaces"
    limit: int = 10

class ModelResult(Model):
    """Model for a single Hugging Face model result"""
    id: str
    downloads: Optional[int] = None
    library_name: Optional[str] = None
    pipeline_tag: Optional[str] = None
    inference: Optional[str] = None
    # Changed from bool to str to handle any value type
    safetensors: str = "false"
    gated: str = "false"

class QueryResponse(Model):
    """Model for receiving query results from the HF Query Agent"""
    explanation: str
    results: List[ModelResult]
    error: Optional[str] = None