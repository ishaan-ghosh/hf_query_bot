#!/usr/bin/env python3
"""
Chat Protocol for Agentverse AI Engine v0.2.0

This module defines the protocol for communication between agents using the AgentChatProtocol.
It includes message structures like ChatMessage and ChatAcknowledgement, as well as content
types like TextContent, ResourceContent, MetadataContent, and session/stream management.

Protocol Digest: proto:5dc989a455cd2d1c93a2fb2342b9b1c5f1e05a15230d121890d1e8c1811b6e2a
"""

from uagents import Model, Protocol
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import uuid4

# Define the chat protocol
chat_proto = Protocol("AgentChatProtocol", version="0.2.0")

class Resource(Model):
    """Resource with URI and metadata"""
    uri: str
    metadata: Dict[str, str]

class TextContent(Model):
    """Text content for chat messages"""
    type: str = "text"
    text: str

class ResourceContent(Model):
    """Resource content for chat messages"""
    type: str = "resource"
    resource_id: str  # UUID4
    resource: Union[Resource, List[Resource]]

class MetadataContent(Model):
    """Metadata content for chat messages"""
    type: str = "metadata"
    metadata: Dict[str, str]

class StartSessionContent(Model):
    """Content to start a chat session"""
    type: str = "start-session"

class EndSessionContent(Model):
    """Content to end a chat session"""
    type: str = "end-session"

class StartStreamContent(Model):
    """Content to start a stream"""
    type: str = "start-stream"
    stream_id: str  # UUID4

class EndStreamContent(Model):
    """Content to end a stream"""
    type: str = "start-stream"  # Note: This matches the schema, though it seems like a typo
    stream_id: str  # UUID4

class ChatMessage(Model):
    """Standard chat message format for agent communication"""
    msg_id: str  # UUID4
    timestamp: datetime
    content: List[Union[
        TextContent, 
        ResourceContent, 
        MetadataContent,
        StartSessionContent,
        EndSessionContent,
        StartStreamContent,
        EndStreamContent
    ]]

class ChatAcknowledgement(Model):
    """Acknowledgement message for received chat messages"""
    timestamp: datetime
    acknowledged_msg_id: str  # UUID4
    metadata: Optional[Dict[str, str]] = None

def create_text_chat(text: str) -> ChatMessage:
    """
    Create a chat message with text content
    
    Args:
        text (str): The text content of the message
        
    Returns:
        ChatMessage: A formatted chat message
    """
    return ChatMessage(
        msg_id=str(uuid4()),
        timestamp=datetime.now(),
        content=[TextContent(text=text)]
    )

def create_metadata_chat(metadata: Dict[str, str]) -> ChatMessage:
    """
    Create a chat message with metadata content
    
    Args:
        metadata (dict): The metadata content of the message
        
    Returns:
        ChatMessage: A formatted chat message
    """
    return ChatMessage(
        msg_id=str(uuid4()),
        timestamp=datetime.now(),
        content=[MetadataContent(metadata=metadata)]
    )

def create_resource_chat(resource_uri: str, resource_metadata: Dict[str, str]) -> ChatMessage:
    """
    Create a chat message with resource content
    
    Args:
        resource_uri (str): The URI of the resource
        resource_metadata (dict): Metadata about the resource
        
    Returns:
        ChatMessage: A formatted chat message
    """
    resource = Resource(uri=resource_uri, metadata=resource_metadata)
    return ChatMessage(
        msg_id=str(uuid4()),
        timestamp=datetime.now(),
        content=[ResourceContent(
            resource_id=str(uuid4()),
            resource=resource
        )]
    )

def create_mixed_chat(text: str, metadata: Dict[str, str]) -> ChatMessage:
    """
    Create a chat message with both text and metadata content
    
    Args:
        text (str): The text content of the message
        metadata (dict): The metadata content of the message
        
    Returns:
        ChatMessage: A formatted chat message
    """
    return ChatMessage(
        msg_id=str(uuid4()),
        timestamp=datetime.now(),
        content=[TextContent(text=text), MetadataContent(metadata=metadata)]
    )

def create_session_start() -> ChatMessage:
    """
    Create a chat message to start a session
    
    Returns:
        ChatMessage: A formatted chat message
    """
    return ChatMessage(
        msg_id=str(uuid4()),
        timestamp=datetime.now(),
        content=[StartSessionContent()]
    )

def create_session_end() -> ChatMessage:
    """
    Create a chat message to end a session
    
    Returns:
        ChatMessage: A formatted chat message
    """
    return ChatMessage(
        msg_id=str(uuid4()),
        timestamp=datetime.now(),
        content=[EndSessionContent()]
    )

def create_stream_start() -> ChatMessage:
    """
    Create a chat message to start a stream
    
    Returns:
        ChatMessage: A formatted chat message with stream start content
    """
    stream_id = str(uuid4())
    return ChatMessage(
        msg_id=str(uuid4()),
        timestamp=datetime.now(),
        content=[StartStreamContent(stream_id=stream_id)]
    ), stream_id

def create_stream_end(stream_id: str) -> ChatMessage:
    """
    Create a chat message to end a stream
    
    Args:
        stream_id (str): The ID of the stream to end
        
    Returns:
        ChatMessage: A formatted chat message
    """
    return ChatMessage(
        msg_id=str(uuid4()),
        timestamp=datetime.now(),
        content=[EndStreamContent(stream_id=stream_id)]
    )

@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx, sender, msg):
    """Default handler for chat messages - can be overridden by the agent"""
    ctx.logger.info(f"Received chat message from {sender}")
    
    # Send acknowledgement
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(),
            acknowledged_msg_id=msg.msg_id
        )
    )

@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx, sender, msg):
    """Default handler for acknowledgements - can be overridden by the agent"""
    ctx.logger.info(f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")