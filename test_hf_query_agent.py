#!/usr/bin/env python3
"""
Test Agent for Hugging Face Query Agent

This agent sends a query to the Hugging Face Query Agent and displays the results.
Modify the QUERY variable below to test different queries.
"""

# ============= MODIFY THIS QUERY AS NEEDED =============
QUERY = "Find me BERT models for sentiment analysis"
RESOURCE_TYPE = "models"  # Options: "models", "datasets", "spaces"
LIMIT = 10
# ======================================================

import sys
import subprocess
import asyncio
from datetime import datetime
from uuid import uuid4

# Import uAgents library
from uagents import Agent, Context, Model

# Import chat protocol
from chat_protocol import chat_proto, create_text_chat, ChatMessage, ChatAcknowledgement

# Import rich, install if not available
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except ImportError:
    print("rich not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich import box

# Initialize console for output
console = Console()

# Create the test agent
test_agent = Agent(
    name="TestQueryAgent",
    seed="test-query-agent-seed",
)

# Define the HF Query Agent address
# Replace this with the actual address of your HF Query Agent
HF_QUERY_AGENT_ADDRESS = "agent1qdpuf6wv00mcc2nzvrr8xlywlx0t69kdcfgtzry66epk5la23gvf2ejjevv"

# Import shared message models
from message_models import QueryRequest, ModelResult, QueryResponse

@test_agent.on_event("startup")
async def send_query(ctx: Context):
    """Send a query to the HF Query Agent when the test agent starts"""
    ctx.logger.info(f"Sending query to HF Query Agent: '{QUERY}'")
    
    # Send the query using both the original model and the chat protocol
    
    # 1. Using the original QueryRequest model
    await ctx.send(
        HF_QUERY_AGENT_ADDRESS, 
        QueryRequest(
            query=QUERY,
            resource_type=RESOURCE_TYPE,
            limit=LIMIT
        )
    )
    
    # 2. Using the chat protocol
    await ctx.send(
        HF_QUERY_AGENT_ADDRESS,
        create_text_chat(QUERY)
    )
    
    ctx.logger.info("Queries sent! Waiting for responses...")

@test_agent.on_message(model=QueryResponse)
async def handle_response(ctx: Context, sender: str, msg: QueryResponse):
    """Handle the response from the HF Query Agent using the original model"""
    try:
        ctx.logger.info(f"Received QueryResponse from {sender}")
        ctx.logger.info(f"Response contains {len(msg.results)} results")
        
        # Check if there was an error
        if msg.error:
            ctx.logger.error(f"Error in response: {msg.error}")
            console.print(f"[bold red]Error: {msg.error}[/bold red]")
            return
        
        # Display the query interpretation
        ctx.logger.info("Displaying query interpretation")
        console.print("\n[bold blue]Query Interpretation (Original Protocol):[/bold blue]")
        console.print(msg.explanation)
        
        # Display the results
        console.print(f"\n[bold green]Found {len(msg.results)} results:[/bold green]")
        
        if not msg.results:
            console.print("[bold yellow]No results found.[/bold yellow]")
            return
        
        # Log each result for debugging
        for i, model in enumerate(msg.results):
            ctx.logger.info(f"Result {i+1}: {model.id}")
        
        try:
            # Create a rich table for the results
            table = Table(
                title="Hugging Face Models (Original Protocol)",
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
            for model in msg.results:
                table.add_row(
                    model.id,
                    f"{model.downloads:,}" if model.downloads else "N/A",
                    model.library_name or "N/A",
                    model.pipeline_tag or "N/A",
                    model.inference or "N/A",
                    "✓" if model.safetensors == "true" else "✗",
                    "✓" if model.gated == "true" else "✗"
                )
            
            # Print the table
            ctx.logger.info("Printing results table")
            console.print(table)
        except Exception as table_error:
            # Fallback to simple text output if rich table fails
            ctx.logger.error(f"Error creating table: {str(table_error)}")
            console.print("\n[bold blue]Results:[/bold blue]")
            for i, model in enumerate(msg.results, 1):
                console.print(f"\n{i}. {model.id}")
                console.print(f"   Downloads: {model.downloads if model.downloads else 'N/A'}")
                console.print(f"   Library: {model.library_name or 'N/A'}")
                console.print(f"   Task: {model.pipeline_tag or 'N/A'}")
                console.print(f"   Inference: {model.inference or 'N/A'}")
                console.print(f"   Safetensors: {'Yes' if model.safetensors == 'true' else 'No'}")
                console.print(f"   Gated: {'Yes' if model.gated == 'true' else 'No'}")
    except Exception as e:
        # Catch any other exceptions
        ctx.logger.error(f"Error handling response: {str(e)}")
        console.print(f"[bold red]Error handling response: {str(e)}[/bold red]")

@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle the response from the HF Query Agent using the chat protocol"""
    try:
        ctx.logger.info(f"Received ChatMessage from {sender}")
        
        # Extract the text content - in v0.2.0, content items have a 'type' field
        text_content = next((item for item in msg.content if hasattr(item, 'type') and item.type == 'text'), None)
        
        if text_content:
            response_text = text_content.text
            
            # Send acknowledgement (v0.2.0 format)
            await ctx.send(
                sender,
                ChatAcknowledgement(
                    timestamp=datetime.now(),
                    acknowledged_msg_id=msg.msg_id
                )
            )
            
            # Display the response
            console.print("\n[bold blue]Response from Chat Protocol:[/bold blue]")
            console.print(response_text)
        else:
            ctx.logger.warning("Received ChatMessage with no text content")
    except Exception as e:
        ctx.logger.error(f"Error handling chat message: {str(e)}")
        console.print(f"[bold red]Error handling chat message: {str(e)}[/bold red]")

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle acknowledgements from the HF Query Agent"""
    ctx.logger.info(f"Got an acknowledgement from {sender} for {msg.acknowledged_msg_id}")

# Include the chat protocol in the agent
test_agent.include(chat_proto)

if __name__ == "__main__":
    test_agent.run()