"""
Chat endpoints for the RAG system.
"""
import uuid
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from ...config.settings import settings
from ...core.rag_pipeline import RAGPipeline
from ...models.chat import (
    ChatRequest, ChatResponse, ChatMessage, MessageRole,
    ConversationHistory, ConversationListResponse, StreamChatChunk
)
from ...models.rag import RAGQuery, RAGConfig
from ..deps import get_rag_pipeline

router = APIRouter()

# In-memory conversation storage (replace with persistent storage in production)
conversations: Dict[str, ConversationHistory] = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Process a chat message and return a response.
    """
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Get or create conversation
        if conversation_id not in conversations:
            conversations[conversation_id] = ConversationHistory(
                conversation_id=conversation_id,
                messages=[]
            )
        
        conversation = conversations[conversation_id]
        
        # Add user message to conversation
        user_message = ChatMessage(
            role=MessageRole.USER,
            content=request.message,
            metadata=request.context
        )
        conversation.messages.append(user_message)
        
        if request.use_rag:
            # Process with RAG pipeline
            rag_query = RAGQuery(
                query=request.message,
                config=RAGConfig(
                    top_k=settings.retrieval_top_k,
                    similarity_threshold=settings.similarity_threshold,
                    max_context_length=settings.max_context_length
                ),
                user_id=request.user_id,
                conversation_id=conversation_id
            )
            
            rag_response = await rag_pipeline.query(rag_query)
            
            # Create assistant message
            assistant_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=rag_response.answer,
                metadata={
                    "sources": rag_response.sources,
                    "retrieval_time": rag_response.context.retrieval_time,
                    "generation_time": rag_response.generation_time
                }
            )
            
            # Add to conversation
            conversation.messages.append(assistant_message)
            
            return ChatResponse(
                status="success",
                message="Chat response generated successfully",
                conversation_id=conversation_id,
                response=assistant_message,
                sources=rag_response.sources,
                usage=rag_response.usage,
                response_time=rag_response.total_time
            )
        else:
            # Direct LLM response without RAG
            from ...providers.base_provider import LLMMessage
            from ...providers.provider_factory import create_llm_provider
            
            llm_provider = create_llm_provider()
            
            # Build conversation context
            llm_messages = []
            for msg in conversation.messages[-10:]:  # Last 10 messages for context
                llm_messages.append(LLMMessage(role=msg.role.value, content=msg.content))
            
            # Generate response
            llm_response = await llm_provider.generate(
                messages=llm_messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            # Create assistant message
            assistant_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=llm_response.content,
                metadata={"direct_llm": True}
            )
            
            # Add to conversation
            conversation.messages.append(assistant_message)
            
            return ChatResponse(
                status="success",
                message="Chat response generated successfully",
                conversation_id=conversation_id,
                response=assistant_message,
                usage=llm_response.usage,
                response_time=llm_response.response_time
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Process a chat message and return a streaming response.
    """
    try:
        if not request.use_rag:
            # For now, only support RAG streaming
            # Direct LLM streaming would require more complex implementation
            raise HTTPException(
                status_code=400, 
                detail="Streaming is currently only supported with RAG enabled"
            )
        
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        async def generate():
            try:
                # Process with RAG pipeline first to get context
                rag_query = RAGQuery(
                    query=request.message,
                    config=RAGConfig(
                        top_k=settings.retrieval_top_k,
                        similarity_threshold=settings.similarity_threshold,
                        max_context_length=settings.max_context_length
                    ),
                    user_id=request.user_id,
                    conversation_id=conversation_id
                )
                
                rag_response = await rag_pipeline.query(rag_query)
                
                # Stream the response in chunks
                content = rag_response.answer
                chunk_size = 50  # Characters per chunk
                
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    is_final = i + chunk_size >= len(content)
                    
                    stream_chunk = StreamChatChunk(
                        conversation_id=conversation_id,
                        chunk=chunk,
                        is_final=is_final,
                        sources=rag_response.sources if is_final else None,
                        usage=rag_response.usage if is_final else None
                    )
                    
                    yield f"data: {stream_chunk.json()}\\n\\n"
                
                yield "data: [DONE]\\n\\n"
                
            except Exception as e:
                error_chunk = StreamChatChunk(
                    conversation_id=conversation_id,
                    chunk=f"Error: {str(e)}",
                    is_final=True
                )
                yield f"data: {error_chunk.json()}\\n\\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming chat failed: {str(e)}")


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(user_id: str = None):
    """
    List all conversations, optionally filtered by user ID.
    """
    try:
        conversation_summaries = []
        
        for conv_id, conversation in conversations.items():
            # Filter by user if specified
            if user_id:
                # Check if any message in conversation has this user_id
                has_user = any(
                    msg.metadata and msg.metadata.get("user_id") == user_id
                    for msg in conversation.messages
                )
                if not has_user:
                    continue
            
            # Create summary
            summary = {
                "conversation_id": conv_id,
                "user_id": user_id,
                "message_count": len(conversation.messages),
                "created_at": conversation.messages[0].timestamp if conversation.messages else None,
                "updated_at": conversation.messages[-1].timestamp if conversation.messages else None,
                "last_message": conversation.messages[-1].content[:100] if conversation.messages else None
            }
            conversation_summaries.append(summary)
        
        return ConversationListResponse(
            status="success",
            message="Conversations retrieved successfully",
            conversations=conversation_summaries,
            total=len(conversation_summaries)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(conversation_id: str):
    """
    Get a specific conversation by ID.
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversations[conversation_id]


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation by ID.
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[conversation_id]
    
    return {"status": "success", "message": "Conversation deleted successfully"}


@router.delete("/conversations")
async def clear_all_conversations():
    """
    Clear all conversations.
    """
    global conversations
    conversations = {}
    
    return {"status": "success", "message": "All conversations cleared successfully"}
