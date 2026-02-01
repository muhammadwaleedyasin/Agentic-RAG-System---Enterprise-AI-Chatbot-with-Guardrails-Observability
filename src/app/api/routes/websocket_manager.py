"""
WebSocket management utilities and handlers.
"""
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Enhanced WebSocket connection manager with room support."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, str] = {}  # user_id -> connection_id
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.rooms: Dict[str, List[str]] = {}  # room_name -> [connection_ids]
        
    async def connect(self, websocket: WebSocket, connection_id: str, 
                     user_id: str = None, metadata: Dict[str, Any] = None):
        """Connect a WebSocket with optional user ID and metadata."""
        await websocket.accept()
        
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        if user_id:
            self.user_connections[user_id] = connection_id
            
        logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
    
    def disconnect(self, connection_id: str):
        """Disconnect a WebSocket and clean up."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
        if connection_id in self.connection_metadata:
            user_id = self.connection_metadata[connection_id].get("user_id")
            del self.connection_metadata[connection_id]
            
            # Remove from user connections
            if user_id and user_id in self.user_connections:
                del self.user_connections[user_id]
        
        # Remove from all rooms
        for room_name, connection_ids in self.rooms.items():
            if connection_id in connection_ids:
                connection_ids.remove(connection_id)
                
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(message)
                return True
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
                return False
        return False
    
    async def send_to_user(self, message: str, user_id: str):
        """Send a message to a specific user."""
        if user_id in self.user_connections:
            connection_id = self.user_connections[user_id]
            return await self.send_personal_message(message, connection_id)
        return False
    
    async def broadcast(self, message: str, exclude_connections: List[str] = None):
        """Broadcast a message to all active connections."""
        exclude_connections = exclude_connections or []
        disconnected = []
        
        for connection_id, websocket in self.active_connections.items():
            if connection_id not in exclude_connections:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {connection_id}: {e}")
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
            
        return len(self.active_connections) - len(disconnected)
    
    def join_room(self, connection_id: str, room_name: str):
        """Add a connection to a room."""
        if connection_id not in self.active_connections:
            return False
            
        if room_name not in self.rooms:
            self.rooms[room_name] = []
            
        if connection_id not in self.rooms[room_name]:
            self.rooms[room_name].append(connection_id)
            logger.info(f"Connection {connection_id} joined room {room_name}")
            return True
        
        return False
    
    def leave_room(self, connection_id: str, room_name: str):
        """Remove a connection from a room."""
        if room_name in self.rooms and connection_id in self.rooms[room_name]:
            self.rooms[room_name].remove(connection_id)
            logger.info(f"Connection {connection_id} left room {room_name}")
            
            # Clean up empty rooms
            if not self.rooms[room_name]:
                del self.rooms[room_name]
            
            return True
        return False
    
    async def broadcast_to_room(self, message: str, room_name: str, 
                               exclude_connections: List[str] = None):
        """Broadcast a message to all connections in a room."""
        if room_name not in self.rooms:
            return 0
            
        exclude_connections = exclude_connections or []
        sent_count = 0
        disconnected = []
        
        for connection_id in self.rooms[room_name]:
            if connection_id not in exclude_connections:
                if connection_id in self.active_connections:
                    try:
                        await self.active_connections[connection_id].send_text(message)
                        sent_count += 1
                    except Exception as e:
                        logger.error(f"Error sending to room {room_name}, connection {connection_id}: {e}")
                        disconnected.append(connection_id)
                else:
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
            
        return sent_count
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a connection."""
        return self.connection_metadata.get(connection_id)
    
    def get_user_connections(self) -> Dict[str, str]:
        """Get all user ID to connection ID mappings."""
        return self.user_connections.copy()
    
    def get_room_members(self, room_name: str) -> List[str]:
        """Get all connection IDs in a room."""
        return self.rooms.get(room_name, []).copy()
    
    def get_active_connections_count(self) -> int:
        """Get count of active connections."""
        return len(self.active_connections)
    
    def get_rooms_info(self) -> Dict[str, int]:
        """Get room names and member counts."""
        return {room_name: len(connections) for room_name, connections in self.rooms.items()}


class MessageHandler:
    """Handle different types of WebSocket messages."""
    
    def __init__(self, manager: WebSocketManager):
        self.manager = manager
        self.handlers = {
            "ping": self.handle_ping,
            "join_room": self.handle_join_room,
            "leave_room": self.handle_leave_room,
            "room_message": self.handle_room_message,
            "direct_message": self.handle_direct_message,
            "get_status": self.handle_get_status
        }
    
    async def handle_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Route message to appropriate handler."""
        message_type = message_data.get("type")
        
        if message_type in self.handlers:
            try:
                await self.handlers[message_type](connection_id, message_data)
            except Exception as e:
                logger.error(f"Error handling message type {message_type}: {e}")
                await self.send_error(connection_id, f"Error processing {message_type}: {str(e)}")
        else:
            await self.send_error(connection_id, f"Unknown message type: {message_type}")
    
    async def handle_ping(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle ping messages."""
        response = {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat(),
            "connection_id": connection_id
        }
        await self.manager.send_personal_message(json.dumps(response), connection_id)
    
    async def handle_join_room(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle room join requests."""
        room_name = message_data.get("room")
        if not room_name:
            await self.send_error(connection_id, "Room name required")
            return
        
        success = self.manager.join_room(connection_id, room_name)
        
        response = {
            "type": "room_joined" if success else "room_join_failed",
            "room": room_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.manager.send_personal_message(json.dumps(response), connection_id)
        
        # Notify other room members
        if success:
            notification = {
                "type": "user_joined_room",
                "room": room_name,
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.manager.broadcast_to_room(
                json.dumps(notification), 
                room_name, 
                exclude_connections=[connection_id]
            )
    
    async def handle_leave_room(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle room leave requests."""
        room_name = message_data.get("room")
        if not room_name:
            await self.send_error(connection_id, "Room name required")
            return
        
        success = self.manager.leave_room(connection_id, room_name)
        
        response = {
            "type": "room_left" if success else "room_leave_failed",
            "room": room_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.manager.send_personal_message(json.dumps(response), connection_id)
        
        # Notify other room members
        if success:
            notification = {
                "type": "user_left_room",
                "room": room_name,
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.manager.broadcast_to_room(json.dumps(notification), room_name)
    
    async def handle_room_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle messages to be broadcast to a room."""
        room_name = message_data.get("room")
        message_content = message_data.get("message")
        
        if not room_name or not message_content:
            await self.send_error(connection_id, "Room and message required")
            return
        
        # Check if connection is in the room
        if connection_id not in self.manager.get_room_members(room_name):
            await self.send_error(connection_id, "Not a member of this room")
            return
        
        # Broadcast message to room
        broadcast_message = {
            "type": "room_message",
            "room": room_name,
            "from_connection": connection_id,
            "message": message_content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        sent_count = await self.manager.broadcast_to_room(
            json.dumps(broadcast_message), 
            room_name,
            exclude_connections=[connection_id]
        )
        
        # Send confirmation to sender
        confirmation = {
            "type": "message_sent",
            "room": room_name,
            "recipients": sent_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.manager.send_personal_message(json.dumps(confirmation), connection_id)
    
    async def handle_direct_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle direct messages between users."""
        target_user = message_data.get("to_user")
        message_content = message_data.get("message")
        
        if not target_user or not message_content:
            await self.send_error(connection_id, "Target user and message required")
            return
        
        # Get sender info
        sender_info = self.manager.get_connection_info(connection_id)
        sender_user = sender_info.get("user_id") if sender_info else None
        
        # Send message to target user
        direct_message = {
            "type": "direct_message",
            "from_user": sender_user,
            "from_connection": connection_id,
            "message": message_content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = await self.manager.send_to_user(json.dumps(direct_message), target_user)
        
        # Send confirmation to sender
        confirmation = {
            "type": "message_sent" if success else "message_failed",
            "to_user": target_user,
            "delivered": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.manager.send_personal_message(json.dumps(confirmation), connection_id)
    
    async def handle_get_status(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle status requests."""
        connection_info = self.manager.get_connection_info(connection_id)
        
        status = {
            "type": "status_response",
            "connection_id": connection_id,
            "connection_info": connection_info,
            "active_connections": self.manager.get_active_connections_count(),
            "rooms": self.manager.get_rooms_info(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.manager.send_personal_message(json.dumps(status), connection_id)
    
    async def send_error(self, connection_id: str, error_message: str):
        """Send error message to connection."""
        error_response = {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.manager.send_personal_message(json.dumps(error_response), connection_id)


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
message_handler = MessageHandler(websocket_manager)


async def handle_websocket_connection(websocket: WebSocket, connection_id: str, 
                                    user_id: str = None, metadata: Dict[str, Any] = None):
    """Handle a WebSocket connection lifecycle."""
    await websocket_manager.connect(websocket, connection_id, user_id, metadata)
    
    try:
        # Send welcome message
        welcome_message = {
            "type": "connection_established",
            "connection_id": connection_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "WebSocket connection established successfully"
        }
        await websocket_manager.send_personal_message(json.dumps(welcome_message), connection_id)
        
        # Message handling loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle message
                await message_handler.handle_message(connection_id, message_data)
                
            except json.JSONDecodeError:
                await message_handler.send_error(connection_id, "Invalid JSON message")
            except Exception as e:
                logger.error(f"Error processing message from {connection_id}: {e}")
                await message_handler.send_error(connection_id, "Message processing error")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket {connection_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket {connection_id} error: {e}")
    finally:
        websocket_manager.disconnect(connection_id)


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    return websocket_manager


def get_message_handler() -> MessageHandler:
    """Get the global message handler instance."""
    return message_handler