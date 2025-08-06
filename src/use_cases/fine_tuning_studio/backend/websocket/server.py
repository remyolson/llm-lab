"""
WebSocket server for real-time updates in Fine-Tuning Studio
"""

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Set

from fastapi import WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client"""

    id: str
    websocket: WebSocket
    experiment_subscriptions: Set[str]
    user_id: str
    connected_at: datetime


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocketClient] = {}
        self.experiment_subscribers: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str = None, user_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()

        if not client_id:
            client_id = str(uuid.uuid4())

        client = WebSocketClient(
            id=client_id,
            websocket=websocket,
            experiment_subscriptions=set(),
            user_id=user_id or "anonymous",
            connected_at=datetime.utcnow(),
        )

        self.active_connections[client_id] = client

        # Send connection confirmation
        await self.send_personal_message(
            {"type": "connection", "status": "connected", "client_id": client_id}, client_id
        )

        logger.info(f"Client {client_id} connected")
        return client_id

    def disconnect(self, client_id: str):
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            client = self.active_connections[client_id]

            # Remove from all experiment subscriptions
            for experiment_id in client.experiment_subscriptions:
                if experiment_id in self.experiment_subscribers:
                    self.experiment_subscribers[experiment_id].discard(client_id)
                    if not self.experiment_subscribers[experiment_id]:
                        del self.experiment_subscribers[experiment_id]

            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: dict, client_id: str):
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            client = self.active_connections[client_id]
            try:
                await client.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients"""
        disconnected_clients = []

        for client_id, client in self.active_connections.items():
            try:
                await client.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    async def broadcast_to_experiment(self, experiment_id: str, message: dict):
        """Broadcast a message to all clients subscribed to an experiment"""
        if experiment_id not in self.experiment_subscribers:
            return

        disconnected_clients = []

        for client_id in self.experiment_subscribers[experiment_id]:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to client {client_id}: {e}")
                    disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    def subscribe_to_experiment(self, client_id: str, experiment_id: str):
        """Subscribe a client to experiment updates"""
        if client_id in self.active_connections:
            client = self.active_connections[client_id]
            client.experiment_subscriptions.add(experiment_id)

            if experiment_id not in self.experiment_subscribers:
                self.experiment_subscribers[experiment_id] = set()
            self.experiment_subscribers[experiment_id].add(client_id)

            logger.info(f"Client {client_id} subscribed to experiment {experiment_id}")

    def unsubscribe_from_experiment(self, client_id: str, experiment_id: str):
        """Unsubscribe a client from experiment updates"""
        if client_id in self.active_connections:
            client = self.active_connections[client_id]
            client.experiment_subscriptions.discard(experiment_id)

            if experiment_id in self.experiment_subscribers:
                self.experiment_subscribers[experiment_id].discard(client_id)
                if not self.experiment_subscribers[experiment_id]:
                    del self.experiment_subscribers[experiment_id]

            logger.info(f"Client {client_id} unsubscribed from experiment {experiment_id}")


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint handler"""
    client_id = None

    try:
        # Accept connection
        client_id = await manager.connect(websocket)

        # Message handling loop
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            # Handle different message types
            message_type = data.get("type")

            if message_type == "subscribe":
                # Subscribe to experiment updates
                experiment_id = data.get("experiment_id")
                if experiment_id:
                    manager.subscribe_to_experiment(client_id, experiment_id)
                    await manager.send_personal_message(
                        {
                            "type": "subscription",
                            "status": "subscribed",
                            "experiment_id": experiment_id,
                        },
                        client_id,
                    )

            elif message_type == "unsubscribe":
                # Unsubscribe from experiment updates
                experiment_id = data.get("experiment_id")
                if experiment_id:
                    manager.unsubscribe_from_experiment(client_id, experiment_id)
                    await manager.send_personal_message(
                        {
                            "type": "subscription",
                            "status": "unsubscribed",
                            "experiment_id": experiment_id,
                        },
                        client_id,
                    )

            elif message_type == "preview_request":
                # Handle live preview request
                await handle_preview_request(data, client_id)

            elif message_type == "ping":
                # Respond to ping with pong
                await manager.send_personal_message(
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()}, client_id
                )

            else:
                # Echo unknown messages back
                await manager.send_personal_message(
                    {"type": "error", "message": f"Unknown message type: {message_type}"}, client_id
                )

    except WebSocketDisconnect:
        if client_id:
            manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if client_id:
            manager.disconnect(client_id)


async def handle_preview_request(data: dict, client_id: str):
    """Handle live preview request"""
    experiment_id = data.get("experiment_id")
    prompt = data.get("prompt")
    preview_type = data.get("type", "single")

    if preview_type == "comparison":
        # Simulate comparison between base and fine-tuned models
        base_response = f"Base model response to: {prompt}"
        finetuned_response = f"Fine-tuned model response to: {prompt}"

        await manager.send_personal_message(
            {
                "type": "preview_response",
                "experiment_id": experiment_id,
                "data": {
                    "type": "comparison",
                    "result": {
                        "prompt": prompt,
                        "baseResponse": base_response,
                        "fineTunedResponse": finetuned_response,
                        "metrics": {
                            "base": {"tokensPerSecond": 45.2, "latency": 450, "perplexity": 12.3},
                            "fineTuned": {
                                "tokensPerSecond": 42.8,
                                "latency": 480,
                                "perplexity": 8.7,
                            },
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                },
            },
            client_id,
        )
    else:
        # Single model response
        response = f"Model response to: {prompt}"

        await manager.send_personal_message(
            {
                "type": "preview_response",
                "experiment_id": experiment_id,
                "data": {
                    "type": "single",
                    "message": {
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.utcnow().isoformat(),
                        "model": "finetuned",
                        "metrics": {"tokensPerSecond": 42.8, "latency": 480, "perplexity": 8.7},
                    },
                },
            },
            client_id,
        )


async def broadcast_training_progress(
    experiment_id: str, epoch: int, loss: float, accuracy: float = None, progress: float = 0
):
    """Broadcast training progress to subscribed clients"""
    message = {
        "type": "training_progress",
        "data": {
            "experimentId": experiment_id,
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "progress": progress,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }

    await manager.broadcast_to_experiment(experiment_id, message)


async def broadcast_deployment_status(model_id: str, status: str, message: str = None):
    """Broadcast deployment status updates"""
    msg = {
        "type": "deployment_status",
        "data": {
            "modelId": model_id,
            "status": status,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }

    await manager.broadcast(msg)


async def broadcast_quality_update(experiment_id: str, metrics: dict):
    """Broadcast quality metrics update"""
    message = {
        "type": "quality_update",
        "data": {
            "experimentId": experiment_id,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }

    await manager.broadcast_to_experiment(experiment_id, message)


async def broadcast_notification(user_id: str, notification: dict):
    """Send notification to a specific user"""
    # Find all connections for this user
    for client_id, client in manager.active_connections.items():
        if client.user_id == user_id:
            await manager.send_personal_message(
                {"type": "notification", "data": notification}, client_id
            )


# Simulation functions for testing
async def simulate_training_progress(experiment_id: str):
    """Simulate training progress for testing"""
    for epoch in range(1, 11):
        await asyncio.sleep(2)  # Simulate training time

        loss = 3.0 - (epoch * 0.2) + (0.1 * (epoch % 2))  # Simulate decreasing loss
        accuracy = 0.5 + (epoch * 0.04)  # Simulate increasing accuracy
        progress = (epoch / 10) * 100

        await broadcast_training_progress(experiment_id, epoch, loss, accuracy, progress)

        # Send quality update every 3 epochs
        if epoch % 3 == 0:
            await broadcast_quality_update(
                experiment_id,
                {
                    "perplexity": 20 - epoch,
                    "coherence": 0.7 + (epoch * 0.02),
                    "relevance": 0.6 + (epoch * 0.03),
                },
            )


async def simulate_deployment(model_id: str):
    """Simulate deployment process"""
    stages = [
        ("preparing", "Preparing model for deployment"),
        ("building", "Building container image"),
        ("pushing", "Pushing to registry"),
        ("deploying", "Deploying to infrastructure"),
        ("testing", "Running health checks"),
        ("active", "Deployment successful"),
    ]

    for status, message in stages:
        await asyncio.sleep(3)
        await broadcast_deployment_status(model_id, status, message)


# Add WebSocket routes to FastAPI app
def setup_websocket_routes(app):
    """Add WebSocket routes to the FastAPI app"""

    @app.websocket("/ws")
    async def websocket_route(websocket: WebSocket):
        await websocket_endpoint(websocket)

    @app.post("/api/simulate/training/{experiment_id}")
    async def simulate_training(experiment_id: str):
        """Endpoint to trigger training simulation"""
        asyncio.create_task(simulate_training_progress(experiment_id))
        return {"message": "Training simulation started"}

    @app.post("/api/simulate/deployment/{model_id}")
    async def simulate_deploy(model_id: str):
        """Endpoint to trigger deployment simulation"""
        asyncio.create_task(simulate_deployment(model_id))
        return {"message": "Deployment simulation started"}


if __name__ == "__main__":
    # For standalone testing
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()
    setup_websocket_routes(app)

    # Get WebSocket server configuration from settings
    try:
        from ....config.settings import get_settings

        settings = get_settings()
        port = settings.server.websocket_port
        host = settings.server.api_host  # Use same host as API server
    except ImportError:
        # Fallback for backward compatibility
        port = 8001
        host = "0.0.0.0"

    uvicorn.run(app, host=host, port=port)
