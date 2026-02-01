"""
Horizontal Scaling Manager for Production-Scale RAG Systems

This module provides comprehensive horizontal scaling capabilities including
distributed processing, load balancing, auto-scaling, and fault tolerance
for large-scale document retrieval systems.
"""

import asyncio
import logging
import time
import json
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict, deque
import weakref
import socket
import uuid
import pickle
from pathlib import Path

# External dependencies
import redis
import aiohttp
import consul
import etcd3
import psutil
import numpy as np
from kubernetes import client, config
import docker

logger = logging.getLogger(__name__)

@dataclass
class NodeInfo:
    """Information about a scaling node"""
    node_id: str
    host: str
    port: int
    status: str  # "active", "inactive", "error", "scaling"
    cpu_usage: float
    memory_usage: float
    load_score: float
    capabilities: List[str]
    last_heartbeat: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions"""
    total_nodes: int
    active_nodes: int
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_response_time: float
    request_rate: float
    queue_length: int
    error_rate: float
    timestamp: float

@dataclass
class ScalingConfig:
    """Configuration for scaling manager"""
    # Auto-scaling thresholds
    cpu_scale_up_threshold: float = 80.0
    cpu_scale_down_threshold: float = 30.0
    memory_scale_up_threshold: float = 80.0
    memory_scale_down_threshold: float = 30.0
    
    # Response time thresholds
    response_time_scale_up_threshold: float = 5.0
    response_time_scale_down_threshold: float = 1.0
    
    # Queue length thresholds
    queue_length_scale_up_threshold: int = 100
    queue_length_scale_down_threshold: int = 10
    
    # Scaling behavior
    min_nodes: int = 2
    max_nodes: int = 20
    scale_up_cooldown: int = 300  # 5 minutes
    scale_down_cooldown: int = 600  # 10 minutes
    
    # Health check settings
    health_check_interval: int = 30
    node_timeout: int = 120
    max_retries: int = 3
    
    # Load balancing
    load_balancing_strategy: str = "least_connections"  # "round_robin", "weighted_round_robin", "least_connections", "consistent_hash"
    
    # Service discovery
    service_discovery_backend: str = "redis"  # "redis", "consul", "etcd", "kubernetes"
    service_discovery_config: Dict[str, Any] = field(default_factory=dict)

class ServiceDiscovery(ABC):
    """Abstract base class for service discovery backends"""
    
    @abstractmethod
    async def register_node(self, node: NodeInfo) -> bool:
        """Register a node with service discovery"""
        pass
    
    @abstractmethod
    async def deregister_node(self, node_id: str) -> bool:
        """Deregister a node from service discovery"""
        pass
    
    @abstractmethod
    async def get_nodes(self) -> List[NodeInfo]:
        """Get all registered nodes"""
        pass
    
    @abstractmethod
    async def update_node_status(self, node_id: str, status: str, metrics: Dict[str, Any]) -> bool:
        """Update node status and metrics"""
        pass

class RedisServiceDiscovery(ServiceDiscovery):
    """Redis-based service discovery"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", key_prefix: str = "rag_nodes:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        try:
            self._redis = redis.from_url(self.redis_url, decode_responses=False)
            self._redis.ping()
            logger.info(f"Connected to Redis service discovery at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis = None
    
    async def register_node(self, node: NodeInfo) -> bool:
        """Register a node with Redis"""
        if not self._redis:
            return False
        
        try:
            key = f"{self.key_prefix}{node.node_id}"
            data = pickle.dumps(node)
            self._redis.setex(key, 300, data)  # 5-minute TTL
            return True
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
            return False
    
    async def deregister_node(self, node_id: str) -> bool:
        """Deregister a node from Redis"""
        if not self._redis:
            return False
        
        try:
            key = f"{self.key_prefix}{node_id}"
            self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to deregister node: {e}")
            return False
    
    async def get_nodes(self) -> List[NodeInfo]:
        """Get all registered nodes from Redis"""
        if not self._redis:
            return []
        
        try:
            pattern = f"{self.key_prefix}*"
            keys = self._redis.keys(pattern)
            nodes = []
            
            for key in keys:
                data = self._redis.get(key)
                if data:
                    node = pickle.loads(data)
                    nodes.append(node)
            
            return nodes
        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")
            return []
    
    async def update_node_status(self, node_id: str, status: str, metrics: Dict[str, Any]) -> bool:
        """Update node status in Redis"""
        if not self._redis:
            return False
        
        try:
            key = f"{self.key_prefix}{node_id}"
            data = self._redis.get(key)
            
            if data:
                node = pickle.loads(data)
                node.status = status
                node.cpu_usage = metrics.get("cpu_usage", 0.0)
                node.memory_usage = metrics.get("memory_usage", 0.0)
                node.load_score = metrics.get("load_score", 0.0)
                node.last_heartbeat = time.time()
                
                updated_data = pickle.dumps(node)
                self._redis.setex(key, 300, updated_data)
                return True
        except Exception as e:
            logger.error(f"Failed to update node status: {e}")
        
        return False

class LoadBalancer:
    """Load balancer for distributing requests across nodes"""
    
    def __init__(self, strategy: str = "least_connections"):
        self.strategy = strategy
        self._connection_counts: Dict[str, int] = defaultdict(int)
        self._round_robin_counter = 0
        self._weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._lock = threading.RLock()
    
    def select_node(self, nodes: List[NodeInfo], request_hash: Optional[str] = None) -> Optional[NodeInfo]:
        """Select optimal node based on load balancing strategy"""
        active_nodes = [node for node in nodes if node.status == "active"]
        
        if not active_nodes:
            return None
        
        with self._lock:
            if self.strategy == "round_robin":
                return self._round_robin_selection(active_nodes)
            elif self.strategy == "weighted_round_robin":
                return self._weighted_round_robin_selection(active_nodes)
            elif self.strategy == "least_connections":
                return self._least_connections_selection(active_nodes)
            elif self.strategy == "consistent_hash":
                return self._consistent_hash_selection(active_nodes, request_hash)
            else:
                # Default to round robin
                return self._round_robin_selection(active_nodes)
    
    def _round_robin_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Round robin node selection"""
        selected = nodes[self._round_robin_counter % len(nodes)]
        self._round_robin_counter += 1
        return selected
    
    def _weighted_round_robin_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Weighted round robin based on node capabilities"""
        # Calculate weights based on inverse load score
        weighted_nodes = []
        for node in nodes:
            weight = self._weights[node.node_id] / max(node.load_score, 0.1)
            weighted_nodes.extend([node] * int(weight * 10))
        
        if weighted_nodes:
            selected = weighted_nodes[self._round_robin_counter % len(weighted_nodes)]
            self._round_robin_counter += 1
            return selected
        else:
            return nodes[0]
    
    def _least_connections_selection(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node with least active connections"""
        min_connections = min(self._connection_counts[node.node_id] for node in nodes)
        candidates = [node for node in nodes 
                     if self._connection_counts[node.node_id] == min_connections]
        
        # Among nodes with same connection count, prefer lower load score
        selected = min(candidates, key=lambda n: n.load_score)
        return selected
    
    def _consistent_hash_selection(self, nodes: List[NodeInfo], request_hash: Optional[str]) -> NodeInfo:
        """Consistent hash-based selection"""
        if not request_hash:
            return self._round_robin_selection(nodes)
        
        # Simple consistent hashing
        hash_value = int(hashlib.md5(request_hash.encode()).hexdigest(), 16)
        node_index = hash_value % len(nodes)
        return nodes[node_index]
    
    def on_request_start(self, node_id: str):
        """Track request start for connection counting"""
        with self._lock:
            self._connection_counts[node_id] += 1
    
    def on_request_end(self, node_id: str):
        """Track request end for connection counting"""
        with self._lock:
            self._connection_counts[node_id] = max(0, self._connection_counts[node_id] - 1)
    
    def update_node_weight(self, node_id: str, weight: float):
        """Update node weight for weighted strategies"""
        with self._lock:
            self._weights[node_id] = weight

class AutoScaler:
    """Auto-scaling logic based on metrics and thresholds"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self._last_scale_up = 0.0
        self._last_scale_down = 0.0
        self._scaling_decisions = deque(maxlen=100)
    
    def should_scale_up(self, metrics: ScalingMetrics) -> Tuple[bool, str]:
        """Determine if system should scale up"""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self._last_scale_up < self.config.scale_up_cooldown:
            return False, "Scale-up cooldown active"
        
        # Check maximum nodes
        if metrics.active_nodes >= self.config.max_nodes:
            return False, "Maximum nodes reached"
        
        reasons = []
        
        # CPU threshold
        if metrics.avg_cpu_usage > self.config.cpu_scale_up_threshold:
            reasons.append(f"CPU usage {metrics.avg_cpu_usage:.1f}% > {self.config.cpu_scale_up_threshold}%")
        
        # Memory threshold
        if metrics.avg_memory_usage > self.config.memory_scale_up_threshold:
            reasons.append(f"Memory usage {metrics.avg_memory_usage:.1f}% > {self.config.memory_scale_up_threshold}%")
        
        # Response time threshold
        if metrics.avg_response_time > self.config.response_time_scale_up_threshold:
            reasons.append(f"Response time {metrics.avg_response_time:.2f}s > {self.config.response_time_scale_up_threshold}s")
        
        # Queue length threshold
        if metrics.queue_length > self.config.queue_length_scale_up_threshold:
            reasons.append(f"Queue length {metrics.queue_length} > {self.config.queue_length_scale_up_threshold}")
        
        if len(reasons) >= 2:  # Require multiple indicators
            self._last_scale_up = current_time
            reason = "; ".join(reasons)
            self._scaling_decisions.append({
                "action": "scale_up",
                "timestamp": current_time,
                "reason": reason,
                "metrics": metrics._asdict()
            })
            return True, reason
        
        return False, "Scale-up thresholds not met"
    
    def should_scale_down(self, metrics: ScalingMetrics) -> Tuple[bool, str]:
        """Determine if system should scale down"""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self._last_scale_down < self.config.scale_down_cooldown:
            return False, "Scale-down cooldown active"
        
        # Check minimum nodes
        if metrics.active_nodes <= self.config.min_nodes:
            return False, "Minimum nodes reached"
        
        reasons = []
        
        # CPU threshold
        if metrics.avg_cpu_usage < self.config.cpu_scale_down_threshold:
            reasons.append(f"CPU usage {metrics.avg_cpu_usage:.1f}% < {self.config.cpu_scale_down_threshold}%")
        
        # Memory threshold
        if metrics.avg_memory_usage < self.config.memory_scale_down_threshold:
            reasons.append(f"Memory usage {metrics.avg_memory_usage:.1f}% < {self.config.memory_scale_down_threshold}%")
        
        # Response time threshold
        if metrics.avg_response_time < self.config.response_time_scale_down_threshold:
            reasons.append(f"Response time {metrics.avg_response_time:.2f}s < {self.config.response_time_scale_down_threshold}s")
        
        # Queue length threshold
        if metrics.queue_length < self.config.queue_length_scale_down_threshold:
            reasons.append(f"Queue length {metrics.queue_length} < {self.config.queue_length_scale_down_threshold}")
        
        if len(reasons) >= 3:  # Be more conservative about scaling down
            self._last_scale_down = current_time
            reason = "; ".join(reasons)
            self._scaling_decisions.append({
                "action": "scale_down",
                "timestamp": current_time,
                "reason": reason,
                "metrics": metrics._asdict()
            })
            return True, reason
        
        return False, "Scale-down thresholds not met"
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling decision history"""
        cutoff_time = time.time() - (hours * 3600)
        return [
            decision for decision in self._scaling_decisions
            if decision["timestamp"] > cutoff_time
        ]

class HealthChecker:
    """Health monitoring for scaling nodes"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self._health_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    async def check_node_health(self, node: NodeInfo) -> Dict[str, Any]:
        """Check health of a specific node"""
        health_data = {
            "node_id": node.node_id,
            "status": "unknown",
            "response_time": float('inf'),
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "error": None,
            "timestamp": time.time()
        }
        
        try:
            start_time = time.time()
            
            # HTTP health check
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                health_url = f"http://{node.host}:{node.port}/health"
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data["status"] = "healthy"
                        response_data = await response.json()
                        health_data.update(response_data)
                    else:
                        health_data["status"] = "unhealthy"
                        health_data["error"] = f"HTTP {response.status}"
            
            health_data["response_time"] = time.time() - start_time
            
        except asyncio.TimeoutError:
            health_data["status"] = "timeout"
            health_data["error"] = "Health check timeout"
        except Exception as e:
            health_data["status"] = "error"
            health_data["error"] = str(e)
        
        # Cache health data
        with self._lock:
            self._health_cache[node.node_id] = health_data
        
        return health_data
    
    async def check_all_nodes(self, nodes: List[NodeInfo]) -> Dict[str, Dict[str, Any]]:
        """Check health of all nodes concurrently"""
        tasks = [self.check_node_health(node) for node in nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_results = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                node_id = nodes[i].node_id
                health_results[node_id] = {
                    "node_id": node_id,
                    "status": "error",
                    "error": str(result),
                    "timestamp": time.time()
                }
            else:
                health_results[result["node_id"]] = result
        
        return health_results
    
    def get_cached_health(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get cached health data for a node"""
        with self._lock:
            return self._health_cache.get(node_id)

class HorizontalScalingManager:
    """Main horizontal scaling manager"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        
        # Initialize components
        self.service_discovery = self._create_service_discovery()
        self.load_balancer = LoadBalancer(config.load_balancing_strategy)
        self.auto_scaler = AutoScaler(config)
        self.health_checker = HealthChecker(config)
        
        # State management
        self._nodes: Dict[str, NodeInfo] = {}
        self._metrics_history: deque = deque(maxlen=1000)
        self._request_queue: asyncio.Queue = asyncio.Queue()
        
        # Background tasks
        self._monitoring_task = None
        self._scaling_task = None
        self._health_check_task = None
        self._running = False
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "scaling_events": 0
        }
    
    def _create_service_discovery(self) -> ServiceDiscovery:
        """Create service discovery backend"""
        backend = self.config.service_discovery_backend
        
        if backend == "redis":
            redis_url = self.config.service_discovery_config.get("redis_url", "redis://localhost:6379")
            return RedisServiceDiscovery(redis_url)
        elif backend == "consul":
            # Would implement ConsulServiceDiscovery
            raise NotImplementedError("Consul service discovery not implemented")
        elif backend == "etcd":
            # Would implement EtcdServiceDiscovery
            raise NotImplementedError("Etcd service discovery not implemented")
        elif backend == "kubernetes":
            # Would implement KubernetesServiceDiscovery
            raise NotImplementedError("Kubernetes service discovery not implemented")
        else:
            raise ValueError(f"Unknown service discovery backend: {backend}")
    
    async def start(self):
        """Start the scaling manager"""
        if self._running:
            logger.warning("Scaling manager already running")
            return
        
        self._running = True
        
        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Horizontal scaling manager started")
    
    async def stop(self):
        """Stop the scaling manager"""
        self._running = False
        
        # Cancel background tasks
        for task in [self._monitoring_task, self._scaling_task, self._health_check_task]:
            if task and not task.done():
                task.cancel()
        
        logger.info("Horizontal scaling manager stopped")
    
    async def register_node(self, host: str, port: int, capabilities: List[str] = None) -> str:
        """Register a new node"""
        node_id = str(uuid.uuid4())
        
        node = NodeInfo(
            node_id=node_id,
            host=host,
            port=port,
            status="active",
            cpu_usage=0.0,
            memory_usage=0.0,
            load_score=0.0,
            capabilities=capabilities or [],
            last_heartbeat=time.time()
        )
        
        # Register with service discovery
        success = await self.service_discovery.register_node(node)
        
        if success:
            self._nodes[node_id] = node
            logger.info(f"Registered node {node_id} at {host}:{port}")
            return node_id
        else:
            logger.error(f"Failed to register node at {host}:{port}")
            raise RuntimeError("Node registration failed")
    
    async def deregister_node(self, node_id: str) -> bool:
        """Deregister a node"""
        success = await self.service_discovery.deregister_node(node_id)
        
        if success:
            self._nodes.pop(node_id, None)
            logger.info(f"Deregistered node {node_id}")
        
        return success
    
    async def route_request(self, request_data: Any, request_hash: Optional[str] = None) -> Tuple[Any, str]:
        """Route request to optimal node"""
        # Get current nodes
        nodes = await self.service_discovery.get_nodes()
        
        # Select optimal node
        selected_node = self.load_balancer.select_node(nodes, request_hash)
        
        if not selected_node:
            raise RuntimeError("No active nodes available")
        
        # Track request start
        self.load_balancer.on_request_start(selected_node.node_id)
        
        try:
            # Forward request to selected node
            result = await self._forward_request(selected_node, request_data)
            
            self._stats["successful_requests"] += 1
            return result, selected_node.node_id
        
        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error(f"Request failed on node {selected_node.node_id}: {e}")
            raise
        
        finally:
            # Track request end
            self.load_balancer.on_request_end(selected_node.node_id)
            self._stats["total_requests"] += 1
    
    async def _forward_request(self, node: NodeInfo, request_data: Any) -> Any:
        """Forward request to specific node"""
        # This is a placeholder - in practice, this would make HTTP requests
        # or use other protocols to communicate with the node
        
        start_time = time.time()
        
        # Simulate request processing
        await asyncio.sleep(0.1)  # Simulate network latency
        
        response_time = time.time() - start_time
        
        # Update statistics
        total_requests = self._stats["total_requests"]
        if total_requests > 0:
            self._stats["avg_response_time"] = (
                (self._stats["avg_response_time"] * total_requests + response_time) 
                / (total_requests + 1)
            )
        else:
            self._stats["avg_response_time"] = response_time
        
        return {"result": "processed", "node_id": node.node_id, "response_time": response_time}
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Collect metrics
                metrics = await self._collect_metrics()
                self._metrics_history.append(metrics)
                
                # Log metrics
                logger.debug(f"Metrics: {metrics.active_nodes} nodes, "
                           f"CPU: {metrics.avg_cpu_usage:.1f}%, "
                           f"Memory: {metrics.avg_memory_usage:.1f}%, "
                           f"Response: {metrics.avg_response_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _scaling_loop(self):
        """Background auto-scaling loop"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self._metrics_history:
                    continue
                
                # Get latest metrics
                latest_metrics = self._metrics_history[-1]
                
                # Check for scaling decisions
                should_scale_up, up_reason = self.auto_scaler.should_scale_up(latest_metrics)
                should_scale_down, down_reason = self.auto_scaler.should_scale_down(latest_metrics)
                
                if should_scale_up:
                    logger.info(f"Scaling up: {up_reason}")
                    await self._scale_up()
                    self._stats["scaling_events"] += 1
                elif should_scale_down:
                    logger.info(f"Scaling down: {down_reason}")
                    await self._scale_down()
                    self._stats["scaling_events"] += 1
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Get all nodes
                nodes = await self.service_discovery.get_nodes()
                
                if nodes:
                    # Check health of all nodes
                    health_results = await self.health_checker.check_all_nodes(nodes)
                    
                    # Update node statuses
                    for node_id, health_data in health_results.items():
                        if health_data["status"] in ["unhealthy", "timeout", "error"]:
                            await self._handle_unhealthy_node(node_id, health_data)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        nodes = await self.service_discovery.get_nodes()
        active_nodes = [node for node in nodes if node.status == "active"]
        
        if active_nodes:
            avg_cpu = sum(node.cpu_usage for node in active_nodes) / len(active_nodes)
            avg_memory = sum(node.memory_usage for node in active_nodes) / len(active_nodes)
        else:
            avg_cpu = 0.0
            avg_memory = 0.0
        
        return ScalingMetrics(
            total_nodes=len(nodes),
            active_nodes=len(active_nodes),
            avg_cpu_usage=avg_cpu,
            avg_memory_usage=avg_memory,
            avg_response_time=self._stats["avg_response_time"],
            request_rate=self._calculate_request_rate(),
            queue_length=self._request_queue.qsize(),
            error_rate=self._calculate_error_rate(),
            timestamp=time.time()
        )
    
    def _calculate_request_rate(self) -> float:
        """Calculate current request rate"""
        # Simple implementation - in practice, would use sliding window
        return self._stats["total_requests"] / max(time.time() - 3600, 1)  # Requests per second over last hour
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        total = self._stats["total_requests"]
        if total == 0:
            return 0.0
        return self._stats["failed_requests"] / total
    
    async def _scale_up(self):
        """Scale up by adding new nodes"""
        # This is a placeholder - in practice, would:
        # 1. Use container orchestration (Docker/Kubernetes) to start new instances
        # 2. Use cloud APIs to launch new VMs
        # 3. Use auto-scaling groups
        
        logger.info("Scaling up (placeholder implementation)")
        # await self._launch_new_node()
    
    async def _scale_down(self):
        """Scale down by removing nodes"""
        # This is a placeholder - in practice, would:
        # 1. Select least utilized node
        # 2. Drain connections gracefully
        # 3. Terminate node
        
        logger.info("Scaling down (placeholder implementation)")
        # await self._terminate_node()
    
    async def _handle_unhealthy_node(self, node_id: str, health_data: Dict[str, Any]):
        """Handle unhealthy node"""
        logger.warning(f"Node {node_id} is unhealthy: {health_data.get('error', 'Unknown error')}")
        
        # Update node status to inactive
        await self.service_discovery.update_node_status(node_id, "inactive", {})
        
        # In practice, would also:
        # 1. Stop routing requests to this node
        # 2. Attempt to restart the node
        # 3. Replace the node if restart fails
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics"""
        recent_metrics = list(self._metrics_history)[-10:] if self._metrics_history else []
        
        return {
            "current_nodes": len(self._nodes),
            "active_nodes": len([n for n in self._nodes.values() if n.status == "active"]),
            "stats": self._stats,
            "recent_metrics": [m._asdict() for m in recent_metrics],
            "scaling_history": self.auto_scaler.get_scaling_history(),
            "config": {
                "min_nodes": self.config.min_nodes,
                "max_nodes": self.config.max_nodes,
                "load_balancing_strategy": self.config.load_balancing_strategy,
                "service_discovery_backend": self.config.service_discovery_backend
            }
        }

# Factory function
def create_scaling_manager(
    min_nodes: int = 2,
    max_nodes: int = 20,
    load_balancing_strategy: str = "least_connections",
    service_discovery_backend: str = "redis",
    redis_url: str = "redis://localhost:6379"
) -> HorizontalScalingManager:
    """Factory function to create scaling manager with sensible defaults"""
    
    config = ScalingConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        load_balancing_strategy=load_balancing_strategy,
        service_discovery_backend=service_discovery_backend,
        service_discovery_config={"redis_url": redis_url}
    )
    
    return HorizontalScalingManager(config)

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create scaling manager
        manager = create_scaling_manager()
        
        # Start manager
        await manager.start()
        
        # Register some nodes
        node1_id = await manager.register_node("localhost", 8001, ["search", "rerank"])
        node2_id = await manager.register_node("localhost", 8002, ["search", "embed"])
        
        # Simulate some requests
        for i in range(10):
            try:
                result, node_id = await manager.route_request(f"request_{i}")
                print(f"Request {i} routed to {node_id}: {result}")
            except Exception as e:
                print(f"Request {i} failed: {e}")
        
        # Get stats
        stats = manager.get_scaling_stats()
        print("\nScaling Stats:")
        print(json.dumps(stats, indent=2, default=str))
        
        # Cleanup
        await manager.deregister_node(node1_id)
        await manager.deregister_node(node2_id)
        await manager.stop()
    
    asyncio.run(main())