#!/usr/bin/python3
"""
Base class for multi-client queue functionality that can be used across
different storage backends (Minio S3, Local filesystem, SSH).
Frontend version - adapted for UI usage.
"""
from abc import ABC, abstractmethod
import time
import uuid


class MultiClientQueueBase(ABC):
    """
    Abstract base class for multi-client queue implementations.
    Provides common logic for handling multiple clients while delegating
    storage-specific operations to subclasses.
    """

    def __init__(self, **kwargs):
        """Initialize the multi-client queue."""
        # Track active client queues
        self.active_clients = {}
        self.client_queues = {}

    @abstractmethod
    def _create_client_queue(self, client_id):
        """
        Create a queue instance for a specific client.
        Must be implemented by subclasses.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            A queue instance for the client
        """
        pass

    @abstractmethod
    def _discover_clients_from_storage(self):
        """
        Discover available clients from the storage backend.
        Must be implemented by subclasses.
        
        Returns:
            Set of client_ids found in storage
        """
        pass

    def discover_new_clients(self):
        """Discover new clients by checking storage backend."""
        try:
            discovered_clients = self._discover_clients_from_storage()
            
            for client_id in discovered_clients:
                if client_id not in self.active_clients:
                    self.active_clients[client_id] = True
                    print(f"[MULTI-CLIENT] Discovered new client: {client_id}")
                    
        except Exception as e:
            print(f"[MULTI-CLIENT] Error discovering clients: {e}")

    def get_client_queue(self, client_id):
        """Get or create a queue for a specific client."""
        if client_id not in self.client_queues:
            client_queue = self._create_client_queue(client_id)
            self.client_queues[client_id] = client_queue
            self.active_clients[client_id] = True
            print(f"[MULTI-CLIENT] Created queue for client: {client_id}")

        return self.client_queues[client_id]

    def get(self, save=True, check_interrupted=True, force_load=False):
        """
        Get request from any available client queue.
        Returns tuple of (data, client_id, client_queue).
        """
        # First, discover any new clients
        self.discover_new_clients()

        # Check all active client queues for requests
        for client_id, is_active in list(self.active_clients.items()):
            if not is_active:
                continue

            try:
                client_queue = self.get_client_queue(client_id)
                # Try to get data from this client's queue (non-blocking)
                data = client_queue.get_non_blocking()
                if data is not None:
                    print(f"[MULTI-CLIENT] Received request from client: {client_id}")
                    return data, client_id, client_queue
            except Exception as e:
                print(f"[MULTI-CLIENT] Error checking client {client_id}: {e}")
                # Mark client as inactive if there's an error
                self.active_clients[client_id] = False

        # If no immediate requests, wait for the first available one
        while True:
            # Periodically rediscover clients
            self.discover_new_clients()

            for client_id, is_active in list(self.active_clients.items()):
                if not is_active:
                    continue

                try:
                    client_queue = self.get_client_queue(client_id)
                    data = client_queue.get_non_blocking()
                    if data is not None:
                        print(f"[MULTI-CLIENT] Received request from client: {client_id}")
                        return data, client_id, client_queue
                except Exception as e:
                    print(f"[MULTI-CLIENT] Error checking client {client_id}: {e}")
                    self.active_clients[client_id] = False

            time.sleep(0.1)  # Small delay to prevent busy waiting

    def publish_to_client(self, data, client_id):
        """Publish response to a specific client."""
        if client_id in self.client_queues:
            client_queue = self.client_queues[client_id]
            client_queue.publish(data)
            print(f"[MULTI-CLIENT] Published response to client: {client_id}")
            self.remove_client(client_id)
            client_queue.clear(self._get_client_queue_name(client_id))
        else:
            print(f"[MULTI-CLIENT] Warning: Client {client_id} not found for publishing")

    def remove_client(self, client_id):
        """Remove a client from the active clients list."""
        if client_id in self.active_clients:
            del self.active_clients[client_id]
        if client_id in self.client_queues:
            del self.client_queues[client_id]
        print(f"[MULTI-CLIENT] Removed client: {client_id}")

    @abstractmethod
    def _get_client_queue_name(self, client_id):
        """
        Get the queue name for a specific client.
        Must be implemented by subclasses.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Queue name string
        """
        pass


class ClientSpecificQueueBase(ABC):
    """
    Abstract base class for client-specific queues with non-blocking get capability.
    """

    @abstractmethod
    def get_non_blocking(self):
        """
        Non-blocking version of get() that returns None if no data is available.
        Must be implemented by subclasses.
        
        Returns:
            Data if available, None otherwise
        """
        pass

    @abstractmethod
    def publish(self, data):
        """
        Publish data to the queue.
        Must be implemented by subclasses.
        
        Args:
            data: Data to publish
        """
        pass

    @abstractmethod
    def clear(self, name):
        """
        Clear the queue.
        Must be implemented by subclasses.
        
        Args:
            name: Queue name to clear
        """
        pass
