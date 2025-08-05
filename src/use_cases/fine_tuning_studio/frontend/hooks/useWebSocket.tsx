'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { WebSocketEvent, TrainingProgressEvent, DeploymentStatusEvent } from '@/types';
import Cookies from 'js-cookie';
import { toast } from 'react-toastify';

interface WebSocketContextType {
  socket: Socket | null;
  connected: boolean;
  subscribe: (event: string, callback: (data: any) => void) => void;
  unsubscribe: (event: string, callback: (data: any) => void) => void;
  emit: (event: string, data: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    // Initialize WebSocket connection
    const token = Cookies.get('auth_token');
    
    if (token) {
      const newSocket = io('http://localhost:8000', {
        auth: {
          token,
        },
        transports: ['websocket'],
        autoConnect: true,
      });

      newSocket.on('connect', () => {
        console.log('WebSocket connected');
        setConnected(true);
      });

      newSocket.on('disconnect', () => {
        console.log('WebSocket disconnected');
        setConnected(false);
      });

      newSocket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        setConnected(false);
      });

      // Handle training progress updates
      newSocket.on('training_progress', (data: TrainingProgressEvent['data']) => {
        console.log('Training progress:', data);
        // This will be handled by individual components that subscribe
      });

      // Handle deployment status updates
      newSocket.on('deployment_status', (data: DeploymentStatusEvent['data']) => {
        console.log('Deployment status:', data);
        toast.info(`Deployment ${data.status}: ${data.message || ''}`);
      });

      // Handle general notifications
      newSocket.on('notification', (data: { type: string; message: string }) => {
        switch (data.type) {
          case 'success':
            toast.success(data.message);
            break;
          case 'error':
            toast.error(data.message);
            break;
          case 'warning':
            toast.warning(data.message);
            break;
          default:
            toast.info(data.message);
        }
      });

      setSocket(newSocket);

      return () => {
        newSocket.close();
      };
    }
  }, []);

  const subscribe = (event: string, callback: (data: any) => void) => {
    if (socket) {
      socket.on(event, callback);
    }
  };

  const unsubscribe = (event: string, callback: (data: any) => void) => {
    if (socket) {
      socket.off(event, callback);
    }
  };

  const emit = (event: string, data: any) => {
    if (socket && connected) {
      socket.emit(event, data);
    }
  };

  const value: WebSocketContextType = {
    socket,
    connected,
    subscribe,
    unsubscribe,
    emit,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

// Custom hooks for specific WebSocket events
export const useTrainingProgress = (experimentId: string) => {
  const { subscribe, unsubscribe } = useWebSocket();
  const [progress, setProgress] = useState<TrainingProgressEvent['data'] | null>(null);

  useEffect(() => {
    const handleProgress = (data: TrainingProgressEvent['data']) => {
      if (data.experimentId === experimentId) {
        setProgress(data);
      }
    };

    subscribe('training_progress', handleProgress);

    return () => {
      unsubscribe('training_progress', handleProgress);
    };
  }, [experimentId, subscribe, unsubscribe]);

  return progress;
};

export const useDeploymentStatus = (modelId: string) => {
  const { subscribe, unsubscribe } = useWebSocket();
  const [status, setStatus] = useState<DeploymentStatusEvent['data'] | null>(null);

  useEffect(() => {
    const handleStatus = (data: DeploymentStatusEvent['data']) => {
      if (data.modelId === modelId) {
        setStatus(data);
      }
    };

    subscribe('deployment_status', handleStatus);

    return () => {
      unsubscribe('deployment_status', handleStatus);
    };
  }, [modelId, subscribe, unsubscribe]);

  return status;
};