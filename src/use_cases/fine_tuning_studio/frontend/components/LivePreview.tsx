import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  IconButton,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Alert,
  Divider,
  Grid,
  ToggleButton,
  ToggleButtonGroup,
  Skeleton,
  Tooltip,
  List,
  ListItem,
  ListItemText,
  Collapse,
  Badge
} from '@mui/material';
import {
  Send as SendIcon,
  Refresh as RefreshIcon,
  CompareArrows as CompareArrowsIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  ContentCopy as ContentCopyIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Circle as CircleIcon
} from '@mui/icons-material';
import { useWebSocket } from '@/hooks/useWebSocket';
import { format } from 'date-fns';
import type { TrainingProgressEvent } from '@/types';

interface LivePreviewProps {
  experimentId: string;
  baseModelId?: string;
  currentCheckpoint?: string;
}

interface PreviewMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  model: 'base' | 'finetuned';
  metrics?: {
    tokensPerSecond?: number;
    latency?: number;
    perplexity?: number;
  };
}

interface ComparisonResult {
  prompt: string;
  baseResponse: string;
  fineTunedResponse: string;
  metrics: {
    base: {
      tokensPerSecond: number;
      latency: number;
      perplexity: number;
    };
    fineTuned: {
      tokensPerSecond: number;
      latency: number;
      perplexity: number;
    };
  };
  timestamp: string;
}

export default function LivePreview({
  experimentId,
  baseModelId = 'meta-llama/Llama-2-7b-hf',
  currentCheckpoint
}: LivePreviewProps) {
  const { subscribe, unsubscribe, send, isConnected } = useWebSocket();
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [mode, setMode] = useState<'single' | 'comparison'>('comparison');
  const [showMetrics, setShowMetrics] = useState(true);
  const [messages, setMessages] = useState<PreviewMessage[]>([]);
  const [comparisons, setComparisons] = useState<ComparisonResult[]>([]);
  const [trainingProgress, setTrainingProgress] = useState<any>(null);
  const [expandedComparison, setExpandedComparison] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Subscribe to training progress updates
    const handleTrainingProgress = (event: TrainingProgressEvent) => {
      if (event.data.experimentId === experimentId) {
        setTrainingProgress(event.data);
      }
    };

    // Subscribe to preview responses
    const handlePreviewResponse = (data: any) => {
      if (data.experimentId === experimentId) {
        if (data.type === 'comparison') {
          setComparisons(prev => [...prev, data.result]);
        } else {
          setMessages(prev => [...prev, data.message]);
        }
        setIsGenerating(false);
      }
    };

    subscribe('training_progress', handleTrainingProgress);
    subscribe('preview_response', handlePreviewResponse);

    return () => {
      unsubscribe('training_progress', handleTrainingProgress);
      unsubscribe('preview_response', handlePreviewResponse);
    };
  }, [experimentId, subscribe, unsubscribe]);

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, comparisons]);

  const handleSendPrompt = () => {
    if (!prompt.trim() || isGenerating) return;

    setIsGenerating(true);

    if (mode === 'comparison') {
      // Request comparison between base and fine-tuned models
      send('preview_request', {
        experimentId,
        type: 'comparison',
        prompt: prompt.trim(),
        baseModelId,
        checkpoint: currentCheckpoint,
        timestamp: new Date().toISOString()
      });
    } else {
      // Request single model response
      const newMessage: PreviewMessage = {
        id: Date.now().toString(),
        role: 'user',
        content: prompt.trim(),
        timestamp: new Date().toISOString(),
        model: 'finetuned'
      };
      setMessages(prev => [...prev, newMessage]);

      send('preview_request', {
        experimentId,
        type: 'single',
        prompt: prompt.trim(),
        checkpoint: currentCheckpoint,
        timestamp: new Date().toISOString()
      });
    }

    setPrompt('');
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const handleRefresh = () => {
    setMessages([]);
    setComparisons([]);
  };

  const formatMetric = (value: number | undefined, suffix: string) => {
    if (value === undefined) return 'N/A';
    return `${value.toFixed(2)} ${suffix}`;
  };

  const getPerformanceColor = (improved: boolean) => {
    return improved ? 'success' : 'warning';
  };

  const calculateImprovement = (base: number, fineTuned: number) => {
    const improvement = ((fineTuned - base) / base) * 100;
    return improvement > 0 ? `+${improvement.toFixed(1)}%` : `${improvement.toFixed(1)}%`;
  };

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">Live Preview</Typography>
        <Box display="flex" alignItems="center" gap={1}>
          {isConnected ? (
            <Chip
              icon={<CircleIcon sx={{ fontSize: 12 }} />}
              label="Connected"
              color="success"
              size="small"
            />
          ) : (
            <Chip
              icon={<CircleIcon sx={{ fontSize: 12 }} />}
              label="Disconnected"
              color="error"
              size="small"
            />
          )}
          <IconButton size="small" onClick={handleRefresh}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Training Progress Bar */}
      {trainingProgress && (
        <Alert severity="info" sx={{ mb: 2 }}>
          <Box>
            <Typography variant="body2" gutterBottom>
              Training in Progress - Epoch {trainingProgress.epoch} | Loss: {trainingProgress.loss?.toFixed(4)}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={trainingProgress.progress}
              sx={{ mt: 1 }}
            />
          </Box>
        </Alert>
      )}

      {/* Mode Toggle */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <ToggleButtonGroup
          value={mode}
          exclusive
          onChange={(e, value) => value && setMode(value)}
          size="small"
        >
          <ToggleButton value="single">
            Single Model
          </ToggleButton>
          <ToggleButton value="comparison">
            <CompareArrowsIcon sx={{ mr: 1 }} />
            Comparison
          </ToggleButton>
        </ToggleButtonGroup>
        <ToggleButton
          value="metrics"
          selected={showMetrics}
          onChange={() => setShowMetrics(!showMetrics)}
          size="small"
        >
          {showMetrics ? <VisibilityIcon /> : <VisibilityOffIcon />}
          Metrics
        </ToggleButton>
      </Box>

      {/* Chat/Comparison Display */}
      <Paper
        sx={{
          height: 400,
          overflowY: 'auto',
          p: 2,
          mb: 2,
          backgroundColor: 'background.default'
        }}
      >
        {mode === 'comparison' ? (
          comparisons.length === 0 ? (
            <Box
              display="flex"
              alignItems="center"
              justifyContent="center"
              height="100%"
            >
              <Typography color="textSecondary">
                Enter a prompt to compare base and fine-tuned model responses
              </Typography>
            </Box>
          ) : (
            <List>
              {comparisons.map((comparison, index) => (
                <ListItem key={index} sx={{ flexDirection: 'column', alignItems: 'stretch' }}>
                  {/* Prompt */}
                  <Box mb={2}>
                    <Typography variant="subtitle2" color="primary" gutterBottom>
                      Prompt:
                    </Typography>
                    <Typography variant="body2" sx={{ pl: 2 }}>
                      {comparison.prompt}
                    </Typography>
                  </Box>

                  {/* Responses Grid */}
                  <Grid container spacing={2}>
                    {/* Base Model Response */}
                    <Grid item xs={12} md={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                            <Typography variant="subtitle2" color="textSecondary">
                              Base Model
                            </Typography>
                            <IconButton
                              size="small"
                              onClick={() => handleCopy(comparison.baseResponse, `base-${index}`)}
                            >
                              {copied === `base-${index}` ? (
                                <CheckCircleIcon fontSize="small" color="success" />
                              ) : (
                                <ContentCopyIcon fontSize="small" />
                              )}
                            </IconButton>
                          </Box>
                          <Typography variant="body2" sx={{ mb: 2 }}>
                            {comparison.baseResponse}
                          </Typography>

                          {showMetrics && (
                            <Box>
                              <Divider sx={{ my: 1 }} />
                              <Grid container spacing={1}>
                                <Grid item xs={6}>
                                  <Box display="flex" alignItems="center" gap={0.5}>
                                    <SpeedIcon fontSize="small" color="action" />
                                    <Typography variant="caption">
                                      {formatMetric(comparison.metrics.base.tokensPerSecond, 'tok/s')}
                                    </Typography>
                                  </Box>
                                </Grid>
                                <Grid item xs={6}>
                                  <Box display="flex" alignItems="center" gap={0.5}>
                                    <MemoryIcon fontSize="small" color="action" />
                                    <Typography variant="caption">
                                      {formatMetric(comparison.metrics.base.latency, 'ms')}
                                    </Typography>
                                  </Box>
                                </Grid>
                              </Grid>
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>

                    {/* Fine-tuned Model Response */}
                    <Grid item xs={12} md={6}>
                      <Card
                        variant="outlined"
                        sx={{
                          borderColor: 'primary.main',
                          borderWidth: 2
                        }}
                      >
                        <CardContent>
                          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                            <Typography variant="subtitle2" color="primary">
                              Fine-tuned Model
                            </Typography>
                            <IconButton
                              size="small"
                              onClick={() => handleCopy(comparison.fineTunedResponse, `finetuned-${index}`)}
                            >
                              {copied === `finetuned-${index}` ? (
                                <CheckCircleIcon fontSize="small" color="success" />
                              ) : (
                                <ContentCopyIcon fontSize="small" />
                              )}
                            </IconButton>
                          </Box>
                          <Typography variant="body2" sx={{ mb: 2 }}>
                            {comparison.fineTunedResponse}
                          </Typography>

                          {showMetrics && (
                            <Box>
                              <Divider sx={{ my: 1 }} />
                              <Grid container spacing={1}>
                                <Grid item xs={6}>
                                  <Box display="flex" alignItems="center" gap={0.5}>
                                    <SpeedIcon fontSize="small" color="action" />
                                    <Typography variant="caption">
                                      {formatMetric(comparison.metrics.fineTuned.tokensPerSecond, 'tok/s')}
                                    </Typography>
                                    <Chip
                                      label={calculateImprovement(
                                        comparison.metrics.base.tokensPerSecond,
                                        comparison.metrics.fineTuned.tokensPerSecond
                                      )}
                                      size="small"
                                      color={getPerformanceColor(
                                        comparison.metrics.fineTuned.tokensPerSecond >
                                        comparison.metrics.base.tokensPerSecond
                                      )}
                                    />
                                  </Box>
                                </Grid>
                                <Grid item xs={6}>
                                  <Box display="flex" alignItems="center" gap={0.5}>
                                    <MemoryIcon fontSize="small" color="action" />
                                    <Typography variant="caption">
                                      {formatMetric(comparison.metrics.fineTuned.latency, 'ms')}
                                    </Typography>
                                    <Chip
                                      label={calculateImprovement(
                                        comparison.metrics.base.latency,
                                        comparison.metrics.fineTuned.latency
                                      )}
                                      size="small"
                                      color={getPerformanceColor(
                                        comparison.metrics.fineTuned.latency <
                                        comparison.metrics.base.latency
                                      )}
                                    />
                                  </Box>
                                </Grid>
                              </Grid>
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>

                  {index < comparisons.length - 1 && <Divider sx={{ my: 3 }} />}
                </ListItem>
              ))}
            </List>
          )
        ) : (
          // Single model chat view
          messages.length === 0 ? (
            <Box
              display="flex"
              alignItems="center"
              justifyContent="center"
              height="100%"
            >
              <Typography color="textSecondary">
                Enter a prompt to test the fine-tuned model
              </Typography>
            </Box>
          ) : (
            <List>
              {messages.map((message) => (
                <ListItem
                  key={message.id}
                  sx={{
                    flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                    alignItems: 'flex-start'
                  }}
                >
                  <Card
                    sx={{
                      maxWidth: '70%',
                      backgroundColor: message.role === 'user' ? 'primary.light' : 'background.paper'
                    }}
                  >
                    <CardContent>
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                        {message.content}
                      </Typography>
                      {message.metrics && showMetrics && (
                        <Box mt={1} pt={1} borderTop={1} borderColor="divider">
                          <Typography variant="caption" color="textSecondary">
                            {formatMetric(message.metrics.tokensPerSecond, 'tok/s')} •{' '}
                            {formatMetric(message.metrics.latency, 'ms')}
                          </Typography>
                        </Box>
                      )}
                      <Typography variant="caption" color="textSecondary" display="block" mt={0.5}>
                        {format(new Date(message.timestamp), 'HH:mm:ss')}
                      </Typography>
                    </CardContent>
                  </Card>
                </ListItem>
              ))}
            </List>
          )
        )}
        <div ref={messagesEndRef} />
      </Paper>

      {/* Input Area */}
      <Box display="flex" gap={1}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder={
            mode === 'comparison'
              ? "Enter a prompt to compare models..."
              : "Enter a prompt to test the fine-tuned model..."
          }
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSendPrompt();
            }
          }}
          multiline
          maxRows={4}
          disabled={isGenerating || !isConnected}
        />
        <Button
          variant="contained"
          onClick={handleSendPrompt}
          disabled={isGenerating || !prompt.trim() || !isConnected}
          startIcon={isGenerating ? <CircularProgress size={20} /> : <SendIcon />}
        >
          {isGenerating ? 'Generating...' : 'Send'}
        </Button>
      </Box>

      {/* Status Messages */}
      {!isConnected && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          WebSocket connection lost. Attempting to reconnect...
        </Alert>
      )}

      {!currentCheckpoint && (
        <Alert severity="info" sx={{ mt: 2 }}>
          No checkpoint available yet. Responses will use the base model until training produces checkpoints.
        </Alert>
      )}
    </Box>
  );
}

// Fix for CircularProgress import
import { CircularProgress } from '@mui/material';
