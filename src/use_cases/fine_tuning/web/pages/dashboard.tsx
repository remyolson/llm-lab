'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  IconButton,
  Chip,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Paper,
  Button,
  Tooltip,
  Skeleton
} from '@mui/material';
import {
  Science as ScienceIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Assessment as AssessmentIcon,
  TrendingUp as TrendingUpIcon,
  CloudUpload as CloudUploadIcon,
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreVertIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Schedule as ScheduleIcon
} from '@mui/icons-material';
import { useQuery, useMutation } from 'react-query';
import { useRouter } from 'next/navigation';
import { format } from 'date-fns';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { useWebSocket } from '@/hooks/useWebSocket';
import type {
  Experiment,
  TrainingJob,
  DeployedModel,
  ExperimentMetrics,
  Notification
} from '@/types';

// Mock API functions (replace with actual API calls)
const fetchDashboardData = async () => {
  // Simulated API call
  return {
    experiments: [],
    recentDeployments: [],
    activeJobs: [],
    systemMetrics: {
      gpuUtilization: 75,
      memoryUsage: 60,
      activeExperiments: 3,
      totalDeployments: 12
    },
    notifications: []
  };
};

const fetchExperiments = async () => {
  // Simulated API call
  return [];
};

const fetchActiveJobs = async () => {
  // Simulated API call
  return [];
};

const fetchDeployments = async () => {
  // Simulated API call
  return [];
};

export default function DashboardPage() {
  const router = useRouter();
  const { subscribe, unsubscribe } = useWebSocket();
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  const [metricsData, setMetricsData] = useState<any[]>([]);

  // Fetch dashboard data
  const { data: dashboardData, isLoading: isDashboardLoading, refetch: refetchDashboard } = useQuery(
    'dashboard',
    fetchDashboardData,
    {
      refetchInterval: 30000 // Refresh every 30 seconds
    }
  );

  // Fetch experiments
  const { data: experiments = [], isLoading: isExperimentsLoading } = useQuery(
    'experiments',
    fetchExperiments
  );

  // Fetch active training jobs
  const { data: activeJobs = [], isLoading: isJobsLoading } = useQuery(
    'activeJobs',
    fetchActiveJobs,
    {
      refetchInterval: 5000 // Refresh every 5 seconds
    }
  );

  // Fetch deployments
  const { data: deployments = [], isLoading: isDeploymentsLoading } = useQuery(
    'deployments',
    fetchDeployments
  );

  // Subscribe to WebSocket events for real-time updates
  useEffect(() => {
    const handleTrainingProgress = (data: any) => {
      // Update metrics data for charts
      setMetricsData(prev => [...prev, {
        timestamp: new Date().toISOString(),
        loss: data.loss,
        accuracy: data.accuracy,
        epoch: data.epoch
      }].slice(-50)); // Keep last 50 data points
    };

    subscribe('training_progress', handleTrainingProgress);

    return () => {
      unsubscribe('training_progress', handleTrainingProgress);
    };
  }, [subscribe, unsubscribe]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
      case 'active':
        return <CheckCircleIcon color="success" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      case 'running':
        return <ScheduleIcon color="primary" />;
      case 'paused':
        return <PauseIcon color="warning" />;
      default:
        return <WarningIcon color="disabled" />;
    }
  };

  const getStatusColor = (status: string): any => {
    switch (status) {
      case 'completed':
      case 'active':
        return 'success';
      case 'failed':
        return 'error';
      case 'running':
        return 'primary';
      case 'paused':
        return 'warning';
      default:
        return 'default';
    }
  };

  const handleExperimentClick = (experimentId: string) => {
    router.push(`/experiments/${experimentId}`);
  };

  const handleDeploymentClick = (deploymentId: string) => {
    router.push(`/deploy/${deploymentId}`);
  };

  const handleNewExperiment = () => {
    router.push('/experiments/new');
  };

  if (isDashboardLoading) {
    return (
      <Box>
        <Grid container spacing={3}>
          {[1, 2, 3, 4].map(i => (
            <Grid item xs={12} sm={6} md={3} key={i}>
              <Skeleton variant="rectangular" height={120} />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  const systemMetrics = dashboardData?.systemMetrics || {
    gpuUtilization: 0,
    memoryUsage: 0,
    activeExperiments: 0,
    totalDeployments: 0
  };

  return (
    <Box>
      {/* Header */}
      <Box mb={4}>
        <Typography variant="h4" gutterBottom>
          Fine-Tuning Studio Dashboard
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Monitor your experiments, deployments, and system resources
        </Typography>
      </Box>

      {/* System Metrics Cards */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Active Experiments
                  </Typography>
                  <Typography variant="h4">
                    {systemMetrics.activeExperiments}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'primary.light' }}>
                  <ScienceIcon />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    GPU Utilization
                  </Typography>
                  <Typography variant="h4">
                    {systemMetrics.gpuUtilization}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={systemMetrics.gpuUtilization}
                    sx={{ mt: 1 }}
                  />
                </Box>
                <Avatar sx={{ bgcolor: 'success.light' }}>
                  <SpeedIcon />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Memory Usage
                  </Typography>
                  <Typography variant="h4">
                    {systemMetrics.memoryUsage}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={systemMetrics.memoryUsage}
                    sx={{ mt: 1 }}
                    color="warning"
                  />
                </Box>
                <Avatar sx={{ bgcolor: 'warning.light' }}>
                  <MemoryIcon />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Total Deployments
                  </Typography>
                  <Typography variant="h4">
                    {systemMetrics.totalDeployments}
                  </Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'secondary.light' }}>
                  <CloudUploadIcon />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Content Grid */}
      <Grid container spacing={3}>
        {/* Active Training Jobs */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">Active Training Jobs</Typography>
              <IconButton size="small" onClick={() => refetchDashboard()}>
                <RefreshIcon />
              </IconButton>
            </Box>

            {isJobsLoading ? (
              <Skeleton variant="rectangular" height={200} />
            ) : activeJobs.length === 0 ? (
              <Box
                display="flex"
                flexDirection="column"
                alignItems="center"
                justifyContent="center"
                height={200}
              >
                <Typography color="textSecondary">No active training jobs</Typography>
                <Button
                  variant="contained"
                  startIcon={<PlayArrowIcon />}
                  sx={{ mt: 2 }}
                  onClick={handleNewExperiment}
                >
                  Start New Experiment
                </Button>
              </Box>
            ) : (
              <List>
                {activeJobs.slice(0, 5).map((job: TrainingJob) => (
                  <ListItem key={job.id}>
                    <ListItemAvatar>
                      {getStatusIcon(job.status)}
                    </ListItemAvatar>
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography variant="body1">
                            Experiment {job.experimentId}
                          </Typography>
                          <Chip
                            label={`Epoch ${job.currentEpoch}/${job.totalEpochs}`}
                            size="small"
                            color="primary"
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <LinearProgress
                            variant="determinate"
                            value={job.progress}
                            sx={{ mt: 1, mb: 0.5 }}
                          />
                          <Typography variant="caption" color="textSecondary">
                            {job.progress}% • Loss: {job.currentLoss?.toFixed(4) || 'N/A'}
                          </Typography>
                        </Box>
                      }
                    />
                    <IconButton size="small">
                      <MoreVertIcon />
                    </IconButton>
                  </ListItem>
                ))}
              </List>
            )}
          </Paper>
        </Grid>

        {/* Recent Deployments */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">Recent Deployments</Typography>
              <Button
                size="small"
                startIcon={<CloudUploadIcon />}
                onClick={() => router.push('/deploy')}
              >
                View All
              </Button>
            </Box>

            {isDeploymentsLoading ? (
              <Skeleton variant="rectangular" height={200} />
            ) : deployments.length === 0 ? (
              <Box
                display="flex"
                flexDirection="column"
                alignItems="center"
                justifyContent="center"
                height={200}
              >
                <Typography color="textSecondary">No deployments yet</Typography>
              </Box>
            ) : (
              <List>
                {deployments.slice(0, 5).map((deployment: DeployedModel) => (
                  <ListItem
                    key={deployment.id}
                    button
                    onClick={() => handleDeploymentClick(deployment.id)}
                  >
                    <ListItemAvatar>
                      {getStatusIcon(deployment.status)}
                    </ListItemAvatar>
                    <ListItemText
                      primary={deployment.name}
                      secondary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Chip
                            label={deployment.provider}
                            size="small"
                            variant="outlined"
                          />
                          <Typography variant="caption" color="textSecondary">
                            {format(new Date(deployment.deployedAt), 'MMM dd, HH:mm')}
                          </Typography>
                        </Box>
                      }
                    />
                    <Chip
                      label={deployment.status}
                      color={getStatusColor(deployment.status)}
                      size="small"
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </Paper>
        </Grid>

        {/* Performance Metrics Chart */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Training Performance Metrics
            </Typography>

            {metricsData.length === 0 ? (
              <Box
                display="flex"
                alignItems="center"
                justifyContent="center"
                height={300}
              >
                <Typography color="textSecondary">
                  No training data available. Start an experiment to see metrics.
                </Typography>
              </Box>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="epoch"
                    label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#8884d8"
                    name="Loss"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#82ca9d"
                    name="Accuracy"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </Paper>
        </Grid>

        {/* Recent Experiments */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">Recent Experiments</Typography>
              <Button
                variant="contained"
                startIcon={<ScienceIcon />}
                onClick={handleNewExperiment}
              >
                New Experiment
              </Button>
            </Box>

            {isExperimentsLoading ? (
              <Skeleton variant="rectangular" height={200} />
            ) : experiments.length === 0 ? (
              <Box
                display="flex"
                flexDirection="column"
                alignItems="center"
                justifyContent="center"
                height={200}
              >
                <Typography color="textSecondary" gutterBottom>
                  No experiments created yet
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Create your first experiment to start fine-tuning models
                </Typography>
              </Box>
            ) : (
              <Grid container spacing={2}>
                {experiments.slice(0, 6).map((experiment: Experiment) => (
                  <Grid item xs={12} sm={6} md={4} key={experiment.id}>
                    <Card
                      sx={{
                        cursor: 'pointer',
                        '&:hover': { boxShadow: 4 }
                      }}
                      onClick={() => handleExperimentClick(experiment.id)}
                    >
                      <CardContent>
                        <Box display="flex" justifyContent="space-between" alignItems="start">
                          <Typography variant="h6" gutterBottom>
                            {experiment.name}
                          </Typography>
                          <Chip
                            label={experiment.status}
                            size="small"
                            color={getStatusColor(experiment.status)}
                          />
                        </Box>
                        <Typography variant="body2" color="textSecondary" gutterBottom>
                          {experiment.description || 'No description'}
                        </Typography>
                        <Box display="flex" gap={1} mt={2}>
                          {experiment.tags.map(tag => (
                            <Chip key={tag} label={tag} size="small" variant="outlined" />
                          ))}
                        </Box>
                        <Typography variant="caption" color="textSecondary" display="block" mt={1}>
                          Created: {format(new Date(experiment.createdAt), 'MMM dd, yyyy')}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
