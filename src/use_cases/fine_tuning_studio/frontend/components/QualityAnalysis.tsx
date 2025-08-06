import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Alert,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  ToggleButton,
  ToggleButtonGroup,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Badge,
  Skeleton
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Timeline as TimelineIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  BugReport as BugReportIcon,
  Psychology as PsychologyIcon,
  Functions as FunctionsIcon,
  AutoGraph as AutoGraphIcon,
  CompareArrows as CompareArrowsIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ScatterChart,
  Scatter,
  ZAxis
} from 'recharts';
import { useQuery } from 'react-query';
import type { ExperimentMetrics, EvaluationResult } from '@/types';

interface QualityAnalysisProps {
  experimentId: string;
  metrics?: ExperimentMetrics;
  onRefresh?: () => void;
}

export default function QualityAnalysis({
  experimentId,
  metrics,
  onRefresh
}: QualityAnalysisProps) {
  const [timeRange, setTimeRange] = useState('all');
  const [comparisonMode, setComparisonMode] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState('loss');
  const [viewMode, setViewMode] = useState<'charts' | 'table'>('charts');

  // Mock data for demonstration
  const mockMetricsData = {
    lossHistory: [
      { epoch: 1, loss: 2.8, valLoss: 2.9 },
      { epoch: 2, loss: 2.3, valLoss: 2.5 },
      { epoch: 3, loss: 1.9, valLoss: 2.2 },
      { epoch: 4, loss: 1.6, valLoss: 2.0 },
      { epoch: 5, loss: 1.4, valLoss: 1.9 },
    ],
    perplexityTrend: [
      { epoch: 1, perplexity: 16.4 },
      { epoch: 2, perplexity: 10.2 },
      { epoch: 3, perplexity: 7.1 },
      { epoch: 4, perplexity: 5.2 },
      { epoch: 5, perplexity: 4.1 },
    ],
    benchmarkScores: [
      { benchmark: 'MMLU', score: 0.72, baseline: 0.65 },
      { benchmark: 'HellaSwag', score: 0.81, baseline: 0.75 },
      { benchmark: 'TruthfulQA', score: 0.68, baseline: 0.60 },
      { benchmark: 'GSM8K', score: 0.45, baseline: 0.38 },
      { benchmark: 'HumanEval', score: 0.52, baseline: 0.42 },
    ],
    qualityMetrics: {
      coherence: 0.85,
      relevance: 0.78,
      fluency: 0.92,
      factuality: 0.73,
      diversity: 0.81,
      consistency: 0.88
    },
    overfittingIndicators: {
      trainValGap: 0.3,
      earlyStoppingEpoch: null,
      generalizationScore: 0.82,
      status: 'healthy'
    }
  };

  const getStatusColor = (value: number, threshold: number = 0.7) => {
    if (value >= threshold) return 'success';
    if (value >= threshold * 0.8) return 'warning';
    return 'error';
  };

  const getOverfittingStatus = () => {
    const gap = mockMetricsData.overfittingIndicators.trainValGap;
    if (gap < 0.2) return { status: 'Healthy', color: 'success', icon: <CheckCircleIcon /> };
    if (gap < 0.5) return { status: 'Minor Overfitting', color: 'warning', icon: <WarningIcon /> };
    return { status: 'Significant Overfitting', color: 'error', icon: <ErrorIcon /> };
  };

  const calculateImprovement = (current: number, baseline: number) => {
    const improvement = ((current - baseline) / baseline) * 100;
    return improvement > 0 ? `+${improvement.toFixed(1)}%` : `${improvement.toFixed(1)}%`;
  };

  const renderMetricsOverview = () => (
    <Grid container spacing={3}>
      {/* Key Metrics Cards */}
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" variant="body2" gutterBottom>
                  Current Loss
                </Typography>
                <Typography variant="h4">
                  {mockMetricsData.lossHistory[mockMetricsData.lossHistory.length - 1].loss.toFixed(3)}
                </Typography>
                <Box display="flex" alignItems="center" gap={0.5} mt={1}>
                  <TrendingDownIcon color="success" fontSize="small" />
                  <Typography variant="caption" color="success.main">
                    -50% from start
                  </Typography>
                </Box>
              </Box>
              <TimelineIcon color="primary" />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" variant="body2" gutterBottom>
                  Perplexity
                </Typography>
                <Typography variant="h4">
                  {mockMetricsData.perplexityTrend[mockMetricsData.perplexityTrend.length - 1].perplexity.toFixed(1)}
                </Typography>
                <Box display="flex" alignItems="center" gap={0.5} mt={1}>
                  <TrendingDownIcon color="success" fontSize="small" />
                  <Typography variant="caption" color="success.main">
                    Improving
                  </Typography>
                </Box>
              </Box>
              <FunctionsIcon color="secondary" />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" variant="body2" gutterBottom>
                  Avg Benchmark Score
                </Typography>
                <Typography variant="h4">
                  {(mockMetricsData.benchmarkScores.reduce((acc, b) => acc + b.score, 0) / mockMetricsData.benchmarkScores.length * 100).toFixed(0)}%
                </Typography>
                <Box display="flex" alignItems="center" gap={0.5} mt={1}>
                  <TrendingUpIcon color="success" fontSize="small" />
                  <Typography variant="caption" color="success.main">
                    Above baseline
                  </Typography>
                </Box>
              </Box>
              <AssessmentIcon color="success" />
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Box>
                <Typography color="textSecondary" variant="body2" gutterBottom>
                  Overfitting Status
                </Typography>
                <Typography variant="h6">
                  {getOverfittingStatus().status}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={mockMetricsData.overfittingIndicators.generalizationScore * 100}
                  color={getOverfittingStatus().color as any}
                  sx={{ mt: 1 }}
                />
              </Box>
              {getOverfittingStatus().icon}
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderLossChart = () => (
    <Paper sx={{ p: 2 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">Loss Curves</Typography>
        <Box display="flex" gap={1}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <Select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <MenuItem value="all">All Epochs</MenuItem>
              <MenuItem value="last10">Last 10</MenuItem>
              <MenuItem value="last5">Last 5</MenuItem>
            </Select>
          </FormControl>
          <IconButton size="small" onClick={onRefresh}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={mockMetricsData.lossHistory}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
          <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
          <RechartsTooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#8884d8"
            strokeWidth={2}
            name="Training Loss"
            dot={{ r: 4 }}
          />
          <Line
            type="monotone"
            dataKey="valLoss"
            stroke="#82ca9d"
            strokeWidth={2}
            name="Validation Loss"
            strokeDasharray="5 5"
            dot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>

      {mockMetricsData.overfittingIndicators.trainValGap > 0.3 && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          <Typography variant="body2">
            The gap between training and validation loss is increasing. Consider:
          </Typography>
          <List dense>
            <ListItem>• Adding more regularization (dropout, weight decay)</ListItem>
            <ListItem>• Reducing model complexity</ListItem>
            <ListItem>• Increasing dataset size</ListItem>
            <ListItem>• Using early stopping</ListItem>
          </List>
        </Alert>
      )}
    </Paper>
  );

  const renderBenchmarkComparison = () => (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Benchmark Performance
      </Typography>

      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={mockMetricsData.benchmarkScores}>
          <PolarGrid />
          <PolarAngleAxis dataKey="benchmark" />
          <PolarRadiusAxis angle={90} domain={[0, 1]} />
          <Radar
            name="Fine-tuned Model"
            dataKey="score"
            stroke="#8884d8"
            fill="#8884d8"
            fillOpacity={0.6}
          />
          <Radar
            name="Baseline"
            dataKey="baseline"
            stroke="#82ca9d"
            fill="#82ca9d"
            fillOpacity={0.3}
          />
          <Legend />
        </RadarChart>
      </ResponsiveContainer>

      <TableContainer sx={{ mt: 2 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Benchmark</TableCell>
              <TableCell align="center">Score</TableCell>
              <TableCell align="center">Baseline</TableCell>
              <TableCell align="center">Improvement</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {mockMetricsData.benchmarkScores.map((benchmark) => (
              <TableRow key={benchmark.benchmark}>
                <TableCell>{benchmark.benchmark}</TableCell>
                <TableCell align="center">
                  <Typography variant="body2" fontWeight={500}>
                    {(benchmark.score * 100).toFixed(1)}%
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2" color="textSecondary">
                    {(benchmark.baseline * 100).toFixed(1)}%
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Chip
                    label={calculateImprovement(benchmark.score, benchmark.baseline)}
                    color={benchmark.score > benchmark.baseline ? 'success' : 'error'}
                    size="small"
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );

  const renderQualityMetrics = () => {
    const qualityData = Object.entries(mockMetricsData.qualityMetrics).map(([key, value]) => ({
      metric: key.charAt(0).toUpperCase() + key.slice(1),
      value: value * 100,
      fullMark: 100
    }));

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Quality Metrics
        </Typography>

        <Grid container spacing={2}>
          {Object.entries(mockMetricsData.qualityMetrics).map(([metric, value]) => (
            <Grid item xs={12} sm={6} md={4} key={metric}>
              <Box>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2" color="textSecondary">
                    {metric.charAt(0).toUpperCase() + metric.slice(1)}
                  </Typography>
                  <Typography variant="body2" fontWeight={500}>
                    {(value * 100).toFixed(0)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={value * 100}
                  color={getStatusColor(value) as any}
                />
              </Box>
            </Grid>
          ))}
        </Grid>

        <Divider sx={{ my: 3 }} />

        <Typography variant="subtitle2" gutterBottom>
          Quality Score Breakdown
        </Typography>

        <ResponsiveContainer width="100%" height={250}>
          <RadarChart data={qualityData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" />
            <PolarRadiusAxis angle={90} domain={[0, 100]} />
            <Radar
              name="Quality Score"
              dataKey="value"
              stroke="#8884d8"
              fill="#8884d8"
              fillOpacity={0.6}
            />
          </RadarChart>
        </ResponsiveContainer>
      </Paper>
    );
  };

  const renderRecommendations = () => {
    const recommendations = [
      {
        type: 'success',
        title: 'Strong Performance',
        description: 'Model shows good generalization with minimal overfitting',
        icon: <CheckCircleIcon />
      },
      {
        type: 'warning',
        title: 'Consider Data Augmentation',
        description: 'Diversity score could be improved with more varied training data',
        icon: <WarningIcon />
      },
      {
        type: 'info',
        title: 'Optimization Suggestion',
        description: 'Try reducing learning rate for final epochs to improve convergence',
        icon: <InfoIcon />
      }
    ];

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Recommendations
        </Typography>

        <List>
          {recommendations.map((rec, index) => (
            <React.Fragment key={index}>
              <ListItem>
                <ListItemIcon>
                  <Box color={`${rec.type}.main`}>
                    {rec.icon}
                  </Box>
                </ListItemIcon>
                <ListItemText
                  primary={rec.title}
                  secondary={rec.description}
                />
              </ListItem>
              {index < recommendations.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List>

        <Box mt={2}>
          <Button
            fullWidth
            variant="outlined"
            startIcon={<BugReportIcon />}
          >
            Run Full Diagnostic
          </Button>
        </Box>
      </Paper>
    );
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5">Quality Analysis</Typography>
        <Box display="flex" gap={1}>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(e, value) => value && setViewMode(value)}
            size="small"
          >
            <ToggleButton value="charts">Charts</ToggleButton>
            <ToggleButton value="table">Table</ToggleButton>
          </ToggleButtonGroup>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            size="small"
          >
            Export Report
          </Button>
        </Box>
      </Box>

      {/* Metrics Overview */}
      {renderMetricsOverview()}

      <Grid container spacing={3} sx={{ mt: 0 }}>
        {/* Loss Chart */}
        <Grid item xs={12} lg={8}>
          {renderLossChart()}
        </Grid>

        {/* Recommendations */}
        <Grid item xs={12} lg={4}>
          {renderRecommendations()}
        </Grid>

        {/* Benchmark Comparison */}
        <Grid item xs={12} lg={6}>
          {renderBenchmarkComparison()}
        </Grid>

        {/* Quality Metrics */}
        <Grid item xs={12} lg={6}>
          {renderQualityMetrics()}
        </Grid>

        {/* Perplexity Trend */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Perplexity Trend
            </Typography>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={mockMetricsData.perplexityTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis />
                <RechartsTooltip />
                <Area
                  type="monotone"
                  dataKey="perplexity"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
