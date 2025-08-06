import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Slider,
  FormControlLabel,
  Radio,
  RadioGroup,
  Tooltip,
  Badge,
  Divider,
  List,
  ListItem,
  ListItemText,
  Skeleton
} from '@mui/material';
import {
  CompareArrows as CompareArrowsIcon,
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Assessment as AssessmentIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  ThumbsUpDown as ThumbsUpDownIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Psychology as PsychologyIcon,
  BarChart as BarChartIcon
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import { useQuery, useMutation } from 'react-query';
import type { ABTest, ABTestCase, ABTestResults, TestMetrics } from '@/types';

interface ABTestingProps {
  experimentIdA?: string;
  experimentIdB?: string;
  onTestComplete?: (results: ABTestResults) => void;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

export default function ABTesting({
  experimentIdA,
  experimentIdB,
  onTestComplete
}: ABTestingProps) {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedTest, setSelectedTest] = useState<ABTest | null>(null);
  const [testCases, setTestCases] = useState<ABTestCase[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentTestCase, setCurrentTestCase] = useState(0);
  const [showNewTestDialog, setShowNewTestDialog] = useState(false);
  const [showResultsDialog, setShowResultsDialog] = useState(false);
  const [newTestForm, setNewTestForm] = useState({
    name: '',
    description: '',
    modelA: experimentIdA || '',
    modelB: experimentIdB || '',
    testType: 'manual' as 'manual' | 'automatic' | 'hybrid'
  });

  // Mock data for demonstration
  const mockTests: ABTest[] = [
    {
      id: 'test1',
      name: 'Customer Support Quality Test',
      description: 'Comparing response quality for customer support queries',
      modelA: 'model_v1',
      modelB: 'model_v2',
      status: 'completed',
      testCases: [],
      results: {
        winnerModel: 'B',
        confidenceScore: 0.87,
        statisticalSignificance: 0.95,
        metrics: {
          modelA: {
            accuracy: 0.72,
            bleuScore: 0.65,
            rougeScore: 0.58,
            humanPreference: 0.35,
            avgLatency: 450
          },
          modelB: {
            accuracy: 0.85,
            bleuScore: 0.78,
            rougeScore: 0.71,
            humanPreference: 0.65,
            avgLatency: 480
          }
        }
      },
      createdAt: '2024-01-15T10:00:00Z',
      updatedAt: '2024-01-15T12:00:00Z'
    }
  ];

  const handleCreateTest = () => {
    // Create new A/B test
    console.log('Creating test:', newTestForm);
    setShowNewTestDialog(false);
    // Reset form
    setNewTestForm({
      name: '',
      description: '',
      modelA: '',
      modelB: '',
      testType: 'manual'
    });
  };

  const handleStartTest = () => {
    setIsRunning(true);
    // Start running through test cases
  };

  const handleStopTest = () => {
    setIsRunning(false);
  };

  const handleAddTestCase = () => {
    const newCase: ABTestCase = {
      id: Date.now().toString(),
      input: '',
      expectedOutput: '',
      modelAOutput: '',
      modelBOutput: '',
      humanPreference: undefined,
      confidence: undefined
    };
    setTestCases([...testCases, newCase]);
  };

  const handleRemoveTestCase = (id: string) => {
    setTestCases(testCases.filter(tc => tc.id !== id));
  };

  const handleTestCaseChange = (id: string, field: keyof ABTestCase, value: any) => {
    setTestCases(testCases.map(tc =>
      tc.id === id ? { ...tc, [field]: value } : tc
    ));
  };

  const handleHumanPreference = (testCaseId: string, preference: 'A' | 'B' | 'tie', confidence: number) => {
    setTestCases(testCases.map(tc =>
      tc.id === testCaseId
        ? { ...tc, humanPreference: preference, confidence }
        : tc
    ));
  };

  const calculateWinRate = (test: ABTest) => {
    if (!test.results) return { modelA: 0, modelB: 0, tie: 0 };

    const total = test.testCases.length;
    const modelAWins = test.testCases.filter(tc => tc.humanPreference === 'A').length;
    const modelBWins = test.testCases.filter(tc => tc.humanPreference === 'B').length;
    const ties = test.testCases.filter(tc => tc.humanPreference === 'tie').length;

    return {
      modelA: (modelAWins / total) * 100,
      modelB: (modelBWins / total) * 100,
      tie: (ties / total) * 100
    };
  };

  const renderTestConfiguration = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Test Configuration
          </Typography>

          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Model A</InputLabel>
                <Select
                  value={experimentIdA || ''}
                  label="Model A"
                  disabled={isRunning}
                >
                  <MenuItem value="model_v1">Model v1 - Base</MenuItem>
                  <MenuItem value="model_v2">Model v2 - Fine-tuned</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Model B</InputLabel>
                <Select
                  value={experimentIdB || ''}
                  label="Model B"
                  disabled={isRunning}
                >
                  <MenuItem value="model_v2">Model v2 - Fine-tuned</MenuItem>
                  <MenuItem value="model_v3">Model v3 - Latest</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>

          <Box mt={3}>
            <Typography variant="subtitle1" gutterBottom>
              Test Cases ({testCases.length})
            </Typography>

            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Input Prompt</TableCell>
                    <TableCell>Expected Output</TableCell>
                    <TableCell width={100}>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {testCases.map((testCase) => (
                    <TableRow key={testCase.id}>
                      <TableCell>
                        <TextField
                          fullWidth
                          size="small"
                          value={testCase.input}
                          onChange={(e) => handleTestCaseChange(testCase.id, 'input', e.target.value)}
                          placeholder="Enter test prompt..."
                          disabled={isRunning}
                        />
                      </TableCell>
                      <TableCell>
                        <TextField
                          fullWidth
                          size="small"
                          value={testCase.expectedOutput}
                          onChange={(e) => handleTestCaseChange(testCase.id, 'expectedOutput', e.target.value)}
                          placeholder="Expected output (optional)..."
                          disabled={isRunning}
                        />
                      </TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => handleRemoveTestCase(testCase.id)}
                          disabled={isRunning}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            <Button
              startIcon={<AddIcon />}
              onClick={handleAddTestCase}
              sx={{ mt: 1 }}
              disabled={isRunning}
            >
              Add Test Case
            </Button>
          </Box>

          <Box mt={3} display="flex" gap={2}>
            {!isRunning ? (
              <Button
                variant="contained"
                startIcon={<PlayArrowIcon />}
                onClick={handleStartTest}
                disabled={testCases.length === 0}
              >
                Start Test
              </Button>
            ) : (
              <Button
                variant="contained"
                color="error"
                startIcon={<StopIcon />}
                onClick={handleStopTest}
              >
                Stop Test
              </Button>
            )}
            <Button
              variant="outlined"
              startIcon={<UploadIcon />}
            >
              Import Test Cases
            </Button>
          </Box>
        </Paper>
      </Grid>
    </Grid>
  );

  const renderTestExecution = () => (
    <Grid container spacing={3}>
      {isRunning && testCases.length > 0 && (
        <Grid item xs={12}>
          <Alert severity="info" sx={{ mb: 2 }}>
            Running test case {currentTestCase + 1} of {testCases.length}
          </Alert>
          <LinearProgress
            variant="determinate"
            value={(currentTestCase / testCases.length) * 100}
          />
        </Grid>
      )}

      {testCases.map((testCase, index) => (
        <Grid item xs={12} key={testCase.id}>
          <Paper sx={{ p: 3 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">
                Test Case #{index + 1}
              </Typography>
              {testCase.humanPreference && (
                <Chip
                  icon={<CheckCircleIcon />}
                  label="Evaluated"
                  color="success"
                  size="small"
                />
              )}
            </Box>

            <Typography variant="subtitle2" gutterBottom>
              Input Prompt:
            </Typography>
            <Typography variant="body2" sx={{ mb: 2, p: 1, bgcolor: 'background.default', borderRadius: 1 }}>
              {testCase.input}
            </Typography>

            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="primary" gutterBottom>
                      Model A Response
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      {testCase.modelAOutput || 'Generating response...'}
                    </Typography>

                    <Box display="flex" gap={1}>
                      <Chip
                        icon={<SpeedIcon />}
                        label="450ms"
                        size="small"
                        variant="outlined"
                      />
                      <Chip
                        icon={<PsychologyIcon />}
                        label="Perplexity: 2.3"
                        size="small"
                        variant="outlined"
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="secondary" gutterBottom>
                      Model B Response
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      {testCase.modelBOutput || 'Generating response...'}
                    </Typography>

                    <Box display="flex" gap={1}>
                      <Chip
                        icon={<SpeedIcon />}
                        label="480ms"
                        size="small"
                        variant="outlined"
                      />
                      <Chip
                        icon={<PsychologyIcon />}
                        label="Perplexity: 2.1"
                        size="small"
                        variant="outlined"
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Box mt={3}>
              <Typography variant="subtitle2" gutterBottom>
                Human Preference:
              </Typography>
              <RadioGroup
                row
                value={testCase.humanPreference || ''}
                onChange={(e) => handleHumanPreference(testCase.id, e.target.value as any, 0.8)}
              >
                <FormControlLabel
                  value="A"
                  control={<Radio color="primary" />}
                  label={
                    <Box display="flex" alignItems="center" gap={1}>
                      <ThumbUpIcon fontSize="small" />
                      Model A is better
                    </Box>
                  }
                />
                <FormControlLabel
                  value="B"
                  control={<Radio color="secondary" />}
                  label={
                    <Box display="flex" alignItems="center" gap={1}>
                      <ThumbUpIcon fontSize="small" />
                      Model B is better
                    </Box>
                  }
                />
                <FormControlLabel
                  value="tie"
                  control={<Radio />}
                  label={
                    <Box display="flex" alignItems="center" gap={1}>
                      <ThumbsUpDownIcon fontSize="small" />
                      Tie
                    </Box>
                  }
                />
              </RadioGroup>

              {testCase.humanPreference && (
                <Box mt={2}>
                  <Typography variant="subtitle2" gutterBottom>
                    Confidence Level:
                  </Typography>
                  <Slider
                    value={testCase.confidence || 0.5}
                    onChange={(e, value) => handleHumanPreference(
                      testCase.id,
                      testCase.humanPreference!,
                      value as number
                    )}
                    min={0}
                    max={1}
                    step={0.1}
                    marks={[
                      { value: 0, label: 'Low' },
                      { value: 0.5, label: 'Medium' },
                      { value: 1, label: 'High' }
                    ]}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                  />
                </Box>
              )}
            </Box>
          </Paper>
        </Grid>
      ))}
    </Grid>
  );

  const renderResults = () => {
    const test = mockTests[0]; // Using mock data for demonstration
    const winRates = calculateWinRate(test);

    return (
      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <TrendingUpIcon color="primary" />
                <Typography variant="subtitle2">Winner</Typography>
              </Box>
              <Typography variant="h4">
                Model {test.results?.winnerModel}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {test.results?.confidenceScore ? `${(test.results.confidenceScore * 100).toFixed(0)}% confidence` : ''}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <AssessmentIcon color="secondary" />
                <Typography variant="subtitle2">Statistical Significance</Typography>
              </Box>
              <Typography variant="h4">
                {test.results?.statisticalSignificance ? `${(test.results.statisticalSignificance * 100).toFixed(0)}%` : 'N/A'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                p-value: 0.03
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <ThumbUpIcon color="success" />
                <Typography variant="subtitle2">Human Preference</Typography>
              </Box>
              <Typography variant="h4">
                {test.results?.metrics.modelB.humanPreference ? `${(test.results.metrics.modelB.humanPreference * 100).toFixed(0)}%` : 'N/A'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                for Model B
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <SpeedIcon color="warning" />
                <Typography variant="subtitle2">Avg Latency Diff</Typography>
              </Box>
              <Typography variant="h4">
                +30ms
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Model B vs Model A
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Win Rate Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Win Rate Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={[
                    { name: 'Model A', value: winRates.modelA },
                    { name: 'Model B', value: winRates.modelB },
                    { name: 'Tie', value: winRates.tie }
                  ]}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {[0, 1, 2].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <RechartsTooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Performance Comparison */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Performance Metrics
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart
                data={[
                  {
                    metric: 'Accuracy',
                    modelA: test.results?.metrics.modelA.accuracy || 0,
                    modelB: test.results?.metrics.modelB.accuracy || 0
                  },
                  {
                    metric: 'BLEU Score',
                    modelA: test.results?.metrics.modelA.bleuScore || 0,
                    modelB: test.results?.metrics.modelB.bleuScore || 0
                  },
                  {
                    metric: 'ROUGE Score',
                    modelA: test.results?.metrics.modelA.rougeScore || 0,
                    modelB: test.results?.metrics.modelB.rougeScore || 0
                  },
                  {
                    metric: 'Human Pref',
                    modelA: test.results?.metrics.modelA.humanPreference || 0,
                    modelB: test.results?.metrics.modelB.humanPreference || 0
                  }
                ]}
              >
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis angle={90} domain={[0, 1]} />
                <Radar name="Model A" dataKey="modelA" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} />
                <Radar name="Model B" dataKey="modelB" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.3} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Detailed Metrics Table */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Detailed Comparison
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    <TableCell align="center">Model A</TableCell>
                    <TableCell align="center">Model B</TableCell>
                    <TableCell align="center">Difference</TableCell>
                    <TableCell align="center">Winner</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>Accuracy</TableCell>
                    <TableCell align="center">{test.results?.metrics.modelA.accuracy?.toFixed(2)}</TableCell>
                    <TableCell align="center">{test.results?.metrics.modelB.accuracy?.toFixed(2)}</TableCell>
                    <TableCell align="center">
                      <Chip
                        label={`+${((test.results?.metrics.modelB.accuracy || 0) - (test.results?.metrics.modelA.accuracy || 0)).toFixed(2)}`}
                        color="success"
                        size="small"
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Chip label="Model B" color="primary" size="small" />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>BLEU Score</TableCell>
                    <TableCell align="center">{test.results?.metrics.modelA.bleuScore?.toFixed(2)}</TableCell>
                    <TableCell align="center">{test.results?.metrics.modelB.bleuScore?.toFixed(2)}</TableCell>
                    <TableCell align="center">
                      <Chip
                        label={`+${((test.results?.metrics.modelB.bleuScore || 0) - (test.results?.metrics.modelA.bleuScore || 0)).toFixed(2)}`}
                        color="success"
                        size="small"
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Chip label="Model B" color="primary" size="small" />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Latency (ms)</TableCell>
                    <TableCell align="center">{test.results?.metrics.modelA.avgLatency}</TableCell>
                    <TableCell align="center">{test.results?.metrics.modelB.avgLatency}</TableCell>
                    <TableCell align="center">
                      <Chip
                        label={`+${(test.results?.metrics.modelB.avgLatency || 0) - (test.results?.metrics.modelA.avgLatency || 0)}ms`}
                        color="warning"
                        size="small"
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Chip label="Model A" color="secondary" size="small" />
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>

            <Box mt={3} display="flex" gap={2}>
              <Button
                variant="contained"
                startIcon={<DownloadIcon />}
              >
                Export Results
              </Button>
              <Button
                variant="outlined"
                startIcon={<BarChartIcon />}
              >
                Generate Report
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    );
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        A/B Testing
      </Typography>
      <Typography variant="body2" color="textSecondary" gutterBottom>
        Compare models side-by-side with statistical significance testing
      </Typography>

      <Tabs
        value={activeTab}
        onChange={(e, v) => setActiveTab(v)}
        sx={{ mb: 3 }}
      >
        <Tab label="Configuration" />
        <Tab label="Test Execution" />
        <Tab label="Results" />
        <Tab label="History" />
      </Tabs>

      {activeTab === 0 && renderTestConfiguration()}
      {activeTab === 1 && renderTestExecution()}
      {activeTab === 2 && renderResults()}
      {activeTab === 3 && (
        <Typography color="textSecondary">Test history will be displayed here</Typography>
      )}

      {/* New Test Dialog */}
      <Dialog
        open={showNewTestDialog}
        onClose={() => setShowNewTestDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Create New A/B Test</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Test Name"
                value={newTestForm.name}
                onChange={(e) => setNewTestForm({ ...newTestForm, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                multiline
                rows={2}
                value={newTestForm.description}
                onChange={(e) => setNewTestForm({ ...newTestForm, description: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Model A</InputLabel>
                <Select
                  value={newTestForm.modelA}
                  onChange={(e) => setNewTestForm({ ...newTestForm, modelA: e.target.value })}
                  label="Model A"
                >
                  <MenuItem value="model1">Model 1</MenuItem>
                  <MenuItem value="model2">Model 2</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Model B</InputLabel>
                <Select
                  value={newTestForm.modelB}
                  onChange={(e) => setNewTestForm({ ...newTestForm, modelB: e.target.value })}
                  label="Model B"
                >
                  <MenuItem value="model2">Model 2</MenuItem>
                  <MenuItem value="model3">Model 3</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl component="fieldset">
                <RadioGroup
                  row
                  value={newTestForm.testType}
                  onChange={(e) => setNewTestForm({ ...newTestForm, testType: e.target.value as any })}
                >
                  <FormControlLabel value="manual" control={<Radio />} label="Manual Evaluation" />
                  <FormControlLabel value="automatic" control={<Radio />} label="Automatic Metrics" />
                  <FormControlLabel value="hybrid" control={<Radio />} label="Hybrid" />
                </RadioGroup>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowNewTestDialog(false)}>Cancel</Button>
          <Button onClick={handleCreateTest} variant="contained">Create Test</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
