// Core Types for Fine-Tuning Studio

export interface User {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'developer' | 'viewer';
  createdAt: string;
  lastLogin?: string;
}

export interface Experiment {
  id: string;
  name: string;
  description?: string;
  status: 'draft' | 'running' | 'completed' | 'failed' | 'paused';
  recipe: Recipe;
  metrics?: ExperimentMetrics;
  createdAt: string;
  updatedAt: string;
  userId: string;
  version: number;
  tags: string[];
}

export interface Recipe {
  name: string;
  description?: string;
  model: ModelConfig;
  dataset: DatasetConfig;
  training: TrainingConfig;
  evaluation?: EvaluationConfig;
  monitoring?: MonitoringConfig;
}

export interface ModelConfig {
  name: string;
  baseModel: string;
  modelType: 'causal_lm' | 'seq2seq' | 'classification';
  useFlashAttention?: boolean;
}

export interface DatasetConfig {
  name: string;
  path: string;
  format: 'jsonl' | 'csv' | 'huggingface' | 'parquet';
  splitRatios?: {
    train: number;
    validation: number;
    test?: number;
  };
  maxSamples?: number;
  preprocessing?: Record<string, any>;
}

export interface TrainingConfig {
  numEpochs: number;
  perDeviceTrainBatchSize: number;
  learningRate: number;
  useLora?: boolean;
  loraRank?: number;
  loraAlpha?: number;
  loraDropout?: number;
  fp16?: boolean;
  bf16?: boolean;
  gradientCheckpointing?: boolean;
}

export interface EvaluationConfig {
  metrics: string[];
  benchmarks?: string[];
  customEvalFunction?: string;
  evalFunctionConfig?: Record<string, any>;
}

export interface MonitoringConfig {
  platforms: string[];
  projectName?: string;
  enableAlerts?: boolean;
  resourceMonitoring?: boolean;
}

export interface ExperimentMetrics {
  loss: number[];
  accuracy?: number[];
  perplexity?: number[];
  learningRate: number[];
  timestamps: string[];
  evaluationResults?: EvaluationResult[];
}

export interface EvaluationResult {
  benchmark: string;
  score: number;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface TrainingJob {
  id: string;
  experimentId: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  currentEpoch?: number;
  totalEpochs?: number;
  estimatedTimeRemaining?: number;
  gpuUtilization?: number;
  memoryUsage?: number;
  currentLoss?: number;
  logMessages: LogMessage[];
  startedAt?: string;
  completedAt?: string;
}

export interface LogMessage {
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
  source?: string;
}

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  path: string;
  format: string;
  size: number;
  samples: number;
  createdAt: string;
  updatedAt: string;
  metadata: DatasetMetadata;
  preview?: DatasetSample[];
}

export interface DatasetMetadata {
  columns?: string[];
  tokenStats?: {
    avgLength: number;
    maxLength: number;
    minLength: number;
    totalTokens: number;
  };
  qualityScore?: number;
  duplicates?: number;
  formatErrors?: number;
}

export interface DatasetSample {
  id: string;
  content: Record<string, any>;
  tokenCount?: number;
  qualityFlags?: string[];
}

export interface DeployedModel {
  id: string;
  name: string;
  experimentId: string;
  checkpointPath: string;
  endpoint?: string;
  status: 'deploying' | 'active' | 'inactive' | 'failed';
  provider: 'huggingface' | 'local' | 'aws' | 'gcp' | 'azure';
  deployedAt: string;
  lastHealthCheck?: string;
  metrics?: DeploymentMetrics;
}

export interface DeploymentMetrics {
  requestCount: number;
  avgLatency: number;
  errorRate: number;
  uptime: number;
  lastUpdated: string;
}

export interface ABTest {
  id: string;
  name: string;
  description?: string;
  modelA: string;
  modelB: string;
  status: 'draft' | 'running' | 'completed';
  testCases: ABTestCase[];
  results?: ABTestResults;
  createdAt: string;
  updatedAt: string;
}

export interface ABTestCase {
  id: string;
  input: string;
  expectedOutput?: string;
  modelAOutput?: string;
  modelBOutput?: string;
  humanPreference?: 'A' | 'B' | 'tie';
  confidence?: number;
}

export interface ABTestResults {
  winnerModel: 'A' | 'B' | 'tie';
  confidenceScore: number;
  statisticalSignificance: number;
  metrics: {
    modelA: TestMetrics;
    modelB: TestMetrics;
  };
}

export interface TestMetrics {
  accuracy?: number;
  bleuScore?: number;
  rougeScore?: number;
  humanPreference: number;
  avgLatency: number;
}

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actionUrl?: string;
}

// WebSocket Event Types
export interface WebSocketEvent {
  type: string;
  data: any;
  timestamp: string;
}

export interface TrainingProgressEvent extends WebSocketEvent {
  type: 'training_progress';
  data: {
    experimentId: string;
    epoch: number;
    loss: number;
    accuracy?: number;
    progress: number;
  };
}

export interface DeploymentStatusEvent extends WebSocketEvent {
  type: 'deployment_status';
  data: {
    modelId: string;
    status: string;
    message?: string;
  };
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  perPage: number;
  totalPages: number;
}

// Form Types
export interface ExperimentForm {
  name: string;
  description?: string;
  recipe: Recipe;
  tags: string[];
}

export interface DatasetUploadForm {
  name: string;
  description?: string;
  file: File;
  format: string;
}

// Chart Data Types
export interface ChartDataPoint {
  x: number | string;
  y: number;
  label?: string;
}

export interface TimeSeriesData {
  timestamp: string;
  value: number;
  label?: string;
}

// Theme Types
export interface ThemeConfig {
  mode: 'light' | 'dark';
  primaryColor: string;
  secondaryColor: string;
}

// Settings Types
export interface UserSettings {
  theme: ThemeConfig;
  notifications: {
    email: boolean;
    browser: boolean;
    training: boolean;
    deployment: boolean;
  };
  defaultSettings: {
    autoSave: boolean;
    autoRefresh: boolean;
    refreshInterval: number;
  };
}
