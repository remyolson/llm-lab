import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  IconButton,
  Grid,
  Card,
  CardContent,
  CardActions,
  Chip,
  LinearProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  InputAdornment,
  Tabs,
  Tab,
  Tooltip,
  List,
  ListItem,
  ListItemText,
  Collapse,
  Badge,
  Divider,
  Skeleton
} from '@mui/material';
import {
  Search as SearchIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
  FilterList as FilterListIcon,
  Assessment as AssessmentIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Visibility as VisibilityIcon,
  CloudUpload as CloudUploadIcon,
  Storage as StorageIcon,
  Timeline as TimelineIcon,
  BugReport as BugReportIcon,
  ContentCopy as ContentCopyIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  DataObject as DataObjectIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useQuery, useMutation } from 'react-query';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Histogram,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import type { Dataset, DatasetSample, DatasetMetadata } from '@/types';

interface DatasetExplorerProps {
  onDatasetSelect?: (dataset: Dataset) => void;
  selectedDatasetId?: string;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export default function DatasetExplorer({
  onDatasetSelect,
  selectedDatasetId
}: DatasetExplorerProps) {
  const [activeTab, setActiveTab] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [showSampleDialog, setShowSampleDialog] = useState(false);
  const [selectedSample, setSelectedSample] = useState<DatasetSample | null>(null);
  const [filters, setFilters] = useState({
    format: 'all',
    qualityScore: [0, 100],
    sizeRange: 'all'
  });
  const [expandedStats, setExpandedStats] = useState(true);

  // Mock data
  const mockDatasets: Dataset[] = [
    {
      id: 'dataset1',
      name: 'Customer Support Conversations',
      description: 'Collection of customer support chat logs',
      path: '/data/customer_support.jsonl',
      format: 'jsonl',
      size: 1024 * 1024 * 50, // 50MB
      samples: 10000,
      createdAt: '2024-01-15T10:00:00Z',
      updatedAt: '2024-01-15T10:00:00Z',
      metadata: {
        columns: ['input', 'output', 'context'],
        tokenStats: {
          avgLength: 256,
          maxLength: 512,
          minLength: 32,
          totalTokens: 2560000
        },
        qualityScore: 0.85,
        duplicates: 23,
        formatErrors: 5
      },
      preview: [
        {
          id: '1',
          content: {
            input: 'How do I reset my password?',
            output: 'To reset your password, please click on the "Forgot Password" link on the login page...',
            context: 'Customer support'
          },
          tokenCount: 45,
          qualityFlags: []
        }
      ]
    }
  ];

  const onDrop = useCallback((acceptedFiles: File[]) => {
    // Handle file upload
    console.log('Files dropped:', acceptedFiles);
    setShowUploadDialog(true);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/json': ['.json', '.jsonl'],
      'text/csv': ['.csv'],
      'application/x-parquet': ['.parquet']
    }
  });

  const handleDatasetClick = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    if (onDatasetSelect) {
      onDatasetSelect(dataset);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  };

  const getQualityColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  const getQualityLabel = (score: number) => {
    if (score >= 0.8) return 'Excellent';
    if (score >= 0.6) return 'Good';
    if (score >= 0.4) return 'Fair';
    return 'Poor';
  };

  const renderDatasetGrid = () => (
    <Grid container spacing={3}>
      {/* Upload Area */}
      <Grid item xs={12}>
        <Paper
          {...getRootProps()}
          sx={{
            p: 3,
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'divider',
            backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
            cursor: 'pointer',
            textAlign: 'center',
            transition: 'all 0.3s'
          }}
        >
          <input {...getInputProps()} />
          <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            {isDragActive ? 'Drop files here' : 'Drag & drop dataset files'}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            or click to browse (JSON, JSONL, CSV, Parquet)
          </Typography>
        </Paper>
      </Grid>

      {/* Search and Filters */}
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                placeholder="Search datasets..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  )
                }}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Format</InputLabel>
                <Select
                  value={filters.format}
                  onChange={(e) => setFilters({ ...filters, format: e.target.value })}
                  label="Format"
                >
                  <MenuItem value="all">All Formats</MenuItem>
                  <MenuItem value="jsonl">JSONL</MenuItem>
                  <MenuItem value="csv">CSV</MenuItem>
                  <MenuItem value="parquet">Parquet</MenuItem>
                  <MenuItem value="huggingface">HuggingFace</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <Button
                fullWidth
                variant="outlined"
                startIcon={<FilterListIcon />}
              >
                More Filters
              </Button>
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* Dataset Cards */}
      {mockDatasets.map((dataset) => (
        <Grid item xs={12} md={6} lg={4} key={dataset.id}>
          <Card
            sx={{
              cursor: 'pointer',
              transition: 'all 0.3s',
              border: selectedDataset?.id === dataset.id ? '2px solid' : '1px solid',
              borderColor: selectedDataset?.id === dataset.id ? 'primary.main' : 'divider',
              '&:hover': {
                boxShadow: 4,
                transform: 'translateY(-2px)'
              }
            }}
            onClick={() => handleDatasetClick(dataset)}
          >
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
                <Typography variant="h6" gutterBottom>
                  {dataset.name}
                </Typography>
                <Chip
                  label={dataset.format.toUpperCase()}
                  size="small"
                  color="primary"
                  variant="outlined"
                />
              </Box>

              <Typography variant="body2" color="textSecondary" gutterBottom>
                {dataset.description}
              </Typography>

              <Box mt={2}>
                <Grid container spacing={1}>
                  <Grid item xs={6}>
                    <Box display="flex" alignItems="center" gap={0.5}>
                      <StorageIcon fontSize="small" color="action" />
                      <Typography variant="caption">
                        {formatFileSize(dataset.size)}
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box display="flex" alignItems="center" gap={0.5}>
                      <DataObjectIcon fontSize="small" color="action" />
                      <Typography variant="caption">
                        {dataset.samples.toLocaleString()} samples
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>

              {dataset.metadata && (
                <Box mt={2}>
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Typography variant="caption" color="textSecondary">
                      Quality Score
                    </Typography>
                    <Chip
                      label={`${(dataset.metadata.qualityScore * 100).toFixed(0)}% - ${getQualityLabel(dataset.metadata.qualityScore)}`}
                      size="small"
                      color={getQualityColor(dataset.metadata.qualityScore)}
                    />
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={dataset.metadata.qualityScore * 100}
                    color={getQualityColor(dataset.metadata.qualityScore)}
                    sx={{ mt: 1 }}
                  />
                </Box>
              )}

              {dataset.metadata?.formatErrors && dataset.metadata.formatErrors > 0 && (
                <Alert severity="warning" sx={{ mt: 2, py: 0.5 }}>
                  <Typography variant="caption">
                    {dataset.metadata.formatErrors} format errors detected
                  </Typography>
                </Alert>
              )}
            </CardContent>
            <CardActions>
              <Button size="small" startIcon={<VisibilityIcon />}>
                Preview
              </Button>
              <Button size="small" startIcon={<AssessmentIcon />}>
                Analyze
              </Button>
              <IconButton size="small">
                <DownloadIcon />
              </IconButton>
            </CardActions>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  const renderDatasetDetails = () => {
    if (!selectedDataset) {
      return (
        <Box display="flex" alignItems="center" justifyContent="center" height={400}>
          <Typography color="textSecondary">
            Select a dataset to view details
          </Typography>
        </Box>
      );
    }

    const tokenDistribution = [
      { range: '0-100', count: 2000 },
      { range: '100-200', count: 3500 },
      { range: '200-300', count: 2800 },
      { range: '300-400', count: 1200 },
      { range: '400+', count: 500 }
    ];

    return (
      <Grid container spacing={3}>
        {/* Dataset Info */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Dataset Information
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText
                  primary="Name"
                  secondary={selectedDataset.name}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Format"
                  secondary={selectedDataset.format.toUpperCase()}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Size"
                  secondary={formatFileSize(selectedDataset.size)}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Samples"
                  secondary={selectedDataset.samples.toLocaleString()}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="Path"
                  secondary={selectedDataset.path}
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        {/* Token Statistics */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">Token Statistics</Typography>
              <IconButton size="small" onClick={() => setExpandedStats(!expandedStats)}>
                {expandedStats ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </IconButton>
            </Box>

            <Collapse in={expandedStats}>
              <Grid container spacing={2} mb={2}>
                <Grid item xs={6} md={3}>
                  <Box>
                    <Typography variant="caption" color="textSecondary">
                      Average Length
                    </Typography>
                    <Typography variant="h6">
                      {selectedDataset.metadata?.tokenStats?.avgLength || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box>
                    <Typography variant="caption" color="textSecondary">
                      Max Length
                    </Typography>
                    <Typography variant="h6">
                      {selectedDataset.metadata?.tokenStats?.maxLength || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box>
                    <Typography variant="caption" color="textSecondary">
                      Min Length
                    </Typography>
                    <Typography variant="h6">
                      {selectedDataset.metadata?.tokenStats?.minLength || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Box>
                    <Typography variant="caption" color="textSecondary">
                      Total Tokens
                    </Typography>
                    <Typography variant="h6">
                      {selectedDataset.metadata?.tokenStats?.totalTokens?.toLocaleString() || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={tokenDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <RechartsTooltip />
                  <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </Collapse>
          </Paper>
        </Grid>

        {/* Quality Analysis */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Quality Analysis
            </Typography>

            <Box mb={2}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="body2">Overall Quality Score</Typography>
                <Chip
                  label={`${((selectedDataset.metadata?.qualityScore || 0) * 100).toFixed(0)}%`}
                  color={getQualityColor(selectedDataset.metadata?.qualityScore || 0)}
                />
              </Box>
              <LinearProgress
                variant="determinate"
                value={(selectedDataset.metadata?.qualityScore || 0) * 100}
                color={getQualityColor(selectedDataset.metadata?.qualityScore || 0)}
              />
            </Box>

            <List dense>
              <ListItem>
                <ListItemText
                  primary={
                    <Box display="flex" alignItems="center" gap={1}>
                      {selectedDataset.metadata?.duplicates === 0 ? (
                        <CheckCircleIcon color="success" fontSize="small" />
                      ) : (
                        <WarningIcon color="warning" fontSize="small" />
                      )}
                      Duplicate Samples
                    </Box>
                  }
                  secondary={`${selectedDataset.metadata?.duplicates || 0} duplicates found`}
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary={
                    <Box display="flex" alignItems="center" gap={1}>
                      {selectedDataset.metadata?.formatErrors === 0 ? (
                        <CheckCircleIcon color="success" fontSize="small" />
                      ) : (
                        <ErrorIcon color="error" fontSize="small" />
                      )}
                      Format Errors
                    </Box>
                  }
                  secondary={`${selectedDataset.metadata?.formatErrors || 0} errors detected`}
                />
              </ListItem>
            </List>

            <Button
              fullWidth
              variant="outlined"
              startIcon={<BugReportIcon />}
              sx={{ mt: 2 }}
            >
              Run Full Quality Check
            </Button>
          </Paper>
        </Grid>

        {/* Sample Preview */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Sample Preview
            </Typography>

            {selectedDataset.preview && selectedDataset.preview.length > 0 ? (
              <Box>
                {selectedDataset.preview.slice(0, 3).map((sample, index) => (
                  <Card key={sample.id} variant="outlined" sx={{ mb: 1 }}>
                    <CardContent sx={{ py: 1 }}>
                      <Typography variant="caption" color="textSecondary">
                        Sample #{index + 1}
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        <strong>Input:</strong> {sample.content.input}
                      </Typography>
                      {sample.content.output && (
                        <Typography variant="body2" sx={{ mt: 0.5 }}>
                          <strong>Output:</strong> {sample.content.output}
                        </Typography>
                      )}
                      <Box display="flex" gap={1} mt={1}>
                        <Chip
                          label={`${sample.tokenCount} tokens`}
                          size="small"
                          variant="outlined"
                        />
                        {sample.qualityFlags && sample.qualityFlags.length > 0 && (
                          sample.qualityFlags.map(flag => (
                            <Chip
                              key={flag}
                              label={flag}
                              size="small"
                              color="warning"
                              variant="outlined"
                            />
                          ))
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                ))}
                <Button
                  fullWidth
                  variant="text"
                  onClick={() => setShowSampleDialog(true)}
                  sx={{ mt: 1 }}
                >
                  View All Samples
                </Button>
              </Box>
            ) : (
              <Typography variant="body2" color="textSecondary">
                No preview available
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Actions */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Grid container spacing={2}>
              <Grid item>
                <Button
                  variant="contained"
                  startIcon={<CheckCircleIcon />}
                  onClick={() => onDatasetSelect && onDatasetSelect(selectedDataset)}
                >
                  Use This Dataset
                </Button>
              </Grid>
              <Grid item>
                <Button
                  variant="outlined"
                  startIcon={<EditIcon />}
                >
                  Edit Dataset
                </Button>
              </Grid>
              <Grid item>
                <Button
                  variant="outlined"
                  startIcon={<ContentCopyIcon />}
                >
                  Duplicate
                </Button>
              </Grid>
              <Grid item>
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                >
                  Export
                </Button>
              </Grid>
              <Grid item>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteIcon />}
                >
                  Delete
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    );
  };

  return (
    <Box>
      <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)} sx={{ mb: 3 }}>
        <Tab label="Browse Datasets" />
        <Tab label="Dataset Details" />
        <Tab label="Upload New" />
        <Tab label="Quality Reports" />
      </Tabs>

      {activeTab === 0 && renderDatasetGrid()}
      {activeTab === 1 && renderDatasetDetails()}
      {activeTab === 2 && (
        <Typography color="textSecondary">Upload interface will be displayed here</Typography>
      )}
      {activeTab === 3 && (
        <Typography color="textSecondary">Quality reports will be displayed here</Typography>
      )}

      {/* Upload Dialog */}
      <Dialog
        open={showUploadDialog}
        onClose={() => setShowUploadDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Upload Dataset</DialogTitle>
        <DialogContent>
          <Typography>Configure upload settings...</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowUploadDialog(false)}>Cancel</Button>
          <Button variant="contained">Upload</Button>
        </DialogActions>
      </Dialog>

      {/* Sample Dialog */}
      <Dialog
        open={showSampleDialog}
        onClose={() => setShowSampleDialog(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>Dataset Samples</DialogTitle>
        <DialogContent>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>Input</TableCell>
                  <TableCell>Output</TableCell>
                  <TableCell>Tokens</TableCell>
                  <TableCell>Quality</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {selectedDataset?.preview?.map((sample) => (
                  <TableRow key={sample.id}>
                    <TableCell>{sample.id}</TableCell>
                    <TableCell>{sample.content.input}</TableCell>
                    <TableCell>{sample.content.output}</TableCell>
                    <TableCell>{sample.tokenCount}</TableCell>
                    <TableCell>
                      {sample.qualityFlags?.length === 0 ? (
                        <CheckCircleIcon color="success" fontSize="small" />
                      ) : (
                        <WarningIcon color="warning" fontSize="small" />
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSampleDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
