'use client';

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  IconButton,
  Chip,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tabs,
  Tab,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Menu,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip,
  Fab,
  InputAdornment,
  Skeleton,
  Alert,
  Checkbox,
  FormControlLabel
} from '@mui/material';
import {
  Add as AddIcon,
  Search as SearchIcon,
  FilterList as FilterListIcon,
  MoreVert as MoreVertIcon,
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  ContentCopy as ContentCopyIcon,
  Archive as ArchiveIcon,
  CompareArrows as CompareArrowsIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  Science as ScienceIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon,
  HourglassEmpty as HourglassEmptyIcon
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useRouter } from 'next/navigation';
import { format } from 'date-fns';
import type { Experiment } from '@/types';

// Mock API functions
const fetchExperiments = async (filters?: any) => {
  // Simulated API call
  return {
    items: [],
    total: 0,
    page: 1,
    perPage: 10,
    totalPages: 0
  };
};

const deleteExperiment = async (id: string) => {
  // Simulated API call
  return { success: true };
};

const duplicateExperiment = async (id: string) => {
  // Simulated API call
  return { success: true, newId: `${id}_copy` };
};

const archiveExperiment = async (id: string) => {
  // Simulated API call
  return { success: true };
};

export default function ExperimentsPage() {
  const router = useRouter();
  const queryClient = useQueryClient();

  // State
  const [selectedTab, setSelectedTab] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([]);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [filterDialogOpen, setFilterDialogOpen] = useState(false);

  // Filters state
  const [filters, setFilters] = useState({
    status: 'all',
    dateRange: 'all',
    tags: [],
    model: 'all'
  });

  // Fetch experiments
  const { data, isLoading, error } = useQuery(
    ['experiments', page, rowsPerPage, searchQuery, filters],
    () => fetchExperiments({
      page: page + 1,
      perPage: rowsPerPage,
      search: searchQuery,
      ...filters
    }),
    {
      keepPreviousData: true
    }
  );

  // Mutations
  const deleteMutation = useMutation(deleteExperiment, {
    onSuccess: () => {
      queryClient.invalidateQueries('experiments');
      setDeleteDialogOpen(false);
      setSelectedExperiment(null);
    }
  });

  const duplicateMutation = useMutation(duplicateExperiment, {
    onSuccess: (data) => {
      queryClient.invalidateQueries('experiments');
      router.push(`/experiments/${data.newId}`);
    }
  });

  const archiveMutation = useMutation(archiveExperiment, {
    onSuccess: () => {
      queryClient.invalidateQueries('experiments');
    }
  });

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, experimentId: string) => {
    setAnchorEl(event.currentTarget);
    setSelectedExperiment(experimentId);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedExperiment(null);
  };

  const handleDelete = () => {
    if (selectedExperiment) {
      deleteMutation.mutate(selectedExperiment);
    }
  };

  const handleDuplicate = () => {
    if (selectedExperiment) {
      duplicateMutation.mutate(selectedExperiment);
    }
    handleMenuClose();
  };

  const handleArchive = () => {
    if (selectedExperiment) {
      archiveMutation.mutate(selectedExperiment);
    }
    handleMenuClose();
  };

  const handleNewExperiment = () => {
    router.push('/experiments/new');
  };

  const handleExperimentClick = (id: string) => {
    router.push(`/experiments/${id}`);
  };

  const handleBulkAction = (action: string) => {
    // Handle bulk actions
    console.log('Bulk action:', action, selectedExperiments);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" fontSize="small" />;
      case 'running':
        return <PlayArrowIcon color="primary" fontSize="small" />;
      case 'failed':
        return <ErrorIcon color="error" fontSize="small" />;
      case 'paused':
        return <PauseIcon color="warning" fontSize="small" />;
      case 'draft':
        return <EditIcon color="disabled" fontSize="small" />;
      default:
        return <HourglassEmptyIcon color="disabled" fontSize="small" />;
    }
  };

  const getStatusColor = (status: string): any => {
    switch (status) {
      case 'completed': return 'success';
      case 'running': return 'primary';
      case 'failed': return 'error';
      case 'paused': return 'warning';
      case 'draft': return 'default';
      default: return 'default';
    }
  };

  const experiments = data?.items || [];

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Experiments
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Manage and monitor your fine-tuning experiments
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={handleNewExperiment}
          size="large"
        >
          New Experiment
        </Button>
      </Box>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={selectedTab}
          onChange={(e, v) => setSelectedTab(v)}
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab label="All Experiments" />
          <Tab label="Active" />
          <Tab label="Completed" />
          <Tab label="Drafts" />
          <Tab label="Archived" />
        </Tabs>
      </Paper>

      {/* Filters and Search */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              placeholder="Search experiments..."
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
              <InputLabel>Status</InputLabel>
              <Select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                label="Status"
              >
                <MenuItem value="all">All Status</MenuItem>
                <MenuItem value="draft">Draft</MenuItem>
                <MenuItem value="running">Running</MenuItem>
                <MenuItem value="completed">Completed</MenuItem>
                <MenuItem value="failed">Failed</MenuItem>
                <MenuItem value="paused">Paused</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<FilterListIcon />}
              onClick={() => setFilterDialogOpen(true)}
            >
              More Filters
            </Button>
          </Grid>
        </Grid>

        {/* Bulk Actions */}
        {selectedExperiments.length > 0 && (
          <Box mt={2} display="flex" gap={1}>
            <Button
              size="small"
              variant="outlined"
              onClick={() => handleBulkAction('delete')}
            >
              Delete Selected ({selectedExperiments.length})
            </Button>
            <Button
              size="small"
              variant="outlined"
              onClick={() => handleBulkAction('archive')}
            >
              Archive Selected
            </Button>
            <Button
              size="small"
              variant="outlined"
              onClick={() => handleBulkAction('export')}
            >
              Export Selected
            </Button>
          </Box>
        )}
      </Paper>

      {/* Experiments Table */}
      <TableContainer component={Paper}>
        {isLoading ? (
          <Box p={3}>
            {[1, 2, 3, 4, 5].map(i => (
              <Skeleton key={i} height={60} sx={{ mb: 1 }} />
            ))}
          </Box>
        ) : experiments.length === 0 ? (
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            p={8}
          >
            <ScienceIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
            <Typography variant="h6" color="textSecondary" gutterBottom>
              No experiments found
            </Typography>
            <Typography variant="body2" color="textSecondary" mb={3}>
              Create your first experiment to start fine-tuning models
            </Typography>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={handleNewExperiment}
            >
              Create Experiment
            </Button>
          </Box>
        ) : (
          <>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell padding="checkbox">
                    <Checkbox
                      indeterminate={
                        selectedExperiments.length > 0 &&
                        selectedExperiments.length < experiments.length
                      }
                      checked={
                        experiments.length > 0 &&
                        selectedExperiments.length === experiments.length
                      }
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedExperiments(experiments.map((exp: Experiment) => exp.id));
                        } else {
                          setSelectedExperiments([]);
                        }
                      }}
                    />
                  </TableCell>
                  <TableCell>Name</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Model</TableCell>
                  <TableCell>Dataset</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Updated</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {experiments.map((experiment: Experiment) => (
                  <TableRow
                    key={experiment.id}
                    hover
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={selectedExperiments.includes(experiment.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedExperiments([...selectedExperiments, experiment.id]);
                          } else {
                            setSelectedExperiments(
                              selectedExperiments.filter(id => id !== experiment.id)
                            );
                          }
                        }}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </TableCell>
                    <TableCell onClick={() => handleExperimentClick(experiment.id)}>
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="body1" fontWeight={500}>
                          {experiment.name}
                        </Typography>
                      </Box>
                      {experiment.description && (
                        <Typography variant="caption" color="textSecondary">
                          {experiment.description}
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center" gap={0.5}>
                        {getStatusIcon(experiment.status)}
                        <Chip
                          label={experiment.status}
                          color={getStatusColor(experiment.status)}
                          size="small"
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {experiment.recipe.model.baseModel}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {experiment.recipe.dataset.name}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {format(new Date(experiment.createdAt), 'MMM dd, yyyy')}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {format(new Date(experiment.updatedAt), 'MMM dd, HH:mm')}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleMenuOpen(e, experiment.id);
                        }}
                      >
                        <MoreVertIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <TablePagination
              component="div"
              count={data?.total || 0}
              page={page}
              onPageChange={(e, newPage) => setPage(newPage)}
              rowsPerPage={rowsPerPage}
              onRowsPerPageChange={(e) => {
                setRowsPerPage(parseInt(e.target.value, 10));
                setPage(0);
              }}
            />
          </>
        )}
      </TableContainer>

      {/* Action Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => {
          handleMenuClose();
          if (selectedExperiment) {
            handleExperimentClick(selectedExperiment);
          }
        }}>
          <ListItemIcon>
            <EditIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Edit</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleDuplicate}>
          <ListItemIcon>
            <ContentCopyIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Duplicate</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => {
          handleMenuClose();
          // Handle A/B test
        }}>
          <ListItemIcon>
            <CompareArrowsIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>A/B Test</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleArchive}>
          <ListItemIcon>
            <ArchiveIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Archive</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => {
          handleMenuClose();
          // Handle export
        }}>
          <ListItemIcon>
            <DownloadIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Export</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => {
          handleMenuClose();
          setDeleteDialogOpen(true);
        }}>
          <ListItemIcon>
            <DeleteIcon fontSize="small" color="error" />
          </ListItemIcon>
          <ListItemText>Delete</ListItemText>
        </MenuItem>
      </Menu>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Delete Experiment</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this experiment? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleDelete}
            color="error"
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Filter Dialog */}
      <Dialog
        open={filterDialogOpen}
        onClose={() => setFilterDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Advanced Filters</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Date Range</InputLabel>
                <Select
                  value={filters.dateRange}
                  onChange={(e) => setFilters({ ...filters, dateRange: e.target.value })}
                  label="Date Range"
                >
                  <MenuItem value="all">All Time</MenuItem>
                  <MenuItem value="today">Today</MenuItem>
                  <MenuItem value="week">This Week</MenuItem>
                  <MenuItem value="month">This Month</MenuItem>
                  <MenuItem value="custom">Custom Range</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Model Type</InputLabel>
                <Select
                  value={filters.model}
                  onChange={(e) => setFilters({ ...filters, model: e.target.value })}
                  label="Model Type"
                >
                  <MenuItem value="all">All Models</MenuItem>
                  <MenuItem value="llama">Llama</MenuItem>
                  <MenuItem value="mistral">Mistral</MenuItem>
                  <MenuItem value="gpt">GPT</MenuItem>
                  <MenuItem value="custom">Custom</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFilterDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={() => {
              setFilterDialogOpen(false);
              // Apply filters
            }}
            variant="contained"
          >
            Apply Filters
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
