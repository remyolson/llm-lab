'use client';

import React, { useState } from 'react';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  Paper,
  TextField,
  Grid,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  Chip,
  Alert,
  Autocomplete,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  InputAdornment
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  Upload as UploadIcon,
  Science as ScienceIcon,
  Settings as SettingsIcon,
  Dataset as DatasetIcon,
  Psychology as PsychologyIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Save as SaveIcon,
  PlayArrow as PlayArrowIcon
} from '@mui/icons-material';
import { useRouter } from 'next/navigation';
import { useMutation } from 'react-query';
import { useForm, Controller } from 'react-hook-form';
import type { Recipe, ExperimentForm } from '@/types';

const steps = ['Basic Information', 'Model Configuration', 'Dataset Setup', 'Training Parameters', 'Review & Launch'];

// Mock API function
const createExperiment = async (data: ExperimentForm) => {
  // Simulated API call
  return { success: true, id: 'exp_123' };
};

export default function NewExperimentPage() {
  const router = useRouter();
  const [activeStep, setActiveStep] = useState(0);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { control, handleSubmit, watch, setValue, formState: { errors } } = useForm<ExperimentForm>({
    defaultValues: {
      name: '',
      description: '',
      tags: [],
      recipe: {
        name: '',
        model: {
          name: '',
          baseModel: 'meta-llama/Llama-2-7b-hf',
          modelType: 'causal_lm',
          useFlashAttention: false
        },
        dataset: {
          name: '',
          path: '',
          format: 'jsonl',
          splitRatios: {
            train: 0.8,
            validation: 0.15,
            test: 0.05
          },
          maxSamples: undefined
        },
        training: {
          numEpochs: 3,
          perDeviceTrainBatchSize: 4,
          learningRate: 2e-5,
          useLora: true,
          loraRank: 8,
          loraAlpha: 16,
          loraDropout: 0.1,
          fp16: true,
          bf16: false,
          gradientCheckpointing: true
        },
        evaluation: {
          metrics: ['perplexity', 'accuracy'],
          benchmarks: []
        },
        monitoring: {
          platforms: ['tensorboard'],
          enableAlerts: true,
          resourceMonitoring: true
        }
      }
    }
  });

  const formData = watch();

  const createMutation = useMutation(createExperiment, {
    onSuccess: (data) => {
      router.push(`/experiments/${data.id}`);
    }
  });

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const onSubmit = (data: ExperimentForm) => {
    createMutation.mutate(data);
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Basic Information
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Provide basic details about your fine-tuning experiment
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="name"
                control={control}
                rules={{ required: 'Experiment name is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Experiment Name"
                    placeholder="e.g., Customer Support Chatbot v2"
                    error={!!errors.name}
                    helperText={errors.name?.message}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="description"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Description (Optional)"
                    placeholder="Describe the purpose and goals of this experiment"
                    multiline
                    rows={4}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="tags"
                control={control}
                render={({ field }) => (
                  <Autocomplete
                    {...field}
                    multiple
                    freeSolo
                    options={['production', 'development', 'testing', 'research']}
                    renderTags={(value, getTagProps) =>
                      value.map((option, index) => (
                        <Chip variant="outlined" label={option} {...getTagProps({ index })} />
                      ))
                    }
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Tags"
                        placeholder="Add tags for organization"
                        helperText="Press Enter to add custom tags"
                      />
                    )}
                    onChange={(_, value) => field.onChange(value)}
                  />
                )}
              />
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Model Configuration
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Select and configure the base model for fine-tuning
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="recipe.model.baseModel"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>Base Model</InputLabel>
                    <Select {...field} label="Base Model">
                      <MenuItem value="meta-llama/Llama-2-7b-hf">Llama 2 7B</MenuItem>
                      <MenuItem value="meta-llama/Llama-2-13b-hf">Llama 2 13B</MenuItem>
                      <MenuItem value="mistralai/Mistral-7B-v0.1">Mistral 7B</MenuItem>
                      <MenuItem value="google/flan-t5-base">Flan-T5 Base</MenuItem>
                      <MenuItem value="custom">Custom Model</MenuItem>
                    </Select>
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="recipe.model.modelType"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>Model Type</InputLabel>
                    <Select {...field} label="Model Type">
                      <MenuItem value="causal_lm">Causal Language Model</MenuItem>
                      <MenuItem value="seq2seq">Sequence-to-Sequence</MenuItem>
                      <MenuItem value="classification">Classification</MenuItem>
                    </Select>
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="recipe.model.useFlashAttention"
                control={control}
                render={({ field }) => (
                  <FormControlLabel
                    control={<Switch {...field} checked={field.value} />}
                    label={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography>Use Flash Attention</Typography>
                        <Tooltip title="Flash Attention optimizes memory usage and speeds up training">
                          <InfoIcon fontSize="small" color="action" />
                        </Tooltip>
                      </Box>
                    }
                  />
                )}
              />
            </Grid>

            {/* Model Information Card */}
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <PsychologyIcon color="primary" />
                    <Typography variant="subtitle1">Model Information</Typography>
                  </Box>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">Parameters</Typography>
                      <Typography variant="body1">7B</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">Context Length</Typography>
                      <Typography variant="body1">4096 tokens</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">Architecture</Typography>
                      <Typography variant="body1">Transformer</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">License</Typography>
                      <Typography variant="body1">Custom</Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );

      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Dataset Setup
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Configure your training dataset
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="recipe.dataset.name"
                control={control}
                rules={{ required: 'Dataset name is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Dataset Name"
                    placeholder="e.g., customer_support_conversations"
                    error={!!errors.recipe?.dataset?.name}
                    helperText={errors.recipe?.dataset?.name?.message}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12}>
              <Controller
                name="recipe.dataset.path"
                control={control}
                rules={{ required: 'Dataset path is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Dataset Path"
                    placeholder="/path/to/dataset or huggingface-dataset-id"
                    error={!!errors.recipe?.dataset?.path}
                    helperText={errors.recipe?.dataset?.path?.message || 'Local path or HuggingFace dataset ID'}
                    InputProps={{
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton>
                            <UploadIcon />
                          </IconButton>
                        </InputAdornment>
                      )
                    }}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="recipe.dataset.format"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>Dataset Format</InputLabel>
                    <Select {...field} label="Dataset Format">
                      <MenuItem value="jsonl">JSONL</MenuItem>
                      <MenuItem value="csv">CSV</MenuItem>
                      <MenuItem value="huggingface">HuggingFace</MenuItem>
                      <MenuItem value="parquet">Parquet</MenuItem>
                    </Select>
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Controller
                name="recipe.dataset.maxSamples"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Max Samples (Optional)"
                    type="number"
                    placeholder="Leave empty to use all samples"
                    InputProps={{
                      inputProps: { min: 1 }
                    }}
                  />
                )}
              />
            </Grid>

            {/* Data Split Configuration */}
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Data Split Configuration
              </Typography>
              <Box sx={{ px: 2 }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Typography variant="body2" color="textSecondary">
                      Training: {Math.round(formData.recipe.dataset.splitRatios?.train * 100)}%
                    </Typography>
                    <Controller
                      name="recipe.dataset.splitRatios.train"
                      control={control}
                      render={({ field }) => (
                        <Slider
                          {...field}
                          value={field.value * 100}
                          onChange={(_, value) => field.onChange((value as number) / 100)}
                          min={0}
                          max={100}
                          color="primary"
                        />
                      )}
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Typography variant="body2" color="textSecondary">
                      Validation: {Math.round(formData.recipe.dataset.splitRatios?.validation * 100)}%
                    </Typography>
                    <Controller
                      name="recipe.dataset.splitRatios.validation"
                      control={control}
                      render={({ field }) => (
                        <Slider
                          {...field}
                          value={field.value * 100}
                          onChange={(_, value) => field.onChange((value as number) / 100)}
                          min={0}
                          max={100}
                          color="secondary"
                        />
                      )}
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Typography variant="body2" color="textSecondary">
                      Test: {Math.round((formData.recipe.dataset.splitRatios?.test || 0) * 100)}%
                    </Typography>
                    <Controller
                      name="recipe.dataset.splitRatios.test"
                      control={control}
                      render={({ field }) => (
                        <Slider
                          {...field}
                          value={(field.value || 0) * 100}
                          onChange={(_, value) => field.onChange((value as number) / 100)}
                          min={0}
                          max={100}
                          color="success"
                        />
                      )}
                    />
                  </Grid>
                </Grid>
              </Box>
            </Grid>
          </Grid>
        );

      case 3:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Training Parameters
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Configure hyperparameters for fine-tuning
              </Typography>
            </Grid>

            {/* Basic Parameters */}
            <Grid item xs={12} md={4}>
              <Controller
                name="recipe.training.numEpochs"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Number of Epochs"
                    type="number"
                    InputProps={{
                      inputProps: { min: 1, max: 100 }
                    }}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <Controller
                name="recipe.training.perDeviceTrainBatchSize"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Batch Size"
                    type="number"
                    InputProps={{
                      inputProps: { min: 1, max: 128 }
                    }}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <Controller
                name="recipe.training.learningRate"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Learning Rate"
                    type="number"
                    InputProps={{
                      inputProps: { step: 0.00001 }
                    }}
                  />
                )}
              />
            </Grid>

            {/* LoRA Configuration */}
            <Grid item xs={12}>
              <Accordion expanded={formData.recipe.training.useLora}>
                <AccordionSummary>
                  <FormControlLabel
                    control={
                      <Controller
                        name="recipe.training.useLora"
                        control={control}
                        render={({ field }) => (
                          <Switch
                            {...field}
                            checked={field.value}
                            onClick={(e) => e.stopPropagation()}
                          />
                        )}
                      />
                    }
                    label={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography>Use LoRA (Low-Rank Adaptation)</Typography>
                        <Tooltip title="LoRA reduces memory usage and training time">
                          <InfoIcon fontSize="small" color="action" />
                        </Tooltip>
                      </Box>
                    }
                    onClick={(e) => e.stopPropagation()}
                  />
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      <Controller
                        name="recipe.training.loraRank"
                        control={control}
                        render={({ field }) => (
                          <TextField
                            {...field}
                            fullWidth
                            label="LoRA Rank"
                            type="number"
                            disabled={!formData.recipe.training.useLora}
                            InputProps={{
                              inputProps: { min: 1, max: 128 }
                            }}
                          />
                        )}
                      />
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Controller
                        name="recipe.training.loraAlpha"
                        control={control}
                        render={({ field }) => (
                          <TextField
                            {...field}
                            fullWidth
                            label="LoRA Alpha"
                            type="number"
                            disabled={!formData.recipe.training.useLora}
                            InputProps={{
                              inputProps: { min: 1, max: 256 }
                            }}
                          />
                        )}
                      />
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Controller
                        name="recipe.training.loraDropout"
                        control={control}
                        render={({ field }) => (
                          <TextField
                            {...field}
                            fullWidth
                            label="LoRA Dropout"
                            type="number"
                            disabled={!formData.recipe.training.useLora}
                            InputProps={{
                              inputProps: { min: 0, max: 1, step: 0.01 }
                            }}
                          />
                        )}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>

            {/* Advanced Options */}
            <Grid item xs={12}>
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Advanced Options</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      <Controller
                        name="recipe.training.fp16"
                        control={control}
                        render={({ field }) => (
                          <FormControlLabel
                            control={<Switch {...field} checked={field.value} />}
                            label="Use FP16 (Mixed Precision)"
                          />
                        )}
                      />
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Controller
                        name="recipe.training.bf16"
                        control={control}
                        render={({ field }) => (
                          <FormControlLabel
                            control={<Switch {...field} checked={field.value} />}
                            label="Use BF16"
                          />
                        )}
                      />
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Controller
                        name="recipe.training.gradientCheckpointing"
                        control={control}
                        render={({ field }) => (
                          <FormControlLabel
                            control={<Switch {...field} checked={field.value} />}
                            label="Gradient Checkpointing"
                          />
                        )}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>
          </Grid>
        );

      case 4:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Review & Launch
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Review your configuration before launching the experiment
              </Typography>
            </Grid>

            {/* Summary Cards */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Experiment Details
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <List dense>
                    <ListItem>
                      <ListItemText
                        primary="Name"
                        secondary={formData.name || 'Not specified'}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Description"
                        secondary={formData.description || 'Not specified'}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Tags"
                        secondary={
                          <Box display="flex" gap={0.5} flexWrap="wrap" mt={0.5}>
                            {formData.tags.map(tag => (
                              <Chip key={tag} label={tag} size="small" />
                            ))}
                          </Box>
                        }
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Model Configuration
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <List dense>
                    <ListItem>
                      <ListItemText
                        primary="Base Model"
                        secondary={formData.recipe.model.baseModel}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Model Type"
                        secondary={formData.recipe.model.modelType}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Flash Attention"
                        secondary={formData.recipe.model.useFlashAttention ? 'Enabled' : 'Disabled'}
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Dataset Configuration
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <List dense>
                    <ListItem>
                      <ListItemText
                        primary="Dataset"
                        secondary={formData.recipe.dataset.name}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Format"
                        secondary={formData.recipe.dataset.format}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Data Split"
                        secondary={`Train: ${Math.round(formData.recipe.dataset.splitRatios.train * 100)}% | Val: ${Math.round(formData.recipe.dataset.splitRatios.validation * 100)}% | Test: ${Math.round((formData.recipe.dataset.splitRatios.test || 0) * 100)}%`}
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Training Parameters
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <List dense>
                    <ListItem>
                      <ListItemText
                        primary="Epochs"
                        secondary={formData.recipe.training.numEpochs}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Batch Size"
                        secondary={formData.recipe.training.perDeviceTrainBatchSize}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Learning Rate"
                        secondary={formData.recipe.training.learningRate}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="LoRA"
                        secondary={formData.recipe.training.useLora ? `Enabled (Rank: ${formData.recipe.training.loraRank})` : 'Disabled'}
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            {/* Resource Estimation */}
            <Grid item xs={12}>
              <Alert severity="info" icon={<MemoryIcon />}>
                <Typography variant="subtitle2" gutterBottom>
                  Estimated Resource Requirements
                </Typography>
                <Typography variant="body2">
                  • GPU Memory: ~16 GB
                  • Training Time: ~2-4 hours
                  • Storage: ~5 GB for checkpoints
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        );

      default:
        return null;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Create New Experiment
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <form onSubmit={handleSubmit(onSubmit)}>
        <Paper sx={{ p: 3, mb: 3 }}>
          {renderStepContent(activeStep)}
        </Paper>

        <Box display="flex" justifyContent="space-between">
          <Button
            disabled={activeStep === 0}
            onClick={handleBack}
          >
            Back
          </Button>
          <Box display="flex" gap={2}>
            <Button
              variant="outlined"
              onClick={() => router.push('/experiments')}
            >
              Cancel
            </Button>
            {activeStep === steps.length - 1 ? (
              <Button
                variant="contained"
                type="submit"
                startIcon={<PlayArrowIcon />}
                disabled={createMutation.isLoading}
              >
                {createMutation.isLoading ? 'Creating...' : 'Launch Experiment'}
              </Button>
            ) : (
              <Button
                variant="contained"
                onClick={handleNext}
              >
                Next
              </Button>
            )}
          </Box>
        </Box>
      </form>
    </Box>
  );
}
