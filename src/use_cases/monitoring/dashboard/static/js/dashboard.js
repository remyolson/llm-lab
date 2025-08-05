/**
 * Dashboard JavaScript
 * 
 * Core functionality for the LLM Monitoring Dashboard including
 * real-time updates, chart management, and user interactions.
 */

// Global dashboard state
const Dashboard = {
    charts: {},
    websocket: null,
    refreshTimer: null,
    config: {},
    
    // Initialize dashboard
    init: function() {
        this.loadConfig();
        this.setupEventHandlers();
        this.initializeCharts();
        this.startRefreshTimer();
        
        if (this.config.enable_websockets) {
            this.initializeWebSocket();
        }
        
        console.log('Dashboard initialized');
    },
    
    // Load dashboard configuration
    loadConfig: function() {
        fetch('/api/v1/config')
            .then(response => response.json())
            .then(config => {
                this.config = config;
                console.log('Dashboard config loaded:', config);
            })
            .catch(error => {
                console.error('Failed to load config:', error);
                this.showNotification('Failed to load configuration', 'error');
            });
    },
    
    // Set up event handlers
    setupEventHandlers: function() {
        // Refresh button
        document.getElementById('refresh-btn')?.addEventListener('click', () => {
            this.refreshDashboard();
        });
        
        // Time range buttons
        document.querySelectorAll('[data-timerange]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setTimeRange(e.target.dataset.timerange);
            });
        });
        
        // Export buttons
        document.querySelectorAll('[data-export]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                this.exportData(e.target.dataset.export);
            });
        });
        
        // Settings button
        document.getElementById('settings-btn')?.addEventListener('click', () => {
            this.showSettings();
        });
        
        // Handle visibility change for performance
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseUpdates();
            } else {
                this.resumeUpdates();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'r':
                        e.preventDefault();
                        this.refreshDashboard();
                        break;
                    case 's':
                        e.preventDefault();
                        this.showSettings();
                        break;
                }
            }
        });
    },
    
    // Initialize charts
    initializeCharts: function() {
        this.createPerformanceChart();
        this.createCostChart();
    },
    
    // Create performance chart
    createPerformanceChart: function() {
        const ctx = document.getElementById('performance-chart');
        if (!ctx) return;
        
        this.charts.performance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Latency (ms)',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    tension: 0.1,
                    fill: true
                }, {
                    label: 'Requests/min',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.1,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    if (context.dataset.label === 'Latency (ms)') {
                                        label += context.parsed.y.toFixed(1) + 'ms';
                                    } else {
                                        label += context.parsed.y.toFixed(0);
                                    }
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Latency (ms)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Requests/min'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    },
    
    // Create cost chart
    createCostChart: function() {
        const ctx = document.getElementById('cost-chart');
        if (!ctx) return;
        
        this.charts.cost = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['OpenAI', 'Anthropic', 'Google', 'Other'],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        '#FF6384',
                        '#36A2EB',
                        '#FFCE56',
                        '#4BC0C0'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                return `${label}: $${value.toFixed(2)} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    },
    
    // Start refresh timer
    startRefreshTimer: function() {
        const interval = (this.config.monitoring?.refresh_interval || 30) * 1000;
        this.refreshTimer = setInterval(() => {
            if (!document.hidden) {
                this.refreshDashboard();
            }
        }, interval);
    },
    
    // Stop refresh timer
    stopRefreshTimer: function() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    },
    
    // Refresh dashboard data
    refreshDashboard: function() {
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            const icon = refreshBtn.querySelector('i');
            icon?.classList.add('fa-spin');
            
            setTimeout(() => {
                icon?.classList.remove('fa-spin');
            }, 1000);
        }
        
        this.loadOverviewMetrics();
        this.loadPerformanceData();
        this.loadCostData();
        this.loadProviderComparison();
        this.loadRecentAlerts();
        this.updateLastRefreshed();
    },
    
    // Load overview metrics
    loadOverviewMetrics: function() {
        fetch('/api/v1/metrics/overview')
            .then(response => response.json())
            .then(data => {
                this.updateOverviewMetrics(data);
                this.updateConnectionStatus(true);
            })
            .catch(error => {
                console.error('Failed to load overview metrics:', error);
                this.updateConnectionStatus(false);
                this.showNotification('Failed to load metrics', 'error');
            });
    },
    
    // Update overview metrics display
    updateOverviewMetrics: function(data) {
        const elements = {
            'total-models': data.total_models,
            'total-requests': this.formatNumber(data.total_requests),
            'avg-latency': this.formatDuration(data.avg_latency),
            'total-cost': this.formatCurrency(data.total_cost)
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value || '-';
                element.classList.add('metric-update');
                setTimeout(() => element.classList.remove('metric-update'), 1000);
            }
        });
        
        // Update uptime
        const uptimeElement = document.getElementById('uptime');
        if (uptimeElement && data.uptime) {
            uptimeElement.textContent = this.formatDuration(data.uptime);
        }
    },
    
    // Load performance data
    loadPerformanceData: function() {
        const timerange = document.querySelector('[data-timerange].active')?.dataset.timerange || '1h';
        
        fetch(`/api/v1/metrics/performance?timerange=${timerange}`)
            .then(response => response.json())
            .then(data => {
                this.updatePerformanceChart(data);
            })
            .catch(error => {
                console.error('Failed to load performance data:', error);
            });
    },
    
    // Update performance chart
    updatePerformanceChart: function(data) {
        if (!this.charts.performance || !data.time_series) return;
        
        const chart = this.charts.performance;
        
        // Sample data for demonstration
        const now = new Date();
        const labels = [];
        const latencyData = [];
        const requestData = [];
        
        for (let i = 23; i >= 0; i--) {
            const time = new Date(now.getTime() - i * 60000);
            labels.push(time.toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit' 
            }));
            
            // Generate sample data
            latencyData.push(Math.random() * 500 + 200);
            requestData.push(Math.random() * 100 + 50);
        }
        
        chart.data.labels = labels;
        chart.data.datasets[0].data = latencyData;
        chart.data.datasets[1].data = requestData;
        chart.update('none');
    },
    
    // Load cost data
    loadCostData: function() {
        fetch('/api/v1/metrics/costs')
            .then(response => response.json())
            .then(data => {
                this.updateCostChart(data);
            })
            .catch(error => {
                console.error('Failed to load cost data:', error);
            });
    },
    
    // Update cost chart
    updateCostChart: function(data) {
        if (!this.charts.cost) return;
        
        const chart = this.charts.cost;
        
        // Sample data for demonstration
        const costData = [
            Math.random() * 50 + 10,  // OpenAI
            Math.random() * 30 + 5,   // Anthropic
            Math.random() * 20 + 2,   // Google
            Math.random() * 15 + 1    // Other
        ];
        
        chart.data.datasets[0].data = costData;
        chart.update('none');
    },
    
    // Load provider comparison
    loadProviderComparison: function() {
        const tbody = document.querySelector('#provider-comparison-table tbody');
        if (!tbody) return;
        
        // Sample data for demonstration
        const providers = [
            {
                provider: 'OpenAI',
                model: 'gpt-4o-mini',
                requests: 1234,
                latency: 0.45,
                success_rate: 99.2,
                cost_per_request: 0.002,
                status: 'online'
            },
            {
                provider: 'Anthropic',
                model: 'claude-3-haiku',
                requests: 856,
                latency: 0.38,
                success_rate: 99.8,
                cost_per_request: 0.001,
                status: 'online'
            },
            {
                provider: 'Google',
                model: 'gemini-1.5-flash',
                requests: 567,
                latency: 0.29,
                success_rate: 98.9,
                cost_per_request: 0.0005,
                status: 'online'
            }
        ];
        
        tbody.innerHTML = providers.map(p => `
            <tr>
                <td><strong>${p.provider}</strong></td>
                <td>${p.model}</td>
                <td>${this.formatNumber(p.requests)}</td>
                <td>${p.latency.toFixed(2)}s</td>
                <td>${p.success_rate.toFixed(1)}%</td>
                <td>${this.formatCurrency(p.cost_per_request)}</td>
                <td><span class="badge bg-success">Online</span></td>
            </tr>
        `).join('');
    },
    
    // Load recent alerts
    loadRecentAlerts: function() {
        fetch('/api/v1/alerts')
            .then(response => response.json())
            .then(data => {
                this.updateAlertsDisplay(data);
            })
            .catch(error => {
                console.error('Failed to load alerts:', error);
            });
    },
    
    // Update alerts display
    updateAlertsDisplay: function(data) {
        const alertsList = document.getElementById('alerts-list');
        const alertCount = document.getElementById('alert-count');
        
        if (!alertsList || !alertCount) return;
        
        const totalAlerts = (data.active_alerts?.length || 0);
        alertCount.textContent = totalAlerts;
        
        if (totalAlerts === 0) {
            alertCount.className = 'badge bg-success';
            alertsList.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-check-circle me-2"></i>
                    No recent alerts
                </div>
            `;
        } else {
            alertCount.className = 'badge bg-danger';
            // Display active alerts (placeholder)
            alertsList.innerHTML = `
                <div class="alert alert-warning alert-sm mb-2">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Sample alert message
                </div>
            `;
        }
    },
    
    // Set time range
    setTimeRange: function(range) {
        document.querySelectorAll('[data-timerange]').forEach(btn => {
            btn.classList.remove('active');
        });
        
        document.querySelector(`[data-timerange="${range}"]`)?.classList.add('active');
        this.loadPerformanceData();
    },
    
    // Export data
    exportData: function(format) {
        this.showNotification(`Exporting data as ${format.toUpperCase()}...`, 'info');
        
        // Placeholder for export functionality
        setTimeout(() => {
            this.showNotification(`Export as ${format.toUpperCase()} will be available soon`, 'warning');
        }, 1000);
    },
    
    // Show settings modal
    showSettings: function() {
        this.showNotification('Settings panel will be available soon', 'info');
    },
    
    // Initialize WebSocket connection
    initializeWebSocket: function() {
        if (typeof io === 'undefined') {
            console.warn('Socket.IO not available, WebSocket disabled');
            return;
        }
        
        this.websocket = io();
        
        this.websocket.on('connect', () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
            this.websocket.emit('subscribe_metrics', { metric_type: 'overview' });
        });
        
        this.websocket.on('disconnect', () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
        });
        
        this.websocket.on('metrics_update', (data) => {
            console.log('Received metrics update:', data);
            if (data.type === 'overview') {
                this.updateOverviewMetrics(data.data);
            }
        });
        
        this.websocket.on('alert_notification', (alert) => {
            this.showNotification(alert.message, alert.severity);
            this.loadRecentAlerts();
        });
    },
    
    // Update connection status
    updateConnectionStatus: function(connected) {
        const statusBadge = document.getElementById('connection-status');
        const wsStatus = document.getElementById('ws-status');
        const apiStatus = document.getElementById('api-status');
        
        if (connected) {
            statusBadge?.classList.remove('bg-danger');
            statusBadge?.classList.add('bg-success');
            statusBadge?.innerHTML = '<i class="fas fa-circle me-1"></i>Connected';
            
            wsStatus?.classList.remove('bg-danger');
            wsStatus?.classList.add('bg-success');
            wsStatus?.textContent = 'Connected';
            
            apiStatus?.classList.remove('bg-danger');
            apiStatus?.classList.add('bg-success');
            apiStatus?.textContent = 'Online';
        } else {
            statusBadge?.classList.remove('bg-success');
            statusBadge?.classList.add('bg-danger');
            statusBadge?.innerHTML = '<i class="fas fa-circle me-1"></i>Disconnected';
            
            wsStatus?.classList.remove('bg-success');
            wsStatus?.classList.add('bg-danger');
            wsStatus?.textContent = 'Disconnected';
            
            apiStatus?.classList.remove('bg-success');
            apiStatus?.classList.add('bg-danger');
            apiStatus?.textContent = 'Offline';
        }
    },
    
    // Update last refreshed timestamp
    updateLastRefreshed: function() {
        const element = document.getElementById('last-updated');
        if (element) {
            const now = new Date();
            element.textContent = now.toISOString().replace('T', ' ').substr(0, 19) + ' UTC';
        }
    },
    
    // Pause updates when tab is hidden
    pauseUpdates: function() {
        this.stopRefreshTimer();
        if (this.websocket) {
            this.websocket.disconnect();
        }
    },
    
    // Resume updates when tab is visible
    resumeUpdates: function() {
        this.startRefreshTimer();
        if (this.config.enable_websockets && !this.websocket?.connected) {
            this.initializeWebSocket();
        }
    },
    
    // Show notification
    showNotification: function(message, type = 'info') {
        const container = document.getElementById('alert-container');
        if (!container) return;
        
        const alertClass = {
            'success': 'alert-success',
            'error': 'alert-danger',
            'warning': 'alert-warning',
            'info': 'alert-info'
        }[type] || 'alert-info';
        
        const alertId = 'alert-' + Date.now();
        const alertHtml = `
            <div id="${alertId}" class="alert ${alertClass} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', alertHtml);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    },
    
    // Utility: Format number
    formatNumber: function(num) {
        if (typeof num !== 'number') return num;
        return new Intl.NumberFormat().format(num);
    },
    
    // Utility: Format duration
    formatDuration: function(seconds) {
        if (typeof seconds !== 'number') return seconds;
        
        if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
        return `${(seconds / 3600).toFixed(1)}h`;
    },
    
    // Utility: Format currency
    formatCurrency: function(amount) {
        if (typeof amount !== 'number') return amount;
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: amount < 0.01 ? 4 : 2
        }).format(amount);
    },
    
    // Cleanup on page unload
    cleanup: function() {
        this.stopRefreshTimer();
        if (this.websocket) {
            this.websocket.disconnect();
        }
        
        // Cleanup charts
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
    }
};

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    Dashboard.init();
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    Dashboard.cleanup();
});

// Global error handler
window.addEventListener('error', function(event) {
    console.error('Dashboard error:', event.error);
    Dashboard.showNotification('An unexpected error occurred', 'error');
});

// Export Dashboard object for debugging
window.Dashboard = Dashboard;