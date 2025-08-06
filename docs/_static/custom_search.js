/**
 * Enhanced Search Functionality for LLM Lab Documentation
 *
 * This script enhances the built-in Sphinx search with:
 * - Search suggestions and autocomplete
 * - Search analytics tracking
 * - Better result ranking
 * - Search history
 */

(function() {
    'use strict';

    // Search enhancement configuration
    const searchConfig = {
        minSearchLength: 2,
        maxSuggestions: 10,
        searchDelay: 300,
        highlightResults: true,
        trackAnalytics: true,
        searchHistory: []
    };

    // Common search terms for autocomplete
    const commonSearchTerms = [
        'provider', 'openai', 'anthropic', 'google', 'benchmark',
        'evaluation', 'fine-tuning', 'monitoring', 'api', 'configuration',
        'installation', 'quickstart', 'tutorial', 'example', 'generate',
        'batch', 'streaming', 'cost', 'tracking', 'metrics', 'results',
        'comparison', 'testing', 'deployment', 'docker', 'kubernetes'
    ];

    /**
     * Initialize enhanced search functionality
     */
    function initEnhancedSearch() {
        const searchBox = document.querySelector('input[name="q"]');
        if (!searchBox) return;

        // Add autocomplete functionality
        setupAutocomplete(searchBox);

        // Track search queries
        setupAnalytics(searchBox);

        // Enhance search results display
        enhanceSearchResults();

        // Add search shortcuts
        setupKeyboardShortcuts(searchBox);
    }

    /**
     * Setup autocomplete functionality
     */
    function setupAutocomplete(searchBox) {
        // Create suggestions container
        const suggestionsContainer = document.createElement('div');
        suggestionsContainer.className = 'search-suggestions';
        suggestionsContainer.style.cssText = `
            position: absolute;
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-height: 200px;
            overflow-y: auto;
            display: none;
            z-index: 1000;
            width: 100%;
        `;
        searchBox.parentNode.style.position = 'relative';
        searchBox.parentNode.appendChild(suggestionsContainer);

        let debounceTimer;

        searchBox.addEventListener('input', function(e) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                const query = e.target.value.toLowerCase();
                if (query.length >= searchConfig.minSearchLength) {
                    showSuggestions(query, suggestionsContainer, searchBox);
                } else {
                    suggestionsContainer.style.display = 'none';
                }
            }, searchConfig.searchDelay);
        });

        // Hide suggestions on click outside
        document.addEventListener('click', function(e) {
            if (!searchBox.parentNode.contains(e.target)) {
                suggestionsContainer.style.display = 'none';
            }
        });
    }

    /**
     * Show search suggestions
     */
    function showSuggestions(query, container, searchBox) {
        const suggestions = getSuggestions(query);

        if (suggestions.length === 0) {
            container.style.display = 'none';
            return;
        }

        container.innerHTML = '';
        suggestions.forEach(suggestion => {
            const item = document.createElement('div');
            item.className = 'suggestion-item';
            item.style.cssText = `
                padding: 8px 12px;
                cursor: pointer;
                transition: background-color 0.2s;
            `;
            item.textContent = suggestion;

            item.addEventListener('mouseenter', () => {
                item.style.backgroundColor = '#f0f0f0';
            });

            item.addEventListener('mouseleave', () => {
                item.style.backgroundColor = 'white';
            });

            item.addEventListener('click', () => {
                searchBox.value = suggestion;
                container.style.display = 'none';
                searchBox.form.submit();
            });

            container.appendChild(item);
        });

        container.style.display = 'block';
    }

    /**
     * Get suggestions based on query
     */
    function getSuggestions(query) {
        // Combine common terms with search history
        const allTerms = [...new Set([...commonSearchTerms, ...searchConfig.searchHistory])];

        // Filter and sort suggestions
        const suggestions = allTerms
            .filter(term => term.toLowerCase().includes(query))
            .sort((a, b) => {
                // Prioritize terms that start with the query
                const aStarts = a.toLowerCase().startsWith(query);
                const bStarts = b.toLowerCase().startsWith(query);
                if (aStarts && !bStarts) return -1;
                if (!aStarts && bStarts) return 1;
                return a.length - b.length;
            })
            .slice(0, searchConfig.maxSuggestions);

        return suggestions;
    }

    /**
     * Setup search analytics tracking
     */
    function setupAnalytics(searchBox) {
        searchBox.form.addEventListener('submit', function(e) {
            const query = searchBox.value.trim();
            if (query && searchConfig.trackAnalytics) {
                trackSearch(query);
            }
        });
    }

    /**
     * Track search query
     */
    function trackSearch(query) {
        // Add to search history
        searchConfig.searchHistory.unshift(query);
        searchConfig.searchHistory = searchConfig.searchHistory.slice(0, 20);

        // Store in localStorage
        try {
            localStorage.setItem('llmlab_search_history', JSON.stringify(searchConfig.searchHistory));
        } catch (e) {
            console.error('Failed to save search history:', e);
        }

        // Send analytics event (if analytics service is configured)
        if (typeof gtag !== 'undefined') {
            gtag('event', 'search', {
                search_term: query,
                page_location: window.location.pathname
            });
        }
    }

    /**
     * Enhance search results display
     */
    function enhanceSearchResults() {
        // Check if we're on the search results page
        if (!window.location.pathname.includes('search.html')) return;

        // Load search history
        try {
            const history = localStorage.getItem('llmlab_search_history');
            if (history) {
                searchConfig.searchHistory = JSON.parse(history);
            }
        } catch (e) {
            console.error('Failed to load search history:', e);
        }

        // Enhance result snippets
        const results = document.querySelectorAll('.search li');
        results.forEach(result => {
            // Add relevance indicators
            const link = result.querySelector('a');
            if (link) {
                const href = link.getAttribute('href');
                if (href.includes('/api/')) {
                    addBadge(result, 'API', '#007bff');
                } else if (href.includes('/examples/')) {
                    addBadge(result, 'Example', '#28a745');
                } else if (href.includes('/guides/')) {
                    addBadge(result, 'Guide', '#ffc107');
                }
            }
        });
    }

    /**
     * Add badge to search result
     */
    function addBadge(element, text, color) {
        const badge = document.createElement('span');
        badge.style.cssText = `
            display: inline-block;
            padding: 2px 6px;
            margin-left: 8px;
            background-color: ${color};
            color: white;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
        `;
        badge.textContent = text;
        element.querySelector('a').appendChild(badge);
    }

    /**
     * Setup keyboard shortcuts
     */
    function setupKeyboardShortcuts(searchBox) {
        document.addEventListener('keydown', function(e) {
            // Ctrl/Cmd + K to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                searchBox.focus();
                searchBox.select();
            }

            // Escape to clear search
            if (e.key === 'Escape' && document.activeElement === searchBox) {
                searchBox.value = '';
                searchBox.blur();
            }
        });

        // Add placeholder hint
        searchBox.placeholder = 'Search docs (Ctrl+K)';
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initEnhancedSearch);
    } else {
        initEnhancedSearch();
    }

})();
