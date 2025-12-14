// API Configuration - Auto-detect URL for Hugging Face
const API_BASE_URL = window.location.origin;

// Session Management
let currentSessionId = null;
let currentAnswer = null;
let currentSources = null;
let currentQuery = null;
let queryHistory = [];

// DOM Elements
const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const btnText = submitBtn.querySelector('.btn-text');
const loader = submitBtn.querySelector('.loader');
const responseSection = document.getElementById('responseSection');
const initialState = document.getElementById('initialState');
const loadingState = document.getElementById('loadingState');
const answerDiv = document.getElementById('answer');
const sourcesDiv = document.getElementById('sources');
const errorToast = document.getElementById('errorToast');
const errorMessage = document.getElementById('errorMessage');
const expandedQueriesDiv = document.getElementById('expandedQueries');
const queriesList = document.getElementById('queriesList');
const exportBtn = document.getElementById('exportBtn');
const githubBtn = document.getElementById('githubBtn');
const historySelect = document.getElementById('historySelect');

// Initialize
function init() {
    currentSessionId = sessionStorage.getItem('sessionId') || generateSessionId();
    sessionStorage.setItem('sessionId', currentSessionId);
    
    // Load history
    const saved = sessionStorage.getItem('queryHistory');
    if (saved) {
        try {
            queryHistory = JSON.parse(saved);
            updateHistoryDropdown();
        } catch (e) {
            queryHistory = [];
        }
    }
    
    setupEventListeners();
    checkHealth();
    
    // Log API URL for debugging
    console.log('API Base URL:', API_BASE_URL);
}

function generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function setupEventListeners() {
    // Submit button
    if (submitBtn) {
        submitBtn.addEventListener('click', handleSubmit);
    }
    
    // Enter key (Ctrl+Enter)
    if (queryInput) {
        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                handleSubmit();
            }
        });
    }
    
    // Example queries
    document.querySelectorAll('.example-item').forEach(item => {
        item.addEventListener('click', () => {
            const query = item.getAttribute('data-query');
            if (query && queryInput) {
                queryInput.value = query;
                handleSubmit();
            }
        });
    });
    
    // Export button
    if (exportBtn) {
        exportBtn.addEventListener('click', exportToPDF);
    }
    
    // GitHub button
    if (githubBtn) {
        githubBtn.addEventListener('click', () => {
            window.open('https://github.com/Adeyemi0/FinSight-RAG-Application-', '_blank');
        });
    }
    
    // History dropdown
    if (historySelect) {
        historySelect.addEventListener('change', (e) => {
            const query = e.target.value;
            if (query && queryInput) {
                queryInput.value = query;
            }
        });
    }
}

// Update history dropdown
function updateHistoryDropdown() {
    if (!historySelect) return;
    
    historySelect.innerHTML = '<option value="">Select a recent query</option>';
    
    queryHistory.slice().reverse().forEach((item, index) => {
        const option = document.createElement('option');
        option.value = item.query;
        option.textContent = item.query.substring(0, 60) + (item.query.length > 60 ? '...' : '');
        historySelect.appendChild(option);
    });
}

// Save to history
function saveToHistory(query, answer) {
    queryHistory.push({
        query,
        answer: answer.substring(0, 500),
        timestamp: new Date().toISOString()
    });
    
    // Keep last 20
    if (queryHistory.length > 20) {
        queryHistory = queryHistory.slice(-20);
    }
    
    try {
        sessionStorage.setItem('queryHistory', JSON.stringify(queryHistory));
        updateHistoryDropdown();
    } catch (e) {
        console.error('Failed to save history:', e);
    }
}

// Main submit handler
async function handleSubmit() {
    if (!queryInput) return;
    
    const query = queryInput.value.trim();
    
    if (!query) {
        showError('Please enter a question');
        return;
    }
    
    // Show loading
    setLoading(true);
    hideError();
    
    // Hide initial state, show loading
    if (initialState) initialState.style.display = 'none';
    if (loadingState) loadingState.style.display = 'flex';
    if (responseSection) responseSection.style.display = 'none';
    
    try {
        const requestData = {
            query,
            ticker: 'ACM',  // Fixed ticker
            doc_types: null,  // No filter
            top_k: 10,
            session_id: currentSessionId
        };
        
        // Use API_BASE_URL which auto-detects for Hugging Face
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data) {
            throw new Error('Empty response from server');
        }
        
        // Store current data for export
        currentQuery = query;
        currentAnswer = data.answer;
        currentSources = data.sources;
        
        // Save to history
        saveToHistory(query, data.answer || '');
        
        // Display results
        displayResults(data);
        
        // Clear input
        queryInput.value = '';
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to process query');
        
        // Show initial state again
        if (loadingState) loadingState.style.display = 'none';
        if (initialState) initialState.style.display = 'flex';
        
    } finally {
        setLoading(false);
    }
}

// Display results
function displayResults(data) {
    if (!data) return;
    
    // Hide loading, show response
    if (loadingState) loadingState.style.display = 'none';
    if (responseSection) responseSection.style.display = 'block';
    
    // Display answer with cache indicator
    if (answerDiv) {
        let answerHTML = formatAnswer(data.answer || 'No answer available');
        
        // Add cache indicator if from cache
        if (data.from_cache) {
            const cacheAge = Math.floor((data.cache_age_seconds || 0) / 60);
            const ageText = cacheAge < 1 ? 'just now' : `${cacheAge}m ago`;
            answerHTML = `
                <div class="cache-indicator">
                    <span class="cache-badge">⚡ Cached</span>
                    <span class="cache-details">Retrieved ${ageText} • Hit ${data.cache_hits || 1}x</span>
                </div>
            ` + answerHTML;
        }
        
        answerDiv.innerHTML = answerHTML;
    }
    
    // Display expanded queries
    if (expandedQueriesDiv && queriesList) {
        if (data.expanded_queries && data.expanded_queries.length > 1) {
            expandedQueriesDiv.style.display = 'block';
            queriesList.innerHTML = data.expanded_queries
                .map(q => `<li>${escapeHtml(q)}</li>`)
                .join('');
        } else {
            expandedQueriesDiv.style.display = 'none';
        }
    }
    
    // Display sources
    if (sourcesDiv) {
        displaySources(data.sources || []);
    }
    
    // Scroll to top of right panel
    const rightPanel = document.querySelector('.right-panel');
    if (rightPanel) {
        rightPanel.scrollTop = 0;
    }
}

// Format answer with enhanced styling
function formatAnswer(answer) {
    if (!answer) return '';
    
    let formatted = escapeHtml(answer);
    
    // Line breaks
    formatted = formatted.replace(/\n/g, '<br>');
    
    // Citations
    formatted = formatted.replace(/\[Source (\d+)\]/g, 
        '<span class="citation">[Source $1]</span>');
    
    // Highlight numbers (currency, percentages, ratios)
    formatted = formatted.replace(/\$[\d,]+\.?\d*[BM]?/g, 
        match => `<span class="highlight-number">${match}</span>`);
    formatted = formatted.replace(/\d+\.?\d*%/g, 
        match => `<span class="highlight-number">${match}</span>`);
    
    // Color-code metrics
    formatted = formatted.replace(/(\d+\.?\d+)(x|:1)/g, 
        '<span class="metric-green">$1$2</span>');
    
    // Bold headers (lines ending with:)
    formatted = formatted.replace(/^(.+:)$/gm, '<strong>$1</strong>');
    
    // Create calculation boxes for formulas
    formatted = formatted.replace(/Formula: ([^\n]+)/g, 
        '<div class="calculation-box"><h4>Formula</h4><div class="calculation-step">$1</div></div>');
    
    return formatted;
}

// Display sources with collapsible cards
function displaySources(sources) {
    if (!sourcesDiv) return;
    
    if (!sources || sources.length === 0) {
        sourcesDiv.innerHTML = '<p style="color: var(--text-muted);">No sources available</p>';
        return;
    }
    
    sourcesDiv.innerHTML = sources.map((source, index) => {
        if (!source) return '';
        
        const docTypeLabel = source.doc_type === '10k' ? '10-K Report' : 
                            source.doc_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        return `
            <div class="source-card" id="source-${index}">
                <div class="source-header" onclick="toggleSource(${index})">
                    <div class="source-title">
                        <span class="source-badge">${docTypeLabel}</span>
                        ${escapeHtml(source.filename || 'Unknown')}
                    </div>
                    <div class="source-similarity">
                        Similarity Score: <strong>${source.similarity_score ? (source.similarity_score * 100).toFixed(0) + '%' : 'N/A'}</strong>
                    </div>
                </div>
                <div class="source-content">
                    <div class="source-details">
                        ${source.ticker ? `<span class="source-detail"><strong>Ticker:</strong> ${escapeHtml(source.ticker)}</span>` : ''}
                        ${source.chunk_id ? `<span class="source-detail"><strong>Chunk:</strong> ${escapeHtml(source.chunk_id)}</span>` : ''}
                    </div>
                    <div class="source-preview">
                        "${escapeHtml(source.text_preview || 'No preview available')}"
                    </div>
                </div>
            </div>
        `;
    }).filter(Boolean).join('');
}

// Toggle source expansion
function toggleSource(index) {
    const card = document.getElementById(`source-${index}`);
    if (card) {
        card.classList.toggle('expanded');
    }
}

// Export to PDF function
function exportToPDF() {
    if (!currentAnswer) {
        showError('No answer to export. Please ask a question first.');
        return;
    }
    
    try {
        // Check if jsPDF is loaded
        if (typeof window.jspdf === 'undefined') {
            showError('PDF library not loaded. Please refresh the page.');
            return;
        }
        
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        // Set title
        doc.setFontSize(18);
        doc.setFont(undefined, 'bold');
        doc.text('FinSight Analytics Report', 20, 20);
        
        // Set subtitle
        doc.setFontSize(10);
        doc.setFont(undefined, 'normal');
        doc.setTextColor(100);
        doc.text(`Generated on ${new Date().toLocaleString()}`, 20, 28);
        
        // Query section
        doc.setFontSize(12);
        doc.setFont(undefined, 'bold');
        doc.setTextColor(0);
        doc.text('Query:', 20, 40);
        
        doc.setFont(undefined, 'normal');
        doc.setFontSize(10);
        const queryLines = doc.splitTextToSize(currentQuery || '', 170);
        doc.text(queryLines, 20, 48);
        
        // Answer section
        let yPos = 48 + (queryLines.length * 7) + 10;
        doc.setFontSize(12);
        doc.setFont(undefined, 'bold');
        doc.text('Answer:', 20, yPos);
        
        yPos += 8;
        doc.setFont(undefined, 'normal');
        doc.setFontSize(10);
        
        // Clean answer text (remove HTML tags and format)
        let cleanAnswer = currentAnswer
            .replace(/<[^>]*>/g, '')  // Remove HTML tags
            .replace(/\[Source \d+\]/g, '')  // Remove citation markers
            .replace(/&nbsp;/g, ' ')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&amp;/g, '&');
        
        const answerLines = doc.splitTextToSize(cleanAnswer, 170);
        
        // Add answer with page breaks if needed
        answerLines.forEach((line, index) => {
            if (yPos > 270) {
                doc.addPage();
                yPos = 20;
            }
            doc.text(line, 20, yPos);
            yPos += 7;
        });
        
        // Sources section
        if (currentSources && currentSources.length > 0) {
            yPos += 10;
            if (yPos > 250) {
                doc.addPage();
                yPos = 20;
            }
            
            doc.setFontSize(12);
            doc.setFont(undefined, 'bold');
            doc.text('Sources:', 20, yPos);
            yPos += 8;
            
            doc.setFontSize(9);
            doc.setFont(undefined, 'normal');
            
            currentSources.forEach((source, index) => {
                if (yPos > 270) {
                    doc.addPage();
                    yPos = 20;
                }
                
                doc.setFont(undefined, 'bold');
                doc.text(`[${index + 1}] ${source.filename || 'Unknown'}`, 20, yPos);
                yPos += 5;
                
                doc.setFont(undefined, 'normal');
                doc.setTextColor(100);
                doc.text(`Type: ${source.doc_type || 'N/A'} | Similarity: ${source.similarity_score ? (source.similarity_score * 100).toFixed(0) + '%' : 'N/A'}`, 20, yPos);
                yPos += 8;
                doc.setTextColor(0);
            });
        }
        
        // Save PDF
        const filename = `FinSight_Analysis_${Date.now()}.pdf`;
        doc.save(filename);
        
    } catch (error) {
        console.error('PDF Export Error:', error);
        showError('Failed to export PDF. Please try again.');
    }
}

// Loading state
function setLoading(isLoading) {
    if (!submitBtn) return;
    
    submitBtn.disabled = isLoading;
    
    if (btnText && loader) {
        btnText.style.display = isLoading ? 'none' : 'inline';
        loader.style.display = isLoading ? 'inline-block' : 'none';
    }
}

// Error handling
function showError(message) {
    if (!errorToast || !errorMessage) {
        alert(message);
        return;
    }
    
    errorMessage.textContent = message;
    errorToast.style.display = 'block';
    
    setTimeout(() => {
        errorToast.style.display = 'none';
    }, 5000);
}

function hideError() {
    if (errorToast) {
        errorToast.style.display = 'none';
    }
}

// Utility: Escape HTML
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Health check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            signal: AbortSignal.timeout(5000)
        });
        
        if (!response.ok) {
            console.warn('API health check failed');
        } else {
            console.log('API health check passed');
        }
    } catch (error) {
        console.error('Cannot connect to API:', error);
        // Don't show error on page load for Hugging Face
        // The space might still be starting up
    }
}

// Make toggleSource available globally
window.toggleSource = toggleSource;

// Initialize on load
init();