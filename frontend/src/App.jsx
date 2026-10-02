import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  Package, 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  AlertTriangle, 
  ShieldCheck, 
  CheckCircle2, 
  X, 
  Search, 
  Sparkles, 
  ArrowRightLeft, 
  Info, 
  ThumbsUp, 
  Edit3, 
  Ban,
  LayoutGrid,
  Table as TableIcon,
  ArrowUp,
  ArrowDown,
  Boxes,
  Upload,
  FileText,
  Download,
  Check,
  AlertCircle,
  FileSpreadsheet,
  Layers,
  Calendar,
  HelpCircle,
  RefreshCw,
  Play,
  Sun,
  Moon,
  RotateCcw
} from 'lucide-react';

// API Base URL from environment variable (e.g. VITE_API_BASE_URL=https://my-backend.onrender.com)
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

const getApiUrl = (endpoint) => {
  if (!endpoint) return '';
  return endpoint.startsWith('http') ? endpoint : `${API_BASE}${endpoint}`;
};

const safeFetchJson = async (endpoint, options = {}) => {
  const url = getApiUrl(endpoint);
  try {
    const res = await fetch(url, options);
    const contentType = res.headers.get('content-type') || '';
    
    if (!res.ok) {
      let errorMsg = `HTTP error ${res.status}`;
      if (contentType.includes('application/json')) {
        const errJson = await res.json().catch(() => ({}));
        errorMsg = errJson.detail || errJson.message || errorMsg;
      } else {
        const rawText = await res.text().catch(() => '');
        console.warn(`Non-JSON response from ${url}:`, rawText);
        errorMsg = rawText.slice(0, 200) || errorMsg;
      }
      throw new Error(errorMsg);
    }

    if (contentType.includes('application/json')) {
      return await res.json();
    } else {
      const rawText = await res.text();
      try {
        return JSON.parse(rawText);
      } catch (parseErr) {
        console.warn(`Failed to parse JSON from ${url}. Raw text:`, rawText);
        throw new Error("Invalid JSON response received from API.");
      }
    }
  } catch (err) {
    console.error(`API fetch error for ${url}:`, err);
    throw err;
  }
};

export default function App() {
  const [products, setProducts] = useState([]);
  const [summary, setSummary] = useState({ total: 0, increase: 0, maintain: 0, reduce: 0, high_risk: 0, total_recommended_order_quantity: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);

  // Theme State (Dark / Light)
  const [theme, setTheme] = useState(() => localStorage.getItem('bfwai_theme') || 'dark');

  useEffect(() => {
    localStorage.setItem('bfwai_theme', theme);
    if (theme === 'light') {
      document.documentElement.classList.add('light-theme');
    } else {
      document.documentElement.classList.remove('light-theme');
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Upload & Pipeline State
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStep, setUploadStep] = useState(0); // 0: idle, 1: Upload, 2: Validating, 3: Creating Processed CSV, 4: Forecasting, 5: Market Intel, 6: Decision Engine, 7: Complete
  const [uploadedFilename, setUploadedFilename] = useState('');
  const [validationSummary, setValidationSummary] = useState(null);
  const [verificationStats, setVerificationStats] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);
  const mainFileInputRef = useRef(null);

  // View Mode: 'table' or 'kanban'
  const [viewMode, setViewMode] = useState('table');

  // Filters & Search
  const [searchQuery, setSearchQuery] = useState('');
  const [decisionFilter, setDecisionFilter] = useState('ALL');
  const [riskFilter, setRiskFilter] = useState('ALL');
  const [categoryFilter, setCategoryFilter] = useState('ALL');

  // Sorting
  const [sortField, setSortField] = useState('product_id');
  const [sortDirection, setSortDirection] = useState('asc');

  // Selected Product & Detail Modal
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [productDetailLoading, setProductDetailLoading] = useState(false);
  const [productDetailData, setProductDetailData] = useState(null);
  const [activeDetailTab, setActiveDetailTab] = useState('overview');

  // Human Approval State
  const [customQty, setCustomQty] = useState(0);
  const [approvalNotes, setApprovalNotes] = useState('');
  const [toastMessage, setToastMessage] = useState(null);

  // Load initial data
  useEffect(() => {
    fetchHealth();
    fetchProducts();
  }, []);

  const fetchHealth = async () => {
    try {
      const data = await safeFetchJson('/health');
      setHealthStatus(data);
    } catch (e) {
      console.warn("Backend health check not reached:", e.message);
    }
  };

  const fetchProducts = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await safeFetchJson('/products');
      setProducts(data.products || []);
      
      const prods = data.products || [];
      const incCount = prods.filter(p => p.decision === 'INCREASE').length;
      const mainCount = prods.filter(p => p.decision === 'MAINTAIN').length;
      const redCount = prods.filter(p => p.decision === 'REDUCE').length;
      const highRiskCount = prods.filter(p => p.stockout_risk === 'HIGH').length;
      const totalRecQty = prods.reduce((acc, p) => acc + (p.recommended_order_quantity || 0), 0);

      setSummary({
        total: prods.length,
        increase: data.summary?.increase ?? incCount,
        maintain: data.summary?.maintain ?? mainCount,
        reduce: data.summary?.reduce ?? redCount,
        high_risk: data.summary?.high_risk ?? highRiskCount,
        total_recommended_order_quantity: totalRecQty
      });
    } catch (err) {
      setError(err.message || 'Failed to connect to decision agent API.');
    } finally {
      setLoading(false);
    }
  };


  // Upload CSV File & Run Pipeline
  const handleFileUpload = async (file) => {
    if (!file || !file.name.endsWith('.csv')) {
      alert('Please select a valid .csv dataset file.');
      return;
    }

    setUploading(true);
    setUploadStep(1); // Upload
    setUploadedFilename(file.name);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      setUploadStep(2); // Validating
      await new Promise(r => setTimeout(r, 400));

      setUploadStep(3); // Creating Processed CSV
      await new Promise(r => setTimeout(r, 400));

      setUploadStep(4); // Forecasting
      await new Promise(r => setTimeout(r, 400));

      setUploadStep(5); // Market Intel
      await new Promise(r => setTimeout(r, 400));

      setUploadStep(6); // Decision Engine

      setUploadStep(6); // Decision Engine

      const data = await safeFetchJson('/api/upload-and-run', {
        method: 'POST',
        body: formData
      });

      setUploadStep(7); // Complete
      setValidationSummary(data.validation_summary);
      setVerificationStats(data.verification);

      const newProds = data.products || [];
      setProducts(newProds);
      
      const summaryInfo = data.pipeline_summary || {};
      const totalRecQty = newProds.reduce((acc, p) => acc + (p.recommended_order_quantity || 0), 0);

      setSummary({
        total: newProds.length,
        increase: summaryInfo.INCREASE ?? newProds.filter(p => p.decision === 'INCREASE').length,
        maintain: summaryInfo.MAINTAIN ?? newProds.filter(p => p.decision === 'MAINTAIN').length,
        reduce: summaryInfo.REDUCE ?? newProds.filter(p => p.decision === 'REDUCE').length,
        high_risk: newProds.filter(p => p.stockout_risk === 'HIGH').length,
        total_recommended_order_quantity: summaryInfo.total_recommended_order_quantity ?? totalRecQty
      });

      showToast(`Successfully analyzed ${data.filename} (${newProds.length} products processed)`);
    } catch (err) {
      alert(`Pipeline error: ${err.message}`);
      setUploadStep(0);
    } finally {
      setUploading(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  // Open Product Details
  const handleOpenDetail = async (product, initialTab = 'overview') => {
    setSelectedProduct(product);
    setActiveDetailTab(initialTab);
    setProductDetailLoading(true);
    setProductDetailData(null);
    setApprovalNotes('');

    try {
      const data = await safeFetchJson(`/products/${product.product_id}`);
      setProductDetailData(data);
      const recQty = data.decision?.recommended_order_quantity ?? product.recommended_order_quantity;
      setCustomQty(recQty);
    } catch (e) {
      console.error("Failed to load product details:", e);
    } finally {
      setProductDetailLoading(false);
    }
  };

  // Human Approval Action Submission
  const handleApprovalSubmit = async (action, overrideQty = null) => {
    if (!selectedProduct) return;
    const pid = selectedProduct.product_id;
    const currentDecision = productDetailData?.decision?.decision || selectedProduct.decision;
    const qtyToSubmit = overrideQty !== null ? overrideQty : customQty;

    const payload = {
      product_id: pid,
      decision: currentDecision,
      action: action,
      modified_quantity: action === 'MODIFY' ? parseInt(qtyToSubmit, 10) : undefined,
      notes: approvalNotes || undefined
    };

    try {
      const record = await safeFetchJson('/approval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      showToast(`Action saved: ${action} for ${record.product_name} (${record.final_quantity} units)`);
      fetchProducts();
      if (productDetailData) {
        setProductDetailData({
          ...productDetailData,
          approval: record
        });
      }
    } catch (e) {
      alert(`Approval error: ${e.message}`);
    }
  };

  // Toast feedback trigger
  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage(null);
    }, 4500);
  };

  // Reset System State to default clean state
  const handleResetSystem = async () => {
    if (!window.confirm("Are you sure you want to reset all pipeline data, approvals, and uploaded results?")) return;
    setLoading(true);
    try {
      await safeFetchJson('/api/reset', { method: 'POST' });
      setProducts([]);
      setSearchQuery('');
      setDecisionFilter('ALL');
      setRiskFilter('ALL');
      setCategoryFilter('ALL');
      setUploadStep(0);
      setUploadedFilename('');
      setValidationSummary(null);
      setVerificationStats(null);
      setSelectedProduct(null);
      setProductDetailData(null);

      setSummary({
        total: 0,
        increase: 0,
        maintain: 0,
        reduce: 0,
        high_risk: 0,
        total_recommended_order_quantity: 0
      });
      showToast('Dashboard and pipeline state cleared. Ready for new CSV upload.');
    } catch (err) {
      console.error("Reset failed:", err);
      alert(`Failed to reset system state: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };


  // Table Column Sorting
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Category counts
  const categoryCounts = useMemo(() => {
    const counts = { ALL: products.length };
    products.forEach(p => {
      if (p.category) {
        counts[p.category] = (counts[p.category] || 0) + 1;
      }
    });
    return counts;
  }, [products]);

  const categories = Object.keys(categoryCounts);

  // Filtered & Sorted Products
  const processedProducts = useMemo(() => {
    return products
      .filter(p => {
        const query = searchQuery.trim().toLowerCase();
        const matchesSearch = !query || 
          p.product_name.toLowerCase().includes(query) ||
          p.product_id.toLowerCase().includes(query) ||
          (p.brand && p.brand.toLowerCase().includes(query));

        const matchesDecision = decisionFilter === 'ALL' || p.decision.toUpperCase() === decisionFilter.toUpperCase();
        const matchesRisk = riskFilter === 'ALL' || p.stockout_risk.toUpperCase() === riskFilter.toUpperCase();
        const matchesCategory = categoryFilter === 'ALL' || p.category.toLowerCase() === categoryFilter.toLowerCase();

        return matchesSearch && matchesDecision && matchesRisk && matchesCategory;
      })
      .sort((a, b) => {
        let valA = a[sortField];
        let valB = b[sortField];

        if (typeof valA === 'string') {
          valA = valA.toLowerCase();
          valB = valB.toLowerCase();
        }

        if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
        if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
        return 0;
      });
  }, [products, searchQuery, decisionFilter, riskFilter, categoryFilter, sortField, sortDirection]);

  // Group products for Kanban View
  const kanbanGroups = useMemo(() => {
    return {
      INCREASE: processedProducts.filter(p => p.decision === 'INCREASE'),
      MAINTAIN: processedProducts.filter(p => p.decision === 'MAINTAIN'),
      REDUCE: processedProducts.filter(p => p.decision === 'REDUCE'),
    };
  }, [processedProducts]);

  return (
    <div className="app-container">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="toast-container">
          <div className="toast">
            <CheckCircle2 size={18} color="#10b981" />
            <span>{toastMessage}</span>
          </div>
        </div>
      )}

      {/* Top Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo">
            <Package size={26} />
          </div>
          <div>
            <h1 className="brand-title">BFWAI AI Inventory Decision Agent</h1>
            <div className="brand-subtitle">
              <span>Data-Driven Operational Stock Optimization</span>
              <span className="pipeline-badge">
                <span className="status-dot"></span>
                Pipeline Active
              </span>
            </div>
          </div>
        </div>

        <div className="header-controls">
          <button 
            className="demo-scenario-btn" 
            style={{ background: 'var(--accent-gradient)', border: 'none', padding: '10px 18px', fontSize: '13px' }}
            onClick={() => setShowUploadModal(true)}
          >
            <Upload size={16} />
            <span>Upload Customer CSV</span>
          </button>

          {/* Dark / Light Mode Toggle */}
          <button 
            className="theme-toggle-btn"
            onClick={toggleTheme}
            title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-main)',
              padding: '8px 14px',
              borderRadius: 'var(--radius-md)',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 600,
              transition: 'var(--transition)'
            }}
          >
            {theme === 'dark' ? <Sun size={15} color="#fbbf24" /> : <Moon size={15} color="#6366f1" />}
            <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
          </button>

          {/* Reset System Button */}
          <button 
            className="reset-btn"
            onClick={handleResetSystem}
            title="Reset all pipeline state, approvals, and reload default dataset"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: 'rgba(239, 68, 68, 0.12)',
              border: '1px solid rgba(239, 68, 68, 0.35)',
              color: '#ef4444',
              padding: '8px 14px',
              borderRadius: 'var(--radius-md)',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 600,
              transition: 'var(--transition)'
            }}
          >
            <RotateCcw size={15} />
            <span>Reset</span>
          </button>

          {/* View Mode Toggle */}
          <div className="view-mode-toggle">
            <button 
              className={`view-mode-btn ${viewMode === 'table' ? 'active' : ''}`}
              onClick={() => setViewMode('table')}
              title="Table View"
            >
              <TableIcon size={15} />
              <span>Table</span>
            </button>
            <button 
              className={`view-mode-btn ${viewMode === 'kanban' ? 'active' : ''}`}
              onClick={() => setViewMode('kanban')}
              title="Decision Matrix"
            >
              <LayoutGrid size={15} />
              <span>Matrix</span>
            </button>
          </div>

          <div className="status-pill">
            <span className="status-dot"></span>
            <span>{products.length} Products</span>
          </div>
        </div>
      </header>

      {/* Top Hero Workflow & Upload Banner */}
      <section style={{ 
        background: 'var(--hero-bg)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-lg)',
        padding: '24px 28px',
        marginBottom: '22px',
        boxShadow: 'var(--shadow-card)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '20px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
              <Sparkles size={20} color="#818cf8" />
              <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '18px', fontWeight: 800, color: 'var(--hero-title-color)' }}>
                Customer Workflow Execution Pipeline
              </h2>
            </div>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '780px' }}>
              Upload raw customer inventory CSV. The system automatically inspects headers, populates missing expected columns with NULL, creates <code>data/processed/customer_input.csv</code>, and executes complete demand forecasting, market intelligence, and inventory decisions.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <input 
              type="file" 
              ref={mainFileInputRef} 
              accept=".csv" 
              style={{ display: 'none' }}
              onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
              disabled={uploading}
            />
            <button 
              className="demo-scenario-btn" 
              style={{ background: 'var(--accent-gradient)', border: 'none', padding: '12px 22px', fontSize: '14px', fontWeight: 700 }}
              onClick={() => mainFileInputRef.current?.click()}
              disabled={uploading}
            >
              <Upload size={18} />
              <span>{uploading ? 'Processing Dataset...' : 'Upload Customer CSV'}</span>
            </button>
          </div>
        </div>

        {/* Workflow Progress Indicators */}
        <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '11.5px', fontWeight: 700, color: '#6366f1', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Pipeline Status Progression:
          </div>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', 
            gap: '8px' 
          }}>
            {[
              { step: 1, label: 'Upload' },
              { step: 2, label: 'Validating' },
              { step: 3, label: 'Processed CSV' },
              { step: 4, label: 'Forecasting' },
              { step: 5, label: 'Market Intel' },
              { step: 6, label: 'Decision Engine' },
              { step: 7, label: 'Complete' }
            ].map(s => {
              const isActive = uploadStep === s.step;
              const isDone = uploadStep > s.step || (uploadStep === 7 && s.step === 7);
              const isPending = uploadStep < s.step && uploadStep !== 7;

              return (
                <div key={s.step} style={{
                  background: isDone ? 'rgba(16, 185, 129, 0.15)' : isActive ? 'rgba(99, 102, 241, 0.25)' : 'var(--card-dark-box)',
                  border: `1px solid ${isDone ? '#10b981' : isActive ? '#6366f1' : 'var(--border-subtle)'}`,
                  borderRadius: 'var(--radius-md)',
                  padding: '7px 10px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  transition: 'var(--transition)'
                }}>
                  <div style={{
                    width: '18px',
                    height: '18px',
                    borderRadius: '50%',
                    background: isDone ? '#10b981' : isActive ? '#6366f1' : 'var(--border-subtle)',
                    color: isPending ? 'var(--text-faint)' : 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '10px',
                    fontWeight: 700
                  }}>
                    {isDone ? '✓' : s.step}
                  </div>
                  <span style={{ 
                    fontSize: '11.5px', 
                    fontWeight: isActive ? 700 : isDone ? 600 : 400,
                    color: isDone ? '#10b981' : isActive ? 'var(--text-main)' : 'var(--text-faint)' 
                  }}>
                    {s.label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="modal-overlay" onClick={() => !uploading && setShowUploadModal(false)}>
          <div className="modal-card" style={{ maxWidth: '680px' }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Upload size={22} color="#818cf8" />
                <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '20px', color: 'var(--text-main)' }}>
                  Upload Customer Dataset CSV
                </h2>
              </div>
              <button className="modal-close-btn" disabled={uploading} onClick={() => setShowUploadModal(false)}>
                <X size={20} />
              </button>
            </div>

            <div style={{ padding: '20px 24px' }}>
              <p style={{ fontSize: '13.5px', color: 'var(--text-muted)', marginBottom: '16px' }}>
                Upload any raw customer CSV containing historical sales and inventory. Missing expected columns are automatically populated with NULL values without stopping execution.
              </p>

              {/* Drag & Drop Upload Zone */}
              <div 
                className={`upload-dropzone ${dragActive ? 'active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: '2px dashed rgba(99, 102, 241, 0.4)',
                  borderRadius: 'var(--radius-lg)',
                  padding: '36px 20px',
                  textAlign: 'center',
                  background: dragActive ? 'rgba(99, 102, 241, 0.15)' : 'var(--bg-surface)',
                  cursor: uploading ? 'wait' : 'pointer',
                  transition: 'var(--transition)'
                }}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  accept=".csv" 
                  style={{ display: 'none' }}
                  onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                  disabled={uploading}
                />
                
                <FileSpreadsheet size={42} color="#818cf8" style={{ marginBottom: '12px' }} />
                <h3 style={{ fontSize: '15px', color: 'var(--text-main)', fontWeight: 600, marginBottom: '6px' }}>
                  {uploading ? 'Processing Customer Dataset...' : 'Click or Drop CSV File Here'}
                </h3>
                <p style={{ fontSize: '12px', color: 'var(--text-faint)' }}>
                  Processes all N products dynamically without hardcoded limits.
                </p>
              </div>

              {/* Close / Done Button */}
              {uploadStep === 7 && (
                <div style={{ marginTop: '20px', textAlign: 'right' }}>
                  <button 
                    className="demo-scenario-btn" 
                    style={{ background: '#10b981', color: 'white', border: 'none', padding: '10px 20px' }}
                    onClick={() => setShowUploadModal(false)}
                  >
                    View Analyzed Dashboard Results
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Validation Summary & Verification Card */}
      {validationSummary && (
        <div style={{ 
          background: 'var(--bg-surface)', 
          border: '1px solid var(--border-subtle)', 
          borderRadius: 'var(--radius-lg)', 
          padding: '20px 24px', 
          marginBottom: '22px',
          boxShadow: 'var(--shadow-card)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <ShieldCheck size={22} color="#10b981" />
              <h3 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-main)' }}>
                Customer Data Validation & Verification Summary
              </h3>
            </div>
            <span style={{ fontSize: '12px', color: 'var(--text-faint)', background: 'var(--card-dark-box)', padding: '3px 10px', borderRadius: 'var(--radius-full)', border: '1px solid var(--border-subtle)' }}>
              Source: {uploadedFilename || 'Uploaded CSV'}
            </span>
          </div>

          {/* Validation Metrics Grid: Rows, Detected Products, Processed Products, Categories, Dates */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '12px', marginBottom: '16px' }}>
            <div style={{ background: 'var(--card-dark-box)', padding: '12px 16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Total Rows</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-main)', marginTop: '2px' }}>{validationSummary.num_rows?.toLocaleString()}</div>
            </div>
            <div style={{ background: 'var(--card-dark-box)', padding: '12px 16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Products Detected</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: '#38bdf8', marginTop: '2px' }}>
                {verificationStats?.uploaded_unique_products ?? validationSummary.num_products}
              </div>
            </div>
            <div style={{ background: 'var(--card-dark-box)', padding: '12px 16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Products Processed</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: '#10b981', marginTop: '2px' }}>
                {verificationStats?.processed_unique_products ?? summary.total}
              </div>
            </div>
            <div style={{ background: 'var(--card-dark-box)', padding: '12px 16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Categories</div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: '#818cf8', marginTop: '2px' }}>{validationSummary.num_categories}</div>
            </div>
            <div style={{ background: 'var(--card-dark-box)', padding: '12px 16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Date Horizon</div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-main)', marginTop: '4px' }}>{validationSummary.date_min} &rarr; {validationSummary.date_max}</div>
            </div>
          </div>

          {/* Skipped Products Parity Check */}
          {verificationStats?.skipped_products && verificationStats.skipped_products.length > 0 && (
            <div style={{ background: 'rgba(244, 63, 94, 0.1)', border: '1px solid rgba(244, 63, 94, 0.3)', padding: '12px 16px', borderRadius: 'var(--radius-md)', marginBottom: '12px' }}>
              <div style={{ fontSize: '12.5px', fontWeight: 700, color: '#f43f5e', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <AlertTriangle size={15} />
                <span>Skipped Products Notification:</span>
              </div>
              {verificationStats.skipped_products.map(sp => (
                <div key={sp.product_id} style={{ fontSize: '12px', color: 'var(--text-main)' }}>
                  <strong>{sp.product_id}</strong>: {sp.reason}
                </div>
              ))}
            </div>
          )}

          {/* Missing Columns Section */}
          {validationSummary.missing_columns && validationSummary.missing_columns.length > 0 ? (
            <div style={{ 
              background: 'rgba(245, 158, 11, 0.08)', 
              border: '1px solid rgba(245, 158, 11, 0.3)', 
              padding: '12px 16px', 
              borderRadius: 'var(--radius-md)', 
              marginBottom: '12px' 
            }}>
              <div style={{ fontSize: '12.5px', fontWeight: 700, color: '#f59e0b', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <AlertCircle size={15} />
                <span>Missing Expected Columns (Auto-Populated as NULL):</span>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {validationSummary.missing_columns.map(col => (
                  <span key={col} style={{ background: 'rgba(245, 158, 11, 0.2)', color: 'var(--text-main)', padding: '3px 9px', borderRadius: '4px', fontSize: '11.5px', fontFamily: 'monospace', fontWeight: 600 }}>
                    {col} = NULL
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ fontSize: '12.5px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
              <CheckCircle2 size={15} />
              <span>All expected columns were present in customer upload.</span>
            </div>
          )}

          {/* Data Quality Warnings Section */}
          {validationSummary.data_warnings && validationSummary.data_warnings.length > 0 && (
            <div style={{ background: 'rgba(244, 63, 94, 0.08)', border: '1px solid rgba(244, 63, 94, 0.25)', padding: '12px 16px', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: '12.5px', fontWeight: 700, color: '#f43f5e', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <AlertTriangle size={15} />
                <span>Data Quality Warnings & Observations:</span>
              </div>
              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {validationSummary.data_warnings.map((warn, i) => (
                  <li key={i} style={{ fontSize: '12px', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ color: '#f43f5e' }}>&bull;</span> {warn}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* 4 Downloads Action Bar (Requirement 10) */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        background: 'var(--bg-surface)', 
        border: '1px solid var(--border-subtle)', 
        borderRadius: 'var(--radius-lg)', 
        padding: '14px 22px', 
        marginBottom: '22px', 
        flexWrap: 'wrap', 
        gap: '12px' 
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Download size={20} color="#818cf8" />
          <div>
            <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-main)', display: 'block' }}>Download Functional Outputs</span>
            <span style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>Generated processed datasets, forecasts, decision input, and decision outputs</span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <button 
            className="demo-scenario-btn"
            style={{ background: 'var(--btn-light-bg)', borderColor: 'var(--border-subtle)', color: 'var(--btn-light-color)' }}
            onClick={() => window.open(getApiUrl('/api/download/processed-customer-csv'), '_blank')}
          >
            <Download size={14} />
            <span>Processed Customer CSV</span>
          </button>
          <button 
            className="demo-scenario-btn"
            style={{ background: 'rgba(99, 102, 241, 0.15)', borderColor: 'rgba(99, 102, 241, 0.4)', color: '#6366f1' }}
            onClick={() => window.open(getApiUrl('/api/download/forecasts'), '_blank')}
          >
            <Download size={14} />
            <span>Forecast CSV</span>
          </button>
          <button 
            className="demo-scenario-btn"
            style={{ background: 'rgba(56, 189, 248, 0.15)', borderColor: 'rgba(56, 189, 248, 0.4)', color: '#0284c7' }}
            onClick={() => window.open(getApiUrl('/api/download/decision-input'), '_blank')}
          >
            <Download size={14} />
            <span>Decision Input JSON</span>
          </button>
          <button 
            className="demo-scenario-btn"
            style={{ background: 'rgba(16, 185, 129, 0.18)', borderColor: '#10b981', color: '#059669', fontWeight: 700 }}
            onClick={() => window.open(getApiUrl('/api/download/decisions'), '_blank')}
          >
            <Download size={14} />
            <span>Final Decisions JSON</span>
          </button>

        </div>
      </div>

      {/* Key Metric KPI Cards (INCREASE, MAINTAIN, REDUCE, Total SKUs, Recommended PO) */}
      <section className="metrics-grid">
        <div 
          className={`metric-card total ${decisionFilter === 'ALL' && riskFilter === 'ALL' ? 'active-filter' : ''}`}
          onClick={() => { setDecisionFilter('ALL'); setRiskFilter('ALL'); }}
        >
          <div className="metric-header">
            <span className="metric-label">Total Catalog</span>
            <div className="metric-icon" style={{ background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8' }}>
              <Package size={17} />
            </div>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">{summary.total}</div>
            <span className="metric-percentage">SKUs</span>
          </div>
          <div className="metric-desc">Processed customer products</div>
        </div>

        <div 
          className={`metric-card increase ${decisionFilter === 'INCREASE' ? 'active-filter' : ''}`}
          onClick={() => { setDecisionFilter('INCREASE'); setRiskFilter('ALL'); }}
        >
          <div className="metric-header">
            <span className="metric-label">Increase Orders</span>
            <div className="metric-icon" style={{ background: 'var(--color-increase-bg)', color: 'var(--color-increase)' }}>
              <TrendingUp size={17} />
            </div>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">{summary.increase}</div>
            <span className="metric-percentage">
              {summary.total > 0 ? Math.round((summary.increase / summary.total) * 100) : 0}%
            </span>
          </div>
          <div className="metric-desc">Stockout risk / replenishment</div>
        </div>

        <div 
          className={`metric-card maintain ${decisionFilter === 'MAINTAIN' ? 'active-filter' : ''}`}
          onClick={() => { setDecisionFilter('MAINTAIN'); setRiskFilter('ALL'); }}
        >
          <div className="metric-header">
            <span className="metric-label">Maintain Stock</span>
            <div className="metric-icon" style={{ background: 'var(--color-maintain-bg)', color: 'var(--color-maintain)' }}>
              <ShieldCheck size={17} />
            </div>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">{summary.maintain}</div>
            <span className="metric-percentage">
              {summary.total > 0 ? Math.round((summary.maintain / summary.total) * 100) : 0}%
            </span>
          </div>
          <div className="metric-desc">Balanced inventory coverage</div>
        </div>

        <div 
          className={`metric-card reduce ${decisionFilter === 'REDUCE' ? 'active-filter' : ''}`}
          onClick={() => { setDecisionFilter('REDUCE'); setRiskFilter('ALL'); }}
        >
          <div className="metric-header">
            <span className="metric-label">Reduce Holdings</span>
            <div className="metric-icon" style={{ background: 'var(--color-reduce-bg)', color: 'var(--color-reduce)' }}>
              <TrendingDown size={17} />
            </div>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">{summary.reduce}</div>
            <span className="metric-percentage">
              {summary.total > 0 ? Math.round((summary.reduce / summary.total) * 100) : 0}%
            </span>
          </div>
          <div className="metric-desc">Surplus stock reduction</div>
        </div>

        <div 
          className="metric-card"
          style={{ cursor: 'default' }}
        >
          <div className="metric-header">
            <span className="metric-label">Total Recommended PO</span>
            <div className="metric-icon" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#34d399' }}>
              <Boxes size={17} />
            </div>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">{summary.total_recommended_order_quantity?.toLocaleString()}</div>
            <span className="metric-percentage">units</span>
          </div>
          <div className="metric-desc">Respecting MOQ constraints</div>
        </div>
      </section>

      {/* Category Pills Filter Bar */}
      <div className="category-pills-bar">
        {categories.map(cat => (
          <button
            key={cat}
            className={`category-pill ${categoryFilter === cat ? 'active' : ''}`}
            onClick={() => setCategoryFilter(cat)}
          >
            <span>{cat === 'ALL' ? 'All Categories' : cat}</span>
            <span className="category-pill-count">{categoryCounts[cat]}</span>
          </button>
        ))}
      </div>

      {/* Search & Filter Controls Bar */}
      <div className="controls-bar">
        <div className="search-input-wrapper">
          <Search className="search-icon" size={16} />
          <input 
            type="text" 
            className="search-input" 
            placeholder="Search products by name, ID, or brand..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button className="search-clear-btn" onClick={() => setSearchQuery('')}>
              <X size={14} />
            </button>
          )}
        </div>

        <div className="filter-dropdowns">
          <select 
            className="filter-select"
            value={decisionFilter}
            onChange={(e) => setDecisionFilter(e.target.value)}
          >
            <option value="ALL">All Decisions</option>
            <option value="INCREASE">Increase</option>
            <option value="MAINTAIN">Maintain</option>
            <option value="REDUCE">Reduce</option>
          </select>

          <select 
            className="filter-select"
            value={riskFilter}
            onChange={(e) => setRiskFilter(e.target.value)}
          >
            <option value="ALL">All Risk Levels</option>
            <option value="HIGH">High Risk</option>
            <option value="MEDIUM">Medium Risk</option>
            <option value="LOW">Low Risk</option>
          </select>
        </div>
      </div>

      {/* Main Content Area: Table vs Kanban */}
      {loading ? (
        <div className="loading-box">
          <div className="spinner"></div>
          <p>Loading inventory decisions...</p>
        </div>
      ) : error ? (
        <div className="loading-box" style={{ color: '#f43f5e' }}>
          <AlertTriangle size={32} />
          <p>{error}</p>
          <button className="demo-scenario-btn" onClick={fetchProducts}>Retry</button>
        </div>
      ) : processedProducts.length === 0 ? (
        <div className="loading-box">
          <Package size={32} />
          <p>No products match the selected criteria.</p>
          <button 
            className="demo-scenario-btn"
            onClick={() => { setSearchQuery(''); setDecisionFilter('ALL'); setRiskFilter('ALL'); setCategoryFilter('ALL'); }}
          >
            Clear Filters
          </button>
        </div>
      ) : viewMode === 'table' ? (
        /* TABLE VIEW */
        <div className="table-card">
          <div className="table-wrapper">
            <table className="product-table">
              <thead>
                <tr>
                  <th className="sortable" onClick={() => handleSort('product_id')}>
                    <span className="th-content">
                      Product {sortField === 'product_id' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th className="sortable" onClick={() => handleSort('available_inventory')}>
                    <span className="th-content">
                      Available Stock {sortField === 'available_inventory' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th className="sortable" onClick={() => handleSort('forecast_7d')}>
                    <span className="th-content">
                      7d Demand {sortField === 'forecast_7d' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th>30d Demand</th>
                  <th>Trend</th>
                  <th className="sortable" onClick={() => handleSort('days_of_stock')}>
                    <span className="th-content">
                      Stock Coverage {sortField === 'days_of_stock' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th className="sortable" onClick={() => handleSort('stockout_risk')}>
                    <span className="th-content">
                      Risk {sortField === 'stockout_risk' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th>Market Signal</th>
                  <th className="sortable" onClick={() => handleSort('decision')}>
                    <span className="th-content">
                      Decision {sortField === 'decision' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th className="sortable" onClick={() => handleSort('recommended_order_quantity')}>
                    <span className="th-content">
                      Recommended PO {sortField === 'recommended_order_quantity' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {processedProducts.map(p => {
                  const coverageRatio = p.days_of_stock / Math.max(1, p.supplier_lead_time);
                  const isStockoutRiskHigh = p.days_of_stock < p.supplier_lead_time;
                  
                  return (
                    <tr 
                      key={p.product_id}
                      className={selectedProduct?.product_id === p.product_id ? 'selected-row' : ''}
                      onClick={() => handleOpenDetail(p)}
                    >
                      <td>
                        <div className="product-title-cell">
                          <span className="product-id-tag">{p.product_id}</span>
                          <span className="product-name-text">{p.product_name}</span>
                          <span className="product-cat-text">{p.category} &bull; {p.brand}</span>
                        </div>
                      </td>

                      <td>
                        <div><strong>{(p.available_inventory ?? 0).toLocaleString()}</strong> units</div>
                        <div style={{ fontSize: '11px', color: 'var(--text-faint)' }}>
                          {(p.current_inventory ?? 0).toLocaleString()} on-hand &bull; {p.reserved_inventory ?? 0} reserved
                        </div>
                      </td>

                      <td>
                        <strong>{(p.forecast_7d ?? 0).toLocaleString()}</strong> units
                      </td>

                      <td>
                        {(p.forecast_30d ?? 0).toLocaleString()} units
                      </td>

                      <td>
                        <span className={`trend-badge ${p.sales_trend}`}>
                          {p.sales_trend === 'increasing' && <TrendingUp size={14} />}
                          {p.sales_trend === 'decreasing' && <TrendingDown size={14} />}
                          {p.sales_trend === 'stable' && <Minus size={14} />}
                          {p.sales_trend}
                        </span>
                      </td>

                      <td>
                        <div className="dos-gauge-container">
                          <div className="dos-value-line">
                            <span><strong>{(p.days_of_stock ?? 0).toFixed(1)}d</strong> stock</span>
                            <span style={{ color: 'var(--text-faint)' }}>Lead: {p.supplier_lead_time}d</span>
                          </div>
                          <div className="dos-bar-track">
                            <div 
                              className={`dos-bar-fill ${isStockoutRiskHigh ? 'danger' : coverageRatio < 2.0 ? 'warning' : 'healthy'}`}
                              style={{ width: `${Math.min(100, Math.max(8, (coverageRatio / 3.0) * 100))}%` }}
                            ></div>
                          </div>
                        </div>
                      </td>

                      <td>
                        <span className={`risk-badge ${(p.stockout_risk || 'low').toLowerCase()}`}>
                          {p.stockout_risk}
                        </span>
                      </td>

                      <td>
                        <span className={`market-badge ${(p.market_signal || 'neutral').toLowerCase()}`}>
                          {p.market_signal}
                        </span>
                      </td>

                      <td>
                        <span className={`decision-badge ${(p.decision || 'maintain').toLowerCase()}`}>
                          {p.decision === 'INCREASE' && <TrendingUp size={12} />}
                          {p.decision === 'MAINTAIN' && <ShieldCheck size={12} />}
                          {p.decision === 'REDUCE' && <TrendingDown size={12} />}
                          {p.decision}
                        </span>
                      </td>

                      <td>
                        <strong>{(p.recommended_order_quantity ?? 0).toLocaleString()}</strong> units
                        {p.recommended_order_quantity > 0 && (
                          <div style={{ fontSize: '10.5px', color: 'var(--text-faint)' }}>
                            MOQ: {p.moq}
                          </div>
                        )}
                      </td>

                      <td>
                        <span className={`approval-badge ${(p.approval_status || 'pending').toLowerCase()}`}>
                          {p.approval_status}
                        </span>
                      </td>

                      <td>
                        <button 
                          className="demo-scenario-btn"
                          style={{ padding: '5px 12px', fontSize: '11.5px' }}
                          onClick={(e) => { e.stopPropagation(); handleOpenDetail(p); }}
                        >
                          View Details
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        /* KANBAN / DECISION MATRIX VIEW */
        <div className="kanban-grid">
          {/* INCREASE Column */}
          <div className="kanban-column">
            <div className="kanban-header" style={{ borderTop: '3px solid var(--color-increase)' }}>
              <div className="kanban-title" style={{ color: 'var(--color-increase)' }}>
                <TrendingUp size={16} />
                <span>INCREASE ORDERS</span>
              </div>
              <span className="kanban-count-pill" style={{ background: 'var(--color-increase-bg)', color: 'var(--color-increase)' }}>
                {kanbanGroups.INCREASE.length} SKUs
              </span>
            </div>
            <div className="kanban-cards-list">
              {kanbanGroups.INCREASE.map(p => (
                <div key={p.product_id} className="kanban-card" onClick={() => handleOpenDetail(p)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <span className="product-id-tag">{p.product_id}</span>
                      <div className="product-name-text" style={{ marginTop: '4px' }}>{p.product_name}</div>
                      <div className="product-cat-text">{p.category}</div>
                    </div>
                    <span className="risk-badge high">{p.stockout_risk}</span>
                  </div>

                  <div style={{ background: 'rgba(10,15,29,0.6)', padding: '8px 10px', borderRadius: '6px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Coverage:</span>
                      <strong>{p.days_of_stock.toFixed(1)}d (Lead: {p.supplier_lead_time}d)</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Recommended PO:</span>
                      <strong style={{ color: 'var(--color-increase)' }}>{p.recommended_order_quantity} units</strong>
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--text-faint)' }}>
                    <span>MOQ: {p.moq} units</span>
                    <span className={`approval-badge ${(p.approval_status || 'pending').toLowerCase()}`}>{p.approval_status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* MAINTAIN Column */}
          <div className="kanban-column">
            <div className="kanban-header" style={{ borderTop: '3px solid var(--color-maintain)' }}>
              <div className="kanban-title" style={{ color: 'var(--color-maintain)' }}>
                <ShieldCheck size={16} />
                <span>MAINTAIN STOCK</span>
              </div>
              <span className="kanban-count-pill" style={{ background: 'var(--color-maintain-bg)', color: 'var(--color-maintain)' }}>
                {kanbanGroups.MAINTAIN.length} SKUs
              </span>
            </div>
            <div className="kanban-cards-list">
              {kanbanGroups.MAINTAIN.map(p => (
                <div key={p.product_id} className="kanban-card" onClick={() => handleOpenDetail(p)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <span className="product-id-tag">{p.product_id}</span>
                      <div className="product-name-text" style={{ marginTop: '4px' }}>{p.product_name}</div>
                      <div className="product-cat-text">{p.category}</div>
                    </div>
                    <span className={`risk-badge ${(p.stockout_risk || 'low').toLowerCase()}`}>{p.stockout_risk}</span>
                  </div>

                  <div style={{ background: 'rgba(10,15,29,0.6)', padding: '8px 10px', borderRadius: '6px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Days of Stock:</span>
                      <strong>{p.days_of_stock.toFixed(1)} days</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Status:</span>
                      <span style={{ color: 'var(--color-maintain)' }}>Balanced Stock</span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--text-faint)' }}>
                    <span>Lead time: {p.supplier_lead_time}d</span>
                    <span className={`approval-badge ${(p.approval_status || 'pending').toLowerCase()}`}>{p.approval_status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* REDUCE Column */}
          <div className="kanban-column">
            <div className="kanban-header" style={{ borderTop: '3px solid var(--color-reduce)' }}>
              <div className="kanban-title" style={{ color: 'var(--color-reduce)' }}>
                <TrendingDown size={16} />
                <span>REDUCE HOLDINGS</span>
              </div>
              <span className="kanban-count-pill" style={{ background: 'var(--color-reduce-bg)', color: 'var(--color-reduce)' }}>
                {kanbanGroups.REDUCE.length} SKUs
              </span>
            </div>
            <div className="kanban-cards-list">
              {kanbanGroups.REDUCE.map(p => (
                <div key={p.product_id} className="kanban-card" onClick={() => handleOpenDetail(p)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <span className="product-id-tag">{p.product_id}</span>
                      <div className="product-name-text" style={{ marginTop: '4px' }}>{p.product_name}</div>
                      <div className="product-cat-text">{p.category}</div>
                    </div>
                    <span className="risk-badge low">Overstock</span>
                  </div>

                  <div style={{ background: 'rgba(10,15,29,0.6)', padding: '8px 10px', borderRadius: '6px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Stock Coverage:</span>
                      <strong>{p.days_of_stock.toFixed(0)} days</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Action:</span>
                      <span style={{ color: 'var(--color-reduce)' }}>Surplus Stock</span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--text-faint)' }}>
                    <span>Trend: {p.sales_trend}</span>
                    <span className={`approval-badge ${(p.approval_status || 'pending').toLowerCase()}`}>{p.approval_status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Product Details Modal */}
      {selectedProduct && (
        <div className="modal-overlay" onClick={() => setSelectedProduct(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div className="modal-header">
              <div>
                <span className="product-id-tag">{selectedProduct.product_id}</span>
                <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '22px', marginTop: '4px', color: 'white' }}>
                  {selectedProduct.product_name}
                </h2>
                <p style={{ fontSize: '12.5px', color: 'var(--text-muted)' }}>
                  {selectedProduct.category} &bull; Brand: {selectedProduct.brand}
                </p>
              </div>
              <button className="modal-close-btn" onClick={() => setSelectedProduct(null)}>
                <X size={20} />
              </button>
            </div>

            {/* Modal Tabs Bar */}
            <div className="modal-tabs-bar">
              <button 
                className={`modal-tab-btn ${activeDetailTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveDetailTab('overview')}
              >
                <Boxes size={15} />
                <span>Overview</span>
              </button>
              <button 
                className={`modal-tab-btn ${activeDetailTab === 'ai_rationale' ? 'active' : ''}`}
                onClick={() => setActiveDetailTab('ai_rationale')}
              >
                <Sparkles size={15} />
                <span>AI Rationale</span>
              </button>
              <button 
                className={`modal-tab-btn ${activeDetailTab === 'alternatives' ? 'active' : ''}`}
                onClick={() => setActiveDetailTab('alternatives')}
              >
                <ArrowRightLeft size={15} />
                <span>Alternatives</span>
              </button>
              <button 
                className={`modal-tab-btn ${activeDetailTab === 'approval' ? 'active' : ''}`}
                onClick={() => setActiveDetailTab('approval')}
              >
                <ShieldCheck size={15} />
                <span>Approval</span>
              </button>
            </div>

            {/* Modal Content */}
            <div className="modal-body">
              {productDetailLoading ? (
                <div className="loading-box" style={{ minHeight: '220px' }}>
                  <div className="spinner"></div>
                  <p>Evaluating inventory metrics & AI rationale...</p>
                </div>
              ) : activeDetailTab === 'overview' ? (
                /* OVERVIEW TAB */
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px', marginBottom: '20px' }}>
                    <div style={{ background: 'rgba(10,15,29,0.6)', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Decision Result</div>
                      <div style={{ marginTop: '4px' }}>
                        <span className={`decision-badge ${(selectedProduct.decision || 'maintain').toLowerCase()}`}>
                          {selectedProduct.decision}
                        </span>
                      </div>
                    </div>

                    <div style={{ background: 'rgba(10,15,29,0.6)', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Recommended Purchase Order</div>
                      <div style={{ fontSize: '20px', fontWeight: 800, color: 'white', marginTop: '4px' }}>
                        {(selectedProduct.recommended_order_quantity || 0).toLocaleString()} <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>units</span>
                      </div>
                    </div>

                    <div style={{ background: 'rgba(10,15,29,0.6)', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Stockout Risk</div>
                      <div style={{ marginTop: '6px' }}>
                        <span className={`risk-badge ${(selectedProduct.stockout_risk || 'low').toLowerCase()}`}>
                          {selectedProduct.stockout_risk}
                        </span>
                      </div>
                    </div>
                  </div>

                  <h4 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase' }}>
                    Inventory & Demand Forecast Parameters
                  </h4>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '10px', fontSize: '12.5px' }}>
                    <div style={{ background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>7-Day Forecast:</span>
                      <strong style={{ float: 'right' }}>{(selectedProduct.forecast_7d || 0).toLocaleString()} units</strong>
                    </div>
                    <div style={{ background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>30-Day Forecast:</span>
                      <strong style={{ float: 'right' }}>{(selectedProduct.forecast_30d || 0).toLocaleString()} units</strong>
                    </div>
                    <div style={{ background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>On-Hand Stock:</span>
                      <strong style={{ float: 'right' }}>{(selectedProduct.current_inventory || 0).toLocaleString()}</strong>
                    </div>
                    <div style={{ background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Reserved Stock:</span>
                      <strong style={{ float: 'right' }}>{(selectedProduct.reserved_inventory || 0).toLocaleString()}</strong>
                    </div>
                    <div style={{ background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Incoming Transit:</span>
                      <strong style={{ float: 'right' }}>{(selectedProduct.incoming_inventory || 0).toLocaleString()}</strong>
                    </div>
                    <div style={{ background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Supplier Lead Time:</span>
                      <strong style={{ float: 'right' }}>{selectedProduct.supplier_lead_time} days</strong>
                    </div>
                    <div style={{ background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Supplier MOQ:</span>
                      <strong style={{ float: 'right' }}>{selectedProduct.moq} units</strong>
                    </div>
                    <div style={{ background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Reorder Point:</span>
                      <strong style={{ float: 'right' }}>{selectedProduct.reorder_point} units</strong>
                    </div>
                    <div style={{ background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: 'var(--radius-md)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Market Signal:</span>
                      <strong style={{ float: 'right', textTransform: 'capitalize' }}>{selectedProduct.market_signal}</strong>
                    </div>
                  </div>
                </div>
              ) : activeDetailTab === 'ai_rationale' ? (
                /* AI RATIONALE TAB */
                <div>
                  <div style={{ background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.3)', padding: '16px', borderRadius: 'var(--radius-md)', marginBottom: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#a5b4fc', fontWeight: 700, fontSize: '14px', marginBottom: '8px' }}>
                      <Sparkles size={16} />
                      <span>Executive AI Analysis Narrative</span>
                    </div>
                    <p style={{ fontSize: '13px', lineHeight: 1.6, color: '#e2e8f0' }}>
                      {productDetailData?.ai_explanation || "Analyzing multi-signal supply chain evidence..."}
                    </p>
                  </div>

                  <h4 style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase' }}>
                    Deterministic Signal Breakdown
                  </h4>
                  <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {productDetailData?.decision?.reasons?.map((reason, idx) => (
                      <li key={idx} style={{ background: 'rgba(10,15,29,0.6)', padding: '10px 14px', borderRadius: 'var(--radius-md)', fontSize: '12.5px', color: 'white', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <CheckCircle2 size={15} color="#10b981" style={{ flexShrink: 0 }} />
                        <span>{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : activeDetailTab === 'alternatives' ? (
                /* ALTERNATIVES TAB */
                <div>
                  <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '14px' }}>
                    Candidate SKUs in the catalog for capital re-allocation:
                  </p>
                  {productDetailData?.alternative_products?.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      {productDetailData.alternative_products.map(alt => (
                        <div key={alt.product_id} style={{ background: 'rgba(10,15,29,0.6)', border: '1px solid var(--border-subtle)', padding: '14px', borderRadius: 'var(--radius-md)' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                            <strong style={{ color: 'white', fontSize: '13.5px' }}>{alt.product_name} ({alt.product_id})</strong>
                            <span style={{ fontSize: '11.5px', color: '#34d399', fontWeight: 600 }}>{alt.sales_trend} demand</span>
                          </div>
                          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{alt.reason}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p style={{ fontSize: '12.5px', color: 'var(--text-faint)' }}>No alternative product re-allocations required.</p>
                  )}
                </div>
              ) : (
                /* APPROVAL TAB */
                <div>
                  <h4 style={{ fontSize: '14px', fontWeight: 700, color: 'white', marginBottom: '10px' }}>
                    Human-in-the-Loop Order Review
                  </h4>
                  <p style={{ fontSize: '12.5px', color: 'var(--text-muted)', marginBottom: '16px' }}>
                    Review the system's recommended purchase order quantity and approve, modify, or reject it.
                  </p>

                  <div style={{ background: 'rgba(10,15,29,0.6)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', marginBottom: '20px' }}>
                    <label style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
                      Order Quantity (Units):
                    </label>
                    <input 
                      type="number" 
                      value={customQty} 
                      onChange={(e) => setCustomQty(e.target.value)}
                      style={{ background: 'rgba(15,23,42,0.9)', border: '1px solid var(--border-subtle)', color: 'white', padding: '8px 12px', borderRadius: 'var(--radius-md)', width: '100%', fontSize: '16px', fontWeight: 700, outline: 'none' }}
                    />
                    <label style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'block', marginTop: '12px', marginBottom: '6px' }}>
                      Justification Notes (Optional):
                    </label>
                    <textarea 
                      rows={2} 
                      value={approvalNotes} 
                      onChange={(e) => setApprovalNotes(e.target.value)}
                      placeholder="Add review notes or business context..."
                      style={{ background: 'rgba(15,23,42,0.9)', border: '1px solid var(--border-subtle)', color: 'white', padding: '8px 12px', borderRadius: 'var(--radius-md)', width: '100%', fontSize: '12.5px', outline: 'none', resize: 'vertical' }}
                    />
                  </div>

                  <div style={{ display: 'flex', gap: '10px' }}>
                    <button 
                      className="demo-scenario-btn" 
                      style={{ background: '#10b981', color: 'white', border: 'none', flex: 1, justifyContent: 'center' }}
                      onClick={() => handleApprovalSubmit('APPROVE')}
                    >
                      <ThumbsUp size={14} /> Approve Order
                    </button>
                    <button 
                      className="demo-scenario-btn" 
                      style={{ background: '#38bdf8', color: 'white', border: 'none', flex: 1, justifyContent: 'center' }}
                      onClick={() => handleApprovalSubmit('MODIFY')}
                    >
                      <Edit3 size={14} /> Modify Order
                    </button>
                    <button 
                      className="demo-scenario-btn" 
                      style={{ background: '#f43f5e', color: 'white', border: 'none', flex: 1, justifyContent: 'center' }}
                      onClick={() => handleApprovalSubmit('REJECT')}
                    >
                      <Ban size={14} /> Reject Order
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
