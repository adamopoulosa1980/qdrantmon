#!/usr/bin/env python3
"""
Markdown Report Generator for Enhanced Qdrant Monitor
Supports Dense, Sparse, and Named vectors
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class MarkdownReportGeneratorEnhanced:
    """Generate Markdown reports for all vector types"""
    
    def __init__(self, monitor_instance):
        self.monitor = monitor_instance
    
    def generate_markdown_report(self, 
                                collection_names: Optional[List[str]] = None,
                                output_dir: str = "./health_reports") -> str:
        """Generate comprehensive Markdown report"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qdrant_health_report_{timestamp}.md"
        filepath = Path(output_dir) / filename
        
        report = self.monitor.generate_report(collection_names)
        md_content = self._build_markdown(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(filepath)
    
    def _build_markdown(self, report: Dict) -> str:
        """Build complete Markdown document"""
        lines = []
        
        lines.append(self._build_header(report))
        lines.append(self._build_executive_summary(report))
        lines.append(self._build_vector_type_overview(report))
        lines.append(self._build_collection_details(report))
        lines.append(self._build_metric_guide(report))
        lines.append(self._build_footer())
        
        return '\n'.join(lines)
    
    def _build_header(self, report: Dict) -> str:
        """Build report header"""
        lines = [
            '# 📊 Qdrant Knowledge Base Health Report',
            '',
            f'**Generated**: {report["timestamp"]}',
            f'**Qdrant URL**: `{report["qdrant_url"]}`',
            f'**Collections Analyzed**: {report["summary"]["total_collections"]}',
            f'**Average Health Score**: {report["summary"]["avg_health_score"]:.1f}/100',
            '',
            '---',
            ''
        ]
        return '\n'.join(lines)
    
    def _build_executive_summary(self, report: Dict) -> str:
        """Build executive summary"""
        lines = ['## 🎯 Executive Summary', '']
        
        avg_score = report['summary']['avg_health_score']
        if avg_score >= 85:
            status = '🟢 **EXCELLENT** - Knowledge base is healthy'
        elif avg_score >= 75:
            status = '🟡 **GOOD** - Minor issues detected'
        elif avg_score >= 50:
            status = '🟠 **FAIR** - Several issues require attention'
        else:
            status = '🔴 **CRITICAL** - Do not use in production'
        
        lines.append(status)
        lines.append('')
        
        if report['summary']['critical_issues']:
            lines.append('### ⚠️ Critical Issues')
            for issue in report['summary']['critical_issues'][:5]:
                lines.append(f'- {issue}')
            lines.append('')
        
        lines.append('### 📈 Health Snapshot')
        lines.append('')
        lines.append('| Collection | Type | Points | Score | Status |')
        lines.append('|-----------|------|--------|-------|--------|')
        
        for coll_name, data in report['collections'].items():
            if data['health']:
                h = data['health']
                vtype = data['vector_type'].upper()
                points = h['point_count']
                score = h['health_score']
                bar = self._get_score_bar(score)
                lines.append(f'| {coll_name} | {vtype} | {points} | {score:.0f} | {bar} |')
        
        lines.append('')
        lines.append('---')
        lines.append('')
        
        return '\n'.join(lines)
    
    def _build_vector_type_overview(self, report: Dict) -> str:
        """Build vector types overview"""
        lines = ['## 🔢 Vector Types Summary', '']
        
        vector_types = report['summary']['vector_types']
        
        lines.append('### Detected Vector Types')
        lines.append('')
        
        type_descriptions = {
            'dense': '🔷 Dense Vectors - Traditional float embeddings with full dimensions',
            'sparse': '🔶 Sparse Vectors - Efficient keyword-based vectors with mostly zeros',
            'named': '🔸 Named Vectors - Multiple different vectors per point',
            'multi': '🔹 Multi-Vector - Combinations of vector types',
            'unknown': '❓ Unknown - Could not determine vector type'
        }
        
        for vtype, count in vector_types.items():
            desc = type_descriptions.get(vtype, f'Unknown type: {vtype}')
            lines.append(f'**{desc}**')
            lines.append(f'  - Collections: {count}')
            lines.append('')
        
        lines.append('### Vector Type Characteristics')
        lines.append('')
        lines.append('| Type | Characteristics | Use Case | Pros | Cons |')
        lines.append('|------|-----------------|----------|------|------|')
        lines.append('| **Dense** | Full float vectors | Semantic search, similarity | Accurate, well-established | High memory, slow for large vocab |')
        lines.append('| **Sparse** | Indices + values | BM25, keyword search | Memory efficient, interpretable | Less semantic, requires vocab |')
        lines.append('| **Named** | Multiple vectors per point | Hybrid search, multi-modal | Flexible, combines approaches | Complex, needs careful tuning |')
        lines.append('')
        lines.append('---')
        lines.append('')
        
        return '\n'.join(lines)
    
    def _build_collection_details(self, report: Dict) -> str:
        """Build detailed collection analysis"""
        lines = ['## 📋 Collection Analysis', '']
        
        for coll_name, data in report['collections'].items():
            lines.append(f'### {coll_name}')
            lines.append('')
            
            vtype = data['vector_type']
            lines.append(f'**Vector Type**: `{vtype.upper()}`')
            lines.append('')
            
            if data['health']:
                h = data['health']
                health_emoji = self._get_health_emoji(h['health_score'])
                lines.append(f'{health_emoji} **Health Score**: {h["health_score"]:.1f}/100')
                lines.append('')
            
            # Vector-specific details
            if vtype == 'dense':
                lines.extend(self._build_dense_details(data))
            elif vtype == 'sparse':
                lines.extend(self._build_sparse_details(data))
            elif vtype == 'named':
                lines.extend(self._build_named_details(data))
            
            if data['health'] and data['health']['quality_issues']:
                lines.append('#### ⚠️ Issues Detected')
                lines.append('')
                for issue in data['health']['quality_issues']:
                    lines.append(f'- {issue}')
                lines.append('')
            
            lines.append('---')
            lines.append('')
        
        return '\n'.join(lines)
    
    @staticmethod
    def _magnitude_status(value: float) -> str:
        if 0.95 <= value <= 1.05:
            return '🟢 OK'
        if 0.8 <= value <= 1.2:
            return '🟡 WARN'
        return '🔴 ISSUE'

    @staticmethod
    def _diversity_status(value: float) -> str:
        if value > 0.7:
            return '🟢 OK'
        if value > 0.3:
            return '🟡 WARN'
        return '🔴 ISSUE'

    @staticmethod
    def _zero_count_status(value: int) -> str:
        return '🟢 OK' if value == 0 else '🔴 ISSUE'

    @staticmethod
    def _duplicate_status(rate_pct: float) -> str:
        if rate_pct < 5:
            return '🟢 OK'
        if rate_pct < 15:
            return '🟡 WARN'
        return '🔴 ISSUE'

    @staticmethod
    def _sparsity_status(pattern: str) -> str:
        if pattern == 'extreme':
            return '🔴 ISSUE'
        if pattern in ('very_high', 'low'):
            return '🟡 WARN'
        return '🟢 OK'

    @staticmethod
    def _non_zero_status(count: int) -> str:
        if count >= 10:
            return '🟢 OK'
        if count >= 3:
            return '🟡 WARN'
        return '🔴 ISSUE'

    @staticmethod
    def _coverage_status(pct: float) -> str:
        if pct >= 10:
            return '🟢 OK'
        if pct >= 0.1:
            return '🟡 WARN'
        return '🔴 ISSUE'

    def _build_dense_details(self, data: Dict) -> List[str]:
        """Build dense vector details"""
        lines = []

        if data['dense_stats']:
            s = data['dense_stats']
            dup_rate = (s['near_duplicate_pairs'] / max(s['total_points'], 1)) * 100
            lines.append('#### 🔷 Dense Vector Metrics')
            lines.append('')
            lines.append('| Metric | Value | Status |')
            lines.append('| --- | --- | --- |')
            lines.append(f'| **Magnitude (avg ± std)** | {s["avg_magnitude"]:.3f} ± {s["std_magnitude"]:.3f} | {self._magnitude_status(s["avg_magnitude"])} |')
            lines.append(f'| **Magnitude (min / max)** | {s["min_magnitude"]:.3f} / {s["max_magnitude"]:.3f} | – |')
            lines.append(f'| **NaN Vectors** | {s["nan_count"]} | {self._zero_count_status(s["nan_count"])} |')
            lines.append(f'| **Zero Vectors** | {s["zero_count"]} | {self._zero_count_status(s["zero_count"])} |')
            lines.append(f'| **Exact Duplicate Pairs** | {s["duplicate_pairs"]} | {self._zero_count_status(s["duplicate_pairs"])} |')
            lines.append(f'| **Near-Duplicate Pairs** | {s["near_duplicate_pairs"]} ({dup_rate:.1f}%) | {self._duplicate_status(dup_rate)} |')
            lines.append(f'| **Diversity** | {s["approximate_diversity"]:.3f} | {self._diversity_status(s["approximate_diversity"])} |')
            lines.append(f'| **Embedding Dimension** | {s["embedding_dim"]} | – |')
            lines.append(f'| **Total Points** | {s["total_points"]} | – |')
            lines.append('')

        return lines

    def _build_sparse_details(self, data: Dict) -> List[str]:
        """Build sparse vector details"""
        lines = []

        if data['sparse_stats']:
            s = data['sparse_stats']
            lines.append('#### 🔶 Sparse Vector Metrics')
            lines.append('')
            lines.append('| Metric | Value | Status |')
            lines.append('| --- | --- | --- |')
            lines.append(f'| **Sparsity** | {s["avg_sparsity"]:.2f}% ({s["sparsity_pattern"]}) | {self._sparsity_status(s["sparsity_pattern"])} |')
            lines.append(f'| **Avg Non-Zero Count** | {s["avg_non_zero_count"]} | {self._non_zero_status(s["avg_non_zero_count"])} |')
            lines.append(f'| **Non-Zero Range** | {s["min_non_zero_count"]} – {s["max_non_zero_count"]} | – |')
            lines.append(f'| **Vocabulary Coverage** | {s["coverage_score"]:.2f}% | {self._coverage_status(s["coverage_score"])} |')
            lines.append(f'| **Total Points** | {s["total_points"]} | – |')
            lines.append('')

        return lines

    def _build_named_details(self, data: Dict) -> List[str]:
        """Build named vector details"""
        lines = []

        if data['named_stats'] and data['named_stats']['vector_names']:
            n = data['named_stats']
            lines.append('#### 🔸 Named Vectors')
            lines.append('')

            for vname in n['vector_names']:
                dim = n['dimensions'].get(vname, '?')
                lines.append(f'##### `{vname}` ({dim} dims)')
                lines.append('')
                lines.append('| Metric | Value | Status |')
                lines.append('| --- | --- | --- |')

                stats = n['stats_per_vector'].get(vname)
                if stats:
                    avg_mag = stats.get('avg_magnitude', 0)
                    std_mag = stats.get('std_magnitude', 0)
                    min_mag = stats.get('min_magnitude', 0)
                    max_mag = stats.get('max_magnitude', 0)
                    nan_count = stats.get('nan_count', 0)
                    diversity = stats.get('diversity', 0)

                    lines.append(f'| **Magnitude (avg ± std)** | {avg_mag:.3f} ± {std_mag:.3f} | {self._magnitude_status(avg_mag)} |')
                    lines.append(f'| **Magnitude (min / max)** | {min_mag:.3f} / {max_mag:.3f} | – |')
                    lines.append(f'| **NaN Count** | {nan_count} | {self._zero_count_status(nan_count)} |')
                    lines.append(f'| **Diversity** | {diversity:.3f} | {self._diversity_status(diversity)} |')
                else:
                    lines.append('| _no statistics available_ | – | – |')

                lines.append('')

        return lines
    
    def _build_metric_guide(self, report: Dict) -> str:
        """Build metric reference guide"""
        lines = [
            '## 📚 Metric Reference Guide',
            '',
            '### Dense Vector Metrics',
            '',
            '#### Embedding Magnitude',
            '- **Optimal**: 0.95–1.05 (L2 normalized)',
            '- **Indicates**: Proper normalization across embeddings',
            '- **Issue if**: <0.85 (truncation) or >1.2 (not normalized)',
            '- **For Legal RAG**: Verify consistent normalization for EN/DA/EL',
            '',
            '#### NaN Vectors',
            '- **Optimal**: 0',
            '- **Indicates**: No extraction/encoding failures',
            '- **Issue if**: Any NaN present (even 1 breaks batch retrieval)',
            '',
            '#### Zero Vectors',
            '- **Optimal**: 0',
            '- **Indicates**: No meaningless embeddings',
            '- **Issue if**: Documents lost during processing',
            '',
            '#### Semantic Diversity',
            '- **Optimal**: >0.70',
            '- **Indicates**: Topic coverage breadth via PCA variance',
            '- **Issue if**: <0.5 (corpus too narrow)',
            '',
            '### Sparse Vector Metrics',
            '',
            '#### Sparsity',
            '- **Typical Range**: 80–99.9%',
            '- **Indicates**: Efficiency of keyword-based encoding',
            '- **Extreme** (>99.9%): Check if encoding is correct',
            '',
            '#### Non-Zero Count',
            '- **Typical Range**: 10–1000+ (depends on vocabulary)',
            '- **Indicates**: How many unique terms per document',
            '- **Low** (<3): May indicate truncation or poor encoding',
            '',
            '#### Vocabulary Coverage',
            '- **Typical Range**: 1–100% (depends on corpus)',
            '- **Indicates**: Percentage of vocabulary actually used',
            '- **Low** (<10%): Sparse encoding may not be capturing content',
            '',
            '### Named Vector Metrics',
            '',
            '**Individual metrics apply per named vector**:',
            '- Monitor each vector independently',
            '- Check consistency across named vectors',
            '- Verify all vectors have similar quality',
            '',
            '---',
            ''
        ]
        
        return '\n'.join(lines)
    
    def _build_footer(self) -> str:
        """Build report footer"""
        lines = [
            '## 📖 How to Use This Report',
            '',
            '1. **Review Vector Types** - Understand which vector types you\'re using',
            '2. **Check Health Scores** - Identify collections needing attention',
            '3. **Examine Vector-Specific Metrics** - Understand what\'s healthy for your type',
            '4. **Identify Issues** - Read specific issues and recommendations',
            '5. **Take Action** - Implement fixes based on metric guides',
            '',
            '## Vector Type Optimization Tips',
            '',
            '### Dense Vectors (Semantic Search)',
            '- ✅ Monitor magnitude consistency across languages',
            '- ✅ Track diversity to ensure topic coverage',
            '- ✅ Check for NaN/zero vectors regularly',
            '- ✅ Evaluate after embedding model changes',
            '',
            '### Sparse Vectors (Keyword Search)',
            '- ✅ Monitor sparsity patterns (aim for 80-99%)',
            '- ✅ Check vocabulary coverage (ensure >10%)',
            '- ✅ Verify non-zero counts are reasonable',
            '- ✅ Test retrieval quality with representative queries',
            '',
            '### Named Vectors (Hybrid Search)',
            '- ✅ Monitor each named vector independently',
            '- ✅ Ensure all vectors have similar quality',
            '- ✅ Test hybrid ranking and weighting',
            '- ✅ Balance between semantic and keyword search',
            '',
            '---',
            '',
            f'**Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            '',
            '_For configuration, see .env file. For detailed usage, consult documentation._'
        ]
        return '\n'.join(lines)
    
    def _get_health_emoji(self, score: float) -> str:
        """Get emoji for health score"""
        if score >= 85:
            return '🟢'
        elif score >= 75:
            return '🟡'
        elif score >= 50:
            return '🟠'
        else:
            return '🔴'
    
    def _get_score_bar(self, score: float) -> str:
        """Create visual score bar"""
        filled = int(score / 10)
        empty = 10 - filled
        return f'{"🟩" * filled}{"⬜" * empty}'
