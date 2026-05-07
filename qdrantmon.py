#!/usr/bin/env python3
"""
Enhanced Qdrant Health Monitor with .env Configuration
Supports all vector types: Dense, Sparse, Named Vectors
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv('.env')
except ImportError:
    print("⚠️  python-dotenv not installed. Using environment variables or defaults.")
    print("   Install: pip install python-dotenv")

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


@dataclass
class VectorTypeInfo:
    """Information about vector configuration"""
    vector_type: str  # 'dense', 'sparse', 'named', 'multi'
    dense_count: int = 0
    sparse_count: int = 0
    named_vectors: List[str] = None  # List of named vector names
    vector_dimensions: Dict[str, int] = None  # {vector_name: dimension}


@dataclass
class DenseVectorStats:
    """Statistics for dense vectors"""
    collection_name: str
    total_points: int
    embedding_dim: int
    avg_magnitude: float
    std_magnitude: float
    min_magnitude: float
    max_magnitude: float
    nan_count: int
    zero_count: int
    duplicate_pairs: int
    near_duplicate_pairs: int
    approximate_diversity: float


@dataclass
class SparseVectorStats:
    """Statistics for sparse vectors"""
    collection_name: str
    total_points: int
    avg_sparsity: float  # % of dimensions that are zero
    avg_non_zero_count: int
    max_non_zero_count: int
    min_non_zero_count: int
    sparsity_pattern: str  # 'uniform', 'skewed', 'extreme'
    coverage_score: float  # How many unique dimensions are used


@dataclass
class NamedVectorStats:
    """Statistics for named vectors"""
    collection_name: str
    vector_names: List[str]
    dimensions: Dict[str, int]  # {name: dimension}
    stats_per_vector: Dict[str, Dict[str, float]]  # {name: {metric: value}}


@dataclass
class SemanticDriftAnalysis:
    collection_name: str
    reference_mean_distance: float
    current_mean_distance: float
    drift_score: float
    clustering_tightness: float
    outlier_count: int
    timestamp: str


@dataclass
class CollectionHealth:
    collection_name: str
    vector_type_info: VectorTypeInfo
    point_count: int
    health_score: float
    quality_issues: List[str]
    recommendations: List[str]
    timestamp: str


class QdrantHealthMonitorEnhanced:
    """Enhanced Qdrant Health Monitor with all vector types support"""
    
    def __init__(self, 
                 qdrant_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 similarity_threshold: float = None,
                 sample_size: int = None):
        """
        Initialize monitor with .env configuration support
        
        Args:
            qdrant_url: Override .env QDRANT_URL
            api_key: Override .env QDRANT_API_KEY
            similarity_threshold: Override .env SIMILARITY_THRESHOLD
            sample_size: Override .env SAMPLE_SIZE
        """
        # Load from environment or use provided values
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.api_key = api_key or os.getenv('QDRANT_API_KEY', None)
        self.similarity_threshold = similarity_threshold or float(os.getenv('SIMILARITY_THRESHOLD', 0.99))
        self.sample_size = sample_size or int(os.getenv('SAMPLE_SIZE', 1000))
        self.output_dir = os.getenv('OUTPUT_DIR', './health_reports')
        
        # Initialize Qdrant client
        if self.api_key:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key
            )
        else:
            self.client = QdrantClient(url=self.qdrant_url)
        
        self.reference_stats: Dict[str, Dict] = {}
        
        logger.info(f"✓ Connected to Qdrant: {self.qdrant_url}")
        logger.info(f"  Sample size: {self.sample_size}")
        logger.info(f"  Similarity threshold: {self.similarity_threshold}")
    
    def detect_vector_type(self, collection_name: str) -> VectorTypeInfo:
        """Detect what vector types are used in collection"""
        logger.info(f"Detecting vector types for: {collection_name}")

        try:
            collection_info = self.client.get_collection(collection_name)
            params = collection_info.config.params
            dense_cfg = getattr(params, 'vectors', None)
            sparse_cfg = getattr(params, 'sparse_vectors', None)

            vector_info = VectorTypeInfo(vector_type='unknown')
            vector_info.named_vectors = []
            vector_info.vector_dimensions = {}

            has_named_dense = isinstance(dense_cfg, dict) and len(dense_cfg) > 0
            has_single_dense = dense_cfg is not None and not isinstance(dense_cfg, dict)
            has_sparse = isinstance(sparse_cfg, dict) and len(sparse_cfg) > 0

            if has_named_dense or (has_single_dense and has_sparse) or (has_sparse and has_named_dense):
                vector_info.vector_type = 'named'
                if has_named_dense:
                    for name, cfg in dense_cfg.items():
                        vector_info.named_vectors.append(name)
                        if hasattr(cfg, 'size'):
                            vector_info.vector_dimensions[name] = cfg.size
                        logger.info(f"  Named dense vector '{name}': dim={getattr(cfg, 'size', '?')}")
                elif has_single_dense:
                    vector_info.named_vectors.append('default')
                    if hasattr(dense_cfg, 'size'):
                        vector_info.vector_dimensions['default'] = dense_cfg.size
                if has_sparse:
                    for name in sparse_cfg.keys():
                        vector_info.named_vectors.append(name)
                        logger.info(f"  Named sparse vector '{name}'")
            elif has_single_dense:
                vector_info.vector_type = 'dense'
                vector_info.dense_count = collection_info.points_count
                if hasattr(dense_cfg, 'size'):
                    vector_info.vector_dimensions['default'] = dense_cfg.size
                logger.info(f"  Vector type: DENSE (dim={getattr(dense_cfg, 'size', '?')})")
            elif has_sparse:
                vector_info.vector_type = 'sparse'
                logger.info(f"  Vector type: SPARSE ({len(sparse_cfg)} sparse vector(s))")
            else:
                logger.warning(f"  Could not determine vector type for {collection_name}")

            return vector_info

        except Exception as e:
            logger.error(f"Error detecting vector type: {e}")
            return VectorTypeInfo(vector_type='unknown')
    
    def analyze_dense_vectors(self, collection_name: str) -> Optional[DenseVectorStats]:
        """Analyze dense vector embeddings"""
        logger.info(f"Analyzing dense vectors for: {collection_name}")
        
        try:
            collection_info = self.client.get_collection(collection_name)
            total_points = collection_info.points_count
            
            if total_points == 0:
                logger.warning(f"Collection {collection_name} is empty")
                return None
            
            # Get vector dimension
            config = collection_info.config.params.vectors
            if isinstance(config, dict):
                # Named vectors - use first one
                embedding_dim = next(iter(config.values())).size
            else:
                embedding_dim = config.size
            
            # Sample points
            sample_size = min(self.sample_size, total_points)
            points, _ = self.client.scroll(
                collection_name=collection_name,
                limit=sample_size,
                with_vectors=True
            )

            if not points:
                return None

            # Extract vectors
            vectors = []
            for p in points:
                if isinstance(p.vector, dict):
                    # Named vectors - use first one
                    vector = next(iter(p.vector.values()))
                else:
                    vector = p.vector
                if vector is None:
                    continue
                vectors.append(vector)

            if not vectors:
                logger.warning(f"No dense vectors retrieved for {collection_name}")
                return None

            vectors = np.array(vectors)
            
            # Compute statistics
            magnitudes = np.linalg.norm(vectors, axis=1)
            avg_magnitude = float(np.mean(magnitudes))
            std_magnitude = float(np.std(magnitudes))
            min_magnitude = float(np.min(magnitudes))
            max_magnitude = float(np.max(magnitudes))
            
            nan_count = int(np.sum(np.isnan(vectors)))
            zero_count = int(np.sum(np.all(vectors == 0, axis=1)))
            
            duplicate_pairs, near_duplicate_pairs = self._detect_duplicates(vectors)
            diversity = self._compute_diversity(vectors)
            
            stats = DenseVectorStats(
                collection_name=collection_name,
                total_points=total_points,
                embedding_dim=embedding_dim,
                avg_magnitude=avg_magnitude,
                std_magnitude=std_magnitude,
                min_magnitude=min_magnitude,
                max_magnitude=max_magnitude,
                nan_count=nan_count,
                zero_count=zero_count,
                duplicate_pairs=duplicate_pairs,
                near_duplicate_pairs=near_duplicate_pairs,
                approximate_diversity=diversity
            )
            
            logger.info(f"✓ Dense vectors: {total_points} points, diversity={diversity:.3f}")
            return stats
        
        except Exception as e:
            logger.error(f"Error analyzing dense vectors: {e}")
            return None
    
    def analyze_sparse_vectors(self, collection_name: str) -> Optional[SparseVectorStats]:
        """Analyze sparse vector embeddings"""
        logger.info(f"Analyzing sparse vectors for: {collection_name}")
        
        try:
            collection_info = self.client.get_collection(collection_name)
            total_points = collection_info.points_count
            
            if total_points == 0:
                return None
            
            # Sample points
            sample_size = min(self.sample_size, total_points)
            points, _ = self.client.scroll(
                collection_name=collection_name,
                limit=sample_size,
                with_vectors=True
            )

            if not points:
                return None

            # Analyze sparsity patterns
            non_zero_counts = []
            all_indices = set()
            
            for p in points:
                vector = p.vector if isinstance(p.vector, dict) else {'sparse': p.vector}
                
                # Handle sparse vector format
                if hasattr(vector, 'indices') and hasattr(vector, 'values'):
                    non_zero = len(vector.indices)
                else:
                    # Assume dict format {'indices': [...], 'values': [...]}
                    indices = vector.get('indices', [])
                    non_zero = len(indices) if indices else 0
                    all_indices.update(indices if indices else [])
                
                non_zero_counts.append(non_zero)
            
            avg_non_zero = float(np.mean(non_zero_counts)) if non_zero_counts else 0
            max_non_zero = int(np.max(non_zero_counts)) if non_zero_counts else 0
            min_non_zero = int(np.min(non_zero_counts)) if non_zero_counts else 0
            
            # Estimate total vocabulary/dimensions
            total_dims = int(max(all_indices)) + 1 if all_indices else max_non_zero
            
            # Sparsity: percentage of zeros
            avg_sparsity = (1 - avg_non_zero / max(total_dims, 1)) * 100
            
            # Coverage: what percentage of vocab is actually used
            coverage = (len(all_indices) / max(total_dims, 1)) * 100 if total_dims > 0 else 0
            
            # Sparsity pattern classification
            if avg_sparsity > 99.9:
                pattern = 'extreme'
            elif avg_sparsity > 99:
                pattern = 'very_high'
            elif avg_sparsity > 95:
                pattern = 'high'
            elif avg_sparsity > 80:
                pattern = 'moderate'
            else:
                pattern = 'low'
            
            stats = SparseVectorStats(
                collection_name=collection_name,
                total_points=total_points,
                avg_sparsity=avg_sparsity,
                avg_non_zero_count=int(avg_non_zero),
                max_non_zero_count=max_non_zero,
                min_non_zero_count=min_non_zero,
                sparsity_pattern=pattern,
                coverage_score=float(coverage)
            )
            
            logger.info(f"✓ Sparse vectors: sparsity={avg_sparsity:.1f}%, coverage={coverage:.1f}%")
            return stats
        
        except Exception as e:
            logger.error(f"Error analyzing sparse vectors: {e}")
            return None
    
    def analyze_named_vectors(self, collection_name: str) -> Optional[NamedVectorStats]:
        """Analyze named vectors (multiple vectors per point)"""
        logger.info(f"Analyzing named vectors for: {collection_name}")
        
        try:
            collection_info = self.client.get_collection(collection_name)
            config = collection_info.config.params.vectors

            if not isinstance(config, dict):
                return None

            vector_names = list(config.keys())
            dimensions = {}
            stats_per_vector = {}
            
            for name in vector_names:
                dimensions[name] = config[name].size
                
                # Analyze this specific named vector
                sample_size = min(self.sample_size, collection_info.points_count)
                points, _ = self.client.scroll(
                    collection_name=collection_name,
                    limit=sample_size,
                    with_vectors=True
                )
                
                vectors = []
                for p in points:
                    if isinstance(p.vector, dict) and name in p.vector:
                        vectors.append(p.vector[name])
                
                if vectors:
                    vectors = np.array(vectors)
                    magnitudes = np.linalg.norm(vectors, axis=1)
                    
                    stats_per_vector[name] = {
                        'avg_magnitude': float(np.mean(magnitudes)),
                        'std_magnitude': float(np.std(magnitudes)),
                        'min_magnitude': float(np.min(magnitudes)),
                        'max_magnitude': float(np.max(magnitudes)),
                        'nan_count': int(np.sum(np.isnan(vectors))),
                        'diversity': float(self._compute_diversity(vectors))
                    }
            
            stats = NamedVectorStats(
                collection_name=collection_name,
                vector_names=vector_names,
                dimensions=dimensions,
                stats_per_vector=stats_per_vector
            )
            
            logger.info(f"✓ Named vectors: {len(vector_names)} vectors")
            return stats
        
        except Exception as e:
            logger.error(f"Error analyzing named vectors: {e}")
            return None
    
    def compute_collection_health(self, collection_name: str) -> Optional[CollectionHealth]:
        """Compute overall health for collection (all vector types)"""
        logger.info(f"Computing health for: {collection_name}")
        
        try:
            collection_info = self.client.get_collection(collection_name)
            
            # Detect vector type
            vector_type_info = self.detect_vector_type(collection_name)
            
            health_score = 100.0
            issues = []
            recommendations = []
            
            # Analyze based on vector type
            if vector_type_info.vector_type == 'dense':
                stats = self.analyze_dense_vectors(collection_name)
                if stats:
                    # Standard dense vector checks
                    if stats.avg_magnitude < 0.8 or stats.avg_magnitude > 1.2:
                        health_score -= 15
                        issues.append(f"Dense embedding magnitude anomaly: {stats.avg_magnitude:.3f}")
                    
                    if stats.nan_count > 0:
                        health_score -= 25
                        issues.append(f"Found {stats.nan_count} NaN vectors")
                    
                    if stats.zero_count > 0:
                        health_score -= 15
                        issues.append(f"Found {stats.zero_count} zero vectors")
                    
                    if stats.approximate_diversity < 0.3:
                        health_score -= 12
                        issues.append(f"Low semantic diversity: {stats.approximate_diversity:.3f}")
            
            elif vector_type_info.vector_type == 'sparse':
                stats = self.analyze_sparse_vectors(collection_name)
                if stats:
                    # Sparse-specific checks
                    if stats.coverage_score < 0.1:
                        health_score -= 20
                        issues.append(f"Very low vocab coverage: {stats.coverage_score:.1f}%")
                    
                    if stats.sparsity_pattern == 'extreme':
                        health_score -= 10
                        issues.append("Extreme sparsity (>99.9%); may indicate encoding issues")
                    
                    if stats.avg_non_zero_count < 3:
                        health_score -= 15
                        issues.append(f"Very low non-zero count: {stats.avg_non_zero_count}")
            
            elif vector_type_info.vector_type == 'named':
                stats = self.analyze_named_vectors(collection_name)
                if stats:
                    # Check each named vector
                    for name, vec_stats in stats.stats_per_vector.items():
                        if vec_stats['nan_count'] > 0:
                            health_score -= 10
                            issues.append(f"Named vector '{name}' has {vec_stats['nan_count']} NaN values")
                        
                        if vec_stats['avg_magnitude'] < 0.8 or vec_stats['avg_magnitude'] > 1.2:
                            health_score -= 8
                            issues.append(f"Vector '{name}' magnitude anomaly: {vec_stats['avg_magnitude']:.3f}")
            
            # General checks (all types)
            try:
                drift = self.detect_semantic_drift(collection_name)
                if drift and drift.drift_score > 0.2:
                    health_score -= 15
                    issues.append(f"Significant drift: {drift.drift_score:.3f}")
            except:
                pass
            
            health_score = max(0, min(100, health_score))
            
            health = CollectionHealth(
                collection_name=collection_name,
                vector_type_info=vector_type_info,
                point_count=collection_info.points_count,
                health_score=health_score,
                quality_issues=issues,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Health Score: {health_score:.1f}/100")
            return health
        
        except Exception as e:
            logger.error(f"Error computing health: {e}")
            return None
    
    def generate_report(self, collection_names: Optional[List[str]] = None) -> Dict:
        """Generate comprehensive report"""
        if collection_names is None:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'qdrant_url': self.qdrant_url,
            'collections': {},
            'summary': {
                'total_collections': len(collection_names),
                'vector_types': {},
                'avg_health_score': 0.0,
                'critical_issues': []
            }
        }
        
        health_scores = []
        
        for coll_name in collection_names:
            logger.info(f"\n=== Analyzing {coll_name} ===")
            
            health = self.compute_collection_health(coll_name)
            
            vector_type = health.vector_type_info.vector_type if health else 'unknown'
            report['summary']['vector_types'][vector_type] = \
                report['summary']['vector_types'].get(vector_type, 0) + 1
            
            # Analyze based on vector type
            dense_stats = None
            sparse_stats = None
            named_stats = None
            
            if health and health.vector_type_info.vector_type == 'dense':
                dense_stats = self.analyze_dense_vectors(coll_name)
            elif health and health.vector_type_info.vector_type == 'sparse':
                sparse_stats = self.analyze_sparse_vectors(coll_name)
            elif health and health.vector_type_info.vector_type == 'named':
                named_stats = self.analyze_named_vectors(coll_name)
            
            report['collections'][coll_name] = {
                'health': asdict(health) if health else None,
                'dense_stats': asdict(dense_stats) if dense_stats else None,
                'sparse_stats': asdict(sparse_stats) if sparse_stats else None,
                'named_stats': {
                    'vector_names': named_stats.vector_names if named_stats else [],
                    'dimensions': named_stats.dimensions if named_stats else {},
                    'stats_per_vector': named_stats.stats_per_vector if named_stats else {}
                } if named_stats else None,
                'vector_type': health.vector_type_info.vector_type if health else 'unknown'
            }
            
            if health:
                health_scores.append(health.health_score)
                if health.health_score < 50:
                    report['summary']['critical_issues'].extend(health.quality_issues)
        
        if health_scores:
            report['summary']['avg_health_score'] = float(np.mean(health_scores))
        
        return report
    
    def detect_semantic_drift(self, collection_name: str) -> Optional[SemanticDriftAnalysis]:
        """Detect embedding drift"""
        try:
            points, _ = self.client.scroll(
                collection_name=collection_name,
                limit=self.sample_size,
                with_vectors=True
            )

            if not points:
                return None

            vectors = []
            for p in points:
                if isinstance(p.vector, dict):
                    vector = next(iter(p.vector.values()))
                else:
                    vector = p.vector
                if vector is None:
                    continue
                vectors.append(vector)

            if not vectors:
                return None

            vectors = np.array(vectors)
            current_mean_distance = self._compute_mean_pairwise_distance(vectors)
            clustering = self._compute_clustering_tightness(vectors)
            
            if collection_name not in self.reference_stats:
                self.reference_stats[collection_name] = {
                    'mean_distance': current_mean_distance
                }
                drift_score = 0.0
            else:
                ref_distance = self.reference_stats[collection_name]['mean_distance']
                drift_score = abs(current_mean_distance - ref_distance) / (ref_distance + 1e-6)
            
            outliers = self._detect_outliers(vectors)
            
            return SemanticDriftAnalysis(
                collection_name=collection_name,
                reference_mean_distance=float(self.reference_stats[collection_name]['mean_distance']),
                current_mean_distance=float(current_mean_distance),
                drift_score=float(drift_score),
                clustering_tightness=float(clustering),
                outlier_count=outliers,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.warning(f"Could not compute drift: {e}")
            return None
    
    def _detect_duplicates(self, vectors: np.ndarray) -> Tuple[int, int]:
        """Detect duplicates"""
        exact_duplicates = 0
        near_duplicates = 0
        
        sample_indices = np.random.choice(
            len(vectors), 
            min(500, len(vectors)), 
            replace=False
        )
        sample_vectors = vectors[sample_indices]
        
        for i in range(len(sample_vectors)):
            for j in range(i + 1, len(sample_vectors)):
                similarity = 1 - cosine(sample_vectors[i], sample_vectors[j])
                if similarity > 0.9999:
                    exact_duplicates += 1
                elif similarity > self.similarity_threshold:
                    near_duplicates += 1
        
        scale_factor = len(vectors) / len(sample_vectors)
        return int(exact_duplicates * scale_factor), int(near_duplicates * scale_factor)
    
    def _compute_diversity(self, vectors: np.ndarray) -> float:
        """Compute semantic diversity"""
        try:
            if len(vectors) < 10:
                return 0.5
            
            scaler = StandardScaler()
            vectors_scaled = scaler.fit_transform(vectors)
            
            pca = PCA(n_components=min(10, vectors_scaled.shape[1]))
            pca.fit(vectors_scaled)
            
            diversity = float(np.sum(pca.explained_variance_ratio_))
            return min(diversity, 1.0)
        except:
            return 0.5
    
    def _compute_mean_pairwise_distance(self, vectors: np.ndarray) -> float:
        """Compute mean pairwise distance"""
        sample_size = min(200, len(vectors))
        indices = np.random.choice(len(vectors), sample_size, replace=False)
        sample = vectors[indices]
        
        distances = []
        for i in range(min(50, len(sample))):
            for j in range(i + 1, len(sample)):
                dist = cosine(sample[i], sample[j])
                distances.append(dist)
        
        return float(np.mean(distances)) if distances else 0.5
    
    def _compute_clustering_tightness(self, vectors: np.ndarray) -> float:
        """Compute clustering tightness"""
        try:
            if len(vectors) < 10:
                return 0.5
            
            scaler = StandardScaler()
            vectors_scaled = scaler.fit_transform(vectors)
            
            pca = PCA(n_components=2)
            pca.fit(vectors_scaled)
            
            return float(np.sum(pca.explained_variance_ratio_))
        except:
            return 0.5
    
    def _detect_outliers(self, vectors: np.ndarray, std_threshold: float = 3.0) -> int:
        """Detect outliers"""
        magnitudes = np.linalg.norm(vectors, axis=1)
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        
        outliers = np.sum(np.abs(magnitudes - mean_mag) > std_threshold * std_mag)
        return int(outliers)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Qdrant Health Monitor with .env Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use .env configuration
  python qdrant_health_monitor_enhanced_env.py --markdown
  
  # Override URL
  python qdrant_health_monitor_enhanced_env.py --url http://qdrant.example.com:6333 --markdown
  
  # Specific collection
  python qdrant_health_monitor_enhanced_env.py --collection ucc_laws --markdown
  
  # Console output
  python qdrant_health_monitor_enhanced_env.py
        """
    )
    parser.add_argument('--url', help='Override QDRANT_URL from .env')
    parser.add_argument('--api-key', help='Override QDRANT_API_KEY from .env')
    parser.add_argument('--collection', help='Analyze specific collection')
    parser.add_argument('--markdown', action='store_true', help='Export as Markdown')
    parser.add_argument('--output-dir', help='Override OUTPUT_DIR from .env')
    
    args = parser.parse_args()
    
    # Create monitor with .env config
    monitor = QdrantHealthMonitorEnhanced(
        qdrant_url=args.url,
        api_key=args.api_key
    )

    collections = [args.collection] if args.collection else None

    if args.markdown:
        from markdown_report_generator_enhanced import MarkdownReportGeneratorEnhanced
        generator = MarkdownReportGeneratorEnhanced(monitor)
        output_dir = args.output_dir or monitor.output_dir
        filepath = generator.generate_markdown_report(collections, output_dir=output_dir)
        print(f"\n✓ Markdown report written to: {filepath}")
        return

    # Generate report
    report = monitor.generate_report(collections)
    
    print("\n" + "="*70)
    print("QDRANT KNOWLEDGE BASE HEALTH REPORT")
    print("="*70)
    print(f"Qdrant URL: {monitor.qdrant_url}")
    print(f"Collections: {report['summary']['total_collections']}")
    print(f"Vector Types: {report['summary']['vector_types']}")
    print(f"Average Health: {report['summary']['avg_health_score']:.1f}/100\n")
    
    for coll_name, data in report['collections'].items():
        vector_type = data['vector_type']
        print(f"\n{coll_name} [{vector_type.upper()}]")
        print("-" * 70)
        
        if data['health']:
            h = data['health']
            print(f"  Health Score: {h['health_score']:.1f}/100")
            
            if h['quality_issues']:
                print(f"  Issues ({len(h['quality_issues'])}):")
                for issue in h['quality_issues'][:3]:
                    print(f"    ⚠️  {issue}")
        
        # Show vector-type specific info
        if vector_type == 'dense' and data['dense_stats']:
            s = data['dense_stats']
            print(f"  Dense Stats:")
            print(f"    Magnitude: {s['avg_magnitude']:.3f} ± {s['std_magnitude']:.3f}")
            print(f"    Diversity: {s['approximate_diversity']:.3f}")
            print(f"    Duplicates: {s['near_duplicate_pairs']}")
        
        elif vector_type == 'sparse' and data['sparse_stats']:
            s = data['sparse_stats']
            print(f"  Sparse Stats:")
            print(f"    Sparsity: {s['avg_sparsity']:.1f}% ({s['sparsity_pattern']})")
            print(f"    Non-zero avg: {s['avg_non_zero_count']}")
            print(f"    Vocab coverage: {s['coverage_score']:.1f}%")
        
        elif vector_type == 'named' and data['named_stats']:
            n = data['named_stats']
            print(f"  Named Vectors:")
            for vname in n['vector_names']:
                print(f"    - {vname}: dim={n['dimensions'].get(vname)}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
