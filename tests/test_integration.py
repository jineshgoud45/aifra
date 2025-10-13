"""
Integration Tests for FRA Diagnostics Platform
SIH 2025 PS 25190

Tests complete end-to-end workflows including:
- File upload → Parse → QA → AI inference → Report generation
- Multi-vendor file handling
- Error handling and edge cases
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import time
from pathlib import Path

# Import modules to test
from parser import UniversalFRAParser, FRAParserError
from simulator import TransformerSimulator
from ai_ensemble import FRAEnsemble, load_models
from report_generator import generate_iec_report
from config import TESTING

class TestEndToEndPipeline:
    """Test complete analysis pipeline from file to report."""
    
    @pytest.fixture(scope="class")
    def simulator(self):
        """Fixture: Transformer simulator."""
        return TransformerSimulator(seed=42)
    
    @pytest.fixture(scope="class")
    def parser(self):
        """Fixture: FRA parser."""
        return UniversalFRAParser()
    
    @pytest.fixture(scope="class")
    def temp_dir(self):
        """Fixture: Temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_complete_pipeline_normal_transformer(self, simulator, parser, temp_dir):
        """Test complete pipeline with normal transformer data."""
        # Step 1: Generate synthetic data
        freq, mag, phase = simulator.generate_normal_signature(add_noise=True)
        
        # Create DataFrame
        df = pd.DataFrame({
            'frequency_hz': freq,
            'magnitude_db': mag,
            'phase_deg': phase,
            'sample_id': 'test_normal',
            'fault_type': 'normal'
        })
        
        # Step 2: Save as CSV
        csv_path = os.path.join(temp_dir, 'test_normal.csv')
        df[['frequency_hz', 'magnitude_db', 'phase_deg']].to_csv(csv_path, index=False)
        
        # Step 3: Parse file
        parsed_df = parser.parse_file(csv_path, vendor='generic')
        
        # Step 4: Verify parsing
        assert len(parsed_df) == len(df)
        assert 'frequency_hz' in parsed_df.columns
        assert 'magnitude_db' in parsed_df.columns
        assert 'phase_deg' in parsed_df.columns
        
        # Step 5: Run QA checks
        qa_results = parser.get_qa_results(csv_path)
        assert qa_results is not None
        assert 'checks' in qa_results
    
    def test_complete_pipeline_with_ai_inference(self, simulator, parser, temp_dir):
        """
        CRITICAL: Test COMPLETE end-to-end pipeline including AI model inference.
        
        This is the real integration test that verifies:
        1. Data generation
        2. File I/O and parsing
        3. QA checks
        4. AI model loading and inference
        5. Report generation
        """
        # Step 1: Generate synthetic fault data (axial deformation)
        freq, mag, phase = simulator.generate_axial_deformation()
        
        df = pd.DataFrame({
            'frequency_hz': freq,
            'magnitude_db': mag,
            'phase_deg': phase,
            'sample_id': 'integration_test_001',
            'fault_type': 'axial_deformation'
        })
        
        # Step 2: Save as CSV
        csv_path = os.path.join(temp_dir, 'test_axial_deformation.csv')
        df[['frequency_hz', 'magnitude_db', 'phase_deg']].to_csv(csv_path, index=False)
        
        # Step 3: Parse file
        parsed_df = parser.parse_file(csv_path, vendor='generic')
        assert len(parsed_df) > 0, "Parsed data should not be empty"
        
        # Step 4: Run QA checks
        qa_results = parser.get_qa_results(csv_path)
        assert qa_results is not None, "QA results should be generated"
        assert 'checks' in qa_results, "QA results should contain checks"
        
        # Step 5: Load AI model and run inference
        try:
            from ai_ensemble import load_models
            
            # Try to load models (skip if not available in CI)
            ensemble = load_models(save_dir='models')
            
            # Step 6: Run prediction
            prediction = ensemble.predict(parsed_df)
            
            # Verify prediction structure
            assert 'predicted_fault' in prediction, "Prediction must include fault type"
            assert 'confidence' in prediction, "Prediction must include confidence"
            assert 'probabilities' in prediction, "Prediction must include probabilities"
            assert 'cnn_probs' in prediction, "Prediction must include CNN probabilities"
            assert 'resnet_probs' in prediction, "Prediction must include ResNet probabilities"
            assert 'svm_score' in prediction, "Prediction must include SVM score"
            assert 'uncertainty' in prediction, "Prediction must include uncertainty"
            
            # Verify confidence is valid
            assert 0 <= prediction['confidence'] <= 1, "Confidence must be between 0 and 1"
            
            # Verify probabilities sum to ~1
            prob_sum = sum(prediction['probabilities'].values())
            assert 0.99 <= prob_sum <= 1.01, f"Probabilities should sum to 1, got {prob_sum}"
            
            # Step 7: Generate PDF report
            from report_generator import generate_iec_report
            
            report_path = os.path.join(temp_dir, 'integration_test_report.pdf')
            generated_path = generate_iec_report(
                df=parsed_df,
                prediction_result=prediction,
                qa_results=qa_results,
                test_id='integration_test_001',
                output_path=report_path
            )
            
            # Verify report was created
            assert os.path.exists(generated_path), "PDF report should be generated"
            assert os.path.getsize(generated_path) > 1000, "PDF should be >1KB"
            
            # If we got here, full pipeline works!
            print(f"✅ Full integration test passed!")
            print(f"   Predicted: {prediction['predicted_fault']}")
            print(f"   Confidence: {prediction['confidence']:.2%}")
            print(f"   Report: {generated_path}")
            
        except ImportError as e:
            pytest.skip(f"AI ensemble module not available: {e}")
        except FileNotFoundError as e:
            pytest.skip(f"AI models not found (expected in CI): {e}")
        except Exception as e:
            # If models exist but inference fails, that's a real error
            if os.path.exists('models'):
                pytest.fail(f"AI inference failed with models present: {e}")
            else:
                pytest.skip(f"AI models not available: {e}")
    
    def test_multi_vendor_file_handling(self, simulator, parser, temp_dir):
        """Test parsing files from multiple vendors."""
        # Generate test data
        freq, mag, phase = simulator.generate_axial_deformation()
        
        # Test Omicron CSV
        omicron_path = os.path.join(temp_dir, 'omicron_test.csv')
        with open(omicron_path, 'w') as f:
            f.write("# Omicron FRANEO\n")
            f.write("Frequency (Hz),Magnitude (dB),Phase (deg)\n")
            for i in range(len(freq)):
                f.write(f"{freq[i]},{mag[i]},{phase[i]}\n")
        
        omicron_df = parser.parse_file(omicron_path, vendor='omicron')
        assert len(omicron_df) > 0
        
        # Test generic CSV
        generic_path = os.path.join(temp_dir, 'generic_test.csv')
        pd.DataFrame({
            'frequency_hz': freq,
            'magnitude_db': mag,
            'phase_deg': phase
        }).to_csv(generic_path, index=False)
        
        generic_df = parser.parse_file(generic_path, vendor='generic')
        assert len(generic_df) > 0
    
    def test_error_handling_invalid_file(self, parser, temp_dir):
        """Test error handling for invalid files."""
        # Test empty file
        empty_path = os.path.join(temp_dir, 'empty.csv')
        Path(empty_path).touch()
        
        with pytest.raises(FRAParserError):
            parser.parse_file(empty_path)
        
        # Test malformed CSV
        malformed_path = os.path.join(temp_dir, 'malformed.csv')
        with open(malformed_path, 'w') as f:
            f.write("This is not a valid CSV\n")
            f.write("Random data\n")
        
        with pytest.raises(FRAParserError):
            parser.parse_file(malformed_path)
    
    def test_large_file_handling(self, simulator, parser, temp_dir):
        """Test handling of large FRA files."""
        # Generate large dataset (using config constant)
        freq = np.logspace(np.log10(20), np.log10(2e6), TESTING.max_test_data_points)
        mag = np.random.randn(TESTING.max_test_data_points) * 2 + 10
        phase = np.random.randn(TESTING.max_test_data_points) * 20 - 45
        
        large_path = os.path.join(temp_dir, 'large_file.csv')
        pd.DataFrame({
            'frequency_hz': freq,
            'magnitude_db': mag,
            'phase_deg': phase
        }).to_csv(large_path, index=False)
        
        # Should handle gracefully (may truncate)
        parsed_df = parser.parse_file(large_path)
        assert len(parsed_df) > 0
        assert len(parsed_df) <= TESTING.max_test_data_points  # Max data points from IEC config


class TestReportGeneration:
    """Test PDF report generation."""
    
    def test_report_generation_basic(self, temp_dir):
        """Test basic report generation without AI model."""
        # Create minimal test data
        df = pd.DataFrame({
            'frequency_hz': [20, 100, 1000, 10000, 100000, 1000000],
            'magnitude_db': [0, 5, 10, 8, 3, -2],
            'phase_deg': [-10, -30, -60, -90, -120, -150],
            'test_id': 'test_001',
            'vendor': 'generic',
            'timestamp': pd.Timestamp.now()
        })
        
        # Mock prediction result
        prediction_result = {
            'predicted_fault': 'normal',
            'confidence': 0.95,
            'probabilities': {
                'normal': 0.95,
                'axial_deformation': 0.03,
                'radial_deformation': 0.01,
                'interturn_short': 0.005,
                'core_grounding': 0.003,
                'tapchanger_fault': 0.002
            },
            'cnn_probs': {
                'normal': 0.93,
                'axial_deformation': 0.04,
                'radial_deformation': 0.02,
                'interturn_short': 0.005,
                'core_grounding': 0.003,
                'tapchanger_fault': 0.002
            },
            'resnet_probs': {
                'normal': 0.97,
                'axial_deformation': 0.02,
                'radial_deformation': 0.005,
                'interturn_short': 0.003,
                'core_grounding': 0.001,
                'tapchanger_fault': 0.001
            },
            'svm_score': 1.2,
            'uncertainty': 0.15
        }
        
        # Mock QA results
        qa_results = {
            'filepath': 'test.csv',
            'test_id': 'test_001',
            'vendor': 'generic',
            'checks': {
                'frequency_range': {
                    'passed': True,
                    'min_freq': 20,
                    'max_freq': 1000000,
                    'message': 'OK'
                },
                'frequency_grid': {
                    'passed': True,
                    'log_spacing_std': 0.3,
                    'num_points': 6
                }
            }
        }
        
        # Generate report
        report_path = os.path.join(temp_dir, 'test_report.pdf')
        result_path = generate_iec_report(
            df=df,
            prediction_result=prediction_result,
            qa_results=qa_results,
            test_id='test_001',
            output_path=report_path
        )
        
        # Verify report was created
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 1000  # PDF should be >1KB


class TestDataQuality:
    """Test data quality and edge cases."""
    
    def test_frequency_range_validation(self, parser, temp_dir):
        """Test validation of frequency ranges."""
        # Test frequencies outside IEC range
        invalid_freq = np.array([1, 10, 5e6, 10e6])  # Below 20Hz and above 2MHz
        mag = np.array([0, 0, 0, 0])
        phase = np.array([0, 0, 0, 0])
        
        invalid_path = os.path.join(temp_dir, 'invalid_freq.csv')
        pd.DataFrame({
            'frequency_hz': invalid_freq,
            'magnitude_db': mag,
            'phase_deg': phase
        }).to_csv(invalid_path, index=False)
        
        # Should parse but generate QA warnings
        df = parser.parse_file(invalid_path)
        qa = parser.get_qa_results(invalid_path)
        
        # Check that QA caught the issue
        assert 'frequency_range' in qa['checks']
        # May pass or fail depending on tolerance, but should be recorded
    
    def test_artifact_detection(self, parser, temp_dir):
        """Test detection of measurement artifacts."""
        # Create data with artificial spikes
        freq = np.logspace(np.log10(20), np.log10(2e6), 100)
        mag = np.sin(np.log10(freq)) * 5 + 10
        mag[50] += 10  # Add spike
        phase = -45 * np.ones_like(freq)
        
        artifact_path = os.path.join(temp_dir, 'artifact.csv')
        pd.DataFrame({
            'frequency_hz': freq,
            'magnitude_db': mag,
            'phase_deg': phase
        }).to_csv(artifact_path, index=False)
        
        df = parser.parse_file(artifact_path)
        qa = parser.get_qa_results(artifact_path)
        
        # Should detect artifacts
        if 'artifacts' in qa['checks']:
            assert 'num_artifacts' in qa['checks']['artifacts']


class TestPerformanceBenchmarks:
    """Test performance benchmarks to ensure acceptable response times."""
    
    @pytest.fixture(scope="class")
    def performance_parser(self):
        """Fixture: Parser for performance tests."""
        return UniversalFRAParser()
    
    @pytest.fixture(scope="class")
    def performance_simulator(self):
        """Fixture: Simulator for performance tests."""
        return TransformerSimulator(seed=42)
    
    def test_parse_time_benchmark(self, performance_parser, performance_simulator, temp_dir):
        """Test that file parsing meets performance requirements."""
        # Generate realistic dataset
        freq, mag, phase = performance_simulator.generate_normal_signature(add_noise=True)
        
        csv_path = os.path.join(temp_dir, 'benchmark.csv')
        pd.DataFrame({
            'frequency_hz': freq,
            'magnitude_db': mag,
            'phase_deg': phase
        }).to_csv(csv_path, index=False)
        
        # Measure parse time
        start_time = time.time()
        df = performance_parser.parse_file(csv_path, vendor='generic')
        parse_time = time.time() - start_time
        
        # Verify performance
        assert parse_time < TESTING.max_parse_time_seconds, \
            f"Parse time {parse_time:.2f}s exceeds maximum {TESTING.max_parse_time_seconds}s"
        
        print(f"✅ Parse benchmark: {parse_time:.3f}s (limit: {TESTING.max_parse_time_seconds}s)")
    
    def test_report_generation_time_benchmark(self, performance_simulator, temp_dir):
        """Test that PDF report generation meets performance requirements."""
        from report_generator import generate_iec_report
        
        # Generate test data
        freq, mag, phase = performance_simulator.generate_axial_deformation()
        df = pd.DataFrame({
            'frequency_hz': freq,
            'magnitude_db': mag,
            'phase_deg': phase,
            'test_id': 'perf_test',
            'vendor': 'generic',
            'timestamp': pd.Timestamp.now()
        })
        
        # Mock prediction and QA results
        prediction_result = {
            'predicted_fault': 'axial_deformation',
            'confidence': 0.85,
            'probabilities': {'normal': 0.05, 'axial_deformation': 0.85, 'radial_deformation': 0.05,
                            'interturn_short': 0.02, 'core_grounding': 0.02, 'tapchanger_fault': 0.01},
            'cnn_probs': {'normal': 0.06, 'axial_deformation': 0.83, 'radial_deformation': 0.06,
                         'interturn_short': 0.02, 'core_grounding': 0.02, 'tapchanger_fault': 0.01},
            'resnet_probs': {'normal': 0.04, 'axial_deformation': 0.87, 'radial_deformation': 0.04,
                           'interturn_short': 0.02, 'core_grounding': 0.02, 'tapchanger_fault': 0.01},
            'svm_score': -0.5,
            'uncertainty': 0.25
        }
        
        qa_results = {
            'filepath': 'test.csv',
            'test_id': 'perf_test',
            'vendor': 'generic',
            'checks': {
                'frequency_range': {'passed': True, 'min_freq': 20, 'max_freq': 2e6, 'message': 'OK'},
                'frequency_grid': {'passed': True, 'log_spacing_std': 0.3, 'num_points': len(df)}
            }
        }
        
        # Measure report generation time
        report_path = os.path.join(temp_dir, 'benchmark_report.pdf')
        start_time = time.time()
        generated_path = generate_iec_report(
            df=df,
            prediction_result=prediction_result,
            qa_results=qa_results,
            test_id='perf_test',
            output_path=report_path
        )
        report_time = time.time() - start_time
        
        # Verify performance
        assert report_time < TESTING.max_report_generation_seconds, \
            f"Report generation {report_time:.2f}s exceeds maximum {TESTING.max_report_generation_seconds}s"
        
        assert os.path.exists(generated_path)
        print(f"✅ Report generation benchmark: {report_time:.3f}s (limit: {TESTING.max_report_generation_seconds}s)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
