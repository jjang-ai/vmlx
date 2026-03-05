# SPDX-License-Identifier: Apache-2.0
"""
Tests for vllm-mlx hardware detection and system info.

Usage:
    pytest tests/test_optimizations.py -v
"""


class TestHardwareDetection:
    """Tests for hardware detection functionality."""

    def test_detect_hardware(self):
        """Test that hardware detection works."""
        from vllm_mlx.optimizations import detect_hardware

        hw = detect_hardware()

        assert hw is not None
        assert hw.chip_name is not None
        assert hw.total_memory_gb > 0
        assert hw.memory_bandwidth_gbs > 0
        assert hw.gpu_cores > 0

    def test_get_system_memory(self):
        """Test that system memory detection works."""
        from vllm_mlx.optimizations import get_system_memory_gb

        memory_gb = get_system_memory_gb()

        assert memory_gb > 0
        assert memory_gb < 1024  # Sanity check: less than 1TB

    def test_hardware_profiles_exist(self):
        """Test that hardware profiles are defined."""
        from vllm_mlx.optimizations import HARDWARE_PROFILES

        assert len(HARDWARE_PROFILES) > 0
        assert "M1" in HARDWARE_PROFILES
        assert "M4 Max" in HARDWARE_PROFILES


def run_quick_test():
    """Run a quick test of hardware detection."""
    from vllm_mlx.optimizations import detect_hardware

    print("=" * 60)
    print("Quick Hardware Detection Test")
    print("=" * 60)

    hw = detect_hardware()
    print("\nHardware Detection:")
    print(f"  Chip: {hw.chip_name}")
    print(f"  Memory: {hw.total_memory_gb:.1f} GB")
    print(f"  Bandwidth: {hw.memory_bandwidth_gbs} GB/s")
    print(f"  GPU Cores: {hw.gpu_cores}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_test()
