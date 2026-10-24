"""Unit tests for PyTorch device dispatch and accelerator management.

Compatible with both standard library `unittest` and `pytest`.
"""

import unittest
from src.models.device import DeviceManager, get_device

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestDeviceManager(unittest.TestCase):
    """Test suite verifying hardware-agnostic tensor device dispatch."""

    def test_torch_availability(self) -> None:
        """Verify TORCH_AVAILABLE flag mirrors module state."""
        self.assertEqual(DeviceManager.is_torch_available(), TORCH_AVAILABLE)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for device tests")
    def test_auto_device_selection(self) -> None:
        """Auto device selection should return CUDA, MPS, or CPU."""
        device = get_device(preferred="auto")
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ("cuda", "mps", "cpu"))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for device tests")
    def test_explicit_cpu_selection(self) -> None:
        """Requesting CPU should always return CPU device."""
        device = get_device(preferred="cpu")
        self.assertEqual(device.type, "cpu")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for device tests")
    def test_diagnostic_info_structure(self) -> None:
        """get_device_info should return a rich diagnostic dictionary."""
        info = DeviceManager.get_device_info()
        self.assertTrue(info["torch_available"])
        self.assertIn("torch_version", info)
        self.assertIn("selected_device", info)
        self.assertIn("device_type", info)
        self.assertIn("os", info)
        self.assertIn("machine", info)
        self.assertIn("cuda_available", info)
        self.assertIn("mps_available", info)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for device tests")
    def test_tensor_allocation_and_math(self) -> None:
        """Ensure tensors can be allocated and computed on selected hardware."""
        device = get_device()
        t1 = torch.tensor([1.0, 2.0, 3.0], device=device)
        t2 = torch.tensor([4.0, 5.0, 6.0], device=device)
        res = t1 + t2
        self.assertEqual(res.device.type, device.type)
        self.assertEqual(res.tolist(), [5.0, 7.0, 9.0])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for device tests")
    def test_empty_cache_execution(self) -> None:
        """empty_cache should execute safely without crashing."""
        device = get_device()
        # Should not raise any exceptions
        DeviceManager.empty_cache(device)


if __name__ == "__main__":
    unittest.main()
