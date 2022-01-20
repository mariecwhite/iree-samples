# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import numpy
import test_util

model_path = "https://storage.googleapis.com/iree-model-artifacts/ssd_mobilenet_v1_320_1.0_uint8.tflite"

class SsdMobilenetV1Uint8Test(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(SsdMobilenetV1Uint8Test, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(SsdMobilenetV1Uint8Test, self).compare_results(iree_results, tflite_results, details)
    for i in range(len(iree_results)):
      self.assertTrue(numpy.isclose(iree_results[i], tflite_results[i], atol=1.0).all())

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()
