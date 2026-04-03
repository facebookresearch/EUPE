import unittest

import torch

from eupe.eval.segmentation.inference import slide_inference


class DummySegmentationModel:
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.calls = 0

    def predict(self, x, rescale_to=(512, 512)):
        self.calls += 1
        batch_size, _, height, width = x.shape
        return torch.full(
            (batch_size, 1, height, width),
            fill_value=self.scale * self.calls,
            dtype=x.dtype,
            device=x.device,
        )


class SlideInferenceTests(unittest.TestCase):
    def test_accumulates_overlapping_windows_without_padding_roundtrip(self):
        model = DummySegmentationModel()
        inputs = torch.zeros(1, 3, 4, 4)

        preds = slide_inference(
            inputs,
            model,
            n_output_channels=1,
            crop_size=(3, 3),
            stride=(2, 2),
        )

        expected = torch.tensor(
            [
                [
                    [
                        [1.0, 1.5, 1.5, 2.0],
                        [2.0, 2.5, 2.5, 3.0],
                        [2.0, 2.5, 2.5, 3.0],
                        [3.0, 3.5, 3.5, 4.0],
                    ]
                ]
            ]
        )
        self.assertTrue(torch.equal(preds.cpu(), expected))

    def test_returns_predictions_on_input_device(self):
        model = DummySegmentationModel(scale=2.0)
        inputs = torch.zeros(1, 3, 3, 3)

        preds = slide_inference(
            inputs,
            model,
            n_output_channels=1,
            crop_size=(2, 2),
            stride=(1, 1),
        )

        self.assertEqual(preds.device, inputs.device)


if __name__ == "__main__":
    unittest.main()
