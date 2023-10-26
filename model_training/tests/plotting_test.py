"""Tests for plot_training_and_test_loss function"""
import unittest
from pathlib import Path

import pytest

from model_training.utils import plot_metric, plot_training_and_test_loss


class PlotTrainingAndTestLossTest(unittest.TestCase):
    """
    Tests for functions that create plots
    """
    def setUp(self):
        self.log_history = [{'loss': 3.6603, 'epoch': 1.08, 'step': 10},
                            {'eval_loss': 3.3250460624694824, 'epoch': 1.08, 'step': 10},
                            {'loss': 3.4554, 'epoch': 2.16, 'step': 20},
                            {'eval_loss': 3.1895878314971924, 'epoch': 2.16, 'step': 20},
                            {'loss': 3.5297, 'epoch': 3.24, 'step': 30},
                            {'eval_loss': 3.0954017639160156, 'epoch': 3.24, 'step': 30},
                            {'loss': 3.2921, 'epoch': 4.32, 'step': 40},
                            {'eval_loss': 3.0169196128845215, 'epoch': 4.32, 'step': 40},
                            {'loss': 3.1995, 'epoch': 5.41, 'step': 50},
                            {'eval_loss': 2.964205026626587,'epoch': 5.41, 'step': 50},
                            {'loss': 3.2171, 'epoch': 6.49, 'step': 60},
                            {'eval_loss': 2.9277913570404053, 'epoch': 6.49, 'step': 60},
                            {'loss': 3.158, 'epoch': 7.57, 'step': 70},
                            {'eval_loss': 2.9022018909454346, 'epoch': 7.57, 'step': 70},
                            {'loss': 3.198, 'epoch': 8.65, 'step': 80},
                            {'eval_loss': 2.8873517513275146, 'epoch': 8.65, 'step': 80},
                            {'loss': 3.153, 'epoch': 9.73, 'step': 90},
                            {'eval_loss': 2.882246255874634, 'epoch': 9.73, 'step': 90}]

        self.metric_log_history = [{"eval_rouge1": 0.81, 'epoch': 1, 'step': 10},
                                   {"eval_rouge1": 0.82, 'epoch': 2, 'step': 20},
                                   {"eval_rouge1": 0.83, 'epoch': 3, 'step': 30},
                                   {"eval_rouge1": 0.84, 'epoch': 4, 'step': 40},
                                   {"eval_rouge1": 0.85, 'epoch': 5, 'step': 50},
                                   {"eval_rouge1": 0.86, 'epoch': 6, 'step': 60},
                                   {"eval_rouge1": 0.87, 'epoch': 7, 'step': 70},
                                   {"eval_rouge1": 0.88, 'epoch': 8, 'step': 80},
                                   {"eval_rouge1": 0.89, 'epoch': 9, 'step': 90},
                                   {"eval_rouge1": 0.90, 'epoch': 10, 'step': 100}]

        self.tests_dir = Path(__file__).resolve().parent
        self.expected_loss_epoch_filepath = (self.tests_dir / "data" /
                                             "expected-loss-plot-epoch.png")
        self.expected_loss_step_filepath = (self.tests_dir / "data" /
                                            "expected-loss-plot-step.png")
        self.test_loss_epoch_filepath = self.tests_dir / "data" / "test-loss-plot-epoch.png"
        self.test_loss_step_filepath = self.tests_dir / "data" / "test-loss-plot-step.png"

        self.expected_metric_epoch_filepath = (self.tests_dir / "data" /
                                            "expected-rouge1-plot-epoch.png")
        self.expected_metric_step_filepath = (self.tests_dir / "data" /
                                            "expected-rouge1-plot-step.png")

        self.test_metric_epoch_filepath = (self.tests_dir / "data" /
                                           "test-rouge1-plot-epoch.png")
        self.test_metric_step_filepath = (self.tests_dir / "data" /
                                          "test-rouge1-plot-step.png")

        if self.test_loss_epoch_filepath.exists():
            self.test_loss_epoch_filepath.unlink()

        if self.test_loss_step_filepath.exists():
            self.test_loss_step_filepath.unlink()

        if self.test_metric_epoch_filepath.exists():
            self.test_metric_epoch_filepath.unlink()

        if self.test_metric_step_filepath.exists():
            self.test_metric_step_filepath.unlink()

    @pytest.mark.model_training
    def test_plot_training_and_test_loss_with_epochs_ideal(self):
        """
        Plot_training_and_test_loss should create a png file with the correct plot
        """
        plot_training_and_test_loss(self.log_history, self.test_loss_epoch_filepath)

        self.assertTrue(self.test_loss_epoch_filepath.exists())

        with open(self.test_loss_epoch_filepath, "rb") as file:
            actual = file.read()

        with open(self.expected_loss_epoch_filepath, "rb") as file:
            expected = file.read()

        self.assertEqual(expected, actual)

    @pytest.mark.model_training
    def test_plot_training_and_test_loss_with_steps_ideal(self):
        """
        Plot_training_and_test_loss should create a png file with the correct plot
        """
        plot_training_and_test_loss(self.log_history, self.test_loss_step_filepath,
                                    plot_epochs=False)

        self.assertTrue(self.test_loss_step_filepath.exists())

        with open(self.test_loss_step_filepath, "rb") as file:
            actual = file.read()

        with open(self.expected_loss_step_filepath, "rb") as file:
            expected = file.read()

        self.assertEqual(expected, actual)

    @pytest.mark.model_training
    def test_plot_metric_with_epochs_ideal(self):
        """
        Plot_metric should create a png file with the correct plot
        """
        plot_metric("eval_rouge1", self.metric_log_history,
                    self.test_metric_epoch_filepath, plot_epochs=True)

        self.assertTrue(self.test_metric_epoch_filepath.exists())

        with open(self.test_metric_epoch_filepath, "rb") as file:
            actual = file.read()

        with open(self.expected_metric_epoch_filepath, "rb") as file:
            expected = file.read()

        self.assertEqual(expected, actual)

    @pytest.mark.model_training
    def test_plot_metric_with_steps_ideal(self):
        """
        Plot_metric should create a png file with the correct plot
        """
        plot_metric("eval_rouge1", self.metric_log_history,
                    self.test_metric_step_filepath, plot_epochs=False)

        self.assertTrue(self.test_metric_step_filepath.exists())

        with open(self.test_metric_step_filepath, "rb") as file:
            actual = file.read()

        with open(self.expected_metric_step_filepath, "rb") as file:
            expected = file.read()

        self.assertEqual(expected, actual)

    def tearDown(self):
        if self.test_loss_epoch_filepath.exists():
            self.test_loss_epoch_filepath.unlink()

        if self.test_loss_step_filepath.exists():
            self.test_loss_step_filepath.unlink()

        if self.test_metric_epoch_filepath.exists():
            self.test_metric_epoch_filepath.unlink()

        if self.test_metric_step_filepath.exists():
            self.test_metric_step_filepath.unlink()
