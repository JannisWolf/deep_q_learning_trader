from unittest import TestCase
from experts.obscure_expert import ObscureExpert
from framework.company import Company
from traders.deep_q_learning_trader import DeepQLearningTrader


class TestDeepQLearningTrader(TestCase):
    def test_create_bah_trader(self):
        expert_a = ObscureExpert(Company.A)
        expert_b = ObscureExpert(Company.B)
        trader = DeepQLearningTrader(expert_a, expert_b, False, False, 'test_color', 'test_name')
        self.assertIsNotNone(trader)
        self.assertEqual(trader.get_color(), 'test_color')
        self.assertEqual(trader.get_name(), 'test_name')
