import random
from collections import deque
from typing import List
import numpy as np
import stock_exchange
from experts.obscure_expert import ObscureExpert
from framework.vote import Vote
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger
from framework.stock_data import StockData
from framework.stock_market_data import StockMarketData

# assure stable results
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(69)




class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading

        # Parameters for neural network
        self.state_size = 2
        self.action_size = 4  # 10
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action = None
        self.last_action_a = None
        self.last_action_b = None
        self.last_portfolio_value = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu',
                                 batch_size=self.batch_size))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        # self.model.summary()
        self.vote2num = {Vote.BUY: 1, Vote.SELL: 2, Vote.HOLD: 3}
        #self.actions = [[Vote.BUY, Vote.BUY],
        #                [Vote.BUY, Vote.SELL],
         #               [Vote.BUY, Vote.HOLD],
         #               [Vote.SELL, Vote.BUY],
          #              [Vote.SELL, Vote.SELL],
           #             [Vote.SELL, Vote.HOLD],
            #            [Vote.HOLD, Vote.BUY],
             #           [Vote.HOLD, Vote.SELL],
              #          [Vote.HOLD, Vote.HOLD]]
        self.actions = [[Vote.BUY, Vote.BUY],
                        [Vote.BUY, Vote.SELL],
                        [Vote.SELL, Vote.BUY],
                        [Vote.SELL, Vote.SELL]]

    def save_trained_model(self, epsiode):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def _remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def _act(self, state, stock_market_data, portfolio, order_list):
        if self.train_while_trading:
            #if False:
            if np.random.rand() <= self.epsilon:
                #print('random action performed, epsilon = ' + str(self.epsilon))
                action = random.randrange(self.action_size)
            else:
                try:
                    if state.ndim == 1:
                        state = np.array([state])
                    act_values = self.model.predict(state)
                except ValueError:
                    print(state)

                action = np.argmax(act_values[0])  # returns action
        else:
            try:
                if state.ndim == 1:
                    state = np.array([state])
                act_values = self.model.predict(state)
            except ValueError:
                print(state)

            action = np.argmax(act_values[0])  # returns action
        company_list = stock_market_data.get_companies()
        vote_a = self.actions[action][0]
        vote_b = self.actions[action][1]

        for company in company_list:
            if company == Company.A:
                stock_data_a = stock_market_data[Company.A]
                self.__follow_action(Company.A, stock_data_a, vote_a, portfolio, order_list)
            elif company == Company.B:
                stock_data_b = stock_market_data[Company.B]
                self.__follow_action(Company.B, stock_data_b, vote_b, portfolio, order_list)
            else:
                assert False
        return action

    def _replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        #print('learn')
        # for state, action, reward, next_state in minibatch:
        #     target = reward
        #     if next_state.ndim == 1:
        #         next_state = np.array([next_state])
        #     #target = reward + self.learning_rate * \
        #     #         np.amax(self.model.predict(next_state)[0])
        #
        #     if state.ndim == 1:
        #         state = np.array([state])
        #     target_f = self.model.predict(state, batch_size=self.batch_size)
        #     target_f[0][action] = target
        #     self.model.fit(state, target_f, epochs=1, verbose=0)
        state = np.array(minibatch)[:,0]
        action = np.array(minibatch)[:,1]
        reward = np.array(minibatch)[:,2]
        next_state = np.array(minibatch)[:,3]

        #if next_state.ndim == 1:
        #    next_state = np.array([next_state])
        #for idx, rew in enumerate(reward):
        #    reward[idx] = rew + 0.1 * \
        #         np.amax(self.model.predict(np.array([next_state[idx]])))
        state_processed = np.array([])
        state = [np.append(state_processed, st) for st in state]
        state = np.array(state)
        if state.ndim == 1:
            state = np.array([state])
        target_f = self.model.predict(state, batch_size=self.batch_size)
        for idx, tar in enumerate(target_f):
            tar[action[idx]] = reward[idx]
        self.model.fit(state, target_f, epochs=1, verbose=0, batch_size=self.batch_size)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def __follow_action(self, company: Company, stock_data: StockData, vote: Vote, portfolio: Portfolio,
                             order_list: List[Order]):
        assert company is not None
        assert stock_data is not None
        assert vote is not None
        assert portfolio is not None
        assert order_list is not None

        if vote == Vote.BUY:
            # buy as many stocks as possible
            stock_price = stock_data.get_last()[-1]
            amount_to_buy = int(portfolio.cash // stock_price)
            logger.debug(f"{self.get_name()}: Got vote to buy {company}: {amount_to_buy} shares a {stock_price}")
            if amount_to_buy > 0:
                order_list.append(Order(OrderType.BUY, company, amount_to_buy))
        elif vote == Vote.SELL:
            # sell as many stocks as possible
            amount_to_sell = portfolio.get_stock(company)
            logger.debug(f"{self.get_name()}: Got vote to sell {company}: {amount_to_sell} shares available")
            if amount_to_sell > 0:
                order_list.append(Order(OrderType.SELL, company, amount_to_sell))
        else:
            # do nothing
            assert vote == Vote.HOLD
            logger.debug(f"{self.get_name()}: Got vote to hold {company}")

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"
    
        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]

        # TODO Compute the current state
        #print(stock_market_data[Company.A].get_dates()[-1])

        order_list = []

        vote_a = self.expert_a.vote(stock_market_data[Company.A])
        vote_b = self.expert_b.vote(stock_market_data[Company.B])

        # convert votes to state
        state = np.array([self.vote2num[vote_a], self.vote2num[vote_b]])

        # TODO Store state as experience (memory) and train the neural network only if trade() was called before at
        #  least once

        # TODO Create actions for current state and decrease epsilon for fewer random actions

        # TODO Save created state, actions and portfolio value for the next call of trade()

        # Decide action
        action = self._act(state, stock_market_data, portfolio, order_list)
        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        #next_state, reward, done, _ = env.step(action)
        #next_state = np.reshape(next_state, [1, 4])
        # Remember the previous state, action, reward, and done
        if self.last_portfolio_value is not None:
            if self.last_portfolio_value <= portfolio.get_value(stock_market_data):
                #reward = 2 - portfolio.get_value(stock_market_data) / self.last_portfolio_value
                reward = 1
            else:
                reward = 0
            self._remember(self.last_state, action, reward, state)
        self.last_portfolio_value = portfolio.get_value(stock_market_data)
        self.last_state = state

        # make next_state the new current state for the next frame.
        #state = next_state
        # done becomes True when the game ends
        # ex) The agent drops the pole
        #if done:
            # print the score and break out of the loop
        #    print("episode: {}/{}, score: {}"
        #         .format(e, episodes, time_t))
        #    break
        # train the agent with the experience of the episode
        if self.train_while_trading:
            if len(self.memory) > self.min_size_of_memory_before_training:
                self._replay()
        return order_list



# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 15
if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    #a=training_data.deepcopy_first_n_items(1002)
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(1000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)
    #training_trader.epsilon = training_trader.epsilon_min

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model(i)
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.subplot(211)
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])


    plt.subplot(212)
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['test'])

    plt.show()
