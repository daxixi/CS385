from torch import nn
from torch import optim
import torch
import random

class LSTMCell(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hc):
        hx, cx = self.lstm(input, hc)
        output = hx.squeeze(1)
        output = self.classifier(output)
        prob = self.softmax(output)
        return prob, (hx, cx)

class Controller():
    def __init__(self, sample_size, input_size=8, output_size=8, hidden_size=100):
        super(Controller, self).__init__()
        self.model = LSTMCell(input_size, output_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.baseline = None
        self.baseline_decay = 0.99

        # memory
        self.memory_size = 500
        self.sample_size = sample_size
        self.memory_idx = 0
        self.memory_reward = [None for _ in range(self.memory_size)]
        self.memory_probs = [None for _ in range(self.memory_size)]
        self.memory_actions = [None for _ in range(self.memory_size)]

    def _rollout(self):
        action = torch.zeros(1, self.input_size)
        hx = torch.zeros(1, self.hidden_size)
        cx = torch.zeros(1, self.hidden_size)
        num_edges = 14
        actions = []
        probs = []
        for i in range(2*num_edges):
            prob, (hx, cx) = self.model(action, (hx, cx))
            m = torch.distributions.Categorical(prob)
            action = torch.zeros(1,self.input_size)
            action[0][m.sample()] = 1
            probs.append(prob.squeeze(0))
            actions.append(action.squeeze(0))
        probs = torch.stack(probs,dim=0)
        actions = torch.stack(actions,dim=0)
        return probs, actions

    def rollout(self):
        probs, actions = self._rollout()
        mask_normal = actions[0:14,]
        mask_reduce = actions[14:28,]
        return mask_normal, mask_reduce, probs, actions

    def _update(self, reward, probs, actions):
        self.optimizer.zero_grad()
        num_edges = 14
        loss = 0
        for i in range(2*num_edges):
            action = actions[i].unsqueeze(0)
            prob = probs[i].unsqueeze(0)
            m = torch.distributions.Categorical(prob)
            loss -= torch.mean(m.log_prob(action)) * reward
        loss.backward(retain_graph=True)

        self.optimizer.step()

    def _compute_reward(self, accuracy_list):
        # scale accuracy to 0-1
        avg_acc = torch.mean(torch.Tensor(accuracy_list)) / 100
        if self.baseline is None:
            self.baseline = avg_acc
        else:
            self.baseline += self.baseline_decay * (avg_acc - self.baseline)
        # reward = accuracy - baseline
        return [accuracy_list[i] / 100 - self.baseline for i in range(len(accuracy_list))]

    # update with memory replay
    def update(self, accuracy_list, probs_list, actions_list):
        # push experiences into memory
        reward_list = self._compute_reward(accuracy_list)
        self._push_memory(self.memory_reward,self.memory_idx,reward_list)
        self._push_memory(self.memory_probs, self.memory_idx, probs_list)
        self._push_memory(self.memory_actions, self.memory_idx, actions_list)
        self.memory_idx += len(accuracy_list)
        if self.memory_idx == self.memory_size:
           self.memory_idx = 0

        # sample experiences from memory, update controller
        reward_list, probs_list, actions_list = self._get_memory()
        for i in range(len(accuracy_list)):
            reward = reward_list[i]
            probs = probs_list[i]
            actions = actions_list[i]
            self._update(reward, probs, actions)

    def _push_memory(self,memory,start_idx,data):
        idx = start_idx
        for i in range(len(data)):
            if idx == self.memory_size:
                idx = 0
            memory[idx] = data[i]
            idx += 1

    def _get_memory(self):
        pool = []
        for i in range(self.memory_size):
            if self.memory_reward[i] is not None:
                pool.append(i)
        pos = random.sample(pool,self.sample_size)
        reward_list = []
        probs_list = []
        actions_list = []
        for p in pos:
            reward_list.append(self.memory_reward[p])
            probs_list.append(self.memory_probs[p])
            actions_list.append(self.memory_actions[p])
        return reward_list, probs_list, actions_list

    # update without memory
    def mc_update(self, accuracy_list, probs_list, actions_list):
        reward_list = self._compute_reward(accuracy_list)
        for i in range(len(accuracy_list)):
            reward = reward_list[i]
            probs = probs_list[i]
            actions = actions_list[i]
            self._update(reward, probs, actions)

if __name__ == '__main__':
    from time import time
    start = time()
    input_size = 8
    output_size = 8
    hidden_size = 100
    controller = Controller(input_size,output_size,hidden_size)

    mask_normal, mask_reduce, probs, actions = controller.rollout()
    print(mask_normal)
    print(mask_reduce)
    print(probs)
    print(probs.shape)
    print(actions)
    print(actions.shape)

    reward = 0.9
    controller._update(reward,probs,actions)
    print(time()-start)