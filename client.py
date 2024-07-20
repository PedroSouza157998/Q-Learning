import connection as cn
import numpy as np

socket = cn.connect(2037)

actions = ["left", "right", "jump"]

epochs = 1500
learning_rate = 0.3
discount_factor = 0.9
epsilon = 0.2 

num_states = 100

try:
    q_table = np.loadtxt('resultado.txt')
    print("Q Table carregada do arquivo resultado.txt")
except OSError:
    q_table = np.zeros((96, 3))
    print("Arquivo resultado.txt não encontrado, iniciando Q Table com zeros")

def get_q(state, action):
    action_number = actions.index(action)
    return q_table[state, action_number]

def update_q(state, action, reward, next_state):
    action_number = actions.index(action)
    best_next_action = np.max(q_table[next_state])
    q_table[state, action_number] = (1 - learning_rate) * q_table[state, action_number] + learning_rate * (reward + discount_factor * best_next_action)


for epoch in range(epochs):
    print("Época: ",epoch)
    state = int('0000000', 2)
    done = False
    last_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action_index = np.random.randint(3)
        else:
            q_values = [get_q(state, action) for action in actions]
            action_index = np.argmax(q_values)

        action = actions[action_index]

        next_state, reward = cn.get_state_reward(socket, action)

        if(last_reward == reward): reward = reward*2
        else: last_reward = reward
        next_state = int(next_state[2:], 2)

        update_q(state, action, reward, next_state)

        state = next_state
        if(reward == -100): done = True
        if(reward > -2): done= True

np.savetxt('resultado.txt', q_table, fmt='%f')

print("Treinamento concluído")