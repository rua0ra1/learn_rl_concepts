v_pre =0
v_new=0
reward=-1

for _ in range(1,2):
    v_new=(0.75*(reward+v_pre))+(0.25*(reward-20))
    v_pre=v_new
    print(v_new)


#final assignment2 sample code
import numpy as np
num_spaces = 3
num_prices = 3

V = np.zeros(num_spaces + 1)
pi = np.ones((num_spaces + 1, num_prices)) / num_prices

print(pi)

for s, pi_s in enumerate(pi):
    print("s: ", s, " pi_s: ",pi_s)
    for a, p in enumerate(pi_s):
        print(f'pi(A={a}|S={s}) = {p.round(2)}    ', end='')
    print()

