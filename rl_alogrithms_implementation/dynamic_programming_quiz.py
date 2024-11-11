v_pre =0
v_new=0
reward=-1

for _ in range(1,100):
    v_new=(0.75*(reward+v_pre))+(0.25*(reward-20))
    v_pre=v_new
    print(v_new)
