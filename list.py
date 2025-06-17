my_list = []
my_list.extend([10, 20, 30, 40])
my_list.insert(1, 15)
my_list.extend([50, 60, 70])
my_list.pop(-1)
my_list.sort(reverse=True)
index_30 = my_list.index(30)
print(my_list)
print("index of 30:", index_30)
