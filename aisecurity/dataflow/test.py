ids = {}
for i in range(100000):
	name = makeRec(str(i)).names
	if "Invalid ID" in names: continue
	ids[name] = i
	print(name)
