import data as d
import train as t
import show as s


print("\rDATA:")
for i in d.train:
    print(i)

print("BEFORE TRAIN")
s.showResult()

t.training()

print("AFTER TRAIN")
s.showResult()

