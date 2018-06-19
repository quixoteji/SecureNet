import numpy as np
import requests
import json


# a = np.random.rand(2,2).tolist().tostring()
# b = np.random.rand(2,2).tolist().tostring()
# 
a = np.random.rand(2,2)
b = np.random.rand(2,2)
c = np.matmul(a,b)
print(c)

a = a.tolist()
b = b.tolist()
a = json.dumps(a)
b = json.dumps(b)


user_info = {'a': a, 'b': b}


# r = requests.post("http://127.0.0.1:5000/test", data=user_info)
c = requests.get("http://127.0.0.1:5000/mlp")
print(c.text)
# j = np.asarray(json.loads(r.text))

# print(typec(j))
# print(j)