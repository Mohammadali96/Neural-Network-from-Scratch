# NN from Scratch
![image](https://github.com/user-attachments/assets/713dfc4c-77f3-4e95-b4be-c8069cb0ff73)

Input layer (0): 〖input:  x〗_i^0͵  〖output:  y_i^0=x〗_i^0  
Middle layer (1): weights: w_ji^1͵  〖input:  x〗_j^1=∑_(i=0)^(n_0)▒〖w_ij y_i 〗͵  〖output:  y_i^1=f(x〗_j^1);  j=neurons in current layer͵  i= neurons in pervious layer 
Output layer (2): weights: w_kj^2͵  〖input:  x〗_k^2=∑_(i=0)^(n_1)▒〖w_jk y_j 〗͵  〖output:  y_j^2=f(x〗_k^2);  k=neurons in current layer͵  j= neurons in pervious layer
