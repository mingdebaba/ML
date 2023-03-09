```python
graph = {
    'A':['B','C'],
    'B':['A','C','D'],
    'C':['A','B','D','E'],
    'D':['B','C','E','F'],
    'E':['C','D'],
    'F':['D']
}
```


```python
def DFS_2(graph, s):  # graph是图，s是开始结点
    stack = []        # 栈
    stack.append(s)   # 开始结点入栈
    v = set()
    v.add(s)          # 无序添加
    # print(s, end=' ')
    flag = 0          # 标记
    while len(stack) > 0:       # 栈非空
        flag = 0
        vertex = stack[-1]      # 查看尾元素
        nodes = graph[vertex]   # 访问结点相连的结点列表
        for w in nodes:
            if w not in v:
                stack.append(w) # 未被访问的相连的下一个结点入栈
                v.add(w)        # 标记已访问
                flag = 1        # 存在未被访问的相连结点
                print(w, end=' ')
                print(vertex + '->' + w)
                break
        if flag == 0:           # 不存在未被访问的相连结点，回溯
            stack.pop()
```


```python
DFS_2(graph, 'E')
```

    C E->C
    A C->A
    B A->B
    D B->D
    F D->F
    


```python
def BFS(graph,s):#graph图  s指的是开始结点
    #需要一个队列
    queue=[]
    queue.append(s)
    seen=set()#看是否访问过该结点
    seen.add(s)
    while (len(queue)>0):
        vertex=queue.pop(0)#保存第一结点，并弹出，方便把他下面的子节点接入
        nodes=graph[vertex]#子节点的数组
        for w in nodes:
            if w not in seen:#判断是否访问过，使用一个数组
                queue.append(w)
                seen.add(w)
        print(vertex)
```


```python
BFS(graph,'B')
```

    B
    A
    C
    D
    E
    F
    


```python
#Confusion 
```
