直接运行dist/pagerank.exe即可。需要保证Data.txt在运行时的同目录下，建议在./dist中运行exe文件



超参数固定为：teleport number = 0.85 | epsilon = 1e-8 | block_size=100

可执行文件与直接运行python文件效果相同，如果需要可执行文件中调整超参数需要重新编译

```
# pip install pyinstaller
pyinstaller -F pagerank.py
```

