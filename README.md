## 注意事项

如果是有共享内存和统一架构，需要改成cudaMallocManaged()或者cudaHostAlloc()

内存不够分配命令：

```