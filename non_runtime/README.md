# Non-Runtime Materials

这个目录用于集中存放不参与主体 Web 服务启动和前端构建的辅助材料，避免散落在项目根目录。

## 目录

- `evaluation_sets/`：检索评估题集、查新测试集。
- `annotation_records/`：检索标注记录，Web 后端会继续把新的标注记录写到这里。
- `notes/`：交接说明、设计 QA、临时报告和其他过程文档。
- `design/`：线框图等设计过程文件。
- `tests/`：回归测试代码。

运行测试：

```powershell
python -m pytest non_runtime/tests -q
```
